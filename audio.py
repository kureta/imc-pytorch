from soundfile import write as write_audio
import numpy as np
import librosa
from librosa import display
from pathlib import Path
from multiprocessing import Pool
import torch
import lightning as L


dir=Path("/home/kureta/Music/Chorale Samples/")


def prepare(fname):
    sound = librosa.load(fname, sr=44100, mono=True)[0]
    stft = librosa.stft(sound, n_fft=1024, hop_length=512, window='hann')
    spec = np.abs(stft)[1:]
    
    return librosa.amplitude_to_db(spec)


with Pool(12) as p:
    spectra = p.map(prepare, dir.glob("*.mp3"))


data = torch.from_numpy(np.concatenate(spectra, axis=1).T)


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.Tanh(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.ae = AE()

    def training_step(self, batch, _):
        pred = self.ae(batch)
        loss = torch.nn.functional.mse_loss(pred, batch)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        with torch.inference_mode():
            pred = self.ae(batch)

        result = np.zeros((513, 1024))
        result[1:, :] = pred.T.cpu().numpy()
        y = librosa.griffinlim(librosa.db_to_amplitude(result), n_iter=100, hop_length=512, n_fft=1024, window='hann')
        write_audio(f'test_{self.global_step}.wav', y, 44100)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class Spec(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = Spec(data)

loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=1024, shuffle=True, num_workers=8, prefetch_factor=64, persistent_workers=True
)

# Model Initialization
model = AutoEncoder()


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(max_epochs=250, limit_val_batches=1, accelerator="gpu", devices=1, profiler="simple")
trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
