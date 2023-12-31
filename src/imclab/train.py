from pathlib import Path

import lightning as L  # noqa
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger  # noqa
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader

from src.imclab.data import Spec
from src.imclab.model import AutoEncoder
from src.imclab.preprocessing import audio_preprocess, image_preprocess, prepare

DEFAULT_DATA_DIR = "/home/kureta/Music/Chorale Samples/"


class LightningAutoEncoder(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = AutoEncoder()

    def training_step(self, batch, _):
        example = batch
        pred = self.model(example)
        loss = F.mse_loss(pred, example)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        example = batch
        with torch.inference_mode():
            pred = self.model(example)

        tensorboard: TensorBoardLogger = self.logger.experiment  # noqa
        tensorboard.add_image(f"spec_{batch_idx}", image_preprocess(pred), self.global_step)  # noqa

        y = audio_preprocess(pred)
        tensorboard.add_audio(f"sample_{batch_idx}", y, self.global_step, sample_rate=44100)  # noqa

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SpecLoader(L.LightningDataModule):
    def __init__(self, data_dir=DEFAULT_DATA_DIR, batch_size=1024, num_workers=8, prefetch_factor=64,
                 persistent_workers=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dataset = None

    def setup(self, stage: str) -> None:
        self.dataset = Spec(prepare(self.data_dir))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset[:self.batch_size], batch_size=self.batch_size, shuffle=False)


def main():
    cli = LightningCLI(LightningAutoEncoder, SpecLoader)  # noqa


if __name__ == "__main__":
    main()
