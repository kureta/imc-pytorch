import lightning as L
import torch
from torchvision import datasets
from torchvision.transforms import transforms


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building a linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )

        # Building a linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
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
        image = batch[0]
        image = image.reshape(-1, 28 * 28)
        pred = self.ae(image)
        loss = torch.nn.functional.mse_loss(pred, image)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, _):
        image = batch[0]
        image = image.reshape(-1, 28 * 28)
        pred = self.ae(image)

        pred = pred.reshape(-1, 1, 28, 28)

        for idx in range(3):
            self.logger.experiment.add_image(
                f"Image/{self.global_step}_{idx}", pred[idx]
            )  # type: ignore

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
# Download the MNIST Dataset
dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=tensor_transform
)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=32, shuffle=True, num_workers=8
)

# Model Initialization
model = AutoEncoder()

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(max_epochs=10, limit_val_batches=1, accelerator="gpu", devices=1)
trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
