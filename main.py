import os
import setuptools

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
import segmentation_models_pytorch as smp

BATCH_SIZE = 16

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        aux_params=dict(
            pooling='avg',
            activation='softmax',
            classes=100,
        )
        self.model = smp.Unet(
            encoder_name="resnet50",
            in_channels=3,
            classes=100,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        _, y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    # Init our model
    mnist_model = Model()

    # Init DataLoader from MNIST Dataset
    train_ds = FakeData(size=10000, image_size=[3, 256, 256], num_classes=100, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        max_epochs=200,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    # Train the model
    trainer.fit(mnist_model, train_loader)

if __name__ == "__main__":
    main()