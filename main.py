import os
import setuptools

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 64

class MNISTModel(LightningModule):
    def __init__(self):
        width = 10000

        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, width)
        self.l2 = torch.nn.Linear(width, width)
        self.l3 = torch.nn.Linear(width, width)
        self.l4 = torch.nn.Linear(width, width)
        self.l5 = torch.nn.Linear(width, width)
        self.l6 = torch.nn.Linear(width, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = torch.relu(self.l6(x))
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def main():
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
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