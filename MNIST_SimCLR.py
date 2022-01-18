import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer


from models.simple_net import SimpleNet


class LitMNIST(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



if __name__ == "__main__":

    # prepare transforms standard to MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # data
    mnist_train = datasets.MNIST(os.getcwd(), train=True, download=False, transform=transform)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)


    model = LitMNIST()
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, mnist_train)
