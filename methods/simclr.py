import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl

from models.conv_net import ConvNet
from simclr_datatransform import SimCLREvalDataTransform, SimCLRTrainDataTransform
from data.mnist_dataloader import MNISTDataModule

class Flatten(nn.Module):

    def __init__(self):
        super().__unit__()

    def forward(self, x):
        return x.view(x.size(0), -1)
        

class ProjectionHead(nn.Module):
    """h --> z"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(self,
        batch_size,
        num_samples,
        loss_temperature=0.5, 
        warmup_epochs=10, 
        max_epochs=100,
        lars_lr=0.1,
        lars_eta=1e-3,
        opt_weight_decay=1e-6,
        **kwargs,
        ):

        super().__init__()
        self.save_hyperparameters()
        # e.g. self.hparams.batch_size

    
    def NT_Xent_loss(self, out_1, out_2, temperature):
        """
        Args:
            out_1: [batch_size, dim]
                Contains outputs through projection head of augment_1 inputs for the batch
            out_2: [batch_size, dim]
                Contains outputs through projection head of augment_2 inputs for the batch 
            
            e.g. out_1[0] and out_2[0] contain two different augmentations of the same input image

        Returns:
            loss : single-element torch.Tensor  

        """
        # concatenate 
        # e.g. out_1 = [x1, x2], out_2 = [y1, y2] --> [x1, x2, y1, y2]
        out = torch.cat([out_1, out_2], dim=0)
    
        n_samples = len(out)  # n_samples = 2N in SimCLR paper

        # similarity matrix
        cov = torch.mm(out, out.t())  # e.g. cov[0] = [x1.x1, x1.x2, x1.y1, x1.y2]
        sim = torch.exp(cov/temperature)

        # create mask to remove diagonal elements from sim matrix
        # mask has False in diagonals, True in off-diagonals
        mask = ~torch.eye(n_samples, device=sim.device).bool()  # e.g. diag(False, False, ..., False)

        # calculate denom in loss (SimCLR paper eq. (1)) for each z_i
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # loss computed across all positive pairs, both (i, j) and (j, i) in a mini-batch
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss
    
    def forward(self, X):
        """
        Args:
            X (torch.Tensor): a batch of images in the tensor format, size [N, C, H, W]
            
        """
        pass
    
    
    


def main():
    pass

    


if __name__ == "__main__":
    main()

