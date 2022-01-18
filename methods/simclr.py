import argparse
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from loss.contrastive_loss import ContrastiveLoss
from models.simple_net import SimpleNet

class SimCLR(nn.Module):

    def __init__(
        self,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        classifier_lr: float,
        temperature: float, 
        proj_output_dim: int, 
        proj_hidden_dim: int, 
    ):

        super().__init__()


        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.classifier_lr = classifier_lr
        self.temperature = temperature

        # encoder
        self.encoder = SimpleNet()

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # classifier
        self.classifier = nn.Linear(self.features_dim, num_classes)

    
    def forward(self, X):
        """
        Args:
            X (torch.Tensor): a batch of images in the tensor format, size [N, C, H, W]
            
        """
        # batch size

        N = X.size(0)
        for k in range(N):
            x = X[k]
            x_1 = self._augment_image(x)
            h_1 = self.encoder(x_1)
            z_1 = self.projector(h_1)

            x_2 = self._augment_image(x)
            h_2 = self.encoder(x_2)
            z_2 = self.encoder(h_2)

    
    
    def _augment_image(self, x):

        # randoming cropping --> resize back to the original size
        x_area = x.size(-2)*x.size(-1)
        random_crop_area = random.uniform(0.08, 1.0)
        crop_size = int(random_crop_area*x_area)
        x_aug = x.transforms.RandomResizedCrop(size=crop_size)

        # random colour distortions
        color_jitter = transforms.ColorJitter() # add in color dropping and increase strength later
        # random Gaussian blur
        kernel_size = int(min(x.size(-2), x.size(-1)) * 0.1)
        sigma = random.uniform(0.1, 2.0)
        gaussian_blur = transforms.GaussianBlur(kernel_size = kernel_size, sigma = sigma)
        
        x_aug = transforms.RandomApply([color_jitter, gaussian_blur], p=0.5)
        
        return x_aug


def main():
    pass


if __name__ == "__main__":
    main()

