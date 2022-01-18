import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

class SimCLR(pl.LightningModule):

    def __init__(self):
        super().__init__()

    
    def forward(self, X):
        pass

    
    def training_step(self, batch):
        pass


    def validation_step(self, batch):
        pass

    
    def simclr_loss(self):
        pass

    
    def configure_optimizers(self):
        pass