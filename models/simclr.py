import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

class SimCLR(BaseModel):

    def __init__(self):
        super().__init__()

    
    def forward(self, X):
        return super().forward(X)

    
    def training_step(self, batch):
        return super().training_step(batch)


    def validation_step(self, batch):
        return super().validation_step(batch)

    
    def simclr_loss(self):
        pass