import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

    
    def forward(self, X):
        pass


    def training_step(self, batch):
        pass

    
    def validation_step(self, batch):
        pass

