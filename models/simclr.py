import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

class SimCLR(BaseModel):

    def __init__(self):
        super().__init__()