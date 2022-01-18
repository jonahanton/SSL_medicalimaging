import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature


    def simclr_loss(self):
        pass
