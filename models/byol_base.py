import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_net import ConvNet

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# BYOL class
class BYOLBase(nn.Module):

    def __init__(self, output_dim, arch='simple'):
        super().__init__()

        # encoder f()
        if arch == 'simple':
            self.encoder = ConvNet()

        # projector
        dim_projector= self.encoder.fc.out_features
        self.projector = MLP(dim = dim_projector, projection_size = output_dim)

    def forward(self, v):

        y = self.encoder(v)
        z = self.projector(y)
        return z

class BYOLOnlineBase(BYOLBase):
    def __init__(self, output_dim, arch='simple'):
        super().__init__(output_dim, arch)

        # predictor
        dim_predictor = output_dim
        self.predictor = MLP(dim = output_dim, projection_size = output_dim)

    def forward(self, v):
        y = self.encoder(v)
        z = self.projector(y)
        q = self.predictor(z)

        return q

if __name__ == "__main__":
    target = BYOLBase(output_dim=10)
    # print(target)
    # print(target.encoder)

    online = BYOLOnlineBase(output_dim=10)
    print(online.encoder)
    # print(online)