import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.cnn_base import ConvNet


class SimCLRBase(nn.Module):
    """SimCLR base class."""

    def __init__(self, arch="resnet18"):
        super().__init__()

        self.backbones_dict = {
            "resnet18": torchvision.models.resnet18(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
            "ConvNet": ConvNet()
        }
        
        try:
            self.backbone = self.backbones_dict[arch]
        except KeyError:
            print(f"Invalid architecture {arch}. Pleases input either 'resnet18','resnet50' or 'ConvNet'.")
            raise KeyError

        
        # add projection head
        dim_proj = self.backbone.fc.in_features
        # SimCLR projects the representation to a 128-dimensional latent space
        projector = nn.Sequential(
            nn.Linear(dim_proj, dim_proj),
            nn.ReLU(),
            nn.Linear(dim_proj, 128),
            )
        self.backbone.fc = projector


    def forward(self, x):

        out = self.backbone(x)
        return out


if __name__ == "__main__":
    pass
