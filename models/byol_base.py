import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.cnn_base import ConvNet


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


class BYOLBase(nn.Module):

    def __init__(self, arch, is_target=False):
        super().__init__()

        self.backbones_dict = {
            "resnet18": torchvision.models.resnet18(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
            "ConvNet": ConvNet()
        }

        try:
            self.backbone = self.backbones_dict[arch]
        except KeyError:
            print(f"Invalid architecture {arch}. Pleases input either 'resnet18' or 'resnet50'.")
            raise KeyError

        # add projection head
        dim_proj = self.backbone.fc.out_features
        self.projector = MLP(dim=dim_proj, projection_size=dim_proj)
        # self.backbone = nn.Sequential(self.backbone, self.projector)
        self.is_target = is_target

        # predictor, only used in the forward pass of the ONLINE model
        if not self.is_target:
            dim_pred = dim_proj
            self.predictor = MLP(dim=dim_pred, projection_size=dim_pred)

        # print(self.parameters)

    def forward(self, x):

        out = self.backbone(x)
        out = self.projector(out)

        if not self.is_target:
            out = self.predictor(x)

        return out


if __name__ == "__main__":
    pass