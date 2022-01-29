import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 


class SimCLRBase(nn.Module):

    def __init__(self, arch="resnet18"):
        super().__init__()

        self.backbones_dict = {
            "resnet18": torchvision.models.resnet18(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
        }
        
        try:
            self.backbone = self.backbones_dict[arch]
        except KeyError:
            print(f"Invalid architecture {arch}. Pleases input either 'resnet18' or 'resnet50'.")
            raise KeyError

        
        # add projection head
        dim_proj = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_proj, dim_proj), nn.ReLU(), self.backbone.fc)


    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    pass
