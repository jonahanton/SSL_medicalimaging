import torch
import torch.nn as nn

from models.simclr_base import SimCLRBase
from models.byol_base import BYOLOnlineBase

from tqdm import tqdm
import os

class DownstreamModel:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']

         # Load in trained network
        if self.args.pre_train_method == "simclr":
            self.model = SimCLRBase(self.args.arch)
        elif self.args.pre_train_method == "byol":
            self.model = BYOLOnlineBase(self.args.arch)


    def load(self):

        print("Loading model")
        
        # load in weights from pretrained model
        checkpoint_filepath = self.args.pretrained_path
        checkpoint = torch.load(checkpoint_filepath)
        state_dict = checkpoint['model_state_dict']

        print(state_dict)
        for k in list(state_dict.keys()):
    
            print(k)

            if k.startswith('backbone.'):
                print(k)
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    print(k)
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        print(state_dict)

        self.model.load_state_dict(state_dict)

        # # If not finetuning (only adding a linear classifier), freeze weights
        # if not self.args.finetune:
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        
        # # Add linear layer 
        # self.linear = nn.Linear(self.model.backbone.fc.out_features, self.args.num_classes)
    

    # def forward(self, x):

    #     # Don't pass through projector
    #     out = self.model.backbone(x)
    #     out = self.linear(out)
    #     return out
