import torch
import torch.nn as nn
import torchvision

from utils import accuracy
from models.cnn_base import ConvNet

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging
import os

class DownstreamModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = kwargs['args']

        # Load in trained network
        # Note that loading in resnet automatically attaches linear layer on the end (with output dim = num_classes)
        self.models_dict = {
            "resnet18": torchvision.models.resnet18(pretrained=False, num_classes=self.args.num_classes),
            "resnet50": torchvision.models.resnet50(pretrained=False, num_classes=self.args.num_classes),
            "ConvNet": ConvNet()
        }

        try:
            self.model = self.models_dict[self.args.arch]
        except KeyError:
            print(f"Invalid architecture {self.argsarch}. Pleases input either 'resnet18', 'resnet50' or 'ConvNet'.")
            raise KeyError


    def load(self):

        print("Loading saved model...")

        checkpoint_filepath = f"./saved_models/ssl_{self.args.pre_train_method}_{self.args.arch}_{self.args.pretrain_dataset_name}_trained_model.pth.tar"

        # load in weights from pretrained model
        # checkpoint_filepath = self.args.pretrained_path
        checkpoint = torch.load(checkpoint_filepath)
        state_dict = checkpoint['model_state_dict']

        if self.args.pre_train_method == "simclr":

            formated_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.") and not k.startswith("backbone.fc"):
                    k = k.replace("backbone.", "")
                    formated_state_dict[k] = v
        
            self.model.load_state_dict(formated_state_dict, strict=False)
            print("Succesfully loaded saved model!")

            # If not finetuning (only adding a linear layer), freeze weights not in fc layer
            if not self.args.finetune:
                for name, param in self.model.named_parameters():
                    if name not in ['fc.weight', 'fc.bias']:
                        param.requires_grad = False

        
        # BYOL 
        elif self.args.pre_train_method == "byol":

            formated_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.") and not k.startswith("backbone.fc"):
                    k = k.replace("backbone.", "")
                    formated_state_dict[k] = v
        
            self.model.load_state_dict(formated_state_dict, strict=False)
            print("Succesfully loaded saved model!")

            # If not finetuning (only adding a linear layer), freeze weights not in fc layer
            if not self.args.finetune:
                for name, param in self.model.named_parameters():
                    if name not in ['fc.weight', 'fc.bias']:
                        param.requires_grad = False


    def forward(self, x):

        out = self.model(x)
        return out
    

    def train(self, optimizer, train_loader, test_loader):
        
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        logging.info(f"Starting downstream training for {self.args.epochs} epochs.")

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            
            running_loss = 0 # keep track of loss per epoch
            top1_train_acc = 0
            top1_test_acc = 0

            for x, y in tqdm(train_loader):
            
                x = x.to(self.args.device)
                y = y.to(self.args.device)


                # forward pass
                logits = self(x)
                loss = self.criterion(logits, y)

                acc = accuracy(logits, y)
                top1_train_acc += acc

                print(f"Loss: {loss}")
                print(f"Train accuracy: {acc}")

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Track accuracy on test set after every epoch
            for x, y in test_loader:

                x = x.to(self.args.device)
                y = y.to(self.args.device)

                with torch.no_grad():
                    logits = self(x)
                    acc = accuracy(logits, y)
                    print(f"Test accuracy: {acc}")

                    top1_test_acc += acc


            training_loss = running_loss/len(train_loader)
            top1_train_acc = top1_train_acc/len(train_loader)
            top1_test_acc = top1_test_acc/len(test_loader)
            # print("Train Loss:", training_loss)
            # print("Train Accuracy:", top1_train_acc)
            # print("Test Accuracy:", top1_test_acc)
            logging.info(f"Epoch: {epoch}\tLoss: {training_loss}")
            logging.info(f"Epoch: {epoch}\tAccuracy: {top1_train_acc}")
            logging.info(f"Epoch: {epoch}\tTest Accuracy: {top1_test_acc}")

        logging.info("Finished training.")       

        # Save model
        if self.args.finetune:
            checkpoint_name = f'ssl_{self.args.pre_train_method}_{self.args.dataset_name}_finetuned_model.pth.tar'
        else:
            checkpoint_name = f'ssl_{self.args.pre_train_method}_{self.args.dataset_name}_linear_model.pth.tar'
        checkpoint_filepath = os.path.join(self.args.outpath, checkpoint_name)

        torch.save( 
                {
                'epoch': self.args.epochs,
                'arch': self.args.arch, 
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_filepath)

        logging.info(f"Model has been saved in directory {self.args.outpath}.")

if __name__ == "__main__":
    pass
