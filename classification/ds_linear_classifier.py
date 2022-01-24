import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simclr_base import SimCLRBase

from tqdm import tqdm
import os

class DSLinearClassifier(nn.Module):

    def __init__(self, premodel=None, *args, **kwargs):
        super().__init__()
        self.args = kwargs['args']

        if premodel is None:
            # Load model
            self.premodel = SimCLRBase(arch=self.args.arch, output_dim=self.args.output_dim)
            checkpoint_name = 'ssl_{self.args.dataset_name}_trained_model.pth.tar'
            checkpoint_filepath = os.path.join(self.args.outpath, checkpoint_name)
            checkpoint = torch.load(checkpoint_filepath)
            self.premodel.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.premodel = premodel
            
        # freeze weights
        for param in self.premodel.parameters():
            param.requires_grad = False
        
        # Add linear layer 
        self.linearlayer = nn.Linear(self.premodel.encoder.fc.out_features, self.args.num_classes)


    def forward(self, x):
        # Don't pass through projection head
        out = self.premodel.encoder(x)
        out = self.linearlayer(out)
        return out

    
    def train(self, train_loader):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([params for params in self.parameters() if params.requires_grad], lr=1e-5)

        for epoch in range(30):
            print("Epoch:", epoch)

            running_loss = 0 
            for x, y in tqdm(train_loader):
                
                out = self(x)
                loss = criterion(out, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            training_loss = running_loss/len(train_loader)
            print("Train Loss:", training_loss)
    

    def test(self, test_loader):

        correct = 0
        total = 0 

        with torch.no_grad():
            for x, y in test_loader:

                out = self(x)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        print(f"ACCURACY on test set: {100 * correct/total} %")


        

if __name__ == "__main__":
    pass