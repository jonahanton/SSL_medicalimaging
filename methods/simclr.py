import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pl_bolts.optimizers.lars import LARS


from simclr_datatransform import SimCLREvalDataTransform, SimCLRTrainDataTransform
from data.mnist_dataloader import MNISTDataModule

from models.conv_net import ConvNet
        

class ProjectionHead(nn.Module):
    """h --> z"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(self,
        batch_size,
        num_samples,
        loss_temperature=0.5, 
        warmup_epochs=10, 
        max_epochs=100,
        lr=0.1,
        opt_weight_decay=1e-6,
        **kwargs,
        ):

        super().__init__()
        self.save_hyperparameters()
        # e.g. self.hparams.batch_size

        # encoder f()
        self.encoder = ConvNet()

        # projection head g()
        self.projection_head = ProjectionHead()

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                                     self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.opt_weight_decay
                                    )

        # not implemented - SimCLR paper uses LARS wrapper
        # optimzer = LARS()
        # linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR()
        return optimizer

    
    def training_step(self, batch, batch_idx):
        
        loss = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.EvalResult(chekpoint_on=loss)
        result.log('avg_val_loss', loss)
        return result

    
    def shared_step(self, batch, batch_idx):
        (x1, x2), y = batch

        # encode: x --> h
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # project: h --> z
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        loss = self.NT_Xent_loss(z1, z2, self.hparams.loss_temperature)
        return loss

    
    def NT_Xent_loss(self, out_1, out_2, temperature):
        """
        Args:
            out_1: [batch_size, dim]
                Contains outputs through projection head of augment_1 inputs for the batch
            out_2: [batch_size, dim]
                Contains outputs through projection head of augment_2 inputs for the batch 
            
            e.g. out_1[0] and out_2[0] contain two different augmentations of the same input image

        Returns:
            loss : single-element torch.Tensor  

        """
        # concatenate 
        # e.g. out_1 = [x1, x2], out_2 = [y1, y2] --> [x1, x2, y1, y2]
        out = torch.cat([out_1, out_2], dim=0)
    
        n_samples = len(out)  # n_samples = 2N in SimCLR paper

        # similarity matrix
        cov = torch.mm(out, out.t())  # e.g. cov[0] = [x1.x1, x1.x2, x1.y1, x1.y2]
        sim = torch.exp(cov/temperature)

        # create mask to remove diagonal elements from sim matrix
        # mask has False in diagonals, True in off-diagonals
        mask = ~torch.eye(n_samples, device=sim.device).bool()  # e.g. diag(False, False, ..., False)

        # calculate denom in loss (SimCLR paper eq. (1)) for each z_i
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # loss computed across all positive pairs, both (i, j) and (j, i) in a mini-batch
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss    


def main():

    # Initial parameters 
    batch_size = 32 
    mnist_height = 28 # default 

    # init data
    dm = MNISTDataModule(batch_size=batch_size) 
    dm.train_transforms = SimCLRTrainDataTransform(mnist_height)
    dm.val_transforms = SimCLREvalDataTransform(mnist_height)

    # realize the data
    dm.prepare_data()
    dm.setup()

    train_samples = len(dm.train_dataloader())

    model = SimCLR(batch_size=batch_size, num_samples=train_samples)
    trainer = pl.Trainer(progress_bar_refresh_rate=10, gpus=0) 
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()

