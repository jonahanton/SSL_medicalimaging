import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging
import os
import sys

class SimCLRTrainer:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        logging.basicConfig(level=logging.DEBUG)


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
        out = torch.cat([out_1, out_2], dim=0)
    
        n_samples = len(out)  # n_samples = 2N in SimCLR paper

        # similarity matrix
        cov = torch.mm(out, out.t())  # e.g. cov[0] = [x1.x1, x1.x2, x1.y1, x1.y2]
        sim = torch.exp(cov/temperature)

        # create mask to remove diagonal elements from sim matrix
        # mask has False in diagonals, True in off-diagonals
        mask = ~torch.eye(n_samples, device=sim.device).bool() 

        # calculate denom in loss (SimCLR paper eq. (1)) for each z_i
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # loss computed across all positive pairs, both (i, j) and (j, i) in a mini-batch
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss   

    
    def train(self, train_loader):

        n_iterations = 0
        logging.info(f"Starting SimCLR training for {self.args.epochs} epochs.")

        for epoch in range(self.args.epochs):
            for batch in tqdm(train_loader):
                
                (x1, x2), y = batch

                self.optimizer.zero_grad()
                # forward pass
                out_1 = self.model(x1)
                out_2 = self.model(x2)
                # loss
                loss = self.NT_Xent_loss(out_1, out_2, temperature=self.args.temperature)
                # backprop
                loss.backward()
                self.optimizer.step()

                n_iterations += 1

            if epoch >= 10:
                self.scheduler.step()

        logging.info("Finished training.")       


