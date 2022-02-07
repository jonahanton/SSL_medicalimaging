import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from models.byol_base import BYOLBase

from tqdm import tqdm
import logging
import os
import math


class EMA():
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.tau_base = 0.996

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.tau + (1 - self.tau) * new

    def tau_decay(self, k, K):
        self.tau = 1 - (1-self.tau_base) * (math.cos(math.pi * k/K) + 1)/2


class BYOLTrainer:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()

        # used for testing
        self.losses = []
        self.moving_average = []
        self.taus = []

        # the output_dim should be a hyperparameter (change parser)
        self.model = BYOLBase(self.args.arch).to(self.args.device) # online net

        self.target_net = BYOLBase(self.args.arch, is_target= True).to(self.args.device) # target net

        self.ema_updater = EMA(tau = 0)

        self.update_moving_average(self.ema_updater, ma_model = self.target_net, current_model = self.model)
        # self.target_net.load_state_dict(self.model.state_dict())

        self.ema_updater.tau = 0.996 # Implement decay

        # Stop forward and back propagation
        for p in self.target_net.parameters():
            p.requires_grad = False

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def criterion(self, q_online, z_target):
        """
        Add in doc strings
        Equation (2) in BYOL paper
        """
        q_online = F.normalize(q_online, dim=-1, p=2)
        z_target = F.normalize(z_target, dim=-1, p=2)
        return 2 - 2 * (q_online * z_target).sum(dim=-1)


    def update_moving_average(self, ema_updater, ma_model, current_model):

        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):

            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = ema_updater.update_average(old_weight, up_weight)


    def train(self, train_loader):

        n_iterations = 0
        logging.info(f"Starting BYOL training for {self.args.epochs} epochs.")

        self.taus.append(self.ema_updater.tau)

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for batch_idx, batch in enumerate(tqdm(train_loader)):

                (x1, x2), _ = batch
                x1 = x1.to(self.args.device)
                x2 = x2.to(self.args.device)

                # forward pass
                q_online = self.model(x1)
                z_target = self.target_net(x2)

                symmetric_q_online = self.model(x2)
                symmetric_z_target = self.target_net(x1)

                # loss
                loss = self.criterion(q_online, z_target)
                symmetric_loss = self.criterion(symmetric_q_online, symmetric_z_target)
                byol_loss = loss + symmetric_loss

                # backprop
                self.optimizer.zero_grad()
                byol_loss.mean().backward()
                self.optimizer.step()

                n_iterations += 1
                running_loss += byol_loss.mean().item()

                # Update weights for target net
                num_batches = len(train_loader)
                self.ema_updater.tau_decay(k = epoch * num_batches + batch_idx, K = num_batches * self.args.epochs)
                self.update_moving_average(self.ema_updater, self.target_net, self.model)
                self.taus.append(self.ema_updater.tau)

            # Scheduler for optimiser - e.g. cosine annealing
            if epoch >= 10:
                self.scheduler.step()

            training_loss = running_loss/len(train_loader)
            self.losses.append(training_loss)

            print("Train Loss:", training_loss)
            logging.debug(f"Epoch: {epoch}\tLoss: {training_loss}")

        logging.info("Finished training.")

        # Save model
        checkpoint_name = f'ssl_{self.args.method}_{self.args.arch}_{self.args.dataset_name}_trained_model.pth.tar'
        checkpoint_filepath = os.path.join(self.args.outpath, checkpoint_name)
        torch.save(
                {
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_filepath)

        logging.info(f"BYOL Model has been saved in directory {self.args.outpath}.")

if __name__ == "__main__":
    pass
