import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from models.byol_base import BYOLBase, BYOLOnlineBase

from tqdm import tqdm
import logging
import os


class EMA():
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.tau + (1 - self.tau) * new


class BYOLTrainer:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()

        # the output_dim should be a hyperparameter (change parser)
        self.target_net = BYOLBase().to(self.args.device)  
        self.model = BYOLOnlineBase().to(self.args.device)  

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)


    def criterion(self, q_online, z_target):
        """
        Add in doc strings
        Equation  (2) in BYOL paper
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

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for batch in tqdm(train_loader):

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
                byol_loss.sum().backward()
                self.optimizer.step()

                n_iterations += 1
                running_loss += byol_loss.sum().item()

                # Update weights for target net
                # self.update_moving_average(ema_updater, self.target_net, self.model)

            # Scheduler for optimiser - e.g. cosine annealing
            if epoch >= 10:
                self.scheduler.step()

            training_loss = running_loss/len(train_loader)
            print("Train Loss:", training_loss)
            logging.debug(f"Epoch: {epoch}\tLoss: {training_loss}")



        logging.info("Finished training.")

        # Save model
        checkpoint_name = 'ssl_{self.args.dataset_name}_trained_model.pth.tar'
        checkpoint_filepath = os.path.join(self.args.outpath, checkpoint_name)
        torch.save(
                {
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_filepath)

        logging.info(f"Model has been saved at {self.args.outpath}.")

if __name__ == "__main__":
    pass
