import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging
import os

# Exponential Moving Average
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
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()

        # logging.basicConfig(level=logging.DEBUG)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def loss_fn(self, q_online, z_target):
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



