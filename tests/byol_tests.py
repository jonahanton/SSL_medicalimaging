import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.byol import BYOLTrainer
from models.byol_base import BYOLBase
from torch.utils.data import DataLoader
from data.get_data import DatasetGetter

import math


arch_choices = ["ConvNet"]

parser = argparse.ArgumentParser()
parser.add_argument('--method', '-m', default='byol', help='type of ssl pretraining technique')
parser.add_argument('--data-path', default='./datasets', help='path to dataset')
parser.add_argument('--dataset-name', default='MNIST', help='dataset name')
parser.add_argument('-a', '--arch', default='ConvNet', choices=arch_choices)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--output-dim', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--n-views', type=int, default=2)
parser.add_argument('--outpath', default='saved_models')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--gpu-index', type=int, default=0)


def test_contrastive_loss(model):
    """ Contrastive loss should go down """
    assert model.losses.sorted()[::-1] == model.loss


def test_model_inputs():
    """ Model inputs should make sense """
    pass


def test_gradient_descent():
    """ Gradient descent should be done only on online net and not on target net """ 
    pass


def test_target_weights():
    """ Target net weights should not move during gradient descent (to save on computation) """
    pass


def test_moving_average():
    """ Target net should be a moving average of the online net"""
    pass

 
def test_random_initialization_in_dowstream_task():
    """ SSL model should be randomly initialised model in downstream task """
    pass


def test_tau_decay(model):
    """ Test code for tau decay """

    # check that the tau_base is 0.996 as in the paper
    assert model.taus[0] == 0.996

    # check that tau value reaches 1 at end of training
    # how to choose relative tolerance (i.e.maximum allowed difference between value a and b)?
    assert math.isclose(model.taus[-1], 1, rel_tol = 1e-5)

    # check that tau is updated at each training step
    total_number_training_steps = parser['epochs'] * len(pretrain_loader)
    assert len(model.taus) == total_number_training_steps

    # check that tau is INCREASED at each training step
    assert model.taus == model.taus.sorted()



if __name__ == "__main__":

    batch_size = 512

    args = parser.parse_args()
    args.lr = 0.3 * (args.batch_size / 256)
    args.weight_decay = 1e-6

    # load data
    pretrain_dataset = DatasetGetter(pretrain=True, train=True, args=args).load()
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.method == "byol":

        model = BYOLBase(arch=args.arch)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(pretrain_loader), eta_min=0, last_epoch=-1)

        with torch.cuda.device(args.gpu_index):
            byol = BYOLTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            byol.train(pretrain_loader)
            print(byol.taus)

    test_contrastive_loss(byol)
    test_model_inputs()
    test_gradient_descent()
    test_target_weights()
    test_moving_average()
    test_random_initialization_in_dowstream_task()
    test_tau_decay(byol)
