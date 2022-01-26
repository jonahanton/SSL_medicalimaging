import argparse

import torch
from methods.byol import BYOLTrainer
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

import numpy as np
import os

from models.simclr_base import SimCLRBase
from models.byol_base import BYOLOnlineBase
from data.generate_views import GenerateViews
from methods.simclr import SimCLRTrainer
from data.get_data import DatasetGetter

# from classification.ds_linear_classifier import DSLinearClassifier


arch_choices = [name for name in models.__dict__
                      if name.islower() and not name.startswith("__")
                      and callable(models.__dict__[name])]
arch_choices.append('simple')

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='./datasets', help='path to dataset')
parser.add_argument('--dataset-name', default='MNIST', help='dataset name')
parser.add_argument('-a', '--arch', default='resnet18', choices=arch_choices)
parser.add_argument('--epochs', default=200, type=int, help='total number of epochs to train for')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--output-dim', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--n-views', type=int, default=2)
parser.add_argument('--outpath', default='saved_models')
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--gpu-index', type=int, default=0)
# parser.add_argument('-ds', '--downstream', action="store_false")

def main():

    args = parser.parse_args()

    # create output directory for pretrained model
    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # load data
    train_dataset = DatasetGetter(args).load()

    # Transformations for BYOL and SimCLR - add parser to this
    if args.dataset_name == 'MNIST':
        args.output_dim = 10
        args.num_classes = 10
        transform_pipeline = transforms.Compose([transforms.RandomResizedCrop(size=28),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        data_transforms = GenerateViews(transform_pipeline, args.n_views)
        train_dataset = datasets.MNIST(args.data_path, train=True, transform=data_transforms, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    # apply ssl pretraining
    #model = SimCLRBase(arch=args.arch, output_dim=args.output_dim)
    model = BYOLOnlineBase(arch=args.arch, output_dim=args.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    with torch.cuda.device(args.gpu_index):
        #simclr = SimCLRTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        #simclr.train(train_loader)
        byol = BYOLTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        byol.train(train_loader)


    # # use pretrained model for linear classification downstream task
    # base_transforms = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Normalize((0.1307,), (0.3081,))])
    # ds_dataset = datasets.MNIST(args.data_path, train=False, transform=base_transforms, download=True)
    # split = int(np.floor(0.2 * len(ds_dataset)))
    # ds_train_dataset, ds_test_dataset = random_split(ds_dataset, [len(ds_dataset) - split, split])
    # ds_train_loader = DataLoader(ds_train_dataset, batch_size=32, drop_last=True)
    # ds_test_loader = DataLoader(ds_test_dataset, batch_size=32, drop_last=True)
    # ds_linear_classifier = DSLinearClassifier(args=args)
    # ds_linear_classifier.train(ds_train_loader)
    # ds_linear_classifier.test(ds_test_loader)

if __name__ == "__main__":
    main()