import argparse

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import numpy as np
import os

from models.simclr_base import SimCLRBase
from data.generate_views import GenerateViews
from methods.simclr import SimCLRTrainer
from classification.ds_linear_classifier import DSLinearClassifier


parser = argparse.ArgumentParser()
parser.add_argument('-data-path', default='./datasets', help='path to dataset')
parser.add_argument('-dataset-name', default='MNIST', help='dataset name')
parser.add_argument('--epochs', default=10, help='total number of epochs to train for')
parser.add_argument('--batch-size', default=256)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--weight-decay', default=1e-6)
parser.add_argument('--output-dim', default=10)
parser.add_argument('--temperature', default=0.5)
parser.add_argument('--n-views', default=2)
parser.add_argument('--arch', default='simple')
parser.add_argument('--outpath', default='saved_models')
parser.add_argument('-ds', '--downstream', action="store_false")
parser.add_argument('--num-classes', default='10')

def main():

    args = parser.parse_args()

    # create output directory for pretrained model
    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)

    # load data
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


        base_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        ds_dataset = datasets.MNIST(args.data_path, train=False, transform=base_transforms, download=True)
        split = int(np.floor(0.2 * len(ds_dataset)))
        ds_train_dataset, ds_test_dataset = random_split(ds_dataset, [len(ds_dataset) - split, split])
        ds_train_loader = DataLoader(ds_train_dataset, batch_size=32, drop_last=True)
        ds_test_loader = DataLoader(ds_test_dataset, batch_size=32, drop_last=True)
    

    # apply ssl pretraining
    model = SimCLRBase(arch=args.arch, output_dim=args.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    simclr = SimCLRTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)

    # use pretrained model for linear classification downstream task
    ds_linear_classifier = DSLinearClassifier(args=args)
    ds_linear_classifier.train(ds_train_loader)
    ds_linear_classifier.test(ds_test_loader)

if __name__ == "__main__":
    main()