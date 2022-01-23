import argparse

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from models.simclr_base import SimCLRBase
from data.generate_views import GenerateViews

from methods.simclr import SimCLRTrainer


parser = argparse.ArgumentParser()
parser.add_argument('-data-path', default='./datasets', help='path to dataset')
parser.add_argument('-dataset-name', default='MNIST', help='dataset name')
parser.add_argument('--epochs', default=1, help='total number of epochs to train for')
parser.add_argument('--batch-size', default=256)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--weight-decay', default=1e-6)
parser.add_argument('--output-dim', default=10)
parser.add_argument('--temperature', default=0.5)
parser.add_argument('--n-views', default=2)
# To do: parse in argument for architecture

def main():

    args = parser.parse_args()

    if args.dataset_name == 'MNIST':
        args.output_dim = 10
        transform_pipeline = transforms.Compose([transforms.RandomResizedCrop(size=28), 
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.1307,), (0.3081,))])
        data_transforms = GenerateViews(transform_pipeline, args.n_views)
        train_dataset = datasets.MNIST(args.data_path, train=True, transform=data_transforms, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SimCLRBase(output_dim=args.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)


    simclr = SimCLRTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)

if __name__ == "__main__":
    main()