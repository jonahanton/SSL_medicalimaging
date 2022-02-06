import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from downstream.ds_model import DownstreamModel
from data.get_data import DatasetGetter

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pre-train-method', default='byol')
parser.add_argument('--arch', default='ConvNet')
parser.add_argument('--data-path', default='./datasets')
parser.add_argument('--dataset-name', default='MNIST')
parser.add_argument('--pretrain-dataset-name', default='MNIST')
parser.add_argument('--outpath', default='./saved_models')
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--n_views', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--finetune', action='store_true')

def main():

    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Load in trained network
    ds_model = DownstreamModel(args=args)
    ds_model.load()
    
    # load train data
    train_dataset = DatasetGetter(args=args, train=True, pretrain=False).load()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # load test data
    test_dataset = DatasetGetter(args=args, train=False, pretrain=False).load()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Training
    optimizer = torch.optim.Adam(ds_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    with torch.cuda.device(args.gpu_index):
        print("Downstream training beginning.")
        ds_model.train(optimizer, train_loader, test_loader)


if __name__ == "__main__":
    main()