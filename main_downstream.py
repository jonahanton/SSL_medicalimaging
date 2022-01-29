import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import accuracy
from tqdm import tqdm

from models.simclr_base import SimCLRBase
from models.byol_base import BYOLOnlineBase
from models.ds_model import DownstreamModel
from data.get_data import DatasetGetter

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pre-train-method', default='simclr')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--data-path', default='./datasets')
parser.add_argument('--dataset-name', default='cifar10')
parser.add_argument('--pretrain-dataset-name', default='cifar10')
parser.add_argument('--pretrained-path', default='./saved_models/ssl_cifar10_trained_model.pth.tar')
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--n_views', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--fine-tune', action='store_true')

def main():

    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Load in trained network
    model = DownstreamModel(args=args)
    model.load()
    
    # # load train data
    # train_dataset = DatasetGetter(args, train=True, pretrain=False).load()
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # # load test data
    # test_dataset = DatasetGetter(args, train=False, pretrain=False).load()
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # # Training
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device)


    # for epoch in range(args.epochs):
    #     top1_train_accuracy = 0
    #     for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
    #         x_batch = x_batch.to(args.device)
    #         y_batch = y_batch.to(args.device)

    #         logits = model(x_batch)
    #         loss = criterion(logits, y_batch)

    #         top1 = accuracy(logits, y_batch, topk=(1,))
    #         top1_train_accuracy += top1[0]

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     top1_train_accuracy /= (counter + 1)

    #     top1_accuracy = 0
    #     top5_accuracy = 0

    #     # Test data
    #     for counter, (x_batch, y_batch) in enumerate(test_loader):
    #         x_batch = x_batch.to(args.device)
    #         y_batch = y_batch.to(args.device)

    #         logits = model(x_batch)

    #         top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
    #         top1_accuracy += top1[0]
    #         top5_accuracy += top5[0]

    #     top1_accuracy /= (counter + 1)
    #     top5_accuracy /= (counter + 1)
    #     print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

if __name__ == "__main__":
    main()