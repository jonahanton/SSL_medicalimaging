import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


if __name__ == "__main__":

    train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


