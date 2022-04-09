import os
import argparse
from pprint import pprint
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt


from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from datasets.transforms import HistogramNormalize

from datasets.custom_chexpert_dataset import CustomChexpertDataset
from datasets.custom_diabetic_retinopathy_dataset import CustomDiabeticRetinopathyDataset
from datasets.custom_montgomery_cxr_dataset import CustomMontgomeryCXRDataset
from datasets.custom_shenzhen_cxr_dataset import CustomShenzhenCXRDataset
from datasets.custom_bach_dataset import CustomBachDataset




def compute_saliency_map(img, model, device, size=10, value=0):

    img = img.to(device)

    occlusion_window = torch.zeros((img.size(0), size, size)).to(device)
    occlusion_window.fill_(value)

    occlusion_scores = torch.zeros((img.size(1), img.size(2)))

    with torch.no_grad():
        orig_feature = model.forward(img.unsqueeze(0)).squeeze(0)
    orig_feature_mag = torch.sqrt(torch.pow(orig_feature, 2).sum())

    pbar = tqdm(range((1+img.size(1)-size)*(1+img.size(2)-size)), desc='Computing features for occluded images')
    
    for i in range(1 + img.size(1) - size):
        for j in range(1 + img.size(2) - size):
            img_occluded = img
            img_occluded[:, i:i+size, j:j+size] = occlusion_window
            with torch.no_grad():
                occluded_feature = model.forward(img_occluded.unsqueeze(0)).squeeze(0)

            occlusion_score = torch.sqrt(torch.pow(orig_feature - occluded_feature, 2).sum()) / orig_feature_mag
            occlusion_scores[i:i+size, j:j+size] += occlusion_score.item()

            pbar.update(1)

    pbar.close(1)

    occlusion_scores /= size**2

    # apply crop 
    occlusion_scores = occlusion_scores[size-1:img.size(1)-size+1, size-1:img.size(2)-size+1]

    return occlusion_scores


def get_dataset(dset, root, split, transform):
    return dset(root, train=(split == 'train'), transform=transform, download=True)



class ResNet18Backbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet18(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class DenseNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.densenet121(pretrained=False)
        del self.model.classifier

        state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))
    
    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


def get_train_loader(dset,
                    data_dir,
                    normalise_dict,
                    hist_norm,
                    batch_size,
                    image_size,
                    random_seed,
                    num_workers=1,
                    pin_memory=True):


    normalize = transforms.Normalize(**normalise_dict)

    # define transforms
    if hist_norm:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            HistogramNormalize(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = get_dataset(dset, data_dir, 'train', transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=RandomSampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return train_loader


def random_sample(dataloader):
    img, label = next(iter(dataloader))
    return img


def prepare_data(dset, data_dir, batch_size, image_size, normalisation, hist_norm):
    if normalisation:
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    train_loader = get_train_loader(dset, data_dir, normalise_dict,
                                                hist_norm, batch_size, image_size, random_seed=0)

    return train_loader


# name: {class, root, num_classes}
DATASETS = {
    'cifar10': [datasets.CIFAR10, './data/CIFAR10', 10],
    'cifar100': [datasets.CIFAR100, './data/CIFAR100', 100],
    'shenzhencxr': [CustomShenzhenCXRDataset, './data/shenzhencxr', 2],
    'montgomerycxr': [CustomMontgomeryCXRDataset, './data/montgomerycxr', 2],
    'diabetic_retinopathy' : [CustomDiabeticRetinopathyDataset, './data/diabetic_retinopathy', 5],
    'chexpert' : [CustomChexpertDataset, './data/chexpert', 5],
    'bach' : [CustomBachDataset, './data/bach', 4],
}



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compute and save saliency map for pretrained model.')
    parser.add_argument('-m', '--model', type=str, default='moco-v2',
                        help='name of the pretrained model to load and evaluate (moco-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='name of the dataset to evaluate on')
    parser.add_argument('--number', type=int, default=1, help='number of images to compute saliency map for')
    parser.add_argument('-i', '--image-size', type=int, default=242, help='the size of the input images')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='the size of the mini-batches in the dataloader')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-ip', '--image-path', type=str, default='', help='path to image to calculate saliency map for')
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # histogram normalization
    hist_norm = False
    if 'mimic-chexpert' in args.model:
        hist_norm = True



    # load pretrained model
    if args.model in ['mimic-chexpert_lr_0.1', 'mimic-chexpert_lr_0.01', 'mimic-chexpert_lr_1.0', 'supervised_d121']:
        model = DenseNetBackbone(args.model)
        feature_dim = 1024
    elif 'mimic-cxr' in args.model:
        if 'r18' in args.model:
            model = ResNet18Backbone(args.model)
            feature_dim = 512
        else:
            model = DenseNetBackbone(args.model)
            feature_dim = 1024
    elif args.model == 'supervised_r18':
        feature_dim = 512
    else:
        model = ResNetBackbone(args.model)
        feature_dim = 2048
    
    model = model.to(args.device)


    # If haven't specified image to compute saliency map for, sample randomly from dataset
    if args.image_path == '':

        # load dataset
        dset, data_dir, num_classes = DATASETS[args.dataset]
        # prepare data loaders
        train_loader = prepare_data(dset, data_dir, args.batch_size, args.image_size,
                                    normalisation=args.norm, hist_norm=hist_norm)

        # compute and save saliency maps
        for i in range(args.number):
            img = random_sample(train_loader)
            saliency_map = compute_saliency_map(img, model, args.device)

    
    else:

        if args.norm:
            normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        else:
            normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
        normalize = transforms.Normalize(**normalise_dict)
        if hist_norm:
            transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            HistogramNormalize(),
        ])
        else: 
            transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])

        img = Image.open(args.img_path).convert('RGB')
        img = transform(img)
        saliency_map = compute_saliency_map(img, model, args.device)


    
    


