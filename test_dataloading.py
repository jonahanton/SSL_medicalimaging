# This script is used to understand how we load data.
# To be deleted in future

import os
import argparse
from pprint import pprint
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, datasets

from datasets import few_shot_dataset
from datasets.custom_chexpert_dataset import CustomChexpertDataset
from datasets.custom_diabetic_retinopathy_dataset import CustomDiabeticRetinopathyDataset
from datasets.custom_montgomery_cxr_dataset import CustomMontgomeryCXRDataset
from datasets.custom_shenzhen_cxr_dataset import CustomShenzhenCXRDataset
from datasets.custom_bach_dataset import CustomBachDataset


# Parser arguments
parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model on few-shot recognition.')
parser.add_argument('-m', '--model', type=str, default='moco-v2',
                    help='name of the pretrained model to load and evaluate (moco-v2 | supervised)')
parser.add_argument('-d', '--dataset', type=str, default='diabetic_retinopathy', help='name of the dataset to evaluate on')
parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
parser.add_argument('--n-way', type=int, default=5, help='the number of classes per episode (n-way) in few-shot evaluation')
parser.add_argument('--n-support', type=int, default=5, help='the number of images per class for fitting (n-support) in few-shot evaluation')
parser.add_argument('--n-query', type=int, default=15, help='the number of images per class for testing (n-query) in few-shot evaluation')
parser.add_argument('--iter-num', type=int, default=600, help='the number of testing episodes in few-shot evaluation')
parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                    help='whether to turn off data normalisation (based on ImageNet values)')
parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
args = parser.parse_args()
args.norm = not args.no_norm
pprint(args)


# name: {class, root, num_classes (not necessary here), metric}
FEW_SHOT_DATASETS = {
    'cifar10': [datasets.CIFAR10, './data/CIFAR10', 10, 'accuracy'],
    'cifar100': [datasets.CIFAR100, './data/CIFAR100', 100, 'accuracy'],
    'shenzhencxr': [CustomShenzhenCXRDataset, './data/shenzhencxr', 2, 'accuracy'],
    'montgomerycxr': [CustomMontgomeryCXRDataset, './data/montgomerycxr', 2, 'accuracy'],
    'diabetic_retinopathy' : [CustomDiabeticRetinopathyDataset, './data/diabetic_retinopathy', 5, 'mean per-class accuracy'],
    'chexpert' : [CustomChexpertDataset, './data/chexpert', 5, 'mean per-class accuracy'],
    'bach' : [CustomBachDataset, './data/bach', 4, 'accuracy'],
}

dset, data_dir, num_classes, metric = FEW_SHOT_DATASETS[args.dataset]

print(dset) # <class 'torchvision.datasets.cifar.CIFAR10'>
print(dir(dset))
print(type(dset))

# print(dset.train_list)
# [['data_batch_1', 'c99cafc152244af753f735de768cd75f'], ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'], ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'], ['data_batch_4', '634d18415352ddfa80567beed471001a'], ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']]


datamgr = few_shot_dataset.SetDataManager(dset, data_dir, num_classes, args.image_size, n_episode=args.iter_num,
                                    n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)



dataloader = datamgr.get_data_loader(aug=False, normalise=args.norm, hist_norm=False)
# print(dataloader)
# print(dir(dataloader))

# Iterate through dataloader

# with torch.no_grad():
#     for i, (data, targets) in enumerate(tqdm(dataloader)):
#         # data is batch
#         print("i", i)
#         for batch in data:
#             print(batch[0]) # tensor that corresponds to image
#             print(batch[0].shape) # 3 X 224 X 224





