import argparse
import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, datasets
import PIL
from PIL import Image
import pickle

from datasets.custom_chexpert_dataset import CustomChexpertDataset
from datasets.custom_diabetic_retinopathy_dataset import CustomDiabeticRetinopathyDataset
from datasets.custom_montgomery_cxr_dataset import CustomMontgomeryCXRDataset
from datasets.custom_shenzhen_cxr_dataset import CustomShenzhenCXRDataset
from datasets.custom_bach_dataset import CustomBachDataset
from datasets.custom_ichallenge_amd_dataset import CustomiChallengeAMDDataset
from datasets.custom_ichallenge_pm_dataset import CustomiChallengePMDataset


# Data classes and functions


def get_dataset(dset, root, split, transform):
    return dset(root, train=(split == 'train'), transform=transform, download=True)


def get_train_valid_test_dset(dset,
                              data_dir,
                              image_size):


    # define transforms
    transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            ])

    
    train_valid_dataset = get_dataset(dset, data_dir, 'train', transform)
    test_dataset = get_dataset(dset, data_dir, 'test', transform)
    dataset = ConcatDataset([train_valid_dataset, test_dataset])

    return dataset



# name: {class, root, num_classes}
DATASETS = {
    'cifar10': [datasets.CIFAR10, './data/CIFAR10', 10],
    'cifar100': [datasets.CIFAR100, './data/CIFAR100', 100],
    'shenzhencxr': [CustomShenzhenCXRDataset, './data/shenzhencxr', 2],
    'montgomerycxr': [CustomMontgomeryCXRDataset, './data/montgomerycxr', 2],
    'diabetic_retinopathy' : [CustomDiabeticRetinopathyDataset, './data/diabetic_retinopathy', 5],
    'chexpert' : [CustomChexpertDataset, './data/chexpert', 5],
    'bach' : [CustomBachDataset, './data/bach', 4],
    'ichallenge_amd' : [CustomiChallengeAMDDataset, './data/ichallenge_amd', 2],
    'ichallenge_pm' : [CustomiChallengePMDataset, './data/ichallenge_pm', 2],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='name of the dataset to compute sub meta dict for')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    args = parser.parse_args()
    pprint(args)

    # load dataset
    dset, data_dir, num_classes = DATASETS[args.dataset]
    d = get_train_valid_test_dset(dset, data_dir, args.image_size)
    print(f'Total dataset size: {len(d)}')

    cl_list = range(num_classes)
    sub_meta = {}
    for cl in cl_list:
        sub_meta[cl] = []

    pbar = tqdm(range(len(d)), desc='Iterating through dataset')
    for i, (data, label) in enumerate(d):
        if i > 10000:
            break
        sub_meta[label].append(data)
        pbar.update(1)
    pbar.close()

    print('Number of images per class')
    for key, item in sub_meta.items():
        print(len(sub_meta[key]))
    
    with open(f'misc/few_shot_submeta/{args.dataset}.pickle', 'wb') as handle:
        pickle.dump(sub_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)