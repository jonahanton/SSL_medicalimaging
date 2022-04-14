# This code is modified from: https://github.com/linusericsson/ssl-invariances/blob/main/eval_synthetic_invariance.py

#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as FT

import os
import PIL
import pickle
import argparse
import logging
from pprint import pprint
from tqdm import tqdm
from pathlib import Path
from itertools import product

import numpy as np
import albumentations 
from scipy.spatial.distance import mahalanobis

from datasets.transforms import HistogramNormalize
from datasets.custom_chexpert_dataset import CustomChexpertDataset
from datasets.custom_diabetic_retinopathy_dataset import CustomDiabeticRetinopathyDataset
from datasets.custom_montgomery_cxr_dataset import CustomMontgomeryCXRDataset
from datasets.custom_shenzhen_cxr_dataset import CustomShenzhenCXRDataset
from datasets.custom_bach_dataset import CustomBachDataset
from datasets.custom_ichallenge_amd_dataset import CustomiChallengeAMDDataset
from datasets.custom_ichallenge_pm_dataset import CustomiChallengePMDataset
from datasets.custom_stoic_dataset import CustomStoicDataset





# Image transformations

def deform(img, points, sigma):
    # convert to numpy
    img = np.array(img)
    # apply deformation
    img = albumentations.geometric.transforms.ElasticTransform(sigma=sigma, always_apply=True, approximate=True)(image=img)['image']
    # return PIL image
    return PIL.Image.fromarray(img)

FT.deform = deform


class ManualTransform(object):
    def __init__(self, name, k, norm, resize=256, crop_size=224):
        self.norm = norm
        name_to_fn = {
            'rotation': 'rotate',
            'translation': 'affine',
            'scale': 'affine',
            'shear': 'affine',
            'resized_crop': 'resized_crop',
            'h_flip': 'hflip',
            'v_flip': 'vflip',
            'deform': 'deform',
            'grayscale': 'rgb_to_grayscale',
            'brightness': 'adjust_brightness',
            'contrast': 'adjust_contrast',
            'saturation': 'adjust_saturation',
            'hue': 'adjust_hue',
            'blur': 'gaussian_blur',
            'sharpness': 'adjust_sharpness',
            'invert': 'invert',
            'equalize': 'equalize',
            'posterize': 'posterize',
        }
        self.fn = name_to_fn[name]
        self.k = k
        self.resize = resize
        self.crop_size = crop_size
        if name == 'rotation':
            self.param_keys = ['angle']
            self.param_vals = [torch.linspace(0, 360, self.k + 1)[:self.k].to(torch.float32).tolist()]
            self.original_idx = 0
        elif name == 'translation':
            self.param_keys = ['translate', 'angle', 'scale', 'shear']
            space = (1 - (crop_size / resize)) / 2
            a = torch.linspace(-space, space, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32)
            translate_params = [(float(a * resize), float(b * resize)) for a, b in product(a, a)]
            self.param_vals = [
                translate_params,
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = translate_params.index((0.0, 0.0))
        elif name == 'scale':
            self.param_keys = ['scale', 'translate', 'angle', 'shear']
            self.param_vals = [
                torch.linspace(1 / 4, 2, self.k).to(torch.float32).tolist(),
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = 0
        elif name == 'shear':
            self.param_keys = ['shear', 'translate', 'angle', 'scale']
            a = torch.linspace(-160, 160, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32).tolist()
            shear_params = [(a, b) for a, b in product(a, a)]
            self.param_vals = [
                shear_params,
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = shear_params.index((0.0, 0.0))
        elif name == 'resized_crop':
            self.param_keys = ['top', 'left', 'height', 'width', 'size']
            n = int(np.sqrt(np.sqrt(self.k)))
            a = (torch.linspace(0, 0.25, n) * resize).to(torch.float32).tolist()
            b = (torch.linspace(0.75, 0.25, n) * resize).to(torch.float32).tolist()
            p = product(a, a, b, b)
            a, b, c, d = tuple(zip(*p))
            self.param_vals = [
                a, b, c, d,
                [(s.item(), s.item()) for s in torch.ones(self.k, dtype=int) * crop_size]
            ]
            self.original_idx = 0
        elif name in ['h_flip', 'v_flip']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'deform':
            torch.manual_seed(0)
            np.random.seed(0)
            self.param_keys = ['points', 'sigma'] # 10, 50, 3, 9
            points = torch.repeat_interleave(torch.arange(2, 10), 32).tolist()
            sigma = torch.linspace(10, 50, 8).to(int).repeat(32).tolist()
            self.param_vals = [
                points,
                sigma
            ]
            self.original_idx = 0
        elif name == 'grayscale':
            self.param_keys = ['aug', 'num_output_channels']
            self.param_vals = [[False, True], [3, 3]]
            self.original_idx = 0
        elif name == 'brightness':
            self.param_keys = ['brightness_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'contrast':
            self.param_keys = ['contrast_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'saturation':
            self.param_keys = ['saturation_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'hue':
            self.param_keys = ['hue_factor']
            self.param_vals = [torch.linspace(-0.5, 0.5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'blur':
            self.param_keys = ['sigma', 'kernel_size']
            self.param_vals = [
                torch.linspace(1e-5, 20.0, self.k).to(torch.float32).tolist(),
                (torch.ones(self.k).to(int) + (crop_size // 20 * 2)).tolist(),
            ]
            self.original_idx = 0
        elif name == 'sharpness':
            self.param_keys = ['sharpness_factor']
            self.param_vals = [torch.linspace(1, 30.0, self.k).to(torch.float32).tolist()]
            self.original_idx = 0
        elif name in ['invert', 'equalize']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'posterize':
            self.param_keys = ['bits']
            self.param_vals = [torch.arange(1, 8).tolist()]
            self.original_idx = 0

    def T(self, image, **params):
        if 'aug' in params:
            if params['aug']:
                del params['aug']
                image = eval(f'FT.{self.fn}(image, **params)')
        elif self.fn == 'translation':
            pass
        else:
            image = eval(f'FT.{self.fn}(image, **params)')

        if self.fn != 'resized_crop':
            image = FT.resize(image, self.resize)
            if self.fn == 'translation':
                image = eval(f'FT.{self.fn}(image, **params)')
            image = FT.center_crop(image, self.crop_size)
        image = FT.pil_to_tensor(image).to(torch.float32)
        image = FT.normalize(image / 255., *self.norm)
        return image

    def __call__(self, x):
        xs = []
        for i in range(self.k):
            params = dict([(k, v[i]) for k, v in zip(self.param_keys, self.param_vals)])
            xs.append(self.T(x, **params))
        return tuple(xs)


def D(a, b): # cosine similarity
    return F.cosine_similarity(a, b, dim=-1).mean()



# Data classes and functions


def get_dataset(dset, root, split, transform):
    return dset(root, train=(split == 'train'), transform=transform, download=True)


def get_train_valid_test_dset(dset,
                              data_dir,
                              normalise_dict,
                              hist_norm,
                              resize,
                              crop_size,
                              manual_transform=None):

    
    normalize = transforms.Normalize(**normalise_dict)

    # define transforms
    if manual_transform is None:
        if hist_norm:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                HistogramNormalize(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        transform = manual_transform

    
    train_valid_dataset = get_dataset(dset, data_dir, 'train', transform)
    test_dataset = get_dataset(dset, data_dir, 'test', transform)
    dataset = ConcatDataset([train_valid_dataset, test_dataset])

    return dataset



# Testing classes and functions

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









# name: {class, root}
DATASETS = {
    'cifar10': [datasets.CIFAR10, './data/CIFAR10'],
    'cifar100': [datasets.CIFAR100, './data/CIFAR100'],
    'shenzhencxr': [CustomShenzhenCXRDataset, './data/shenzhencxr'],
    'montgomerycxr': [CustomMontgomeryCXRDataset, './data/montgomerycxr'],
    'diabetic_retinopathy' : [CustomDiabeticRetinopathyDataset, './data/diabetic_retinopathy'],
    'chexpert' : [CustomChexpertDataset, './data/chexpert'],
    'bach' : [CustomBachDataset, './data/bach'],
    'ichallenge_amd' : [CustomiChallengeAMDDataset, './data/ichallenge_amd'],
    'ichallenge_pm' : [CustomiChallengePMDataset, './data/ichallenge_pm'],
    'stoic': [CustomStoicDataset, './data/stoic', 2, 'accuracy'],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='name of the dataset to evaluate on')
    parser.add_argument('--model', default='default', type=str,
                        help='model to evaluate invariance of')
    parser.add_argument('--transform', default='rotation', type=str,
                        help='transform to evaluate invariance of (rotation/translation/colour jitter/blur etc.)')
    parser.add_argument('--device', default='cuda', type=str, help='GPU device')
    parser.add_argument('--num-images', default=100, type=int, help='number of images to evaluate invariance on.')
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--resize', default=256, type=int, help='resize')
    parser.add_argument('--crop-size', default=224, type=int, help='crop size')
    parser.add_argument('--k', default=None, type=int, help='number of transformations')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    args = parser.parse_args()
    args.norm = not args.no_norm
    del args.no_norm
    pprint(args)

    # histogram normalization
    hist_norm = False
    if 'mimic-chexpert' in args.model:
        hist_norm = True


    # set-up logging
    log_fname = f'{args.dataset}.log'
    if not os.path.isdir(f'./logs/invariances/{args.model}/{args.transform}'):
        os.makedirs(f'./logs/invariances/{args.model}/{args.transform}')
    log_path = os.path.join(f'./logs/invariances/{args.model}/{args.transform}', log_fname)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)
    logging.info(args)


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
        model = ResNet18Backbone(args.model)
        feature_dim = 512
    else:
        model = ResNetBackbone(args.model)
        feature_dim = 2048
    
    model = model.to(args.device)


    if args.k is not None:
        k = args.k
    elif args.transform in ['h_flip', 'v_flip', 'grayscale', 'invert', 'equalize']:
        k = 2
    elif args.transform == 'posterize':
        k = 7
    else:
        k = 256

    if args.norm:
        mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        mean_std = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}

    transform = ManualTransform(args.transform, k, norm=mean_std, resize=args.resize, crop_size=args.crop_size)
    
    # load datasets
    dset, data_dir = DATASETS[args.dataset]
    clean_dataset = get_train_valid_test_dset(dset, data_dir, normalise_dict, hist_norm, args.resize, args.crop_size)
    dataset = get_train_valid_test_dset(dset, data_dir, normalise_dict, hist_norm, args.resize, args.crop_size,
                                         manual_transform=transform)

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)


    if os.path.exists(f'./invariances/results/{args.model}_{args.dataset}_feature_cov_matrix.pth'):
        print(f'Found precomputed covariance matrix for {args.model} on {args.dataset}, skipping it')
        logging.info(f'Found precomputed covariance matrix for {args.model} on {args.dataset}, skipping it')
    else:
        print(f'Computing covariance matrix for {args.model} on dataset {args.dataset}')
        logging.info(f'Computing covariance matrix for {args.model} on dataset {args.dataset}')

        # Calculate (approx.) mean and covariance matrix, 
        # from > 1000 sampled images (10% of full dataset, or full dataset if contains < 1000 images)
        if len(clean_dataset) < 1000:
            clean_dataloader = DataLoader(clean_dataset, batch_size=args.batch_size)
        else:
            random_idx = np.random.choice(np.arange(len(clean_dataset)), max(1000, int(0.1*len(clean_dataset))))
            sub_sampler = SubsetRandomSampler(random_idx)
            clean_dataloader = DataLoader(clean_dataset, batch_size=args.batch_size, sampler=sub_sampler)


        all_features = []
        with torch.no_grad():
            progress = tqdm(clean_dataloader)
            for data, _ in progress:
                data = data.to(args.device)
                features = model(data).detach().cpu()
                all_features.append(features)
        all_features = torch.cat(all_features)

        mean_feature = all_features.mean(dim=0)
        cov_matrix = np.cov(all_features, rowvar=False)

        torch.save(mean_feature, f'./invariances/results/{args.model}_{args.dataset}_mean_feature.pth')
        torch.save(cov_matrix, f'./invariances/results/{args.model}_{args.dataset}_feature_cov_matrix.pth')


    # Calculate invariances
    L = torch.zeros((args.num_images, k))
    S = torch.zeros((args.num_images, k))

    mean_feature = torch.load(f'./invariances/results/{args.model}_{args.dataset}_mean_feature.pth')
    cov_matrix = torch.load(f'./invariances/results/{args.model}_{args.dataset}_feature_cov_matrix.pth')
    
    # # ensure inv_cov_matrix is positive semi-definite (so can calculate Choleksy decomp.)
    # epsilon = 1e-10
    # while True:
    #     inv_cov_matrix = np.linalg.inv(cov_matrix)
    #     eig_vals = np.linalg.eigvalsh(inv_cov_matrix)
    #     if len(eig_vals[eig_vals < 0]) > 0:
    #         cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
    #         epsilon *= 10
    #     else:
    #         break
    
    epsilon = 1e-6
    cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    cholesky_matrix = torch.linalg.cholesky(torch.from_numpy(inv_cov_matrix).to(torch.float32))


    def get_same_batch(sampler, d1, d2):
        for i in sampler:
            img1, _ = d1[i]
            img2, _ = d2[i]
            yield (torch.stack(img1), img2.unsqueeze(0))

    sampler = np.random.choice(np.arange(len(dataset)), args.num_images)
    batch_generator = get_same_batch(sampler, dataset, clean_dataset)    
    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            data, clean_data = next(batch_generator)
            clean_feature = model(clean_data.to(args.device)).detach().cpu()
            features = model(data.to(args.device)).detach().cpu()

            a = (mean_feature - clean_feature) @ cholesky_matrix

            for j in range(features.shape[0]):
                b = (mean_feature - features[j]) @ cholesky_matrix
                S[i, j] = D(a, b) # cosine similarity
                L[i, j] = mahalanobis(clean_feature, features[j], inv_cov_matrix) # mahalanobis distance


    L = torch.from_numpy(np.nanmean(L, axis=0))
    S = torch.from_numpy(np.nanmean(S, axis=0))
    print(f'{args.model} on {args.transform} with dataset {args.dataset}:')
    print(f'\t distance {L.mean():.6f} and similarity {S.mean():.6f}')
    logging.info(f'{args.model} on {args.transform} with dataset {args.dataset}:')
    logging.info(f'\t distance {L.mean():.6f} and similarity {S.mean():.6f}')