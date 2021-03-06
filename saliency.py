#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint
import logging
import copy
import math

import medpy.io as medpy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from torchvision import datasets, transforms, models
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from models.backbones import ResNetBackbone, ResNet18Backbone, DenseNetBackbone

from datasets.transforms import HistogramNormalize



# Compute an occlusion-based saliency map for a given image and model
def compute_saliency_map(img, model, device, size=10, value=0):
    img = img.to(device)

    occlusion_window = torch.zeros((img.size(0), size, size)).to(device)
    occlusion_window.fill_(value)

    occlusion_scores = np.zeros((img.size(1), img.size(2)))

    with torch.no_grad():
        orig_feature = model.forward(img.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    orig_feature_mag = np.sqrt((orig_feature**2).sum())

    pbar = tqdm(range((1+img.size(1)-size)*(1+img.size(2)-size)), desc='Computing features for occluded images')
    for i in range(1 + img.size(1) - size):
        for j in range(1 + img.size(2) - size):
            img_occluded = img.clone()
            img_occluded[:, i:i+size, j:j+size] = occlusion_window
            with torch.no_grad():
                occluded_feature = model.forward(img_occluded.unsqueeze(0)).squeeze(0).cpu().detach().numpy()

            occlusion_score = np.sqrt(((orig_feature - occluded_feature)**2).sum()) / orig_feature_mag
            occlusion_scores[i:i+size, j:j+size] += occlusion_score

            pbar.update(1)

    pbar.close()

    occlusion_scores /= size**2

    # apply crop
    occlusion_scores = occlusion_scores[size-1:img.size(1)-size+1, size-1:img.size(2)-size+1]

    return occlusion_scores


def get_dataset(dset, root, split, transform):
    return dset(root, train=(split == 'train'), transform=transform, download=True)


def obtain_and_pre_process_img(img_path, img_size, normalisation, hist_norm, stoic=False):
    
    if normalisation:
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    normalize = transforms.Normalize(**normalise_dict)

    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    if stoic:
        # Load in mha style image using medpy library
        img, _ = medpy.load(img_path)
        # Extract a slice at a given depth
        depth = math.floor(img.shape[2]*0.5)
        img = img[:,:,depth]
        # Clip the pixel values
        min_value, max_value = -1000, 250
        np.clip(img, min_value, max_value, out=img)
        # Rescale to [0,1]
        img = (img - min_value) / (max_value - min_value)
        # Rescale to [0,255] and uint8 (as needed for loading L) 
        # Then convert to RGB
        img = Image.fromarray((img* 255).astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(img_path).convert('RGB')

    img = transform(img)
    if hist_norm:
        histnorm = HistogramNormalize()
        normalized_img = histnorm(img)
    else:
        normalized_img = normalize(img)

    return img, normalized_img


def crop(img, img_size):
    """ Apply center crop."""
    crop = transforms.CenterCrop(img_size)
    return crop(img)




# dataset: {image_name, image_path}
IMAGES = {
    'bach' : ['iv001.tif', './sample_images/bach/iv001.tif'],
    'chestx' : ['00000001_000.png', './sample_images/chestx/00000001_000.png'],
    'chexpert' : ['patient00001_view1_frontal.jpg','./sample_images/chexpert/patient00001_view1_frontal.jpg'],
    'diabetic_retinopathy' : ['34680_left.jpeg', './sample_images/diabetic_retinopathy/34680_left.jpeg'],
    'ichallenge_amd' : ['AMD_A0001.jpg', './sample_images/ichallenge_amd/AMD_A0001.jpg'],
    'ichallenge_pm' : ['H0009.jpg', './sample_images/ichallenge_pm/H0009.jpg'],
    'montgomerycxr' : ['MCUCXR_0001_0.png', './sample_images/montgomerycxr/MCUCXR_0001_0.png'],
    'shenzhencxr' : ['CHNCXR_0076_0.png', './sample_images/shenzhencxr/CHNCXR_0076_0.png'],
    'stoic' : ['8622.mha', 'sample_images/stoic/8622.mha'],
    'imagenet' : ['goldfish.jpeg', 'sample_images/imagenet/goldfish.jpeg'],
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compute and save saliency maps for pretrained model.')
    parser.add_argument('-d', '--datasets', nargs='+', type=str, default='', help='datasets to calculate saliency maps for', required=True)
    parser.add_argument('-m', '--model', type=str, default='moco-v2',
                        help='name of the pretrained model to load and evaluate (moco-v2 | supervised)')
    parser.add_argument('-i', '--image-size', type=int, default=242, help='the size of the images')
    parser.add_argument('-c', '--crop-size', type=int, default=224, help='the size of the images post centre crop')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
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
        model = ResNet18Backbone(args.model)
        feature_dim = 512
    else:
        model = ResNetBackbone(args.model)
        feature_dim = 2048
    
    model = model.to(args.device)


    # set-up logging
    log_fname = f'{args.model}.log'
    if not os.path.isdir(f'./logs/saliency/stoic'):
        os.makedirs(f'./logs/saliency/stoic')
    log_path = os.path.join(f'./logs/saliency/stoic', log_fname)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)
    logging.info(args)


    if args.datasets == '':
        print('No datasets specified!')
    else:

        for dataset in args.datasets:

            stoic = dataset == 'stoic'

            image_name, image_path = IMAGES[dataset]


            # set-up output path
            outpath_base = f'saliency_maps/{args.model}/{dataset}'
            if not os.path.isdir(outpath_base):
                os.makedirs(outpath_base)
            out_heat = os.path.join(outpath_base, 'heatmap.png')
            out_super = os.path.join(outpath_base, 'superimposed.png')

            img, normalized_img = obtain_and_pre_process_img(image_path, args.image_size, args.norm, hist_norm, stoic)
            saliency_map = compute_saliency_map(normalized_img, model, args.device)
            
            cropped_img = crop(img, args.crop_size)
            permuted_img = cropped_img.permute((1, 2, 0))

            superimposed_img = plt.imshow(permuted_img, cmap='gray')
            superimposed_img = plt.imshow(saliency_map, cmap='jet', alpha=0.6)
            plt.axis('off')
            plt.savefig(out_super)
            
            plt.imsave(out_heat, saliency_map, cmap='jet')

            # calculate attentative diffusion (percentage of attention map with values about its mean)
            mean_attention = np.mean(saliency_map)
            attentive_diffusion = 100 * (saliency_map > mean_attention).sum() / np.prod(saliency_map.size)
            print(f'Attentive diffusion on dataset {dataset} with model {args.model}: {attentive_diffusion:.2f}%')
            logging.info(f'Attentive diffusion on dataset {dataset} with model {args.model}: {attentive_diffusion:.2f}%')

    
    


