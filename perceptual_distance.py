import argparse
import os
import logging
from pprint import pprint

import lpips
import torch
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm


def open_and_convert_image(impath, image_size):

    im = Image.open(impath).convert('RGB')

    # rescale to image_size, convert PIL Image to torch tensor
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    im_torch = transform(im)

    # rescale pixels from range [0, 1] (default for transforms.ToTensor()) to [-1, 1]
    im_torch = 2*im_torch - 1
    # output tensor should have size [1, C, H, W]
    im_torch = im_torch.unsqueeze(0)

    return im_torch

    
def perceptual_distance(im1, im2, device):

    im1 = im1.to(device)
    im2 = im2.to(device)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_squeeze = lpips.LPIPS(net='squeeze').to(device)

    d_alex = loss_fn_alex.forward(im1, im2)
    d_vgg = loss_fn_vgg.forward(im1, im2)
    d_squeeze = loss_fn_squeeze.forward(im1, im2)

    return d_alex.cpu().detach().numpy().item(), d_vgg.cpu().detach().numpy().item(), d_squeeze.cpu().detach().numpy().item()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute and save perceptual distances between two images.')
    parser.add_argument('-im1', '--image-1', type=str, default='', help='path for image 1')
    parser.add_argument('-im2', '--image-2', type=str, default='', help='path for image 2')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU (cuda | cpu)')
    args = parser.parse_args()
    pprint(args)

    if args.image_1 == '' or args.image_2 == '':
        pass 
    
    else:
        
        im1 = open_and_convert_image(args.image_1, args.image_size)
        im2 = open_and_convert_image(args.image_2, args.image_size)

        d_alex, d_vgg, d_squeeze = perceptual_distance(im1, im2, args.device)
        
        results_dict = {
            'AlexNet' : d_alex,
            'VGG' : d_vgg,
            'SqueezeNet' : d_squeeze,
        }
        print(f'Perceptual distance between images {args.image_1} and {args.image_2}:')
        pprint(results_dict)
