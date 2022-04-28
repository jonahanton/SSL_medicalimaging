import argparse
from pprint import pprint
import csv
from collections import defaultdict

import lpips   
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import medpy.io as medpy


def open_and_convert_image(impath, image_size):

    if impath[-4:] == '.mha':
        n = 5
        img, _ = medpy.load(impath)
        img = img[:,:,n]
        img = Image.fromarray(img).convert("RGB")

    else:
        img = Image.open(impath).convert('RGB')

    # rescale to image_size, convert PIL Image to torch tensor
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    im_torch = transform(img)

    # rescale pixels from range [0, 1] (default for transforms.ToTensor()) to [-1, 1]
    im_torch = 2*im_torch - 1
    # output tensor should have size [1, C, H, W]
    im_torch = im_torch.unsqueeze(0)

    return im_torch

# We use the Perceptual Similarity Metric library, 
# from the paper "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al., 2018)
    
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

original_images = {
    'bach' : ['iv001.tif', './sample_images/bach/iv001.tif'],
    'chestx' : ['00000001_000.png', './sample_images/chestx/00000001_000.png'],
    'chexpert' : ['patient00001_view1_frontal.jpg','./sample_images/chexpert/patient00001_view1_frontal.jpg'],
    'diabetic_retinopathy' : ['34680_left.jpeg', './sample_images/diabetic_retinopathy/34680_left.jpeg'],
    'ichallenge_amd' : ['AMD_A0001.jpg', './sample_images/ichallenge_amd/AMD_A0001.jpg'],
    'ichallenge_pm' : ['H0009.jpg', './sample_images/ichallenge_pm/H0009.jpg'],
    'montgomerycxr' : ['MCUCXR_0001_0.png', './sample_images/montgomerycxr/MCUCXR_0001_0.png'],
    'shenzhencxr' : ['CHNCXR_0076_0.png', './sample_images/shenzhencxr/CHNCXR_0076_0.png'],
    'stoic' : ['8622.mha', 'sample_images/stoic/8622.mha'],  
    'imagenet' : ['goldfish.jpeg', 'sample_images/imagenet/goldfish.jpeg']
}

reconstructed_images = {
    'bach' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_iv001.tif',
                'swav': './reconstructed_images/swav/swav_True_iv001.tif',
                'byol': './reconstructed_images/byol/byol_True_iv001.tif',
                'pirl': './reconstructed_images/pirl/pirl_True_iv001.tif',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_iv001.tif',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_iv001.tif',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_iv001.tif',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_iv001.tif',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_iv001.tif',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_iv001.tif',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_iv001.tif',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_iv001.tif',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_iv001.tif'
                },
    'chestx' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_00000001_000.png',
                'swav': './reconstructed_images/swav/swav_True_00000001_000.png',
                'byol': './reconstructed_images/byol/byol_True_00000001_000.png',
                'pirl': './reconstructed_images/pirl/pirl_True_00000001_000.png',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_00000001_000.png',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_00000001_000.png',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_00000001_000.png',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_00000001_000.png',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_00000001_000.png',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_00000001_000.png',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_00000001_000.png',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_00000001_000.png',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_00000001_000.png'
                },
    'chexpert' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_patient00001_view1_frontal.jpg',
                'swav': './reconstructed_images/swav/swav_True_patient00001_view1_frontal.jpg',
                'byol': './reconstructed_images/byol/byol_True_patient00001_view1_frontal.jpg',
                'pirl': './reconstructed_images/pirl/pirl_True_patient00001_view1_frontal.jpg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_patient00001_view1_frontal.jpg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_patient00001_view1_frontal.jpg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_patient00001_view1_frontal.jpg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_patient00001_view1_frontal.jpg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_patient00001_view1_frontal.jpg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_patient00001_view1_frontal.jpg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_patient00001_view1_frontal.jpg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_patient00001_view1_frontal.jpg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_patient00001_view1_frontal.jpg'
                },
    'diabetic_retinopathy' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_34680_left.jpeg',
                'swav': './reconstructed_images/swav/swav_True_34680_left.jpeg',
                'byol': './reconstructed_images/byol/byol_True_34680_left.jpeg',
                'pirl': './reconstructed_images/pirl/pirl_True_34680_left.jpeg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_34680_left.jpeg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_34680_left.jpeg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_34680_left.jpeg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_34680_left.jpeg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_34680_left.jpeg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_34680_left.jpeg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_34680_left.jpeg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_34680_left.jpeg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_34680_left.jpeg'
                },
    'ichallenge_amd' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_AMD_A0001.jpg',
                'swav': './reconstructed_images/swav/swav_True_AMD_A0001.jpg',
                'byol': './reconstructed_images/byol/byol_True_AMD_A0001.jpg',
                'pirl': './reconstructed_images/pirl/pirl_True_AMD_A0001.jpg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_AMD_A0001.jpg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_AMD_A0001.jpg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_AMD_A0001.jpg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_AMD_A0001.jpg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_AMD_A0001.jpg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_AMD_A0001.jpg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_AMD_A0001.jpg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_AMD_A0001.jpg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_AMD_A0001.jpg'
                },
    'ichallenge_pm' :{'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_H0009.jpg',
                'swav': './reconstructed_images/swav/swav_True_H0009.jpg',
                'byol': './reconstructed_images/byol/byol_True_H0009.jpg',
                'pirl': './reconstructed_images/pirl/pirl_True_H0009.jpg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_H0009.jpg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_H0009.jpg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_H0009.jpg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_H0009.jpg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_H0009.jpg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_H0009.jpg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_H0009.jpg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_H0009.jpg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_H0009.jpg'
                },
    'montgomerycxr' :{'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_MCUCXR_0001_0.png',
                'swav': './reconstructed_images/swav/swav_True_MCUCXR_0001_0.png',
                'byol': './reconstructed_images/byol/byol_True_MCUCXR_0001_0.png',
                'pirl': './reconstructed_images/pirl/pirl_True_MCUCXR_0001_0.png',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_MCUCXR_0001_0.png',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_MCUCXR_0001_0.png',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_MCUCXR_0001_0.png',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_MCUCXR_0001_0.png',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_MCUCXR_0001_0.png',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_MCUCXR_0001_0.png',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_MCUCXR_0001_0.png',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_MCUCXR_0001_0.png',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_MCUCXR_0001_0.png'
                },
    'shenzhencxr' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_CHNCXR_0076_0.png',
                'swav': './reconstructed_images/swav/swav_True_CHNCXR_0076_0.png',
                'byol': './reconstructed_images/byol/byol_True_CHNCXR_0076_0.png',
                'pirl': './reconstructed_images/pirl/pirl_True_CHNCXR_0076_0.png',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_CHNCXR_0076_0.png',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_CHNCXR_0076_0.png',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_CHNCXR_0076_0.png',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_CHNCXR_0076_0.png',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_CHNCXR_0076_0.png',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_CHNCXR_0076_0.png',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_CHNCXR_0076_0.png',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_CHNCXR_0076_0.png',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_CHNCXR_0076_0.png'
                },
    'stoic' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_8622.jpeg',
                'swav': './reconstructed_images/swav/swav_True_8622.jpeg',
                'byol': './reconstructed_images/byol/byol_True_8622.jpeg',
                'pirl': './reconstructed_images/pirl/pirl_True_8622.jpeg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_8622.jpeg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_8622.jpeg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_8622.jpeg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_8622.jpeg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_8622.jpeg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_8622.jpeg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_8622.jpeg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_8622.jpeg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_8622.jpeg'
                },
    'imagenet' : {'simclr-v1' : './reconstructed_images/simclr-v1/simclr-v1_True_goldfish.jpeg',
                'swav': './reconstructed_images/swav/swav_True_goldfish.jpeg',
                'byol': './reconstructed_images/byol/byol_True_goldfish.jpeg',
                'pirl': './reconstructed_images/pirl/pirl_True_goldfish.jpeg',
                'moco-v2': './reconstructed_images/moco-v2/moco-v2_True_goldfish.jpeg',
                'mimic-chexpert_lr_0.01': './reconstructed_images/mimic-chexpert_lr_0.01/mimic-chexpert_lr_0.01_True_goldfish.jpeg',
                'mimic-chexpert_lr_0.1': './reconstructed_images/mimic-chexpert_lr_0.1/mimic-chexpert_lr_0.1_True_goldfish.jpeg',
                'mimic-chexpert_lr_1.0': './reconstructed_images/mimic-chexpert_lr_1.0/mimic-chexpert_lr_1.0_True_goldfish.jpeg',
                'mimic-cxr_r18_lr_1e-4': './reconstructed_images/mimic-cxr_r18_lr_1e-4/mimic-cxr_r18_lr_1e-4_True_goldfish.jpeg',
                'mimic-cxr_d121_lr_1e-4': './reconstructed_images/mimic-cxr_d121_lr_1e-4/mimic-cxr_d121_lr_1e-4_True_goldfish.jpeg',
                'supervised_r50': './reconstructed_images/supervised_r50/supervised_r50_True_goldfish.jpeg',
                'supervised_r18': './reconstructed_images/supervised_r18/supervised_r18_True_goldfish.jpeg',
                'supervised_d121': './reconstructed_images/supervised_d121/supervised_d121_True_goldfish.jpeg'
                },
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute and save perceptual distances between two images.')
    # parser.add_argument('-im1', '--image-1', type=str, default='', help='path for image 1')
    # parser.add_argument('-im2', '--image-2', type=str, default='', help='path for image 2')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('--device', type=str, default='cpu', help='CUDA or CPU (cuda | cpu)')
    args = parser.parse_args()
    pprint(args)
    
    results_dict_alex = {}
    results_dict_vgg = {}
    results_dict_squeezenet = {}
    csv_columns = ['model']

    for dataset in original_images:
        #im1_name = original_images[dataset][0]
        im1_path = original_images[dataset][1]
        im1 = open_and_convert_image(im1_path, args.image_size)
        results_dict_alex[dataset] = {}
        results_dict_vgg[dataset] = {}
        results_dict_squeezenet[dataset] = {}

        csv_columns.append(dataset)
        
        for model in reconstructed_images[dataset]:
            im2_path = reconstructed_images[dataset][model]
            im2 = open_and_convert_image(im2_path, args.image_size)

            d_alex, d_vgg, d_squeeze = perceptual_distance(im1, im2, args.device)
            
            # results_dict[dataset][model] = {
            #     'AlexNet' : d_alex,
            #     'VGG' : d_vgg,
            #     'SqueezeNet' : d_squeeze,
            # }
            # print(f'Perceptual distance between images {im1_name} and {im2_path}:')
            
            # pprint(results_dict[dataset][model])

            results_dict_alex[dataset][model] = d_alex
            results_dict_vgg[dataset][model] = d_vgg
            results_dict_squeezenet[dataset][model] = d_squeeze
    
    # Flip the nested structure on the results for csv saving
    flipped = defaultdict(dict)
    for key, val in results_dict_alex.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    results_dict_alex = dict(flipped)

    flipped = defaultdict(dict)
    for key, val in results_dict_vgg.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    results_dict_vgg = dict(flipped)

    flipped = defaultdict(dict)
    for key, val in results_dict_squeezenet.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    results_dict_squeezenet = dict(flipped)

    try:
        with open('./results/perceptual-distance/alexnet_distances.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for model in results_dict_alex:
                row = dict({'model': model}, **results_dict_alex[model])
                writer.writerow(row)
        
        with open('./results/perceptual-distance/vgg_distances.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for model in results_dict_vgg:
                row = dict({'model': model}, **results_dict_vgg[model])
                writer.writerow(row)
        
        with open('./results/perceptual-distance/squeezenet_distances.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for model in results_dict_squeezenet:
                row = dict({'model': model}, **results_dict_squeezenet[model])
                writer.writerow(row)
    except IOError:
        print("I/O error")