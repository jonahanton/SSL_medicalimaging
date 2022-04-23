# How Well Do Self-Supervised Models Transfer to Medical Imaging?
This repository contains the codebase for all experiments for the Software Engineering Group Project `How Well Do Self-Supervised Models Transfer to Medical Imaging?` (Imperial MSc AI 2022). <br />
Authors: Jonah Anton, Liam Castelli, Wan Hee Tang, Venus Cheung, Mathilde Outters, Mun Fai Chan

Much of the code is taken/adapted from the codebase for the CVPR 2021 paper [How Well Do Self-Supervised Models Transfer?](https://arxiv.org/abs/2011.13377)

```
@inproceedings{Ericsson2021HowTransfer,
    title = {{How Well Do Self-Supervised Models Transfer?}},
    year = {2021},
    booktitle = {CVPR},
    author = {Ericsson, Linus and Gouk, Henry and Hospedales, Timothy M.},
    url = {http://arxiv.org/abs/2011.13377},
    arxivId = {2011.13377}
}
```

The code for the synthetic invariance evaluation is adapted from:

```
@misc{ericsson2021selfsupervised,
      title={Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks},
      author={Linus Ericsson and Henry Gouk and Timothy M. Hospedales},
      year={2021},
      eprint={2111.11398},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Code for the deep prior image reconstruction is adapted from the codebase (https://github.com/DmitryUlyanov/deep-image-prior) for the paper [Deep Image Prior](https://arxiv.org/abs/1711.10925):

```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```

## Files

    ├── data
    ├── datasets
    │   ├── .                   
    ├── invariances
    ├── logs
    ├── models
    ├── misc
    ├── sample_images
    ├── saliency_maps
    ├── reconstruction
    ├── reconstructed_images         


## Pre-trained Models
We evaluate the following pretrained ResNet50 models (with links)

| Model | URL |
|-------|-----|
| PIRL | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| MoCo-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| SimCLR-v1 | https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip |
| BYOL | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| SwAV | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| Supervised_r50 | Weights from `torchvision.models.resnet50(pretrained=True)` |

We also evaluate the supervised pretrained ResNet18 and DenseNet121 models

| Model | URL |
|-------|-----|
| Supervised_r18 | Weights from `torchvision.models.resnet18(pretrained=True)` |
| Supervised_d121 | Weights from `torchvision.models.densenet121(pretrained=True)` |

We also evaluate the following pretrained medical domain-specific SSL pretrained models (with links to models provided in the linked github repos)
| Model | URL |
|-------|-----|
| MIMIC-CheXpert | Models found in github repo https://github.com/facebookresearch/CovidPrognosis, model URL https://dl.fbaipublicfiles.com/CovidPrognosis/pretrained_models/ |
| MoCo-CXR | Models found in github repo https://github.com/stanfordmlgroup/MoCo-CXR  |

Note that for MIMIC-CheXpert we use the following model names (three different MoCo pretraining learning rates: 0.01, 0.1, 1.0): 
1. mimic-chexpert_lr_0.01_bs_128_fd_128_qs_65536.pt 
2. mimic-chexpert_lr_0.1_bs_128_fd_128_qs_65536.pt 
3. mimic-chexpert_lr_1.0_bs_128_fd_128_qs_65536.pt 

All MIMIC-CheXpert models use a DenseNet121 backbone. <br />

Note that for MoCo-CXR we use both the ResNet18 and DenseNet121 pretrained models (both with learning rate 1e-4), found in URLs:
1. https://storage.googleapis.com/moco-cxr/r8w-00001.pth.tar 
2. https://storage.googleapis.com/moco-cxr/d1w-00001.pth.tar 

**Note 1**: For SimCLR-v1, the TensorFlow checkpoints need to be downloaded manually and converted into PyTorch format (using https://github.com/tonylins/simclr-converter).

**Note 2**: In order to convert BYOL, you may need to install some packages by running:
```
pip install jax jaxlib dill git+https://github.com/deepmind/dm-haiku
```

## Datasets
[To do - Liam]

## Few-shot
[To do - Jonah]

## Many-shot (Finetune)
[To do - Jonah]

## Many-shot (Linear)
[To do - Jonah]

## Saliency Maps
[To do - Jonah]

## Deep Image Prior
[To do - Mun Fai, Mathilde]

## Perceptual Distance
[To do - Jonah]

## Invariances
[To do - Jonah]
