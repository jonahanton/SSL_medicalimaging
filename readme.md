# How Well (and Why) Do Self-Supervised Models Transfer to Medical Imaging? 
This repository contains the codebase for all experiments for the Software Engineering Group Project `How Well (and Why) Do Self-Supervised Models Transfer to Medical Imaging?` (Imperial MSc AI 2022). <br />
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

Code for image reconstruction is adapted from: 

```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}

@inproceedings{ZhaoICLR2021, 
    author = {Nanxuan Zhao and Zhirong Wu and Rynson W.H. Lau and Stephen Lin}, 
    title = {What Makes Instance Discrimination Good for Transfer Learning?}, 
    booktitle = {ICLR}, 
    year = {2021} 
}
```


## Pre-trained Models
We evaluate the following pretrained ResNet50 models (with links)

| Model | URL |
|-------|-----|
| PIRL | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| MoCo-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| SimCLR-v1 | https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip |
| BYOL | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| SwAV | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| Supervised | Weights from `torchvision.models.resnet50(pretrained=True)` |

**Note 1**: For SimCLR-v1, the TensorFlow checkpoints need to be downloaded manually and converted into PyTorch format (using https://github.com/tonylins/simclr-converter).

**Note 2**: In order to convert BYOL, you may need to install some packages by running:
```
pip install jax jaxlib dill git+https://github.com/deepmind/dm-haiku
```
