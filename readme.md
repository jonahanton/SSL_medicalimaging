# How Well (and Why) Do Self-Supervised Models Transfer to Medical Imaging? 
This repository contains the codebase for all experiments for the Software Engineering Group Project `How Well (and Why) Do Self-Supervised Models Transfer to Medical Imaging?` (Imperial MSc AI 2022). <br />
Authors: Jonah Anton, Liam Castelli, Wan Hee Tang, Venus Cheung, Mathilde Outters, Mun Fai Chan


## Pre-trained Models
We evaluate the following pretrained ResNet50 models (with links)

| Model | URL |
|-------|-----|
| PIRL | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| MoCo-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| SimCLR-v2 | https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0 |
| BYOL | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| SwAV | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| Supervised | Weights from `torchvision.models.resnet50(pretrained=True)` |

**Note 1**: For SimCLR-v2, the TensorFlow checkpoints need to be downloaded manually and converted into PyTorch format (using https://github.com/tonylins/simclr-converter and https://github.com/Separius/SimCLRv2-Pytorch, respectively).

**Note 2**: In order to convert BYOL, you may need to install some packages by running:
```
pip install jax jaxlib dill git+https://github.com/deepmind/dm-haiku
```
