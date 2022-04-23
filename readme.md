# How Well Do Self-Supervised Models Transfer to Medical Imaging?
This repository contains the codebase for all experiments for the Software Engineering Group Project `How Well Do Self-Supervised Models Transfer to Medical Imaging?` (Imperial MSc AI 2022). <br />
Authors: Jonah Anton, Liam Castelli, Wan Hee Tang, Venus Cheung, Mathilde Outters, Mun Fai Chan

Much of the code is adapted from the codebase for the CVPR 2021 paper [How Well Do Self-Supervised Models Transfer?](https://arxiv.org/abs/2011.13377)

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

The code for the invariance evaluation is adapted from the codebase for paper [Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks](https://arxiv.org/abs/2111.11398):

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

The code for the deep prior image reconstruction is adapted from the codebase (https://github.com/DmitryUlyanov/deep-image-prior) for the paper [Deep Image Prior](https://arxiv.org/abs/1711.10925):

```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```

## Methods

We evaluate the transfer peformance of several self-supervised pretrained models on medical image classification tasks. We also perform the same evaluation on a selection of supervised pretrained models and self-supervised medical domain-specific pretrained models (both pretrained on X-ray datasets). The pretrained models, datasets and evaulation methods are detailed in this readme.

## Directory Structure

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


## Pretrained Models
We evaluate the following pretrained ResNet50 models (with links)

| Model | URL |
|-------|-----|
| PIRL | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| MoCo-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| SimCLR-v1 | https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip |
| BYOL | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| SwAV | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| Supervised_r50 | Weights from `torchvision.models.resnet50(pretrained=True)` |

To download and prepare all the above models in the same format, run python download_and_prepare_models.py. This will save the prepared models in a directory `models/`. <br />

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

**Note 1**: For MIMIC-CheXpert we use the following model names (three different MoCo pretraining learning rates: 0.01, 0.1, 1.0): 
1. `mimic-chexpert_lr_0.01_bs_128_fd_128_qs_65536.pt` 
2. `mimic-chexpert_lr_0.1_bs_128_fd_128_qs_65536.pt`
3. `mimic-chexpert_lr_1.0_bs_128_fd_128_qs_65536.pt` 

All MIMIC-CheXpert models use a DenseNet121 backbone. <br />

**Note 2**: For MoCo-CXR we use both the ResNet18 and DenseNet121 pretrained models (both with learning rate 1e-4), found in URLs:
1. https://storage.googleapis.com/moco-cxr/r8w-00001.pth.tar 
2. https://storage.googleapis.com/moco-cxr/d1w-00001.pth.tar 

**Note 3**: For SimCLR-v1, the TensorFlow checkpoints need to be downloaded manually and converted into PyTorch format (using https://github.com/tonylins/simclr-converter).

**Note 4**: In order to convert BYOL, you may need to install some packages by running:
```
pip install jax jaxlib dill git+https://github.com/deepmind/dm-haiku
```

## Datasets
The data directory should be set up with the following structure:

    ├── data
        ├── bach
            ├── ICIAR2018_BACH_Challenge
        ├── chestx
            ├── Data_Entry_2017.csv
            ├── images
        ├── chexpert
            ├── CheXpert-v1.0
        ├── CIFAR10
            ├── cifar-10-batches-py
        ├── diabetic_retinopathy
            ├── train
            ├── trainLabels.csv
            ├── test
            ├── testLabels.csv
        ├── ichallenge_amd
            ├── Training400
        ├── ichallenge_pm
            ├── PALM-Training400
        ├── montgomerycxr
            ├── montgomery_metadata.csv
            ├── MontgomerySet
        ├── shenzhencxr
            ├── shenzhencxr_metadata.csv
            ├── ChinaSet_AllFiles
        ├── stoic
            ├── metadata
            ├── data
         
    
Links for where to download each dataset are given here:
[BACH](https://zenodo.org/record/3632035),
[ChestX](https://www.kaggle.com/nih-chest-xrays/data),
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/),
[CIFAR10](https://pytorch.org/vision/stable/datasets.html),
[Diabetic Retinopathy](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data),
[iChallenge-AMD](https://ai.baidu.com/broad/subordinate?dataset=amd),
[iChallenge-PM](https://ai.baidu.com/broad/subordinate?dataset=pm),
[Montgomery-CXR](https://openi.nlm.nih.gov/faq#faq-tb-coll),
[Shenzhen-CXR](https://openi.nlm.nih.gov/faq#faq-tb-coll),
[STOIC](https://registry.opendata.aws/stoic2021-training/)

### Note:
Downloading and unpacking the files above into the relevant directory should yield the structure above. A few of the datasets need additional tinkering to get into the desired format, and we give the instructions for those datasets here:

**ChestX**: Unpacking into the chestx directory, the various image folders (images_001, images_002,...,images_012) were combined so that all image files were contained directly in a single images directory as in the above structure. This can be done by repeated usage of:
```
mv images_0XX/* images/
```
**Diabetic Retionpathy**: To unpack the various train.zip and test.zip files we used the following commands:
```
cat train.zip.* > train.zip
unzip train.zip
cat test.zip.* > test.zip
unzip test.zip
```
Should you encounter problems, see the following discussion which we found was helpful to unpack the various zip files: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/12545. If problems persist, check the MD5 hashes given here: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/12543

**Montgomery-CXR/Shenzhen-CXR**:  The links are under the "I have heard about the Tuberculosis collection. Where can I get those images ?" section

## Few-shot
We provide the code for few-shot evaluation in few_shot.py. We use the technique of Prototypical Networks [Prototypical Networks for Few-Shot Learning](https://arxiv.org/abs/1703.05175).

For example, to evaluate MoCo-v2 on the dataset ShenzhenCXR in a 2-way 20-shot setup, run:
```
python few_shot.py --dataset shenzhen --model moco-v2 --n-way 2 --n-support 20
```
This will save a log of the run (with the results) in the filepath `logs/few-shot/moco-v2/shezhencxr.log`. The test accuracy should be close to 73.76% ± 0.66%. <br />

Or, to evaluate the MIMIC-CheXpert (lr = 0.01) model on ChestX-ray8 in a 5-way 20-shot setup, run:
```
python few_shot.py --dataset chestx --model mimic-chexpert_lr_0.01 --n-way 5 --n-support 20
```
This will save a log of the run (with the results) in the filepath `logs/mimic-chexpert_lr_0.01/moco-v2/chestx.log`. The test accuracy should be close to 33.73% ± 0.45%. <br />

**Note**: <br />
Within few_shot.py a dictionary is produced for the specified dataset where a list is created for each class containing all images for that class. This is neccessary for the random sampling of images during a few-shot episode. However, the creation of this dictionary, `sub_meta`, requires one complete pass over the entire dataset. For the larger datasets we use, namely CheXpert, ChestX-ray8 and EyePACS (diabetic retinopathy), we found that this process is extremely slow. Therefore, to prevent the creation of the sub_meta dict from stratch every time few_shot.py is called for these datasets, the script `datasets/prepare_submeta.py` will create the sub_meta dict (for a maximum of 10,000 images) and store it as a pickle file. This can then be re-loaded in when few_shot.py is called for these datasets.<br />
E.g., to run `datasets/prepare_submeta.py` for CheXpert:
```
python -m datasets.prepare_submeta --dataset chexpert
``` 
The pickle file will be saved in the filepath `misc/few_shot_submeta/chexpert.pickle` and will be automatically loaded by few_shot.py when called with `--dataset chexpert`.

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
