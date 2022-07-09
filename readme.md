# How Well Do Self-Supervised Models Transfer to Medical Imaging?
This repository contains the codebase for all experiments for the Software Engineering Group Project `How Well Do Self-Supervised Models Transfer to Medical Imaging?` (Imperial MSc AI 2022). <br />
Authors: [Jonah Anton](https://github.com/jonahanton), [Liam Castelli](https://github.com/mailingliam02), [Wan Hee Tang](https://github.com/wh-tang), Venus Cheung, [Mathilde Outters](https://github.com/mathildeoutters), [Mun Fai Chan](https://github.com/ChanMunFai)

Abstract:
`
Self-supervised learning approaches have seen success transferring within domain for medical imaging, however there has been no large scale attempt to compare the transferability of self-supervised models against each other on medical images. In this study, we compare the generalisability of seven self-supervised models, two of which were trained in-domain, against supervised baselines across nine different medical datasets. We find that ImageNet pretrained self-supervised models are more generalisable, and benefit significantly from in-domain training for in-domain downstream tasks. However, this training drastically reduces performance for out-of-domain downstream tasks. Our investigation of the feature representations suggests that this trend may be due to the models learning to focus too heavily on specific areas.
`

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

# Table of Contents 
   * [Directory Structure](#Directory-Structure)
   * [Methods](#Methods)
      * [Pretrained Models](#Pretrained-Models)
      * [Datasets](#Datasets)
   * [Training](#Training)
      * [Few Shot](#Few-shot) 
      * [Many-shot(Finetune)](#Many-shot(Finetune))
      * [Many-shot(Linear)](#Many-shot(Linear))
  * [Deep Image Prior](#Deep-Image-Prior)
      * [Perceptual Distance](#Perceptual-Distance)
  * [Invariances](#Invariances)
      

# Directory Structure

    ├── data
    ├── datasets                  
    ├── invariances
    ├── logs
    ├── models
    ├── misc
    ├── sample_images
    ├── saliency_maps
    ├── reconstruction
    ├── reconstructed_images  

# Methods

We evaluate the transfer peformance of several self-supervised pretrained models on medical image classification tasks. We also perform the same evaluation on a selection of supervised pretrained models and self-supervised medical domain-specific pretrained models (both pretrained on X-ray datasets). The pretrained models, datasets and evaulation methods are detailed in this readme.

       
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
[ChestX-ray14](https://www.kaggle.com/nih-chest-xrays/data),
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/),
[CIFAR10](https://pytorch.org/vision/stable/datasets.html),
[EyePACS (Diabetic Retinopathy)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data),
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
**EyePACS (Diabetic Retionpathy)**: To unpack the various train.zip and test.zip files we used the following commands:
```
cat train.zip.* > train.zip
unzip train.zip
cat test.zip.* > test.zip
unzip test.zip
```
Should you encounter problems, see the following discussion which we found was helpful to unpack the various zip files: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/12545. If problems persist, check the MD5 hashes given here: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/12543

**Montgomery-CXR/Shenzhen-CXR**:  The links are under the "I have heard about the Tuberculosis collection. Where can I get those images ?" section


### Additional information:
For all datasets, the labels are converted to binary where possible. For CheXpert, this is done through many-to-one. All other pathologies are labelled as negative, and only the most common pathology, which for both datasets is Pleural Effusion, is assigned a positive label. For datasets with textual labels, like Montgomery and Shenzhen, we treat any abnormal X-ray as a positive label. A similar approach was taken with the iChallenge-PM dataset, combining the high myopia and pathological myopia into a single positive label. The datasets BACH and ChestX-ray8, which have multiclass categorical labels, are treated as ordinal.


# Training 

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
This will save a log of the run (with the results) in the filepath `logs/few-shot/mimic-chexpert_lr_0.01/chestx.log`. The test accuracy should be close to 33.73% ± 0.45%. <br />

**Note**: <br />
Within few_shot.py a dictionary is produced for the specified dataset where a list is created for each class containing all images for that class. This is neccessary for the random sampling of images during a few-shot episode. However, the creation of this dictionary, `sub_meta`, requires one complete pass over the entire dataset. For the larger datasets we use, namely CheXpert, ChestX-ray8 and EyePACS (diabetic retinopathy), we found that this process is extremely slow. Therefore, to prevent the creation of the sub_meta dict from stratch every time few_shot.py is called for these datasets, the script `datasets/prepare_submeta.py` will create the sub_meta dict (for a maximum of 10,000 images) and store it as a pickle file. This can then be re-loaded in when few_shot.py is called for these datasets.<br />
E.g., to run `datasets/prepare_submeta.py` for CheXpert:
```
python -m datasets.prepare_submeta --dataset chexpert
``` 
The pickle file will be saved in the filepath `misc/few_shot_submeta/chexpert.pickle` and will be automatically loaded by few_shot.py when called with `--dataset chexpert`.

## Many-shot (Finetune)
We provide the code for finetuning in finetune.py. By default, the pretrained model will be finetuned (with a linear classification head attached on) for 5000 steps with a batch size of 64, using SGD with Nesterov Momentum = 0.9 and a Cosine Annealing learning rate. The flat --early-stopping implements early stopping (with a patience = 3 by default (checked every 200 steps)). By default, the learning rate is set to 1e-2 and the weight decay to 1e-8, although a hyperparamter search can be initiated using the flat --search. By default random resized crop and random horizontal flip data augmentations will be applied for finetuning. 

For example, to evaluate MoCo-v2 on the dataset CheXpert (with early stopping), run:
```
python finetune.py --dataset chexpert --model moco-v2 --early-stopping
```
This will save a log of the run (with the results on the test set) in the filepath `logs/finetune/moco-v2/chexpert.log`. With early stopping implemented, the test accuracy (pleural effusion, many-to-one) should be close to 79.96%.

## Many-shot (Linear)
We provide the code for linear evaluation in linear.py. This will train linear regression on top of the features from the specified frozen model backbone. By default a hyperparameter search is first performed to find the l2 regularisation constant (`C` in sklearn), choosing between 45 logarithmically spaced points between 1e-6 and 1e5. To instead specify a C value directly, use the input flag --C.

For example, to evaluate PIRL on the dataset EyePACS (diabetic retinopathy), run:
```
python linear.py --dataset diabetic_retinopathy --model pirl 
```
This will save a log of the run (with the results on the test set) in the filepath `logs/linear/pirl/diabetic_retinopathy.log`. The test accuracy should be close to 31.51%, using C value 5623.413.

## Saliency Maps
We use the task-agnostic occlusion-based saliency method proposed in the paper [How Well Do Self-Supervised Models Transfer?](https://arxiv.org/abs/2011.13377) [Erricson et al., 2021]. A 10x10 occlusion mask is passed over the input image and the average feature distance is computed for each pixel. 

For example, to compute the saliency maps for the sample image `/patient00001_view1_frontal.jpg` from CheXpert, where this image is stored in the directory  `sample_images/chexpert/`, with the model MoCo-v2, run:
```
python saliency.py --dataset chexpert --model moco-v2 
```
This will save a log of the run in the filepath `logs/saliency/moco-v2.log`, which contains the attentive diffusion value for the produced saliency map. For the sample image from CheXpert with MoCo-v2, the attentive diffusion should be close to 48.64%. The produced saliency map (and the figure with the saliency map superimposed on top of the original image) will be saved in the directory `saliency_maps/moco-v2/chexpert`.

## Deep Image Prior
Using the methodology from the paper [What makes instance discrimination good for transfer learning?](https://arxiv.org/abs/2006.06606), which relies on the feature inversion algorithm [Deep Image Prior](https://arxiv.org/abs/1711.10925), we studied the ability to **reconstruct RGB images** from the features extracted by our pre-trained models. The code for such reconstructions can be found in ```reconstruction.py```.


**Note**: <br />
The reconstructed images will by default be saved into a ```reconstructed_images/``` directory. 
<br />
The paths to the sample images to be reconstructed are defined whithin ```reconstruction.py``` as a dictionary with structure ```{dataset_name: [file_name, file_path]}```.
<br />

### Perceptual Distance 
To quantify the quality of the reconstructed images, we use the **perceptual distance** metric from [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) in ```perceptual_distance.py```. A good reconstruction has low perceptual distance score.

Once the images have been reconstructed, run the following command to compute the perceptual distance score between original images and reconstructions:

```
python perceptual_distance.py 
```

**Note**: <br />
The code uses the same dictionary structure as described above and the reconstructed image paths are stored in a nested dictionary, with structure ```{dataset_name: {model_name: reconstructed_image_path}}```.
<br />
The perceptual distances will be computed by three different networks (AlexNet, VGG, SqueezeNet) and saved in three corresponding .csv files under ```results/perceptual-distance```.
<br />

## Invariances
We measure the invariances of features extracted from different pretrained models using the cosine similarity metric proposed in [Why Do Self-Supervised Models Transfer? Investigating teh Impact of Invariance on Downstream Tasks](https://arxiv.org/abs/2111.11398) [Ericsson et al., 2021]. The code is adapted from the original [GitHub repository](https://github.com/linusericsson/ssl-invariances) from this paper.

It is possible to either:

A. Compute invariances to synthetic data augmentations, e.g., rotations, horizontal / vertical flips, shears, hue transforms, etc. For example, to compute the invariance of CheXpert to horizontal flip augmentations for the MoCo pretrained model, run
```
python -m invariances.invariances --dataset chexpert --model moco-v2 --transform rotation
```
This will save a log of the run in the filepath `logs/invariances/moco-v2/rotation/chexpert.log`, containing the cosine similarity and Mahalonobis distance. Note that the files do not already exist (from previous ones), this will compute the covariance matrix and mean feature for the dataset CheXpert with MoCo-v2 and save it to the filepaths `misc/invariances/covmatrices/moco-v2_chexpert_feature_cov_matrix.pth`, `misc/invariances/covmatrices/moco-v2_chexpert_mean_feature.pth` respectively.

B. Compute invariances to different views of the same patient. This is only compatible with the CheXpert and EyePACS datasets, which both contain multiple images from different views of the same patient. For example, to compute the multi-view invariance of EyePACS, with the model SwAV, run:
```
python -m invariances.invariances_multiview --dataset diabetic_retinopathy --model swav
```
This will save a log of the run in the filepath `logs/invariances/swav/multi_view/diabetic_retinopathy.log`, containing the cosine similarity and Mahalonobis distance. Note that the files do not already exist (from previous ones), this will compute the covariance matrix and mean feature for the dataset CheXpert with MoCo-v2 and save it to the filepaths `misc/invariances/covmatrices/swav_diabetic_retinopathy_feature_cov_matrix.pth`, `misc/invariances/covmatrices/swav_diabetic_retinopathy_mean_feature.pth` respectively. 
