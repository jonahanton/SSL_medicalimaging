#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/g21mscprj03/SSL/out/few-shot/%j.out

export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/g21mscprj03/SSL

dset=bach
# python few_shot.py -d $dset -m simclr-v1 --no-norm --n-way 2 --n-support 20
python few_shot.py -d $dset -m swav --n-way 2 --n-support 20
python few_shot.py -d $dset -m byol --n-way 2 --n-support 20
python few_shot.py -d $dset -m pirl --n-way 2 --n-support 20
python few_shot.py -d $dset -m moco-v2 --n-way 2 --n-support 20
python few_shot.py -d $dset -m mimic-chexpert_lr_0.01 --n-way 2 --n-support 20
python few_shot.py -d $dset -m mimic-chexpert_lr_0.1 --n-way 2 --n-support 20
python few_shot.py -d $dset -m mimic-chexpert_lr_1.0 --n-way 2 --n-support 20
python few_shot.py -d $dset -m mimic-cxr_r18_lr_1e-4 --n-way 2 --n-support 20
python few_shot.py -d $dset -m mimic-cxr_d121_lr_1e-4 --n-way 2 --n-support 20
python few_shot.py -d $dset -m supervised_r50 --n-way 2 --n-support 20
python few_shot.py -d $dset -m supervised_r18 --n-way 2 --n-support 20
python few_shot.py -d $dset -m supervised_d121 --n-way 2 --n-support 20


