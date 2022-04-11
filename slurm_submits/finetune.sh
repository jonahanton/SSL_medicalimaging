#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/g21mscprj03/SSL/out/finetune/%j.out


export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/g21mscprj03/SSL

# dset=chexpert
dset=diabetic_retinopathy
# python finetune.py -d $dset -m mimic-chexpert_lr_0.1 --early-stopping -b 16
python finetune.py -d $dset -m simclr-v1 --no-norm --early-stopping
