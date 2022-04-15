#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/g21mscprj03/SSL/out/invariances/%j.out


export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/g21mscprj03/SSL

dset=chexpert
python invariances_chexpert.py --dataset $dset --model moco-v2 
python invariances_chexpert.py --dataset $dset --model simclr-v1 --no-norm
python invariances_chexpert.py --dataset $dset --model pirl 
python invariances_chexpert.py --dataset $dset --model swav 
python invariances_chexpert.py --dataset $dset --model byol
python invariances_chexpert.py --dataset $dset --model supervised_r18 
python invariances_chexpert.py --dataset $dset --model supervised_d121 
python invariances_chexpert.py --dataset $dset --model supervised_r50
python invariances_chexpert.py --dataset $dset --model mimic-chexpert_lr_0.01
python invariances_chexpert.py --dataset $dset --model mimic-cxr_d121_lr_1e-4

