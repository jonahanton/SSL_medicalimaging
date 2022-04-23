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

dset=diabetic_retinopathy
python -m invariances.invariances_multiview --dataset $dset --model moco-v2 
python -m invariances.invariances_multiview --dataset $dset --model simclr-v1 --no-norm
python -m invariances.invariances_multiview --dataset $dset --model pirl 
python -m invariances.invariances_multiview --dataset $dset --model swav 
python -m invariances.invariances_multiview --dataset $dset --model byol
python -m invariances.invariances_multiview --dataset $dset --model supervised_r18 
python -m invariances.invariances_multiview --dataset $dset --model supervised_d121 
python -m invariances.invariances_multiview --dataset $dset --model supervised_r50


