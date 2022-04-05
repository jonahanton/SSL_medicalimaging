#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --output=out/fewshot_%j.out


export PATH=/vol/bitbucket/${USER}/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
# TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
# uptime

cd /vol/bitbucket/jla21/SEGP/SSL
python few_shot.py -d cifar10 -m moco-v2 --n-way 5 --n-support 20

