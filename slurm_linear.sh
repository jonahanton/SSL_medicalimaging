#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH –-mail-type=ALL
#SBATCH –-mail-user=jla21

export PATH=/vol/bitbucket/${USER}/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm

echo starting script
cd /vol/bitbucket/jla21/SEGP/SSL
python linear.py -d cifar10 -m moco-v2 -c 316227.7712565657

/usr/bin/nvidia-smi
uptime

