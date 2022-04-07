#!/bin/bash

dset=shenzhen_cxr

#SBATCH --gres=gpu:1
#SBATCH --output=out/linear_%j.out

export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
# TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
# uptime

cd /vol/bitbucket/g21mscprj03/SSL

# python linear.py -d cifar10 -m moco-v2 -c 316227.7712565657
python linear.py -d $dset -m moco-v2
