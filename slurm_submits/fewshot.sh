#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/g21mscprj03/SSL/out/few-shot/%j.out

dset=shenzhencxr
model=swav

export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/g21mscprj03/SSL

# python few_shot.py -d $dset -m $model --n-way 5 --n-support 20
python few_shot.py -d $dset -m simclr-v1 --no-norm --n-way 5 --n-support 20