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

transform=translation
dset=diabetic_retinopathy
k=20
python invariances.py --dataset $dset --model moco-v2 --transform $transform --k $k
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm --k $k
python invariances.py --dataset $dset --model pirl --transform $transform --k $k
python invariances.py --dataset $dset --model swav --transform $transform --k $k
python invariances.py --dataset $dset --model byol --transform $transform --k $k
python invariances.py --dataset $dset --model supervised_r18 --transform $transform --k $k
python invariances.py --dataset $dset --model supervised_d121 --transform $transform --k $k
python invariances.py --dataset $dset --model supervised_r50 --transform $transform --k $k

