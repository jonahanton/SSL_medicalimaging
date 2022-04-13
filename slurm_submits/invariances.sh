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

transform=rotation
dset=montgomerycxr
python invariances.py --dataset $dset --model mimic-chexpert_lr_0.01 --transform $transform
python invariances.py --dataset $dset --model mimic-cxr_d121_lr_1e-4 --transform $transform
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform

transform=h_flip
dset=montgomerycxr
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform

transform=rotation
dset=diabetic_retinopathy
python invariances.py --dataset $dset --model mimic-chexpert_lr_0.01 --transform $transform
python invariances.py --dataset $dset --model mimic-cxr_d121_lr_1e-4 --transform $transform
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform

transform=h_flip
dset=diabetic_retinopathy
python invariances.py --dataset $dset --model mimic-chexpert_lr_0.01 --transform $transform
python invariances.py --dataset $dset --model mimic-cxr_d121_lr_1e-4 --transform $transform
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform


transform=rotation
dset=ichallenge_pm
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform

transform=h_flip
dset=ichallenge_pm
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform

transform=h_flip
dset=ichallenge_amd
python invariances.py --dataset $dset --model moco-v2 --transform $transform
python invariances.py --dataset $dset --model simclr-v1 --transform $transform --no-norm
python invariances.py --dataset $dset --model pirl --transform $transform
python invariances.py --dataset $dset --model swav --transform $transform
python invariances.py --dataset $dset --model byol --transform $transform
python invariances.py --dataset $dset --model supervised_r18 --transform $transform
python invariances.py --dataset $dset --model supervised_d121 --transform $transform