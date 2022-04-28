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

# transform=rotation
# dset=diabetic_retinopathy
# k=20
# python -m invariances.invariances --dataset $dset --model supervised_r50 --transform $transform --k $k
# transform=rotation
# dset=ichallenge_pm
# k=20
# python -m invariances.invariances --dataset $dset --model supervised_r50 --transform $transform --k $k

# transform=h_flip
# dset=shenzhencxr
# python -m invariances.invariances --dataset $dset --model supervised_r50 --transform $transform
# transform=h_flip
# dset=montgomerycxr
# python -m invariances.invariances --dataset $dset --model supervised_r50 --transform $transform


# transform=hue
# dset=bach
# k=20
# python -m invariances.invariances --dataset $dset --model supervised_r50 --transform $transform --k $k

transform=h_flip
dset=chestx
model=simclr-v1
python -m invariances.invariances --dataset $dset --model $model --transform $transform --no-norm
model=moco-v2
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=swav
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=byol
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=pirl
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=supervised_r50
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=supervised_r18
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=supervised_d121
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=mimic-chexpert_lr_0.01
python -m invariances.invariances --dataset $dset --model $model --transform $transform 
model=mimic-cxr_d121_lr_1e-4
python -m invariances.invariances --dataset $dset --model $model --transform $transform 