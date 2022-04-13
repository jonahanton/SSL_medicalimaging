#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/g21mscprj03/SSL/out/reconstruction/%j.out

export PATH=/vol/bitbucket/g21mscprj03/sslvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/g21mscprj03/SSL

input_dir = './data/diabetic_retinopathy/train/10_left.jpeg'
max_iter = 50
clip = true

python reconstruction -m swav --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m byol --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m pirl --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m moco-v2 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m mimic-chexpert_lr_0.01 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m mimic-chexpert_lr_0.1 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m mimic-chexpert_lr_1.0 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m mimic-cxr_r18_lr_1e-4 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m -m mimic-cxr_d121_lr_1e-4 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m supervised_r50 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m supervised_r18 --input_dir $input_dir --max_iter $max_iter --clip $clip
python reconstruction -m supervised_d121 --input_dir $input_dir --max_iter $max_iter --clip $clip



