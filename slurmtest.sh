#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<your_username> # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
echo Starting Script
python3 ~/SSL/main_pretrain.py -m simclr --dataset-name CheXpert --epochs 5
/usr/bin/nvidia-smi
uptime