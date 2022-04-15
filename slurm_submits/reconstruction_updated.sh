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

max_iter=3000
clip=True
dset=imagenet

# python reconstruction.py -m simclr-v1 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr stoic imagenet --max_iter $max_iter --clip $clip
# python reconstruction.py -m swav -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m byol -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m pirl -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m moco-v2 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m mimic-chexpert_lr_0.01 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m mimic-chexpert_lr_0.1 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m mimic-chexpert_lr_1.0 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m mimic-cxr_r18_lr_1e-4 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m mimic-cxr_d121_lr_1e-4 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m supervised_r50 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m supervised_r18 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip
# python reconstruction.py -m supervised_d121 -d bach chestx chexpert diabetic_retinopathy ichallenge_amd ichallenge_pm montgomerycxr shenzhencxr --max_iter $max_iter --clip $clip

python reconstruction.py -m simclr-v1 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m swav -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m byol -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m pirl -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m moco-v2 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m mimic-chexpert_lr_0.01 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m mimic-chexpert_lr_0.1 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m mimic-chexpert_lr_1.0 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m mimic-cxr_r18_lr_1e-4 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m mimic-cxr_d121_lr_1e-4 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m supervised_r50 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m supervised_r18 -d $dset --max_iter $max_iter --clip $clip
python reconstruction.py -m supervised_d121 -d $dset --max_iter $max_iter --clip $clip
