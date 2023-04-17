#!/bin/bash

#SBATCH -J cf_fish
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 5:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2,falcon3

module load any/python/3.8.3-conda

conda activate n2s_env

cd swinia

python train.py +experiment=fmd_deconvolution +backbone=unet project=fmd_deconvolution data.part=cf_fish \
                model.mode=noise2same model.lambda_inv=0 model.lambda_inv_deconv=4 \
                training.amp=False
#                model.lambda_bound=0.1 model.regularization_key=deconv

