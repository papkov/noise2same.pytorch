#!/bin/bash

#SBATCH -J hela
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 10:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2,falcon3

module load any/python/3.8.3-conda
conda activate torch38
cd noise2same || return

python train.py +backbone=unet +denoiser=deconv2same +experiment=microtubules_generated  \
                project=noise2same-ssi-mt-gen-v2 \
#                denoiser.lambda_inv=0 denoiser.lambda_inv_deconv=2 \
                denoiser.regularization_key=image denoiser.lambda_bound=0.1 \
                psd.pad_mode=constant