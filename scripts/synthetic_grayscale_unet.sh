#!/bin/bash

#SBATCH -J sgsu-g50
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 15:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2,falcon3,falcon4,falcon5,falcon6

module load any/python/3.8.3-conda
module load cuda/11.3.1

conda activate n2s_env

cd swinia

python train.py +experiment=synthetic_grayscale +backbone=unet project=synthetic_grayscale model.mode=noise2same \
                data.noise_type=gaussian data.noise_param=50 \
