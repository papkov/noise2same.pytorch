#!/bin/bash

#SBATCH -J bsd68
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 80:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --exclude=falcon2,falcon3,falcon4,falcon5


module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same/noise2same.pytorch

python train.py +experiment=bsd68 +backbone=swinia project=bsd68 \
                training.batch_size=64 model.mode=noise2self training.crop=64 backbone.window_size=8
