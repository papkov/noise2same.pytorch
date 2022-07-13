#!/bin/bash

#SBATCH -J bsd68
#SBATCH --output=slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load any/python/3.8.3-conda

conda activate n2s

cd noise2same/noise2same.pytorch

python evaluate.py +experiment=bsd68 +backbone=swinir project=noise2same-bsd68 training.batch_size=8 \
                   +checkpoint=/gpfs/space/home/chizhov/noise2same/noise2same.pytorch/checkpoints/model_last.pth
