#!/bin/bash

#SBATCH -J bsd68s
#SBATCH --output=slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 15:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load any/python/3.8.3-conda

conda activate n2s

cd noise2same/noise2same.pytorch

python train.py +experiment=bsd68 +backbone=swinir project=noise2same-bsd68 \
                training.batch_size=8
