#!/bin/bash

#SBATCH -J hanzi
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --exclude=falcon3

module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same/noise2same.pytorch

python train.py +experiment=hanzi +backbone=unet model.mode=noise2self project=hanzi
