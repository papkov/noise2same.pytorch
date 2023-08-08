#!/bin/bash

#SBATCH -J imagenet
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 15:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1

module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same || return

python train.py +backbone=swinia +experiment=imagenet project=imagenet
