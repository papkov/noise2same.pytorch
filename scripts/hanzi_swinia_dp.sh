#!/bin/bash

#SBATCH -J hanzi
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
# SBATCH --gres=gpu:a100-40g:2

module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same || return

python train.py +backbone=swinia +experiment=hanzi project=hanzi
