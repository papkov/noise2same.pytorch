#!/bin/bash

#SBATCH -J poiss30
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon4

module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same || return

python train.py +backbone=swinia +experiment=synthetic project=synthetic \
                dataset.noise_param=30 dataset.noise_type=poisson # dataset.standardize=False
