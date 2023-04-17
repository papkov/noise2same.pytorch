#!/bin/bash

#SBATCH -J fpoiss550
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon4

module load any/python/3.8.3-conda
module load cuda/11.3.1

conda activate n2s_env

cd swinia

python train.py +experiment=synthetic +backbone=swinia project=synthetic \
                backbone.window_size=8 model.mode=noise2self training.steps=50000 training.val_batch_size=4 \
                data.noise_type=poisson data.noise_param=[5,50] data.standardize=False