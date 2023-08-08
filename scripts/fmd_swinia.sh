#!/bin/bash

#SBATCH -J fmd
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1

module load any/python/3.8.3-conda
conda activate n2s_env

cd noise2same || return

python train.py +experiment=fmd +backbone=swinia project=fmd \
       dataset.part=cf_fish dataset.mean=11.0421 dataset.std=23.3389
#       dataset.part=cf_mice dataset.mean=16.7063 dataset.std=20.6395
#       dataset.part=tp_mice dataset.mean=29.1322 dataset.std=24.1591

