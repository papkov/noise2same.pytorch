#!/bin/bash

#SBATCH -J tp_mice
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 5:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --nodelist=falcon1

module load any/python/3.8.3-conda

conda activate n2s_env

cd swinia

python train.py +experiment=fmd +backbone=unet project=fmd model.mode=noise2self data.part=tp_mice
