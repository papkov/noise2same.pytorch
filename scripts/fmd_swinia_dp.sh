#!/bin/bash

#SBATCH -J cf_fish
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH --nodelist=falcon1


module load any/python/3.8.3-conda
module load cuda/11.3.1

conda activate n2s_env

cd swinia

python train.py +experiment=fmd +backbone=swinia project=fmd model.mode=noise2self data.part=cf_fish