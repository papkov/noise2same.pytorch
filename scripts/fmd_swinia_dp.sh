#!/bin/bash

#SBATCH -J tp_mice_pegasus1
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:2
#SBATCH --exclude=falcon2,falcon3


module load any/python/3.8.3-conda
module load cuda/11.3.1

conda activate n2s_env

cd swinia

python train.py +experiment=fmd +backbone=swinia project=fmd model.mode=noise2self data.part=tp_mice \
                backbone.embed_dim=96
