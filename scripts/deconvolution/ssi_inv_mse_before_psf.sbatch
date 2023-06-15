#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --job-name=ssi
#SBATCH --output=./logs_ssi_inv_mse_before_psf/log-%J.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhail.papkov@gmail.com

module load any/python/3.8.3-conda cuda/11.3.1
source activate torch38
bash /gpfs/space/home/papkov/noise2same/scripts/ssi_inv_mse_before_psf.sh