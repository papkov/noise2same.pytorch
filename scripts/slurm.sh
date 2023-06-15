#!/bin/bash
srun --pty \
     --partition=gpu \
      --nodes=1 \
      --time=24:00:00 \
      --gres=gpu:tesla:1 \
      --mem=16000 \
      --cpus-per-task=8 \
      --job-name=ssi \
      --mail-type=ALL \
      --mail-user=mikhail.papkov@gmail.com \
      bash