#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=Base64Finetune
#SBATCH --output=alpaca_base64.out

python finetune.py