#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=llamaguard_hb_alpaca
#SBATCH --output=llamaguard_hb_alpaca.out

python classifier_finetune.py