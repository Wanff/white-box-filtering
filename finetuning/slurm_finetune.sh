#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=llamaguard_harmbench_alpaca
#SBATCH --output=llamaguard_harmbench_alpaca.out

python classifier_finetune.py