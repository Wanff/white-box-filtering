#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=llamaguard_generated_new
#SBATCH --output=llamaguard_generated_new.out

python classifier_finetune.py