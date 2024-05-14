#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00
#SBATCH --job-name=llamaguard_generated_new
#SBATCH --output=llamaguard_generated_new.out

python classifier_finetune.py --file_spec 'generated_'