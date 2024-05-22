#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=fpr_seed_4
#SBATCH --output=fpr_seed_4.out

python eval.py --probe_type 'mlp'