#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=eval
#SBATCH --output=eval_5.out

python eval.py --probe_type 'sk'