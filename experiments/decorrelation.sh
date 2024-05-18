#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=decorrelation
#SBATCH --output=decorrelation_5.out

python decorrelation.py --probe_type 'mlp'