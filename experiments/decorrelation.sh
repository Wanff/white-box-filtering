#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=decorrelation
#SBATCH --output=decorrelation.out

python decorrelation.py