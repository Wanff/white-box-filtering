#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=llamaguard_harmbench_alpaca_seeds
#SBATCH --output=llamaguard_harmbench_alpaca_seeds.out

for seed in 0 1 2 3 4; do
    python classifier_finetune.py \
        --seed $seed
    echo "Done with seed $seed"
done