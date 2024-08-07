#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=gemma_fpr_get
#SBATCH --output=gemma_fpr_get.out

# python eval.py --probe_type 'mlp'

# GET FPRs
for seed in 0 1 2 3 4
do
    python temp_fpr_preds_get.py --seed $seed
done

echo "Done"