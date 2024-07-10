#!/bin/bash
#SBATCH --job-name=savedatasts
#SBATCH --output=savedatasts.out
#SBATCH --error=savedatasts.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=s.abbas@deakin.edu.au
#SBATCH --time=10-00:00:00


module purge

eval "$(conda shell.bash hook)"
conda activate base
python save_datasets.py --exclude_force_outliers --full_dataset 
