#!/bin/bash
#SBATCH --job-name=diep_pes_lr0.001_ew
#SBATCH --output=diep_pes_lr0.001_ew.out
#SBATCH --error=diep_pes_lr0.001_ew.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=50
#SBATCH --mail-user=s.abbas@deakin.edu.au
#SBATCH --time=10-00:00:00


module purge

eval "$(conda shell.bash hook)"
conda activate base

python train_pes_mp.py --model dft --dataset mp_pes --no-uses --ew 0.1 --sw 0 --fw 1 --lr 0.0001 --lg --exclude_force_outliers  --full_dataset

# pesm3gnet_5_forcelimit_20
# python train_pes_mp.py --model dft --dataset mp_pes --no-uses --fw 1 --lr 0.0001 --lg --exclude_force_outliers  --full_dataset --forcelimit 20 --max_n 1 --max_l 1