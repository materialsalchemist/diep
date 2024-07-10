#!/bin/bash
#SBATCH --job-name=diep_pes_2_2
#SBATCH --output=diep_pes_2_2.out
#SBATCH --error=diep_pes_2_2.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=s.abbas@deakin.edu.au
#SBATCH --time=10-00:00:00


module purge

eval "$(conda shell.bash hook)"
conda activate base


python train_pes_mp.py --model dft --dataset mp_pes --no-uses --fw 1 --lr 0.0001 --lg --exclude_force_outliers  --full_dataset --forcelimit 10 --max_n 2 --max_l 2