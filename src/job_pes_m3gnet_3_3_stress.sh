#!/bin/bash
#SBATCH --job-name=m3gnet_pes_stress
#SBATCH --output=m3gnet_pes_stress.out
#SBATCH --error=m3gnet_pes_stress.err
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
python train_pes_mp.py --model m3gnet --dataset mp_pes --uses --fw 1 --sw 0.1 --lr 0.0001 --lg --exclude_force_outliers --full_dataset
