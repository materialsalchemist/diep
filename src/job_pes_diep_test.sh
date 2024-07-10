#!/bin/bash
#SBATCH --job-name=test_pes_mp_diep
#SBATCH --output=test_pes_mp_diep.out
#SBATCH --error=test_pes_mp_diep.err
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

python test_pes_mp.py --model dft --dataset mp_pes --no-uses --fw 1 --lr 0.0001 --lg --exclude_force_outliers  --full_dataset  --ckpt_path "logs/PES_2_DFT_training_lg_mp_pes_fw_1.0_lr_0.0001_full_dataset_True_exclude_force_outliers_True_epochs_500/epoch=238-step=1245907.ckpt"
