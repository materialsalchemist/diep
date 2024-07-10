#!/bin/bash
#SBATCH --job-name=savemodel
#SBATCH --output=savemodel.out
#SBATCH --error=savemodel.err
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
python3 save_model.py