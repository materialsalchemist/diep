#!/bin/bash
#SBATCH --job-name=test_mym3gnet
#SBATCH --output=test_mym3gnet.out
#SBATCH --error=test_mym3gnet.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=50
#SBATCH --mail-user=s.abbas@deakin.edu.au
#SBATCH --time=10-00:00:00


module purge

eval "$(conda shell.bash hook)"
conda activate base
python test.py mym3gnet
