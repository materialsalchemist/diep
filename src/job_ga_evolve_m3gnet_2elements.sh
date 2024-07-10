#!/bin/bash
#SBATCH --job-name=ga_evolve_m3gnet_2elements
#SBATCH --output=ga_evolve_m3gnet_2elements.out
#SBATCH --error=ga_evolve_m3gnet_2elements.err
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

python ga_evolve_2elements.py m3gnet