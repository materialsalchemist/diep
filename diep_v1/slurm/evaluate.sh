#!/bin/bash
#SBATCH --job-name=diep-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/diep_eval_%j.out
#SBATCH --error=logs/diep_eval_%j.err
# Add your cluster's --partition / --qos / --account lines above.

# Evaluates the released model on its own fold-1 test split (19,395 structures) and writes
# metrics plus per-structure predictions. Needs the graph cache from `python -m diep_pyg.matpes build`.
#
#   sbatch slurm/evaluate.sh
#   MODEL=path/to/exported/model SPLIT=val sbatch slurm/evaluate.sh

set -u
RELEASE_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_ROOT=${DIEP_MATPES_ROOT:-$RELEASE_DIR/data}
CACHE_NAME=${CACHE_NAME:-DIEPDataset}
MODEL=${MODEL:-$RELEASE_DIR/models/diep_fold1}
SPLIT=${SPLIT:-test}
cd "$RELEASE_DIR"
mkdir -p logs results

# Activate the environment from environment.yml first.
export PYTHONPATH="${RELEASE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
NAME=$(basename "$MODEL")

python -m diep_pyg.evaluate \
    --model "$MODEL" \
    --root "$DATA_ROOT" \
    --cache-name "$CACHE_NAME" \
    --split "$SPLIT" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    --out "results/${NAME}_${SPLIT}.json" \
    --predictions "results/${NAME}_${SPLIT}.npz"
