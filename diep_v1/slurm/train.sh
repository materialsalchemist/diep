#!/bin/bash
#SBATCH --job-name=diep-matpes
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/diep_train_%A_%a.out
#SBATCH --error=logs/diep_train_%A_%a.err
# Add your cluster's --partition / --qos / --account lines above.

# Trains DIEP on MatPES r2SCAN. With the defaults this is the exact command that produced
# models/diep_fold1: fold 1, seed 43, 300 epochs, canonical triplet frame.
#
#   sbatch slurm/train.sh                       # fold 1 (the released model)
#   sbatch --array=0-2 slurm/train.sh           # all three folds
#   FOLD=1 bash slurm/train.sh                  # without SLURM
#   sbatch --export=ALL,TRIPLET_FRAME=bond slurm/train.sh    # smooth bond-anchored frame
#
# Needs the graph cache and fold artifacts first (see README, "Training from scratch"):
#   python -m diep_pyg.matpes build --root "$DATA_ROOT"
#   python -m diep_pyg.matpes folds --root "$DATA_ROOT" --folds 3
#
# GPU MEMORY: use a card with >= 44 GB (the released model trained on an L40S). Batches are
# 32 structures with no size cap, and the two largest MatPES cells (216 atoms, ~574k
# triplets each) need ~23 GB on their own, so a 32 GB V100 runs out of memory in ~5% of
# epochs. HOST MEMORY: peak RSS was 46 GiB (fold 1); 32 GB was OOM-killed, so ask for 64 GB.
#
# WALLTIME: ~15 min/epoch on one L40S (measured), so 300 epochs is ~76 GPU-hours.
#
# RESUMING: a cell resumes from checkpoints/last.ckpt automatically, and --requeue plus
# Lightning's SIGUSR1 handling continues a preempted task. last.ckpt is rewritten only
# when val_Total_Loss improves, so a resume restarts from the best epoch so far. On
# resume, early-stopping patience is restored from the checkpoint and --patience is ignored.

set -u
RELEASE_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_ROOT=${DIEP_MATPES_ROOT:-$RELEASE_DIR/data}
CACHE_NAME=${CACHE_NAME:-DIEPDataset}
cd "$RELEASE_DIR"
mkdir -p logs

# Activate the environment from environment.yml, e.g.:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate diep-matpes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${RELEASE_DIR}${PYTHONPATH:+:$PYTHONPATH}"

FOLD=${SLURM_ARRAY_TASK_ID:-${FOLD:-1}}
SEED=$(( 42 + FOLD ))
ARTIFACTS="artifacts_full/fold${FOLD}"
TRIPLET_FRAME=${TRIPLET_FRAME:-canonical}
case "$TRIPLET_FRAME" in
    canonical) CELL="fold${FOLD}" ;;
    bond) CELL="fold${FOLD}_bond" ;;
    *) echo "TRIPLET_FRAME must be canonical or bond, got '${TRIPLET_FRAME}'" >&2; exit 1 ;;
esac
OUT_DIR="${DATA_ROOT}/runs_diep/diep_${CELL}"
WORKERS=${SLURM_CPUS_PER_TASK:-8}

for f in "${DATA_ROOT}/${ARTIFACTS}/splits.json" "${DATA_ROOT}/${CACHE_NAME}/pyg_graph.pt"; do
    if [[ ! -f "$f" ]]; then
        echo "missing $f -- build the cache and folds first (see README)" >&2
        exit 1
    fi
done

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} fold=${FOLD} seed=${SEED} triplet_frame=${TRIPLET_FRAME}"
echo "data=${DATA_ROOT} cache=${CACHE_NAME} out=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "start: $(date)"

# srun so Lightning's SLURM requeue handling works. --cpus-per-task is explicit because
# SLURM >= 22.05 stopped propagating it to srun, which would starve the dataloader.
LAUNCH=()
[[ -n "${SLURM_JOB_ID:-}" ]] && LAUNCH=(srun --cpus-per-task="$WORKERS")

${LAUNCH[@]+"${LAUNCH[@]}"} python -m diep_pyg.train \
    --root "$DATA_ROOT" \
    --cache-name "$CACHE_NAME" \
    --fold "$FOLD" \
    --artifacts-name "$ARTIFACTS" \
    --out-dir "$OUT_DIR" \
    --tag "$CELL" \
    --triplet-frame "$TRIPLET_FRAME" \
    --max-epochs 300 \
    --patience 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --loss l1_loss \
    --energy-weight 1.0 \
    --force-weight 1.0 \
    --stress-weight 0.1 \
    --num-workers "$WORKERS" \
    --accelerator gpu \
    --devices 1 \
    --seed "$SEED" \
    --no-progress-bar
rc=$?

echo "end: $(date) (exit $rc)"
# Propagate the exit code, or a crashed cell reports COMPLETED to sacct.
exit $rc
