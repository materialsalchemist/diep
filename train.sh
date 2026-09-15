#!/bin/bash
# Four-GPU training with bounded recovery from saved checkpoints.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

SAVE_DIR=${SAVE_DIR:-logs/mp_pes_pyg_v2}
LOG_ROOT="$SAVE_DIR/logs"
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTARTS=0

trap 'exit 130' INT
trap 'exit 143' TERM

while true; do
  LATEST_CKPT=$(find "$LOG_ROOT" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  RESUME_ARGS=()
  if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARGS=(--resume "$LATEST_CKPT")
  fi

  RUN_VERSION="run_$(date -u +%Y%m%dT%H%M%S)_$$"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src python3 -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=4 scripts/train_mp_pes_pyg.py \
    --datasets matpes \
    --data-dir data/mp_pes \
    --save-dir "$SAVE_DIR" \
    --log-version "$RUN_VERSION" \
    --epochs 1000 \
    --max-atoms 150 \
    --max-atoms-per-batch 1000 \
    --accelerator gpu \
    --devices 4 \
    --integral-mode sum \
    "${RESUME_ARGS[@]}" "$@"

  EXIT_CODE=$?
  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Training completed successfully (exit 0)."
    break
  fi
  if [[ $EXIT_CODE -eq 130 || $EXIT_CODE -eq 143 ]]; then
    exit "$EXIT_CODE"
  fi
  RESTARTS=$((RESTARTS + 1))
  if (( RESTARTS > MAX_RESTARTS )); then
    echo "Training failed repeatedly; stopping after $MAX_RESTARTS retries (last exit $EXIT_CODE)."
    exit "$EXIT_CODE"
  fi
  echo "Training exited with code $EXIT_CODE. Retry $RESTARTS/$MAX_RESTARTS in 15s..."
  sleep 15
done
