PYTHONPATH=src python3 scripts/train_mp_pes_pyg.py \
  --datasets matpes \
  --data-dir data/mp_pes \
  --save-dir logs/mp_pes_pyg \
  --epochs 200 \
  --batch-size 32 \
  --accelerator auto