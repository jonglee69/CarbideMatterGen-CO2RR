#!/usr/bin/env bash
# Auto-restarting wrapper: the UMA relabeller occasionally dies in native code
# (no Python traceback). It is resumable (skips material_ids already in the
# JSONL), so just relaunch until both splits are complete.
cd /home/jonglee69/mattergen/CarbideMatterGen
PY=/home/jonglee69/.venv_fairchem/bin/python
for attempt in $(seq 1 40); do
  n=$(wc -l < outputs/uma_labels/uma_labels_train.jsonl 2>/dev/null || echo 0)
  v=$(wc -l < outputs/uma_labels/uma_labels_val.jsonl 2>/dev/null || echo 0)
  if [ "$n" -ge 279 ] && [ "$v" -ge 35 ]; then
    echo "=== RELABEL COMPLETE: train=$n val=$v ==="; break
  fi
  echo "--- attempt $attempt (train=$n/279 val=$v/35) ---"
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 $PY -u scripts/04_uma_relabel.py \
    --labels data_mfg --out outputs/uma_labels --device cuda \
    --max-facets 1 --site-cap 4 --max-steps 30 >> outputs/uma_relabel.log 2>&1
  sleep 5
done
