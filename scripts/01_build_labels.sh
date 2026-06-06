#!/usr/bin/env bash
# Step 1 — build HER physics labels for every TM carbide in alex_mp_20.
set -euo pipefail

SRC_DIR="${SRC_DIR:-$HOME/.cache/mattergen/alex_mp_20}"   # MatterGen source CSVs (train.csv / val.csv with a `cif` column)
OUT_DIR="${OUT_DIR:-$(dirname "$0")/../data}"
WORKERS="${WORKERS:-48}"

python -m carbidemattergen.labeling \
  --source-dir "$SRC_DIR" \
  --output-dir "$OUT_DIR" \
  --workers "$WORKERS"
