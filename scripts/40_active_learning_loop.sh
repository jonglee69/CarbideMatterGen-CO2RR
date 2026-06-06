#!/usr/bin/env bash
# Step 7 — active-learning loop: close the gap between the cheap composition
# proxies (Stage A labels) and the real surface ΔG_H* (Stage B fairchem / DFT).
#
# One iteration:
#   a. generate              (20_generate.sh)
#   b. cheap proxy screen    (30_cheap_screen.py)        -> top-K bulk candidates
#   c. fairchem ΔG_H* screen (carbidemattergen.surface_screen) on top-K
#   d. DFT-validate a small batch with the best fairchem ΔG_H* (50_dft_validate.sh)
#   e. RECALIBRATE the _H_ADS_CARBIDE proxy table in hydrogen.py against the new
#      (composition -> measured ΔG_H*) pairs, then rebuild labels (01) + overlay
#      (02) and re-fine-tune (10). The proxy gets sharper every round, so the
#      generator's hit-rate for genuinely Pt-like, stable, conductive carbides
#      climbs each iteration.
#
# This loop is the scientific core of the project: it is what lets a bulk
# generative model converge on a *surface* property it cannot see directly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ITER="${ITER:-0}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a fairchem OC20/OC22 checkpoint}"
TOPK="${TOPK:-200}"

echo "=== active-learning iteration $ITER ==="
bash "$ROOT/scripts/20_generate.sh"
python "$ROOT/scripts/30_cheap_screen.py" \
  --gen-dir "$ROOT/outputs/gen" \
  --out "$ROOT/outputs/screen/cheap_iter${ITER}.csv"

# Materialize top-K structures into a flat dir for the surface screen.
SCREEN_DIR="$ROOT/outputs/screen/topk_iter${ITER}"
python - "$ROOT/outputs/screen/cheap_iter${ITER}.csv" "$ROOT/outputs/gen" "$SCREEN_DIR" "$TOPK" <<'PY'
import sys, csv, shutil
from pathlib import Path
csv_path, gen_root, out_dir, topk = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
index = {p.name: p for p in [*gen_root.rglob('*.cif'), *gen_root.rglob('*POSCAR*')]}
with open(csv_path) as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= topk: break
        src = index.get(row['source'])
        if src: shutil.copy(src, out_dir / src.name)
print('materialized', out_dir)
PY

python -m carbidemattergen.surface_screen \
  --candidates "$SCREEN_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out "$ROOT/outputs/screen/fairchem_iter${ITER}.jsonl"

echo "--- review fairchem_iter${ITER}.jsonl, DFT-validate the best (50_dft_validate.sh),"
echo "--- then recalibrate _H_ADS_CARBIDE and rerun 01/02/10 with ITER=$((ITER+1)). ---"
