#!/usr/bin/env bash
# Step 2 — overlay the carbide labels onto the MatterGen alex_mp_20 cache.
# Symlinks the original cache and drops one <property>.json per label, aligned
# to structure_id.npy. The original cache is never mutated.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_SRC="${CACHE_SRC:-$HOME/.cache/mattergen/alex_mp_20}"   # MatterGen's processed cache (has train/ and val/)
OVERLAY_CACHE="${OVERLAY_CACHE:-$ROOT/cache/overlay_carbide}"
LABELS_DIR="${LABELS_DIR:-$ROOT/data}"

# build_overlay() iterates the train/val splits, reads
# carbide_labeled_{split}.csv from LABELS_DIR, and writes the property JSONs
# defined in carbidemattergen.cache_overlay.NEW_PROPS.
python -m carbidemattergen.cache_overlay \
  --source-cache "$CACHE_SRC" \
  --overlay "$OVERLAY_CACHE" \
  --labels "$LABELS_DIR"

echo "overlay ready at $OVERLAY_CACHE"
