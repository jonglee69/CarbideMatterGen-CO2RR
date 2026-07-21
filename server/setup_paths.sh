#!/bin/bash
# Run once from the repo root on the CNU GPU server:  bash server/setup_paths.sh
# Points QE at the bundled pseudopotentials and rewrites the absolute pseudo_dir
# baked into the pre-generated QE inputs so they run anywhere.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PSEUDO="$REPO/server/pseudos"

# 1) env vars used by scripts/55 (and 74 via it) when REGENERATING inputs
export QE_PSEUDO_DIR="$PSEUDO"
export QE_SSSP_JSON="$REPO/server/sssp_efficiency.json"

# 2) rewrite the old absolute path inside ALREADY-generated .in files
OLD="/home/jonglee69/software/qe_pseudo/active"
n=$(grep -rl "$OLD" "$REPO/outputs" 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  grep -rl "$OLD" "$REPO/outputs" | xargs sed -i "s#$OLD#$PSEUDO#g"
fi
echo "pseudo_dir -> $PSEUDO   (rewrote $n input files)"
echo "add to your shell:  export QE_PSEUDO_DIR=$PSEUDO  QE_SSSP_JSON=$REPO/server/sssp_efficiency.json"
