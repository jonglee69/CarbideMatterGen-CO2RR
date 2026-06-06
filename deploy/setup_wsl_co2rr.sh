#!/usr/bin/env bash
# =============================================================================
# CarbideMatterGen — WSL setup for the CO2RR-to-CO pipeline.
#
# Assumes mattergen is ALREADY installed and working in your environment
# (RTX 5070 / CUDA). This script only:
#   1. installs CarbideMatterGen editable (so `carbidemattergen` is importable),
#   2. idempotently whitelists the new CO2RR + economic property names in your
#      installed mattergen's globals.py (PROPERTY_SOURCE_IDS),
#   3. sanity-checks the imports.
#
# It does NOT touch your mattergen install otherwise, does NOT download model
# weights, and is safe to re-run. Activate your mattergen venv first (or set
# VENV=/path/to/venv).
# =============================================================================
set -euo pipefail

[ -n "${VENV:-}" ] && source "${VENV}/bin/activate"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "CarbideMatterGen root: ${ROOT}"

# --- 1. editable install --------------------------------------------------
echo ">>> python -m pip install -e ${ROOT}"
python -m pip install -e "${ROOT}"

# --- 2. locate installed mattergen + patch PROPERTY_SOURCE_IDS -------------
python - <<'PY'
import os, re, sys
try:
    import mattergen
except Exception as e:
    sys.exit(f"ERROR: cannot import mattergen ({e}). Activate the right venv.")

pkg = os.path.dirname(mattergen.__file__)
gp = os.path.join(pkg, "common", "utils", "globals.py")
if not os.path.exists(gp):
    sys.exit(f"ERROR: globals.py not found at {gp}")

NEW = [
    # economic
    "expensive_metal_fraction",
    # CO2RR-to-CO
    "mean_co_affinity", "best_site_co_affinity", "frac_co_selective_sites",
    "mean_cooh_affinity", "co2rr_selectivity",
    # HER (in case this is a fresh mattergen that never had the carbide patch)
    "mean_h_affinity", "best_site_h_affinity", "frac_thermoneutral_sites",
    "d_band_proxy", "valence_electron_conc", "carbon_fraction",
    "metal_mixing_entropy", "n_metal_species", "metal_cn_mean",
    "mc_coordination", "metal_packing_density", "metal_percolation_dim",
]

src = open(gp).read()
m = re.search(r"PROPERTY_SOURCE_IDS\s*=\s*\[(.*?)\]", src, re.S)
if not m:
    sys.exit("ERROR: PROPERTY_SOURCE_IDS list not found in globals.py")
block = m.group(1)
missing = [p for p in NEW if f'"{p}"' not in block and f"'{p}'" not in block]
if not missing:
    print(f"[patch] globals.py already whitelists all {len(NEW)} names — nothing to do.")
else:
    add = "".join(f'    "{p}",\n' for p in missing)
    insert = block.rstrip() + "\n    # --- CarbideMatterGen CO2RR/economic additions ---\n" + add
    new_src = src[:m.start(1)] + insert + src[m.end(1):]
    bak = gp + ".carbidebak"
    if not os.path.exists(bak):
        open(bak, "w").write(src)
        print(f"[patch] backup written: {bak}")
    open(gp, "w").write(new_src)
    print(f"[patch] added {len(missing)} names to {gp}: {missing}")
PY

# --- 3. sanity check ------------------------------------------------------
echo ">>> import check"
python - <<'PY'
from carbidemattergen import co2rr, co_surface_screen, labeling, cache_overlay
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS as P
need = ["best_site_co_affinity", "frac_co_selective_sites", "mean_cooh_affinity",
        "co2rr_selectivity", "expensive_metal_fraction"]
miss = [n for n in need if n not in P]
print("carbidemattergen import: OK")
print("overlay props:", len(cache_overlay.NEW_PROPS))
print("CO2RR names whitelisted:", "OK" if not miss else f"MISSING {miss}")
PY

echo ""
echo "Setup complete. Next: see docs/co2rr_design.md (end-to-end) and"
echo "deploy/HANDOFF_CO2RR_WSL.md (what to run with local Claude)."
