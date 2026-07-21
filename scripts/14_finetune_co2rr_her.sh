#!/bin/bash
# =============================================================================
# CarbideMatterGen — Step 11 (CO2RR): fine-tune MatterGen on the CO2RR-to-CO
# property set PLUS the economic constraint (9-property adapter from base).
#
# CO2RR sibling of scripts/11_finetune_economic.sh. SAME recipe, SAME overlay
# cache (the CO2RR + economic labels are additive — rebuild the overlay with
# scripts/02_inject_labels.sh so the *_co_* JSONs exist). Only the conditioning
# set changes from the HER descriptors to the CO2RR-to-CO descriptors; the
# economic descriptor (P9) is identical.
#
# PORTABLE: no hard-coded /home/work paths. Override via env, or let it
# auto-detect the installed mattergen package (works on the WSL/RTX-5070 box).
#   VENV            virtualenv to activate (optional; default: current env)
#   MATTERGEN_ROOT  mattergen repo root (optional; default: pip-installed pkg)
#   BASE_MODEL      mattergen_base checkpoint dir (REQUIRED)
#
# Conditioning set (9). metal_percolation_dim stays EXCLUDED (high NaN).
#   P1 energy_above_hull        (stability; MatterGen base label)
#   P2 valence_electron_conc    (d-band filling -> CO backdonation. REPLACES dft_band_gap:
#                              that label is sparse in alex_mp_20 and cut the training
#                              set 279->63. Metallicity is guaranteed by the group-4/5/6
#                              carbide palette itself and verified by DFT DOS downstream.)
#   P3 best_site_co_affinity    (PRIMARY: most CO-releasing site -> 0)
#   P4 frac_co_selective_sites  (PRIMARY: density of CO-releasing sites -> high)
#   P5 mean_cooh_affinity       (CO2 activation / onset -> low)
#   P6 metal_mixing_entropy     (medium/high entropy -> high)
#   P7 min_dG_H                 (HER SUPPRESSION: drive H binding WEAK / positive. Replaces
#                              co2rr_selectivity: DFT showed every over-binding carbide loses
#                              current to HER; the one selective structure had ΔG_H > 0.)
#   P8 metal_packing_density    (metallic-framework / conductivity proxy)
#   P9 carbon_fraction          (MECHANISM: C enrichment deepens eps_d -> weaker CO)
# =============================================================================
set -euo pipefail

[ -n "${VENV:-}" ] && source "${VENV}/bin/activate"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${ROOT}/configs/hydra_overlay"

if [ -z "${BASE_MODEL:-}" ]; then
  echo "ERROR: set BASE_MODEL to your mattergen_base checkpoint dir." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/finetune_co2rr_her}"
export OUTPUT_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PROJECT_ROOT must be the inner mattergen package dir (contains conf/).
if [ -n "${MATTERGEN_ROOT:-}" ]; then
  export PROJECT_ROOT="${MATTERGEN_ROOT}/mattergen"
else
  export PROJECT_ROOT="$(python -c 'import os,mattergen;print(os.path.dirname(mattergen.__file__))')"
fi
echo "PROJECT_ROOT=${PROJECT_ROOT}"

export CARBIDE_CACHE_ROOT="${CARBIDE_CACHE_ROOT:-${ROOT}/cache/overlay_uma_her}"
export OVERLAY_CACHE="${CARBIDE_CACHE_ROOT}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${OUTPUT_DIR}"

P1="energy_above_hull"
P2="valence_electron_conc"
P3="best_site_co_affinity"
P4="frac_co_selective_sites"
P5="mean_cooh_affinity"
P6="metal_mixing_entropy"
P7="min_dG_H"
P8="metal_packing_density"
P9="carbon_fraction"

MAX_EPOCHS="${MAX_EPOCHS:-600}"
# RTX 5070 (12 GB): 16 is safe. A100: raise to 64.
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
FULL_FT="${FULL_FT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=============================================================="
echo " CarbideMatterGen — CO2RR HER-SELECTIVE fine-tune (9-property adapter)"
echo "   base model    : ${BASE_MODEL}"
echo "   output dir    : ${OUTPUT_DIR}"
echo "   overlay cache : ${CARBIDE_CACHE_ROOT}"
echo "   max epochs    : ${MAX_EPOCHS}   batch: ${BATCH_SIZE}   lr: ${LR}"
echo "   primary heads : ${P3}, ${P4}    economic: ${P9}"
echo "=============================================================="

mattergen-finetune \
  hydra.searchpath="[file://${OVERLAY}]" \
  adapter.model_path="${BASE_MODEL}" \
  adapter.full_finetuning=${FULL_FT} \
  data_module=alex_mp_20_carbide \
  data_module.properties="[${P1},${P2},${P3},${P4},${P5},${P6},${P7},${P8},${P9}]" \
  data_module.batch_size.train=${BATCH_SIZE} \
  data_module.batch_size.val=${BATCH_SIZE} \
  data_module.num_workers.train=${NUM_WORKERS} \
  data_module.num_workers.val=1 \
  lightning_module.optimizer_partial.lr=${LR} \
  lightning_module.scheduler_partials.0.scheduler.patience=30 \
  lightning_module.scheduler_partials.0.monitor="loss_val" \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P1}=${P1} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P2}=${P2} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P3}=${P3} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P4}=${P4} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P5}=${P5} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P6}=${P6} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P7}=${P7} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P8}=${P8} \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${P9}=${P9} \
  trainer.max_epochs=${MAX_EPOCHS} \
  trainer.check_val_every_n_epoch=1 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  "$@"
