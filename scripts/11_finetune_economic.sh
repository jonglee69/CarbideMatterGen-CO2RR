#!/bin/bash
# =============================================================================
# CarbideMatterGen — Step 11: fine-tune MatterGen on the carbide HER property
# set PLUS an economic constraint (fresh 9-property adapter from mattergen_base).
#
# Same recipe as scripts/10 but adds a 9th conditioning property,
# expensive_metal_fraction, so generation can be steered toward affordable,
# non-platinum-group catalysts. Stage B showed clean d-block d-band centres
# match/beat the PGMs, yet the unconstrained generator favoured them; condition
# expensive_metal_fraction -> 0.0 at generation to bias toward economically
# clean compositions (Re and cheap light REE allowed; PGM + heavy REE penalised).
# Output dir is separate so the baseline 8-property model is kept as a comparison.
#
# Conditioning set (9). metal_percolation_dim stays EXCLUDED (38.8% NaN).
#   P1 energy_above_hull        (stability; MatterGen base label)
#   P2 dft_band_gap             (metallicity: drive -> 0; MatterGen base label)
#   P3 best_site_h_affinity     (PRIMARY: most-thermoneutral site -> 0)
#   P4 frac_thermoneutral_sites (PRIMARY: density of active sites -> high)
#   P5 mean_h_affinity          (average H binding)
#   P6 metal_mixing_entropy     (medium/high entropy -> high)
#   P7 valence_electron_conc    (HEA electronic descriptor)
#   P8 metal_packing_density    (metallic-framework / conductivity proxy)
#   P9 expensive_metal_fraction (NEW: PGM + heavy-REE fraction -> 0 at gen)
# =============================================================================
set -euo pipefail

source /home/jonglee69/.venv_mattergen/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${ROOT}/configs/hydra_overlay"
BASE_MODEL="${BASE_MODEL:-/home/jonglee69/mattergen/checkpoints/mattergen_base}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/finetune_economic}"
export OUTPUT_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="/home/jonglee69/mattergen/mattergen"
# Carbide overlay cache built by scripts/02_inject_labels.sh
export CARBIDE_CACHE_ROOT="${CARBIDE_CACHE_ROOT:-${ROOT}/cache/overlay_carbide}"
# data_module/alex_mp_20_carbide.yaml reads OVERLAY_CACHE; keep both in sync.
export OVERLAY_CACHE="${CARBIDE_CACHE_ROOT}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${OUTPUT_DIR}"

# 9 conditioning properties (8 carbide + economic)
P1="energy_above_hull"
P2="dft_band_gap"
P3="best_site_h_affinity"
P4="frac_thermoneutral_sites"
P5="mean_h_affinity"
P6="metal_mixing_entropy"
P7="valence_electron_conc"
P8="metal_packing_density"
P9="expensive_metal_fraction"

MAX_EPOCHS="${MAX_EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-5}"
FULL_FT="${FULL_FT:-true}"
# Host now idle (OxideMatterGen training ended) -> use all 4 cores by default.
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=============================================================="
echo " CarbideMatterGen — fine-tune (9-property economic adapter from base)"
echo "   base model    : ${BASE_MODEL}"
echo "   output dir    : ${OUTPUT_DIR}"
echo "   overlay cache : ${CARBIDE_CACHE_ROOT}"
echo "   max epochs    : ${MAX_EPOCHS}   batch: ${BATCH_SIZE}   lr: ${LR}"
echo "   num_workers   : ${NUM_WORKERS}  (CPU-constrained host)"
echo "   primary heads : ${P3}, ${P4}"
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
