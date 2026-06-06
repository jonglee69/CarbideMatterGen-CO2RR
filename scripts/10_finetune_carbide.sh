#!/bin/bash
# =============================================================================
# CarbideMatterGen — Step 10: fine-tune MatterGen on the carbide HER property
# set (fresh adapter from mattergen_base).
#
# Derived from OxideMatterGen scripts/28_finetune_denovo_v6.sh — the verified
# `mattergen-finetune` + property_embeddings_adapt override syntax. Adapted for:
#   - the 8 carbide conditioning properties (HER core + entropy + framework)
#   - the carbide overlay cache and data_module
#   - a CPU-constrained host (nproc=4, possibly shared with a MatterSim job):
#     num_workers kept low to avoid starving a co-running process.
#
# Conditioning set (8) — start with the HER core + entropy + a conductivity
# proxy; metal_percolation_dim is intentionally EXCLUDED from training (38.8%
# NaN in the labels, see docs/data_qc.md) and used only as a screen-time filter.
#   P1 energy_above_hull        (stability; MatterGen base label)
#   P2 dft_band_gap             (metallicity: drive -> 0; MatterGen base label)
#   P3 best_site_h_affinity     (PRIMARY: most-thermoneutral site -> 0)
#   P4 frac_thermoneutral_sites (PRIMARY: density of active sites -> high)
#   P5 mean_h_affinity          (average H binding)
#   P6 metal_mixing_entropy     (medium/high entropy -> high)
#   P7 valence_electron_conc    (HEA electronic descriptor)
#   P8 metal_packing_density    (metallic-framework / conductivity proxy)
# =============================================================================
set -euo pipefail

source /home/jonglee69/.venv_mattergen/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${ROOT}/configs/hydra_overlay"
BASE_MODEL="${BASE_MODEL:-/home/jonglee69/mattergen/checkpoints/mattergen_base}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/finetune_carbide}"
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

# 8 conditioning properties
P1="energy_above_hull"
P2="dft_band_gap"
P3="best_site_h_affinity"
P4="frac_thermoneutral_sites"
P5="mean_h_affinity"
P6="metal_mixing_entropy"
P7="valence_electron_conc"
P8="metal_packing_density"

MAX_EPOCHS="${MAX_EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-5}"
FULL_FT="${FULL_FT:-true}"
# CPU-constrained host: keep dataloader workers low when a MatterSim/other job
# shares the 4 cores. Override NUM_WORKERS=4 if the host is otherwise idle.
NUM_WORKERS="${NUM_WORKERS:-2}"

echo "=============================================================="
echo " CarbideMatterGen — fine-tune (8-property adapter from base)"
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
  data_module.properties="[${P1},${P2},${P3},${P4},${P5},${P6},${P7},${P8}]" \
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
  trainer.max_epochs=${MAX_EPOCHS} \
  trainer.check_val_every_n_epoch=1 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  "$@"
