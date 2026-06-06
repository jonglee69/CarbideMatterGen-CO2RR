#!/usr/bin/env bash
# Step 4 — generate candidate carbides with classifier-free guidance toward
# Pt-like HER. Drives best_site_h_affinity → 0, frac_thermoneutral_sites high,
# metal_mixing_entropy high (medium/high entropy), dft_band_gap → 0 (metallic).
#
# Conditions on the SAME 8 properties used during fine-tuning (10_finetune)
# — the conditioning set at generation must match training. Uses the verified
# mattergen-generate CLI (OxideMatterGen scripts/29): --diffusion_guidance_factor,
# --record_trajectories=False, best-val checkpoint.
#
# Single A100: the two guidance weights run SEQUENTIALLY (not in parallel) to
# use the full GPU per run and avoid OOM. Generation is chunked so a failure
# loses at most one chunk; completed chunks are skipped on re-run (.done marker).
set -euo pipefail

source /home/jonglee69/.venv_mattergen/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${ROOT}/outputs/finetune_economic}"     # 9-property economic model
GEN_ROOT="${GEN_ROOT:-${ROOT}/outputs/gen_economic}"
CKPT_EPOCH="${CKPT_EPOCH:-best}"

# Per-guidance count = N_CHUNKS × NUM_BATCHES × BATCH_SIZE.
# Economic pilot default: 4 × 100 × 50 = 20,000 per weight -> 40,000 total.
# Generation is now BIASED toward clean compositions (expensive_metal_fraction
# -> 0), so clean-HEC yield is far higher than the unconstrained run.
N_CHUNKS="${N_CHUNKS:-4}"
NUM_BATCHES="${NUM_BATCHES:-100}"
BATCH_SIZE="${BATCH_SIZE:-50}"
WEIGHTS="${WEIGHTS:-1.0 2.5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="/home/jonglee69/mattergen/mattergen"

# Target property values (must reference exactly the 9 trained properties).
# Fire dict syntax: no spaces. expensive_metal_fraction:0.0 steers generation
# toward economically clean compositions (no PGM / heavy REE; Re + cheap REE ok).
COND="{energy_above_hull:0.0"
COND="${COND},dft_band_gap:0.0"
COND="${COND},best_site_h_affinity:0.0"
COND="${COND},frac_thermoneutral_sites:1.0"
COND="${COND},mean_h_affinity:0.0"
COND="${COND},metal_mixing_entropy:1.6"
COND="${COND},valence_electron_conc:6.0"
COND="${COND},metal_packing_density:0.04"
COND="${COND},expensive_metal_fraction:0.0}"

per_w=$((N_CHUNKS * NUM_BATCHES * BATCH_SIZE))
echo "=============================================================="
echo " CarbideMatterGen — Stage A generation"
echo "   model      : ${MODEL}  (checkpoint=${CKPT_EPOCH})"
echo "   weights    : ${WEIGHTS}  (run sequentially)"
echo "   per weight : ${N_CHUNKS} × ${NUM_BATCHES} × ${BATCH_SIZE} = ${per_w}"
echo "   condition  : ${COND}"
echo "   output     : ${GEN_ROOT}"
echo "=============================================================="

for W in ${WEIGHTS}; do
  TAG="w$(echo "${W}" | tr '.' 'p')"
  OUT_ROOT="${GEN_ROOT}/${TAG}"
  mkdir -p "${OUT_ROOT}"
  for (( i=0; i<N_CHUNKS; i++ )); do
    chunk="${OUT_ROOT}/chunk_${i}"
    if [ -f "${chunk}/.done" ]; then
      echo ">>> [${TAG}] chunk ${i} already done ($(ls "${chunk}"/*.cif 2>/dev/null | wc -l) CIFs), skip"
      continue
    fi
    rm -rf "${chunk}"; mkdir -p "${chunk}"
    echo ">>> [${TAG}] chunk ${i}/${N_CHUNKS}: ${NUM_BATCHES}×${BATCH_SIZE} (w=${W})"
    mattergen-generate \
      --model_path="${MODEL}" \
      --output_path="${chunk}" \
      --checkpoint_epoch="${CKPT_EPOCH}" \
      --sampling_config_name=default \
      --properties_to_condition_on="${COND}" \
      --diffusion_guidance_factor="${W}" \
      --batch_size="${BATCH_SIZE}" \
      --num_batches="${NUM_BATCHES}" \
      --record_trajectories=False || { echo ">>> [warn] ${TAG} chunk ${i} failed, continuing"; continue; }
    if [ -f "${chunk}/generated_crystals_cif.zip" ]; then
      unzip -oq "${chunk}/generated_crystals_cif.zip" -d "${chunk}/"
    fi
    n=$(ls "${chunk}"/*.cif 2>/dev/null | wc -l)
    if [ "${n}" -ge $((NUM_BATCHES * BATCH_SIZE * 90 / 100)) ]; then
      touch "${chunk}/.done"; echo ">>> [${TAG}] chunk ${i} complete: ${n} CIFs"
    else
      echo ">>> [${TAG}] chunk ${i} underproduced: ${n} CIFs"
    fi
  done
  echo ">>> [${TAG}] total: $(ls "${OUT_ROOT}"/chunk_*/*.cif 2>/dev/null | wc -l) CIFs"
done
echo "generation done → ${GEN_ROOT}"
