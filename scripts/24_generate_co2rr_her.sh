#!/usr/bin/env bash
# Step 4 (CO2RR economic) — generate candidate carbides with classifier-free
# guidance toward selective, affordable CO2-to-CO catalysts. Drives
# best_site_co_affinity -> 0 (CO releases), frac_co_selective_sites high,
# mean_cooh_affinity low (CO2 activates), co2rr_selectivity high (CO over H2),
# metal_mixing_entropy high (HEC), dft_band_gap -> 0 (metallic), and
# expensive_metal_fraction -> 0 (no PGM / heavy REE).
#
# CO2RR sibling of scripts/21_generate_economic.sh. Conditions on the SAME 9
# properties used in 11_finetune_co2rr_economic.sh (the conditioning set at
# generation MUST match training).
#
# PORTABLE + RTX-5070-tuned. Override via env:
#   VENV            virtualenv to activate (optional; default: current env)
#   MATTERGEN_ROOT  mattergen repo root (optional; default: pip-installed pkg)
#   MODEL           fine-tuned model dir (default: outputs/finetune_co2rr_her)
set -euo pipefail

[ -n "${VENV:-}" ] && source "${VENV}/bin/activate"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${ROOT}/outputs/finetune_co2rr_her}"
GEN_ROOT="${GEN_ROOT:-${ROOT}/outputs/gen_co2rr_her}"
CKPT_EPOCH="${CKPT_EPOCH:-best}"

# Per-guidance count = N_CHUNKS × NUM_BATCHES × BATCH_SIZE.
# RTX 5070 default: 4 × 8 × 16 = 512 per weight → 1,024 total (w=1.0 + w=2.5).
# A100: e.g. 4 × 100 × 50 = 20,000 per weight.
N_CHUNKS="${N_CHUNKS:-4}"
NUM_BATCHES="${NUM_BATCHES:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WEIGHTS="${WEIGHTS:-1.0 2.5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${MATTERGEN_ROOT:-}" ]; then
  export PROJECT_ROOT="${MATTERGEN_ROOT}/mattergen"
else
  export PROJECT_ROOT="$(python -c 'import os,mattergen;print(os.path.dirname(mattergen.__file__))')"
fi

# Target property values (must reference exactly the 9 trained properties).
# Fire dict syntax: no spaces. The palette itself (group-4/5/6 TMs, see
# element_policy.py) enforces manufacturability, so no cost/leach knob is needed.
# The two mechanism levers our DFT identified are conditioned instead:
#   carbon_fraction:0.5      C enrichment deepens eps_d -> weaker CO binding
#   valence_electron_conc:5.5  higher d-filling (W/Mo/Cr-rich) -> CO backdonation
# best_site_co_affinity:0.0 is DELIBERATE extrapolation: the palette caps at
# -0.20 (pure W) in mean field, so reaching the CO window REQUIRES site-level
# (high-entropy) diversity -- that is the hypothesis under test.
COND="{energy_above_hull:0.0"
COND="${COND},valence_electron_conc:5.5"
COND="${COND},best_site_co_affinity:0.0"
COND="${COND},frac_co_selective_sites:1.0"
COND="${COND},mean_cooh_affinity:-0.2"
COND="${COND},metal_mixing_entropy:1.6"
COND="${COND},min_dG_H:0.3"
COND="${COND},metal_packing_density:0.04"
COND="${COND},carbon_fraction:0.5}"

per_w=$((N_CHUNKS * NUM_BATCHES * BATCH_SIZE))
echo "=============================================================="
echo " CarbideMatterGen — Stage A generation (CO2RR-to-CO, MANUFACTURABLE)"
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
