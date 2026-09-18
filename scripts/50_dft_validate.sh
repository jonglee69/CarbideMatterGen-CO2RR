#!/usr/bin/env bash
# Step 8 — DFT validation of top HER candidates (Quantum ESPRESSO).
# Validates (i) bulk stability / metallicity (DOS at E_F) and (ii) ΔG_H* on the
# best surface termination identified by the fairchem screen.
#
# Reuse the OxideMatterGen QE toolchain (scripts 57-60 there); this is a thin
# placeholder pointing at the candidate list produced by 40_active_learning_loop.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANDIDATES="${CANDIDATES:-$ROOT/outputs/screen/fairchem_iter0.jsonl}"

echo "DFT validation entrypoint."
echo "Candidates: $CANDIDATES"
echo "TODO: wire to QE vc-relax -> scf -> nscf/dos (metallicity) and a"
echo "slab + H* scf for ΔG_H*. The OxideMatterGen QE build/run scripts"
echo "(57_qe_input_gen.py, 59_qe_dft_chain.sh, 60_build_qe72_gpu.sh) are"
echo "directly reusable for the bulk part."
