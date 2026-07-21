#!/bin/bash
# One-shot: locate/install Quantum ESPRESSO, then run the winner (100) surface
# relaxations on the CNU Backend.AI server. Defensive + resumable — it always
# tries to end with a working pw.x and produced ΔG results.
#
#   cd /home/work/CarbideMattergen && bash server/run_all.sh
#   (long job? run:  nohup bash server/run_all.sh &  then  tail -f server/run_all.log)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
LOG="$REPO/server/run_all.log"; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
say(){ echo -e "\n=== $* ==="; }

say "0. environment"
echo "host: $(hostname)   date: $(date)"
nvidia-smi -L 2>/dev/null || echo "  (no nvidia-smi)"
echo -n "  nvcc: ";     command -v nvcc || echo "-"
echo -n "  nvfortran: ";command -v nvfortran || echo "-"
echo -n "  gfortran: "; command -v gfortran || echo "-"
echo -n "  mpirun: ";   command -v mpirun || echo "-"
echo -n "  conda/mamba: "; (command -v mamba || command -v conda) || echo "-"
echo -n "  python3: ";  command -v python3 || echo "-"

say "1. pseudo paths + python deps"
bash server/setup_paths.sh || true
export QE_PSEUDO_DIR="$REPO/server/pseudos"
export QE_SSSP_JSON="$REPO/server/sssp_efficiency.json"
python3 -m pip install -q --user ase numpy 2>/dev/null || pip install -q ase numpy || \
  echo "  (pip install failed — install ase/numpy manually if scripts/56 errors)"

say "2. obtain pw.x"
PW=""
find_pw(){ command -v pw.x 2>/dev/null || true; }

# 2a. already on PATH (module preloaded, image-provided)
PW="$(find_pw)"

# 2b. environment modules
if [ -z "$PW" ] && command -v module >/dev/null 2>&1; then
  for m in $(module avail 2>&1 | tr ' ' '\n' | grep -ioE "[a-z0-9._/-]*(quantum[-_]?espresso|^qe|/qe)[a-z0-9._/-]*" | head -4); do
    echo "  trying module load $m"; module load "$m" 2>/dev/null && PW="$(find_pw)" && [ -n "$PW" ] && break
  done
fi

# 2c. GPU build from source (only if NVHPC nvfortran present) — what we prefer
if [ -z "$PW" ] && command -v nvfortran >/dev/null 2>&1; then
  echo "  nvfortran found -> attempting QE GPU build (this takes ~20-40 min)"
  if bash server/build_qe_gpu.sh; then PW="$REPO/server/qe/bin/pw.x"; fi
fi

# 2d. conda CPU fallback (reliable everywhere)
if [ -z "$PW" ]; then
  CONDA="$(command -v mamba || command -v conda || true)"
  if [ -n "$CONDA" ]; then
    echo "  installing QE (CPU) via conda-forge ..."
    "$CONDA" install -y -c conda-forge qe >/dev/null 2>&1 || \
    "$CONDA" install -y -c conda-forge quantum-espresso >/dev/null 2>&1 || true
    PW="$(find_pw)"
  fi
fi

if [ -z "$PW" ]; then
  echo "!! Could not obtain pw.x automatically."
  echo "   Options: 'module load <espresso>', or 'conda install -c conda-forge qe',"
  echo "   or build GPU QE (see server/build_qe_gpu.sh / SERVER_README.md)."
  exit 1
fi
echo "  using pw.x = $PW"
QE_BIN="$(dirname "$PW")"; export PATH="$QE_BIN:$PATH"

say "3. run the surface relaxations (resumable)"
# GPU build -> 1 rank/GPU;  CPU -> up to 16 ranks
if echo "$PW" | grep -qi gpu; then
  MPI="mpirun -np 1"; NPOOL=1
else
  N=$(nproc 2>/dev/null || echo 8); [ "$N" -gt 16 ] && N=16
  MPI="mpirun --use-hwthread-cpus -np $N"; NPOOL=4
fi
echo "  launcher: $MPI   (npool $NPOOL)"
python3 scripts/56_qe_surface_run.py --dft outputs/dft_dual/dft_relax_B \
    --pw "$PW" --mpi "$MPI" --npool "$NPOOL"

say "DONE — ΔG results"
cat outputs/dft_dual/dft_relax_B/surface_dft_dG.csv 2>/dev/null || echo "  (no CSV — check log above)"
echo "full log: $LOG"
