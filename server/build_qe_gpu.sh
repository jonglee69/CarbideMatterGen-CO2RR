#!/bin/bash
# Build Quantum ESPRESSO 7.3.1 with GPU (NVHPC) support for the A100 (cc80).
# Requires the NVHPC SDK (nvfortran). Installs it via conda if missing and conda
# is available. Best-effort + fully logged — QE GPU builds sometimes need a tweak;
# if configure/make fails, read server/build_qe.log and tell Claude the error.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/server/qe-src"; PREFIX="$REPO/server/qe"
BLOG="$REPO/server/build_qe.log"; : > "$BLOG"
exec > >(tee -a "$BLOG") 2>&1
echo "=== QE GPU build $(date) ==="

# --- ensure nvfortran (NVHPC) ---
if ! command -v nvfortran >/dev/null 2>&1; then
  CONDA="$(command -v mamba || command -v conda || true)"
  if [ -n "$CONDA" ]; then
    echo "## nvfortran missing -> installing NVHPC via conda (large, be patient) ..."
    "$CONDA" install -y -c nvidia nvhpc 2>&1 | tail -3 || true
  fi
fi
command -v nvfortran >/dev/null 2>&1 || { echo "!! no nvfortran; cannot GPU-build. Use CPU QE (conda install -c conda-forge qe)."; exit 1; }
echo "## nvfortran: $(command -v nvfortran)"

# --- locate CUDA that ships with NVHPC ---
NVBIN="$(dirname "$(readlink -f "$(command -v nvfortran)")")"
NVROOT="$(cd "$NVBIN/../.." && pwd)"                 # .../Linux_x86_64/<ver>
CUDA_DIR="$(ls -d "$NVROOT"/cuda/*/ 2>/dev/null | sort -V | tail -1)"
[ -z "$CUDA_DIR" ] && CUDA_DIR="$(ls -d "$NVROOT"/cuda 2>/dev/null | tail -1)"
CUDART="$(basename "${CUDA_DIR%/}" 2>/dev/null)"; echo "$CUDART" | grep -qE '^[0-9]+\.[0-9]+' || CUDART="12.3"
# A100 compute capability
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
[ -z "$CC" ] && CC=80
echo "## CUDA_DIR=$CUDA_DIR  runtime=$CUDART  cc=$CC"

# --- fetch source ---
mkdir -p "$SRC"; cd "$SRC"
if [ ! -d q-e ]; then
  git clone --depth 1 -b qe-7.3.1 https://gitlab.com/QEF/q-e.git || \
  git clone --depth 1 -b qe-7.3 https://gitlab.com/QEF/q-e.git
fi
cd q-e

# --- configure + build pw/dos/projwfc ---
echo "## configure ..."
./configure --with-cuda="${CUDA_DIR%/}" --with-cuda-cc="$CC" --with-cuda-runtime="$CUDART" \
            --enable-openmp FC=nvfortran F90=nvfortran CC=nvc 2>&1 | tail -25
echo "## make (this takes a while) ..."
make -j"$(nproc)" pw dos projwfc 2>&1 | tail -15

mkdir -p "$PREFIX/bin"
cp -f bin/pw.x bin/dos.x bin/projwfc.x "$PREFIX/bin/" 2>/dev/null
if [ -x "$PREFIX/bin/pw.x" ]; then
  echo "## OK -> $PREFIX/bin/pw.x"
else
  echo "!! build did not produce pw.x — see $BLOG"; exit 1
fi
