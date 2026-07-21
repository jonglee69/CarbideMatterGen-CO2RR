# CarbideMatterGen — running the QE GPU calculations on the CNU Backend.AI server

This bundle carries everything needed to **resume the DFT surface relaxations** of the
three winner carbides on the GPU server. The 86 GB of QE scratch (`tmp/`, wavefunctions)
was intentionally left behind — it is regenerable. Only code + inputs + pseudopotentials
(~10 MB) travel via git.

## 0. Clone (in the VS Code terminal, already on the session)
```bash
cd /home/work
git clone -b co2rr-surface-dft-analysis \
    https://github.com/jonglee69/CarbideMatterGen-CO2RR.git CarbideMattergen
cd CarbideMattergen
```

## 1. Point QE at the bundled pseudopotentials
```bash
bash server/setup_paths.sh
export QE_PSEUDO_DIR=$PWD/server/pseudos
export QE_SSSP_JSON=$PWD/server/sssp_efficiency.json
```

## 2. Get Quantum ESPRESSO (pick one)
First check what the session already has:
```bash
nvidia-smi -L ; nvcc --version 2>/dev/null ; which pw.x nvfortran mpirun 2>/dev/null
module avail 2>&1 | grep -i -E "espresso|quantum|qe|nvhpc" | head
```
- **If a QE module exists:**  `module load <quantum-espresso>` → `pw.x` on PATH.
- **GPU build (fastest QE, needs NVHPC SDK)**: if `nvfortran` is present, build QE 7.3.1
  with `./configure --with-cuda=$NVHPC_CUDA_HOME --with-cuda-cc=<arch> --enable-openmp`.
- **Quick CPU fallback (works everywhere):**
  ```bash
  conda install -c conda-forge quantum-espresso   # or: mamba install -c conda-forge qe
  ```
  Also install the Python deps the driver needs: `pip install ase numpy`.

## 3. Resume the surface relaxations (the paused job)
`outputs/dft_dual/dft_relax_B/` already contains 3 finished CrTiV slabs (skipped
automatically); 5 remain (CrTiV `*H`, and CrTiZr clean/CO/COOH/H).

**GPU** (1 rank per GPU):
```bash
python scripts/56_qe_surface_run.py --dft outputs/dft_dual/dft_relax_B \
    --pw $(which pw.x) --mpi "mpirun -np 1" --npool 1
```
**CPU** (use the box's cores, e.g. 16):
```bash
python scripts/56_qe_surface_run.py --dft outputs/dft_dual/dft_relax_B \
    --pw $(which pw.x) --mpi "mpirun --use-hwthread-cpus -np 16" --npool 4
```
It is **resumable** (skips any slab whose `relax.out` says `JOB DONE`) and writes
`outputs/dft_dual/dft_relax_B/surface_dft_dG.csv` (ΔG_CO/COOH/H) at the end.

## 4. (Optional) regenerate inputs from scratch
```bash
python scripts/74_relax_winners_gpu.py    # rewrites the relax.in using QE_PSEUDO_DIR
```

## Notes
- Gas references (CO/CO2/H2) are bundled under
  `outputs/dft_dual/dft_relax_moTiZr/_gas/` — the ΔG assembly reuses them.
- Slab settings mirror the MoTiZr flagship: Γ-point, bottom half fixed,
  `forc_conv_thr=1e-2`, `nstep=80` (metal force noise).
- The SSH key `.claude/id_container` is **git-ignored** and never leaves your machine.
