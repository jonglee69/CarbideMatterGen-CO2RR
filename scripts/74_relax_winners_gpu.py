#!/usr/bin/env python
"""Generate GPU DFT full-relaxation inputs for the two remaining winners
hec3-CrTiV and hec3-CrTiZr (100), mirroring the hec3-MoTiZr flagship relax
(Gamma, bottom half fixed, forc_conv 1e-2 for metal force noise, nstep 80).

Writes clean + adslab relax.in for CO/COOH/H under outputs/dft_dual/dft_relax_B/
and a surface_manifest.csv whose gas rows point to the ALREADY-computed MoTiZr
_gas references (same C/O/H pseudopotentials). Then run with GPU via scripts/56:

    source /home/jonglee69/software/qe_gpu_env.sh
    python scripts/56_qe_surface_run.py --dft outputs/dft_dual/dft_relax_B \
        --pw $QE_GPU_BIN/pw.x --mpi "mpirun -np 1" --npool 1
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "s55", str(Path(__file__).resolve().parent / "55_qe_surface_inputs.py"))
s55 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s55)

ROOT = Path(__file__).resolve().parent.parent
TARG = ROOT / "outputs/dft_dual/targets"
OUT = ROOT / "outputs/dft_dual/dft_relax_B"
GAS_DIR = ROOT / "outputs/dft_dual/dft_relax_moTiZr/_gas"   # reuse computed CO/CO2/H2
CANDS = ["hec3_CrTiV", "hec3_CrTiZr"]
ADS = ["CO", "COOH", "H"]


def tune_metal(relax_in: Path):
    """Relax force threshold 1e-3 -> 1e-2 and nstep -> 80 (metal force noise)."""
    t = relax_in.read_text()
    t = re.sub(r"forc_conv_thr\s*=\s*[0-9.eEdD+-]+", "forc_conv_thr    = 1.0d-2", t)
    t = re.sub(r"nstep\s*=\s*\d+", "nstep            = 80", t)
    relax_in.write_text(t)


def main():
    man = []
    for cand in CANDS:
        for ads in ADS:
            for kind in ("clean", "ads"):
                cif = TARG / cand / f"facet100_{ads}_{kind}.cif"
                cdir = OUT / cand / f"facet100_{ads}_{kind}"
                cdir.mkdir(parents=True, exist_ok=True)
                prefix = f"{cand}_f100_{ads}_{kind}"
                info = s55.write_slab(str(cif), cdir, prefix, calc="relax", kmode="gamma")
                tune_metal(cdir / "relax.in")
                man.append({"formula": cand, "facet": "100", "adsorbate": ads,
                            "kind": kind, "calc": "relax", "infile": "relax.in",
                            "dir": str(cdir), "prefix": prefix,
                            "natoms": info["natoms"], "nfixed": info["nfixed"],
                            "kpts": "gamma", "dG_uma": "", "src_cif": str(cif)})
                print(f"  wrote {prefix} ({info['natoms']} atoms, {info['nfixed']} fixed)")
    # gas rows -> reuse already-computed MoTiZr references
    for g in ("CO", "CO2", "H2"):
        man.append({"formula": "_gas", "facet": "", "adsorbate": g, "kind": "_gas",
                    "calc": "relax", "infile": "relax.in", "dir": str(GAS_DIR / g),
                    "prefix": g, "natoms": "", "nfixed": "", "kpts": "gamma",
                    "dG_uma": "", "src_cif": ""})
    OUT.mkdir(parents=True, exist_ok=True)
    mf = OUT / "surface_manifest.csv"
    cols = sorted({k for d in man for k in d})
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(man)
    print(f"\nwrote {len(man)} rows -> {mf}")


if __name__ == "__main__":
    main()
