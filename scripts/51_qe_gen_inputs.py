#!/usr/bin/env python
"""Step 7 (QE) — generate Quantum ESPRESSO inputs for the fairchem-selected top-6
economic-HEC CO2RR candidates. For each structure writes a vc-relax input (cell +
positions, metallic smearing). scf + DOS inputs are written by the driver after
relaxation (they need the relaxed geometry). SSSP 1.3.0 PBE efficiency PPs.

Usage:
  python scripts/51_qe_gen_inputs.py --top6 outputs/screen_co2rr/top6.csv \
      --cand-dir outputs/screen_co2rr/candidates --out outputs/qe
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from ase.io import read, write
from ase.data import atomic_masses, atomic_numbers

PSEUDO_DIR = Path("/home/jonglee69/software/qe_pseudo/active")
SSSP_JSON = Path("/home/jonglee69/software/qe_pseudo/sssp_efficiency.json")

# SSSP's atompaw-Wentzcovitch REE PAWs crash QE 7.3.1 during PAW init (silent
# MPI_ABORT, no CRASH file). Override the rare earths with norm-conserving
# pseudo-dojo pseudos (La = standard; Ce/Pr/Nd = the "3+" 4f-in-core set, which is
# physically apt for these metallic carbides where 4f is localised). All linked
# under PSEUDO_DIR with canonical <El>.upf names.
REE_OVERRIDE = {"La": "La.upf", "Ce": "Ce.upf", "Pr": "Pr.upf", "Nd": "Nd.upf"}

ECUTWFC = 60.0   # Ry (covers SSSP USPP/PAW + the NC REE pseudos)
ECUTRHO = 480.0  # Ry (8x; safe for the USPP augmentation in the mix)
KDENS = 30.0     # MP grid n_i = ceil(KDENS / a_i[Å]); metallic-friendly


def mp_grid(atoms):
    import numpy as np
    abc = atoms.cell.lengths()
    return [max(1, int(np.ceil(KDENS / a))) for a in abc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top6", required=True, type=Path)
    ap.add_argument("--cand-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sssp = json.load(open(SSSP_JSON))
    df = pd.read_csv(args.top6)
    name2path = {p.name: p for p in args.cand_dir.glob("*.cif")}

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, r in df.iterrows():
        atoms = read(str(name2path[r["source"]]))
        elems = sorted(set(atoms.get_chemical_symbols()))
        pseudos = {e: REE_OVERRIDE.get(e, sssp[e]["filename"]) for e in elems}
        cdir = args.out / r["formula"]
        cdir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "control": {
                "calculation": "vc-relax", "restart_mode": "from_scratch",
                "prefix": r["formula"], "outdir": "./tmp", "pseudo_dir": str(PSEUDO_DIR),
                "tprnfor": True, "tstress": True, "forc_conv_thr": 1.0e-3,
                "nstep": 150, "verbosity": "low",
            },
            "system": {
                "ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                "occupations": "smearing", "smearing": "mv", "degauss": 0.02,
            },
            "electrons": {"conv_thr": 1.0e-6, "mixing_beta": 0.3, "electron_maxstep": 200},
            "ions": {"ion_dynamics": "bfgs"},
            "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
        }
        kpts = tuple(mp_grid(atoms))
        write(str(cdir / "vc-relax.in"), atoms, format="espresso-in",
              input_data=input_data, pseudopotentials=pseudos, kpts=kpts)
        manifest.append({"formula": r["formula"], "natoms": len(atoms),
                         "elements": "".join(elems), "kpts": "x".join(map(str, kpts)),
                         "dir": str(cdir)})
        print(f"  {r['formula']:14s} nat={len(atoms):2d}  k={kpts}  pp={[pseudos[e] for e in elems]}")

    pd.DataFrame(manifest).to_csv(args.out / "manifest.csv", index=False)
    print(f"\nwrote {len(manifest)} vc-relax inputs → {args.out}")


if __name__ == "__main__":
    main()
