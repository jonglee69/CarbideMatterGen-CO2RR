#!/usr/bin/env python
"""Campaign-B convex-hull inputs: vc-relax for the 3 winner SQS bulks and the
elemental references (Ti/Zr/Mo/Cr/V/C), all with the SAME SSSP pseudopotentials
used in the Campaign-B surface DFT (script 55), so formation energies cancel.

    python scripts/71_bulk_hull_inputs.py --out outputs/qe_hullB
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import read, write

SSSP = json.load(open("/home/jonglee69/software/qe_pseudo/sssp_efficiency.json"))
PDIR = "/home/jonglee69/software/qe_pseudo/active"
ECUTWFC, ECUTRHO = 60.0, 480.0

# elemental references (0 K ground-state structures)
ELEMS = {
    "Ti": ("hcp", dict(a=2.951, c=4.684)),
    "Zr": ("hcp", dict(a=3.232, c=5.147)),
    "Mo": ("bcc", dict(a=3.147)),
    "Cr": ("bcc", dict(a=2.884)),
    "V":  ("bcc", dict(a=3.024)),
    "C":  ("diamond", dict(a=3.567)),
}
CANDS = ["hec3_MoTiZr", "hec3_CrTiV", "hec3_CrTiZr"]


def pp(symbols):
    return {e: SSSP[e]["filename"] for e in sorted(set(symbols))}


def kmesh(atoms, target=40.0):
    return tuple(max(2, int(round(target / L))) for L in atoms.cell.lengths())


def write_vcrelax(atoms, name, outdir: Path):
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    idata = {
        "control": {"calculation": "vc-relax", "prefix": name, "outdir": "./tmp",
                    "pseudo_dir": PDIR, "tprnfor": True, "tstress": True,
                    "nstep": 200, "forc_conv_thr": 1e-3, "etot_conv_thr": 1e-4},
        "system": {"ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        "electrons": {"conv_thr": 1e-6, "mixing_beta": 0.3, "electron_maxstep": 200},
        "ions": {}, "cell": {"cell_dofree": "all"},
    }
    write(str(d / "vc-relax.in"), atoms, format="espresso-in", input_data=idata,
          pseudopotentials=pp(atoms.get_chemical_symbols()), kpts=kmesh(atoms))
    return len(atoms), kmesh(atoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/qe_hullB"))
    ap.add_argument("--cif-dir", type=Path, default=Path("outputs/sqs_hec"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("== elemental references ==")
    for e, (struct, kw) in ELEMS.items():
        a = bulk(e, struct, **kw)
        n, k = write_vcrelax(a, f"elem_{e}", args.out)
        print(f"  elem_{e:2s} ({struct}) nat={n} k={k}")
    print("== candidates ==")
    for nm in CANDS:
        a = read(str(args.cif_dir / f"{nm}.cif"))
        n, k = write_vcrelax(a, nm, args.out)
        print(f"  {nm:14s} {a.get_chemical_formula():18s} nat={n} k={k}")
    print(f"\nwrote inputs -> {args.out}")


if __name__ == "__main__":
    main()
