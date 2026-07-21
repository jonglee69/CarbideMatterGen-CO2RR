#!/usr/bin/env python
"""Step — UMA bulk relaxation, applied to EVERY structure before any surface screen.

Fair-comparison fix (this bit matters).

The mixing ladder compares structures of very different provenance:

    training / MP references (HfMoC2, Ta2W2C3 ...)  -> DFT-relaxed
    MatterGen output                                -> raw diffusion samples
    SQS supercells                                  -> ideal lattices, never relaxed
    binary baselines                                -> experimental lattice constants

Cutting a slab from an *unrelaxed* bulk gives a strained surface. We measured
exactly what that does: generated candidates whose ΔG_CO was essentially perfect
(HfMoWC5: -0.006) came back with U_L = -4.04 V, against -0.45 V for the
DFT-relaxed HfMoC2. The *COOH dissociates on the strained facet and the limiting
potential collapses. That gap is an artefact of geometry, not of chemistry, and
comparing the two as-is would have buried the generated candidates for the wrong
reason.

So: every rung of the ladder is relaxed (cell + positions) with the SAME UMA
calculator that later screens its surfaces.

    python scripts/37_uma_relax_bulk.py --in outputs/sqs_hec --out outputs/sqs_hec_relaxed
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from ase.io import read, write
    from ase.optimize import LBFGS
    from ase.filters import FrechetCellFilter
    from carbidemattergen.co_surface_screen import _load_fairchem_calc

    calc = _load_fairchem_calc("uma-s-1p1", args.device, "omat")
    args.dst.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src.glob("*.cif"))
    print(f"relaxing {len(files)} structures (UMA, cell+positions)", flush=True)
    for i, fp in enumerate(files, 1):
        out = args.dst / fp.name
        if out.exists():
            continue
        try:
            atoms = read(str(fp))
            atoms.calc = calc
            e0 = atoms.get_potential_energy()
            dyn = LBFGS(FrechetCellFilter(atoms), logfile=None)
            dyn.run(fmax=args.fmax, steps=args.steps)
            e1 = atoms.get_potential_energy()
            write(str(out), atoms)
            print(f"  [{i}/{len(files)}] {fp.stem:36s} nat={len(atoms):3d} "
                  f"dE={e1 - e0:+8.3f} eV conv={dyn.converged()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(files)}] {fp.stem:36s} FAILED: {exc}", flush=True)
    print(f"\nrelaxed -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
