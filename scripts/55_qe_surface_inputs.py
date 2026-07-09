#!/usr/bin/env python
"""Phase A (DFT) — write QE inputs for the CO2RR surface free-energy validation.

Consumes the geometry manifest emitted by ``54_co2rr_surface_expand.py``
(``outputs/qe_surface/targets/dft_targets.csv``: one row per exported
clean-slab / adslab CIF, with formula, facet, adsorbate, UMA ΔG and atom
counts). For every distinct slab it writes a fixed-cell ionic ``relax`` input
(bottom half frozen, metallic smearing), and once it writes molecule-in-a-box
``relax`` inputs for the gas references CO, CO2 and H2.

The DFT ΔG then mirrors the UMA recipe (co_surface_screen), now with QE energies:

    ΔG_CO*   = E(*CO)   − E(*) − E(CO_g)                 + 0.10
    ΔG_COOH* = E(*COOH) − E(*) − E(CO2_g) − 1/2 E(H2_g)  + 0.41
    ΔG_H*    = E(*H)    − E(*) − 1/2 E(H2_g)             + 0.24

Pseudos / cutoffs / smearing match the bulk validation (52_qe_run.py): SSSP
efficiency set with the 4f-in-core NC override for La/Ce/Pr/Nd.

    python scripts/55_qe_surface_inputs.py \
        --manifest outputs/qe_surface/targets/dft_targets.csv \
        --out outputs/qe_surface/dft
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms

PSEUDO_DIR = "/home/jonglee69/software/qe_pseudo/active"
SSSP_JSON = "/home/jonglee69/software/qe_pseudo/sssp_efficiency.json"
REE_OVERRIDE = {"La": "La.upf", "Ce": "Ce.upf", "Pr": "Pr.upf", "Nd": "Nd.upf"}
ECUTWFC, ECUTRHO = 60.0, 480.0
KDENS_SLAB = 30.0   # in-plane k density (Å); out-of-plane forced to 1
FIX_FRAC = 0.5
GAS_BOX = 14.0      # Å cubic cell for isolated molecules

# Gas reference geometries (Å), relaxed by QE.
GAS = {
    "H2":  ("H2",  [[0, 0, 0], [0, 0, 0.74]]),
    "CO":  ("CO",  [[0, 0, 0], [0, 0, 1.128]]),
    "CO2": ("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]),
    "H2O": ("H2O", [[0, 0, 0], [0.7575, 0.5865, 0], [-0.7575, 0.5865, 0]]),
}


def load_pseudos(symbols):
    sssp = json.load(open(SSSP_JSON))
    pp = {}
    for e in sorted(set(symbols)):
        pp[e] = REE_OVERRIDE.get(e, sssp[e]["filename"])
    return pp


def fix_bottom(atoms, frac=FIX_FRAC):
    z = atoms.positions[:, 2]
    cutoff = z.min() + frac * (z.max() - z.min())
    atoms.set_constraint(FixAtoms(mask=z <= cutoff))
    return int((z <= cutoff).sum())


def kgrid_slab(atoms):
    L = atoms.cell.lengths()
    return (max(1, int(np.ceil(KDENS_SLAB / L[0]))),
            max(1, int(np.ceil(KDENS_SLAB / L[1]))), 1)


def write_slab(cif, cdir, prefix, calc="scf", kmode="gamma", kmesh=None):
    """Write a slab QE input.

    calc='scf'  -> single-point on the (UMA-relaxed) geometry. On a 16 GB GPU
                   the dense-grid USPP/PAW slabs only fit at the Gamma point
                   (k>1 pushes >12 GB and thrashes); ΔG = adslab-clean at the
                   SAME cell/k cancels the bulk of the Gamma-point error. This
                   is the DFT//UMA cross-check of the surrogate ΔG.
    calc='relax'-> full ionic relaxation (bottom half fixed). Much heavier;
                   reserve for a few top sites.
    """
    cdir.mkdir(parents=True, exist_ok=True)
    atoms = read(str(cif))
    nfix = fix_bottom(atoms) if calc == "relax" else 0
    pseudos = load_pseudos(atoms.get_chemical_symbols())
    if kmesh is not None:            # explicit override (e.g. k-convergence ladder)
        kpts = tuple(kmesh)
    else:
        kpts = None if kmode == "gamma" else kgrid_slab(atoms)
    # tprnfor only for relax. The NVHPC/GPU USPP force routine segfaults on the
    # larger low-symmetry slabs (>~46 atoms) AFTER SCF converges; single-point
    # ΔG needs only the energy, so skip forces for scf and dodge the crash.
    ctrl = {"calculation": calc, "prefix": prefix, "outdir": "./tmp",
            "pseudo_dir": PSEUDO_DIR, "tprnfor": (calc == "relax"),
            "tstress": False, "verbosity": "low"}
    idata = {
        "control": ctrl,
        "system": {"ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        "electrons": {"conv_thr": 1.0e-6, "mixing_beta": 0.3, "electron_maxstep": 150},
    }
    if calc == "relax":
        ctrl.update({"nstep": 200, "forc_conv_thr": 1.0e-3})
        idata["electrons"]["conv_thr"] = 1.0e-7
        idata["ions"] = {"ion_dynamics": "bfgs"}
    infile = "scf.in" if calc == "scf" else "relax.in"
    write(str(cdir / infile), atoms, format="espresso-in",
          input_data=idata, pseudopotentials=pseudos, kpts=kpts)
    return {"prefix": prefix, "dir": str(cdir), "natoms": len(atoms),
            "nfixed": nfix, "kpts": ("gamma" if kpts is None else list(kpts)),
            "infile": infile, "calc": calc}


def write_gas_relax(name, cdir):
    from ase import Atoms
    cdir.mkdir(parents=True, exist_ok=True)
    sym, pos = GAS[name]
    atoms = Atoms(sym, positions=pos)
    atoms.set_cell([GAS_BOX, GAS_BOX, GAS_BOX]); atoms.center(); atoms.pbc = True
    pseudos = load_pseudos(atoms.get_chemical_symbols())
    idata = {
        "control": {"calculation": "relax", "prefix": name, "outdir": "./tmp",
                    "pseudo_dir": PSEUDO_DIR, "tprnfor": True, "nstep": 100,
                    "verbosity": "low", "forc_conv_thr": 1.0e-3},
        # molecule in a box: Γ only, Martyna–Tuckerman decoupling, light smearing
        "system": {"ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "assume_isolated": "mt", "occupations": "smearing",
                   "smearing": "gauss", "degauss": 0.005},
        "electrons": {"conv_thr": 1.0e-8, "mixing_beta": 0.3},
        "ions": {"ion_dynamics": "bfgs"},
    }
    write(str(cdir / "relax.in"), atoms, format="espresso-in",
          input_data=idata, pseudopotentials=pseudos, kpts=None)
    return {"prefix": name, "dir": str(cdir), "natoms": len(atoms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calc", choices=("scf", "relax"), default="scf",
                    help="scf = single-point DFT//UMA (default, GPU-tractable); "
                         "relax = full ionic relaxation (heavy)")
    ap.add_argument("--kpts", choices=("gamma", "auto"), default="gamma",
                    help="gamma (default, fits 16 GB GPU) or density-based grid")
    ap.add_argument("--kmesh", default=None,
                    help="explicit Monkhorst-Pack mesh 'Nx,Ny,Nz' overriding --kpts "
                         "(for k-convergence tests, e.g. '2,2,1')")
    ap.add_argument("--gas", action="store_true",
                    help="also (re)write gas-reference inputs")
    args = ap.parse_args()
    kmesh = tuple(int(x) for x in args.kmesh.split(",")) if args.kmesh else None

    rows = list(csv.DictReader(open(args.manifest)))
    man = []
    seen = set()
    for r in rows:
        formula, facet, ads = r["formula"], r["facet"], r["adsorbate"]
        for kind in ("clean", "ads"):
            cif = Path(r[f"{kind}_cif"])
            prefix = f"{formula}_f{facet}_{ads}_{kind}"
            cdir = args.out / formula / f"facet{facet}_{ads}_{kind}"
            if prefix in seen:
                continue
            seen.add(prefix)
            info = write_slab(cif, cdir, prefix, calc=args.calc, kmode=args.kpts, kmesh=kmesh)
            info.update({"formula": formula, "facet": facet, "adsorbate": ads,
                         "kind": kind, "dG_uma": r.get("dG_uma"),
                         "src_cif": str(cif)})
            man.append(info)
            print(f"  {prefix:38s} nat={info['natoms']:3d} {args.calc} k={info['kpts']}")

    # gas references needed for the adsorbates present (CO->CO; COOH->CO2,H2;
    # H->H2; OH/O->H2O,H2). already computed at 60/480 (relax); rewrite on request.
    ads_present = {r["adsorbate"] for r in rows}
    need = {"H2"}
    if "CO" in ads_present:   need |= {"CO"}
    if "COOH" in ads_present: need |= {"CO2"}
    if ads_present & {"OH", "O"}: need |= {"H2O"}
    gases = [g for g in ("CO", "CO2", "H2", "H2O") if g in need]
    if args.gas:
        for g in gases:
            cdir = args.out / "_gas" / g
            info = write_gas_relax(g, cdir)
            info.update({"formula": "_gas", "facet": "", "adsorbate": g,
                         "kind": "gas", "infile": "relax.in", "calc": "relax"})
            man.append(info)
            print(f"  gas {g:4s} -> {cdir}")
    else:
        # carry forward the existing gas rows so 56 can find them
        for g in gases:
            man.append({"formula": "_gas", "facet": "", "adsorbate": g,
                        "kind": "gas", "infile": "relax.in", "calc": "relax",
                        "prefix": g, "dir": str(args.out / "_gas" / g)})

    mf = args.out / "surface_manifest.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for d in man for k in d}))
        w.writeheader(); w.writerows(man)
    print(f"\nwrote {len(man)} QE inputs; manifest -> {mf}")


if __name__ == "__main__":
    main()
