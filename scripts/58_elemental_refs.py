#!/usr/bin/env python
"""Elemental reference DFT (QE) for candidate FORMATION energies (thesis item 2).

Runs a vc-relax for each constituent element in its (approximate) ground-state
structure using the SAME pseudopotentials / cutoffs / smearing as the carbide
scf runs (51_qe_gen_inputs.py), so E_f = [E(carbide) - sum n_i E_i]/N is on a
single consistent QE-PBE footing. Metals use a dense k-mesh (unlike the Gamma-only
surface slabs) since small cells need it.

Reference structures (experimental lattice constants; vc-relax refines them):
  W  bcc a=3.165 | Zr hcp a=3.232 c=5.147 | Y hcp a=3.647 c=5.731
  La fcc a=5.303 | Ce fcc a=5.161 | Pr fcc a=5.161 | C  diamond a=3.567
(La/Pr are dhcp experimentally; fcc is a <~10 meV/atom approximation -- flagged in
the thesis. C reference = diamond, robust in PBE; graphite would shift E_f of every
carbide by a per-C constant only.)

Outputs outputs/qe/elemental_refs.csv (element, structure, natoms, E_eV, E_per_atom_eV).
Resumable: skips an element whose relax.out already says JOB DONE.

    OMP_NUM_THREADS=1 PYTHONPATH=$PWD /home/jonglee69/.venv_fairchem/bin/python \
        scripts/58_elemental_refs.py --pw <cpu pw.x> --mpi "mpirun -np 4" --npool 4
"""
from __future__ import annotations
import argparse, csv, re, subprocess
from pathlib import Path
from ase.build import bulk
from ase.io import write

RY = 13.605693122994
PSEUDO_DIR = Path("/home/jonglee69/software/qe_pseudo/active")
ECUTWFC, ECUTRHO = 60.0, 480.0
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs/qe/_elements"; OUTDIR.mkdir(parents=True, exist_ok=True)

# element -> (ase structure, kwargs, pseudo file matching the carbide runs)
ELEMENTS = {
    "W":  (("W", "bcc"),     dict(a=3.165),            "W_pbe_v1.2.uspp.F.UPF"),
    "Zr": (("Zr", "hcp"),    dict(a=3.232, c=5.147),   "Zr_pbe_v1.uspp.F.UPF"),
    "Y":  (("Y", "hcp"),     dict(a=3.647, c=5.731),   "Y_pbe_v1.uspp.F.UPF"),
    "La": (("La", "fcc"),    dict(a=5.303),            "La.upf"),
    "Ce": (("Ce", "fcc"),    dict(a=5.161),            "Ce.upf"),
    "Pr": (("Pr", "fcc"),    dict(a=5.161),            "Pr.upf"),
    "C":  (("C", "diamond"), dict(a=3.567),            "C.pbe-n-kjpaw_psl.1.0.0.UPF"),
}


def kgrid(atoms, target=45.0):
    import numpy as np
    L = atoms.cell.lengths()
    return tuple(max(8, int(round(target / x))) for x in L)


def done(out: Path):
    return out.exists() and "JOB DONE" in out.read_text(errors="ignore")


def final_E(out: Path):
    if not out.exists():
        return None
    e = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)", out.read_text(errors="ignore"))
    return float(e[-1]) * RY if e else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", default="/home/jonglee69/software/qe_build/q-e-qe-7.3.1/bin/pw.x")
    ap.add_argument("--mpi", default="mpirun --use-hwthread-cpus -np 4")
    ap.add_argument("--npool", type=int, default=4)
    ap.add_argument("--run", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()
    mpi = args.mpi.split()
    qe_bin = str(Path(args.pw).parent)

    rows = []
    for el, ((sym, struct), kw, pseudo) in ELEMENTS.items():
        atoms = bulk(sym, struct, **kw)
        cdir = OUTDIR / el; cdir.mkdir(exist_ok=True)
        infile, outfile = "relax.in", "relax.out"
        idata = {
            "calculation": "vc-relax", "prefix": el, "outdir": "./tmp",
            "pseudo_dir": str(PSEUDO_DIR), "tstress": True, "tprnfor": True,
            "forc_conv_thr": 1e-4, "etot_conv_thr": 1e-5,
            "ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
            "occupations": "smearing", "smearing": "mv", "degauss": 0.02,
            "conv_thr": 1e-8, "electron_maxstep": 200,
            "ion_dynamics": "bfgs", "cell_dynamics": "bfgs", "press": 0.0,
        }
        write(cdir / infile, atoms, format="espresso-in", input_data=idata,
              pseudopotentials={sym: pseudo}, kpts=kgrid(atoms))
        out = cdir / outfile
        if args.run and not done(out):
            k = kgrid(atoms)
            print(f"[run ] {el:2s} {struct:7s} nat={len(atoms)} k={k} pp={pseudo}", flush=True)
            with open(out, "w") as f:
                subprocess.run(mpi + [args.pw, "-npool", str(args.npool), "-in", infile],
                               cwd=cdir, stdout=f, stderr=subprocess.STDOUT,
                               env={"PATH": f"{qe_bin}:/usr/bin:/bin", "OMP_NUM_THREADS": "1",
                                    "ESPRESSO_PSEUDO": str(PSEUDO_DIR)})
        E = final_E(out)
        n = len(atoms)
        ok = done(out)
        print(f"[{'ok' if ok else 'INCOMPLETE'}] {el}: E={E} eV  E/atom={E/n if E else None}", flush=True)
        rows.append({"element": el, "structure": struct, "natoms": n,
                     "pseudo": pseudo, "E_eV": E,
                     "E_per_atom_eV": round(E / n, 6) if E else None})

    csvf = ROOT / "outputs/qe/elemental_refs.csv"
    with open(csvf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csvf}")


if __name__ == "__main__":
    main()
