#!/usr/bin/env python
"""Phase A (DFT) — run the CO2RR surface relaxations and assemble ΔG.

Reads ``surface_manifest.csv`` (from 55_qe_surface_inputs.py), runs each
``relax`` (resumable: skipped when its out already says 'JOB DONE'), parses the
final total energy, and combines clean-slab + adslab + gas-reference energies
into the CHE free energies ΔG_CO*, ΔG_COOH*, ΔG_H* per (formula, facet).

Works with either the CPU QE build (default) or a GPU pw.x: pass --pw / --mpi
(e.g. ``--pw /path/to/gpu/pw.x --mpi "mpirun -np 1"``). GPU runs typically use
one MPI rank per GPU; CPU runs use several.

    python scripts/56_qe_surface_run.py --dft outputs/qe_surface/dft
    # GPU:
    python scripts/56_qe_surface_run.py --dft outputs/qe_surface/dft \
        --pw /home/jonglee69/software/qe_gpu/bin/pw.x --mpi "mpirun -np 1" --npool 1
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

RY = 13.605693122994  # eV per Ry
DG_CORR = {"CO": 0.10, "COOH": 0.41, "H": 0.24, "OH": 0.30, "O": 0.05}
DEF_QE_BIN = "/home/jonglee69/software/qe_build/q-e-qe-7.3.1/bin"


def done(out: Path) -> bool:
    return out.exists() and "JOB DONE" in out.read_text(errors="ignore")


def run(cmd, cwd, outfile, qe_bin):
    env = dict(os.environ, OMP_NUM_THREADS="1", PATH=f"{qe_bin}:{os.environ['PATH']}")
    with open(cwd / outfile, "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)


def final_energy_eV(out: Path):
    if not out.exists():
        return None
    t = out.read_text(errors="ignore")
    e = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)", t)
    return float(e[-1]) * RY if e else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dft", required=True, type=Path, help="dir with surface_manifest.csv")
    ap.add_argument("--pw", default=f"{DEF_QE_BIN}/pw.x")
    ap.add_argument("--mpi", default="mpirun --use-hwthread-cpus -np 4")
    ap.add_argument("--npool", type=int, default=4)
    ap.add_argument("--only", default=None, help="run a single formula")
    ap.add_argument("--max-atoms", type=int, default=None,
                    help="skip RUNNING slabs above this atom count (e.g. GPU 16GB cap ~70); "
                         "still assembled into ΔG if already computed elsewhere (CPU)")
    ap.add_argument("--min-atoms", type=int, default=None,
                    help="skip RUNNING slabs below this atom count (CPU complement to --max-atoms)")
    ap.add_argument("--run/--no-run", dest="run", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    qe_bin = str(Path(args.pw).parent)
    mpi = args.mpi.split()
    rows = list(csv.DictReader(open(args.dft / "surface_manifest.csv")))
    if args.only:
        rows = [r for r in rows if r["formula"] in (args.only, "_gas")]

    # 1) run every relaxation (resumable)
    energies = {}
    for r in rows:
        cdir = Path(r["dir"])
        infile = r.get("infile") or "relax.in"
        outfile = infile.replace(".in", ".out")
        out = cdir / outfile
        nat = int(r.get("natoms") or 0)
        too_big = bool(nat and args.max_atoms and nat > args.max_atoms)
        too_small = bool(nat and args.min_atoms and nat < args.min_atoms)
        if args.run and not done(out) and not (too_big or too_small):
            print(f"[run ] {r['prefix']} ({r.get('natoms','?')} atoms, {infile})", flush=True)
            run(mpi + [args.pw, "-npool", str(args.npool), "-in", infile],
                cdir, outfile, qe_bin)
        elif (too_big or too_small) and not done(out):
            why = f">{args.max_atoms}" if too_big else f"<{args.min_atoms}"
            print(f"[skip-size {why}] {r['prefix']} ({nat} atoms)", flush=True)
        e = final_energy_eV(out)
        energies[r["prefix"]] = e
        flag = "ok" if done(out) else "INCOMPLETE"
        print(f"[{flag}] {r['prefix']}: E={e} eV", flush=True)

    # 2) gas references
    def gas(name):
        return energies.get(name)
    e_CO, e_CO2, e_H2, e_H2O = gas("CO"), gas("CO2"), gas("H2"), gas("H2O")
    ref = {"CO": (lambda: e_CO),
           "COOH": (lambda: (e_CO2 + 0.5 * e_H2) if (e_CO2 and e_H2) else None),
           "H": (lambda: 0.5 * e_H2 if e_H2 else None),
           "OH": (lambda: (e_H2O - 0.5 * e_H2) if (e_H2O and e_H2) else None),
           "O": (lambda: (e_H2O - e_H2) if (e_H2O and e_H2) else None)}

    # 3) assemble ΔG per (formula, facet, adsorbate)
    results = {}
    for r in rows:
        if r["formula"] == "_gas":
            continue
        key = (r["formula"], r["facet"], r["adsorbate"])
        d = results.setdefault(key, {})
        d[r["kind"]] = energies.get(r["prefix"])
        d["dG_uma"] = r.get("dG_uma")

    out_rows = []
    for (formula, facet, ads), d in sorted(results.items()):
        e_ads, e_clean = d.get("ads"), d.get("clean")
        e_ref = ref[ads]()
        dG = None
        if None not in (e_ads, e_clean, e_ref):
            dG = round(e_ads - e_clean - e_ref + DG_CORR[ads], 4)
        out_rows.append({"formula": formula, "facet": facet, "adsorbate": ads,
                         "dG_dft_eV": dG, "dG_uma_eV": d.get("dG_uma"),
                         "E_ads_eV": e_ads, "E_clean_eV": e_clean})
        print(f"  {formula:12s} f{facet:4s} {ads:4s}  ΔG_DFT={dG}  (UMA {d.get('dG_uma')})")

    csvf = args.dft / "surface_dft_dG.csv"
    with open(csvf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {csvf}")


if __name__ == "__main__":
    main()
