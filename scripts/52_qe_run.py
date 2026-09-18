#!/usr/bin/env python
"""Step 7 (QE) — driver: for each top-6 candidate run the bulk validation chain
vc-relax -> scf -> dos, then parse stability + metallicity descriptors.

- vc-relax: relax cell+positions (input from 51_qe_gen_inputs.py).
- scf:      single point on the relaxed geometry (denser k, tprnfor/tstress).
- dos.x:    density of states; metallicity = finite DOS at E_F.

Resumable: a stage is skipped if its output already says 'JOB DONE'. Run inside
any env with ASE (e.g. .venv_fairchem). QE binaries + pseudos are wired below.

Usage:  python scripts/52_qe_run.py --qe-dir outputs/qe [--only CeY2ZrWC7]
"""
from __future__ import annotations
import argparse, os, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write

QE_BIN = "/home/jonglee69/software/qe_build/q-e-qe-7.3.1/bin"
PSEUDO_DIR = "/home/jonglee69/software/qe_pseudo/active"
PW = f"{QE_BIN}/pw.x"; DOSX = f"{QE_BIN}/dos.x"
MPI = ["mpirun", "--use-hwthread-cpus", "-np", "4"]
POOL = ["-npool", "4"]
ECUTWFC, ECUTRHO = 60.0, 480.0
KDENS_SCF = 40.0   # denser k for the DOS scf than the relax grid


def done(out: Path) -> bool:
    return out.exists() and "JOB DONE" in out.read_text(errors="ignore")


def run(cmd, cwd, outfile):
    env = dict(os.environ, OMP_NUM_THREADS="1", PATH=f"{QE_BIN}:{os.environ['PATH']}")
    with open(cwd / outfile, "w") as f:
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def mp_grid(atoms, dens):
    return [max(1, int(np.ceil(dens / a))) for a in atoms.cell.lengths()]


def write_scf(atoms, cdir, prefix, pseudos):
    kpts = mp_grid(atoms, KDENS_SCF)
    idata = {
        "control": {"calculation": "scf", "prefix": prefix, "outdir": "./tmp",
                    "pseudo_dir": PSEUDO_DIR, "tprnfor": True, "tstress": True, "verbosity": "low"},
        "system": {"ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        "electrons": {"conv_thr": 1.0e-8, "mixing_beta": 0.3, "electron_maxstep": 200},
    }
    write(str(cdir / "scf.in"), atoms, format="espresso-in",
          input_data=idata, pseudopotentials=pseudos, kpts=tuple(kpts))


def write_dos(cdir, prefix, efermi):
    (cdir / "dos.in").write_text(
        "&DOS\n"
        f"  prefix='{prefix}', outdir='./tmp', fildos='{prefix}.dos',\n"
        f"  Emin={efermi-10:.3f}, Emax={efermi+10:.3f}, DeltaE=0.05, degauss=0.01\n"
        "/\n")


def parse_pw(out: Path) -> dict:
    t = out.read_text(errors="ignore")
    d = {}
    e = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)", t)
    if e: d["E_Ry"] = float(e[-1])
    fe = re.findall(r"the Fermi energy is\s+(-?\d+\.\d+)", t)
    if fe: d["E_fermi_eV"] = float(fe[-1])
    p = re.findall(r"P=\s+(-?\d+\.\d+)", t)
    if p: d["P_kbar"] = float(p[-1])
    d["converged"] = "JOB DONE" in t
    return d


def dos_at_ef(dosfile: Path, efermi: float) -> float:
    if not dosfile.exists(): return float("nan")
    rows = [l.split() for l in dosfile.read_text().splitlines() if l.strip() and not l.startswith("#")]
    arr = np.array([[float(x) for x in r[:2]] for r in rows if len(r) >= 2])
    i = int(np.argmin(np.abs(arr[:, 0] - efermi)))
    return float(arr[i, 1])  # states/eV/cell at E_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qe-dir", required=True, type=Path)
    ap.add_argument("--only", default=None, help="run a single candidate by formula")
    args = ap.parse_args()

    man = pd.read_csv(args.qe_dir / "manifest.csv")
    if args.only:
        man = man[man.formula == args.only]
    summary = []
    for _, row in man.iterrows():
        cdir = Path(row["dir"]); prefix = row["formula"]
        print(f"\n=== {prefix} ({row['natoms']} atoms) ===", flush=True)

        # 1) vc-relax
        if not done(cdir / "vc-relax.out"):
            print("  vc-relax...", flush=True)
            run(MPI + [PW] + POOL + ["-in", "vc-relax.in"], cdir, "vc-relax.out")
        if not done(cdir / "vc-relax.out"):
            print("  vc-relax FAILED; skipping", flush=True)
            summary.append({"formula": prefix, "stage": "vc-relax FAILED"}); continue

        relaxed = read(str(cdir / "vc-relax.out"), index=-1)
        pseudos = {e: __import__("json").load(open("/home/jonglee69/software/qe_pseudo/sssp_efficiency.json")).get(e, {}).get("filename")
                   for e in sorted(set(relaxed.get_chemical_symbols()))}
        # apply REE override (NC pseudos)
        for e in list(pseudos):
            pseudos[e] = {"La": "La.upf", "Ce": "Ce.upf", "Pr": "Pr.upf", "Nd": "Nd.upf"}.get(e, pseudos[e])

        # 2) scf on relaxed geometry
        if not done(cdir / "scf.out"):
            write_scf(relaxed, cdir, prefix, pseudos)
            print("  scf...", flush=True)
            run(MPI + [PW] + POOL + ["-in", "scf.in"], cdir, "scf.out")
        scf = parse_pw(cdir / "scf.out")

        # 3) dos
        ef = scf.get("E_fermi_eV", 0.0)
        if not (cdir / f"{prefix}.dos").exists():
            write_dos(cdir, prefix, ef)
            print("  dos...", flush=True)
            run(MPI + [DOSX] + POOL + ["-in", "dos.in"], cdir, "dos.out")
        gef = dos_at_ef(cdir / f"{prefix}.dos", ef)

        rec = {"formula": prefix, "natoms": int(row["natoms"]),
               "E_Ry": scf.get("E_Ry"), "E_fermi_eV": ef,
               "P_kbar": scf.get("P_kbar"), "dos_at_Ef": round(gef, 3),
               "metallic": bool(gef > 0.1), "scf_converged": scf.get("converged")}
        summary.append(rec)
        print(f"  -> E_F={ef:.2f} eV  DOS(E_F)={gef:.2f} states/eV  metallic={rec['metallic']}", flush=True)

    df = pd.DataFrame(summary)
    df.to_csv(args.qe_dir / "qe_summary.csv", index=False)
    print("\n=== QE bulk validation summary ===")
    print(df.to_string(index=False))
    print(f"\nsaved {args.qe_dir/'qe_summary.csv'}")


if __name__ == "__main__":
    main()
