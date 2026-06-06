#!/usr/bin/env python
"""Step 7+ (QE) — element-projected DOS (PDOS) for the top-6 CO2RR candidates.

Runs projwfc.x on each candidate's EXISTING scf save (no SCF recompute — it just
projects the saved Kohn-Sham wavefunctions onto atomic orbitals), then sums the
per-atom-orbital PDOS into per-element contributions and tabulates each element's
share of the density of states at E_F (its contribution to metallic conduction).

Resumable + gentle (default -np 2) so it can run alongside 52_qe_run.py.

Usage:  python scripts/53_qe_pdos.py --qe-dir outputs/qe [--only Pr3Y2WC6] [-np 2]
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

QE_BIN = "/home/jonglee69/software/qe_build/q-e-qe-7.3.1/bin"
PROJWFC = f"{QE_BIN}/projwfc.x"


def scf_done(cdir: Path) -> bool:
    o = cdir / "scf.out"
    return o.exists() and "JOB DONE" in o.read_text(errors="ignore")


def efermi(cdir: Path):
    t = (cdir / "scf.out").read_text(errors="ignore")
    m = re.findall(r"the Fermi energy is\s+(-?\d+\.\d+)", t)
    return float(m[-1]) if m else None


def run_projwfc(cdir: Path, prefix: str, ef: float, np_: int):
    (cdir / "projwfc.in").write_text(
        "&PROJWFC\n"
        f"  prefix='{prefix}', outdir='./tmp',\n"
        f"  filpdos='{prefix}',\n"
        f"  Emin={ef-12:.3f}, Emax={ef+12:.3f}, DeltaE=0.05, ngauss=0, degauss=0.01\n"
        "/\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", PATH=f"{QE_BIN}:{os.environ['PATH']}")
    cmd = ["mpirun", "--use-hwthread-cpus", "-np", str(np_), PROJWFC, "-in", "projwfc.in"]
    with open(cdir / "projwfc.out", "w") as f:
        subprocess.run(cmd, cwd=cdir, stdout=f, stderr=subprocess.STDOUT, env=env)
    return "JOB DONE" in (cdir / "projwfc.out").read_text(errors="ignore")


def collect_pdos(cdir: Path, prefix: str):
    """Sum per-atom-orbital pdos_atm files into a per-element DOS table.

    projwfc nspin=1 pdos_atm file columns: E(eV), ldos(E), pdos(E)[per m...].
    The element is encoded in the filename: <prefix>.pdos_atm#N(El)_wfc#M(l)."""
    files = glob.glob(f"{cdir}/{prefix}.pdos_atm#*")
    if not files:
        return None
    energy = None
    per_el: dict[str, np.ndarray] = {}
    for fp in files:
        m = re.search(r"pdos_atm#\d+\(([A-Za-z]+)\)", os.path.basename(fp))
        if not m:
            continue
        el = m.group(1)
        data = np.array([[float(x) for x in l.split()]
                         for l in Path(fp).read_text().splitlines()
                         if l.strip() and not l.lstrip().startswith("#")])
        if energy is None:
            energy = data[:, 0]
        ldos = data[:, 1]  # orbital-summed local DOS for this atomic wfc
        per_el[el] = per_el.get(el, np.zeros_like(ldos)) + ldos
    df = pd.DataFrame({"E_eV": energy})
    for el, v in per_el.items():
        df[el] = v
    df["total"] = df[[c for c in df.columns if c != "E_eV"]].sum(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qe-dir", required=True, type=Path)
    ap.add_argument("--only", default=None)
    ap.add_argument("-np", dest="np_", type=int, default=2)
    args = ap.parse_args()

    rows = []
    for cdir in sorted(args.qe_dir.glob("*/")):
        prefix = cdir.name
        if args.only and prefix != args.only:
            continue
        if not scf_done(cdir):
            continue
        ef = efermi(cdir)
        if ef is None:
            continue
        if not glob.glob(f"{cdir}/{prefix}.pdos_atm#*"):
            print(f"[{prefix}] projwfc...", flush=True)
            run_projwfc(cdir, prefix, ef, args.np_)
        pdos = collect_pdos(cdir, prefix)
        if pdos is None:
            print(f"[{prefix}] no pdos produced", flush=True); continue
        pdos.to_csv(cdir / f"{prefix}.pdos_elements.csv", index=False)
        # per-element contribution at E_F
        i = int(np.argmin(np.abs(pdos["E_eV"] - ef)))
        elems = [c for c in pdos.columns if c not in ("E_eV", "total")]
        tot = pdos["total"].iloc[i] or np.nan
        share = {el: round(100 * pdos[el].iloc[i] / tot, 1) for el in elems}
        rec = {"formula": prefix, "E_fermi": ef, "DOS_Ef_total": round(tot, 2)}
        rec.update({f"%{el}": share[el] for el in elems})
        rows.append(rec)
        print(f"  {prefix}: DOS(E_F)={tot:.2f}  shares%={share}", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.qe_dir / "qe_pdos_Ef_contributions.csv", index=False)
        print("\n=== per-element share of DOS(E_F) ===")
        print(df.to_string(index=False))
        print("saved", args.qe_dir / "qe_pdos_Ef_contributions.csv")


if __name__ == "__main__":
    main()
