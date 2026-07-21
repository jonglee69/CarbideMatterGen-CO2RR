#!/usr/bin/env python
"""Campaign-B bulk DOS/PDOS for the 3 Ti-based winners (metallicity + element-
projected conduction character) -- the carbide analog of the REE DOS/PDOS behind
Figure 4.1(c,d). Uses the relaxed structures from the E_hull run (outputs/qe_hullB)
and the SAME SSSP pseudopotentials.

Pipeline per candidate:  scf (GPU pw.x, 3x3x3) -> dos.x -> projwfc.x
Outputs: outputs/qe_bulk_dos/hec3_<c>/{scf.out,<c>.dos,<c>.pdos_atm#*} and a
combined outputs/qe_bulk_dos/dos_pdos_Ef_B.csv (DOS(E_F) + per-element % at E_F).

    OMP_NUM_THREADS=1 python scripts/73_bulk_dos_B.py
"""
from __future__ import annotations
import csv, glob, os, re, subprocess
from pathlib import Path
import json
import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent.parent
HB = ROOT / "outputs/qe_hullB"
OUT = ROOT / "outputs/qe_bulk_dos"; OUT.mkdir(parents=True, exist_ok=True)
# CPU build for everything: the GPU pw.x double-loads libnvomp + system libgomp
# (via libfftw3_omp) and crashes with "libgomp: TODO". The CPU build is the same
# QE 7.3.1 and is what produced the REE DOS/PDOS (scripts/52,53).
CPU_BIN = "/home/jonglee69/software/qe_build/q-e-qe-7.3.1/bin"
PW = f"{CPU_BIN}/pw.x"
MPI = ["mpirun", "--use-hwthread-cpus", "-np", "8"]
SSSP = json.load(open("/home/jonglee69/software/qe_pseudo/sssp_efficiency.json"))
PDIR = "/home/jonglee69/software/qe_pseudo/active"
ECUTWFC, ECUTRHO = 60.0, 480.0
KPTS = (3, 3, 3)
Ry = 13.605693

CANDS = ["hec3_MoTiZr", "hec3_CrTiV", "hec3_CrTiZr"]


def pp(symbols):
    return {e: SSSP[e]["filename"] for e in sorted(set(symbols))}


def run(cmd, cwd, logname, env_extra=None):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if env_extra:
        env.update(env_extra)
    with open(cwd / logname, "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)


def done(p):
    return p.exists() and "JOB DONE" in p.read_text(errors="ignore")


def efermi(scf_out):
    t = scf_out.read_text(errors="ignore")
    m = re.findall(r"the Fermi energy is\s+(-?\d+\.\d+)", t)
    return float(m[-1]) if m else None


def scf(cand):
    d = OUT / cand; d.mkdir(exist_ok=True)
    if done(d / "scf.out"):
        print(f"  [{cand}] scf already done"); return d
    atoms = read(str(HB / cand / "vc-relax.out"), index=-1)
    idata = {
        "control": {"calculation": "scf", "prefix": cand, "outdir": "./tmp",
                    "pseudo_dir": PDIR, "tprnfor": False, "tstress": False,
                    "disk_io": "low", "verbosity": "high"},
        "system": {"ibrav": 0, "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        "electrons": {"conv_thr": 1e-6, "mixing_beta": 0.3, "electron_maxstep": 200},
    }
    write(str(d / "scf.in"), atoms, format="espresso-in", input_data=idata,
          pseudopotentials=pp(atoms.get_chemical_symbols()), kpts=KPTS)
    print(f"  [{cand}] running scf (CPU x8, k={KPTS}) ...", flush=True)
    run(MPI + [PW, "-in", "scf.in"], d, "scf.out")
    print(f"  [{cand}] scf {'OK' if done(d/'scf.out') else 'FAILED'}", flush=True)
    return d


def dos_proj(cand, d):
    ef = efermi(d / "scf.out")
    if ef is None:
        print(f"  [{cand}] no Ef -> skip dos"); return
    # total DOS
    if not (d / f"{cand}.dos").exists():
        (d / "dos.in").write_text(
            "&DOS\n"
            f"  prefix='{cand}', outdir='./tmp', fildos='{cand}.dos',\n"
            f"  Emin={ef-14:.3f}, Emax={ef+10:.3f}, DeltaE=0.02, degauss=0.01\n/\n")
        run(MPI + [f"{CPU_BIN}/dos.x", "-in", "dos.in"], d, "dos.log")
    # projected DOS
    if not glob.glob(str(d / f"{cand}.pdos_atm#*")):
        (d / "projwfc.in").write_text(
            "&PROJWFC\n"
            f"  prefix='{cand}', outdir='./tmp', filpdos='{cand}',\n"
            f"  Emin={ef-14:.3f}, Emax={ef+10:.3f}, DeltaE=0.02, ngauss=0, degauss=0.01\n/\n")
        run(MPI + [f"{CPU_BIN}/projwfc.x", "-in", "projwfc.in"], d, "projwfc.log")
    print(f"  [{cand}] dos+projwfc done (Ef={ef:.3f} eV)", flush=True)


def parse(cand, d):
    ef = efermi(d / "scf.out")
    # total DOS(E_F): dos file columns E(eV) dos int_dos
    dosf = d / f"{cand}.dos"
    dos_ef = None
    if dosf.exists():
        a = np.loadtxt(dosf, comments="#")
        dos_ef = float(np.interp(ef, a[:, 0], a[:, 1]))
    # element PDOS at E_F: sum ldos (col 1 after E) over atoms of each element
    per_el = {}
    for fp in glob.glob(str(d / f"{cand}.pdos_atm#*")):
        m = re.search(r"pdos_atm#\d+\(([A-Za-z]+)\)", os.path.basename(fp))
        if not m:
            continue
        el = m.group(1)
        arr = np.loadtxt(fp, comments="#")
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        ldos = float(np.interp(ef, arr[:, 0], arr[:, 1]))
        per_el[el] = per_el.get(el, 0.0) + ldos
    tot = sum(per_el.values()) or 1.0
    row = {"candidate": cand, "E_fermi": round(ef, 4),
           "DOS_Ef_total": round(dos_ef, 4) if dos_ef is not None else "",
           "DOS_Ef_proj_sum": round(tot, 4)}
    for el, v in per_el.items():
        row[f"pct_{el}"] = round(100 * v / tot, 2)
    return row


def main():
    rows = []
    for c in CANDS:
        d = scf(c)
        if done(d / "scf.out"):
            dos_proj(c, d)
            rows.append(parse(c, d))
    if rows:
        cols = ["candidate", "E_fermi", "DOS_Ef_total", "DOS_Ef_proj_sum"]
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(OUT / "dos_pdos_Ef_B.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print("wrote", OUT / "dos_pdos_Ef_B.csv")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
