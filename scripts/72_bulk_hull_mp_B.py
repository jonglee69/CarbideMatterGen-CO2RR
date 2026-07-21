#!/usr/bin/env python
"""Campaign-B convex-hull stability: e_above_hull for the 3 Ti-based winners,
using their QE E_f (from outputs/qe_hullB) against the Materials Project hull
of competing carbides. Same recipe as script 60 but for the SSSP Campaign-B runs.

    MP_API_KEY=... PYTHONPATH=$PWD python scripts/72_bulk_hull_mp_B.py
"""
import os, re, glob, csv, itertools
from pathlib import Path
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mp_api.client import MPRester

Ry = 13.605693
ROOT = Path(__file__).resolve().parent.parent
HB = ROOT / "outputs/qe_hullB"

CANDS = {
    "hec3-MoTiZr": {"Mo": 11, "Ti": 11, "Zr": 10, "C": 32},
    "hec3-CrTiV":  {"Cr": 11, "Ti": 11, "V": 10,  "C": 32},
    "hec3-CrTiZr": {"Cr": 11, "Ti": 11, "Zr": 10, "C": 32},
}


def final_E(p):
    t = open(p, errors="ignore").read()
    m = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", t, re.M)
    return float(m[-1]) * Ry if m else None


def natoms(p):
    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", open(p, errors="ignore").read())
    return int(m.group(1)) if m else None


def load_elem():
    e = {}
    for p in glob.glob(str(HB / "elem_*/vc-relax.out")):
        el = p.split("elem_")[1].split("/")[0]
        if "JOB DONE" in open(p, errors="ignore").read():
            e[el] = final_E(p) / natoms(p)
    return e


def qe_Ef(comp_counts, eref):
    E = final_E(str(HB / f"{name_to_dir(comp_counts)}/vc-relax.out"))
    N = sum(comp_counts.values())
    return (E - sum(n * eref[e] for e, n in comp_counts.items())) / N


def name_to_dir(_):  # replaced per-candidate below
    raise RuntimeError


def main():
    key = os.environ["MP_API_KEY"]
    eref = load_elem()
    print("elemental refs (eV/atom):", {k: round(v, 3) for k, v in eref.items()})
    rows = []
    with MPRester(key, monty_decode=False) as mpr:
        for name, counts in CANDS.items():
            dirn = "hec3_" + name.split("-")[1]
            E = final_E(str(HB / dirn / "vc-relax.out"))
            N = sum(counts.values())
            Ef = (E - sum(n * eref[e] for e, n in counts.items())) / N
            comp = Composition("".join(f"{e}{n}" for e, n in counts.items()))
            els = sorted(counts)
            subsys = ["-".join(sorted(c)) for k in range(1, len(els) + 1)
                      for c in itertools.combinations(els, k)]
            docs = mpr.materials.thermo.search(
                chemsys=subsys, thermo_types=["GGA_GGA+U"],
                fields=["formula_pretty", "formation_energy_per_atom"])
            pdentries = [PDEntry(Composition(e), 0.0) for e in els]
            for d in docs:
                ef = d.get("formation_energy_per_atom")
                if ef is None:
                    continue
                c = Composition(d["formula_pretty"])
                pdentries.append(PDEntry(c, ef * c.num_atoms))
            pd = PhaseDiagram(pdentries)
            hull_form = pd.get_hull_energy_per_atom(comp)
            e_above = Ef - hull_form
            decomp = pd.get_decomposition(comp)
            products = ", ".join(e.composition.reduced_formula for e in decomp)
            rows.append({"candidate": name, "Ef_QE": round(Ef, 4),
                         "hull_MP": round(hull_form, 4),
                         "e_above_hull": round(e_above, 4),
                         "n_MP": len(pdentries) - len(els),
                         "decomposition": products})
            print(f"{name:14s} Ef(QE)={Ef:+.3f}  hull(MP)={hull_form:+.3f}  "
                  f"E_hull={e_above:+.3f} eV/atom  -> {products}")
    out = HB / "hull_stability_B.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
