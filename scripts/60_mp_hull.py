#!/usr/bin/env python
"""Formation energy (QE) + convex-hull stability (Materials Project) -- thesis item 2.

E_f(candidate) is computed from the QE total energy (qe_summary.csv) and the QE
elemental references (elemental_refs.csv, script 58), so the candidate sits on a
single consistent QE-PBE footing:
    E_f/atom = [E(cand) - sum_i n_i E_i/atom] / N_atoms .
The competing convex hull is built from MP GGA entries for the same chemical space,
and the decomposition (hull) formation energy at the candidate composition is read
off. e_above_hull ~ E_f(QE) - E_hull(MP). Because formation energies are element-
referenced, QE-vs-MP(VASP) differences largely cancel (residual ~0.1 eV/atom; noted).

    MP_API_KEY=... PYTHONPATH=$PWD /home/jonglee69/.venv_fairchem/bin/python scripts/60_mp_hull.py
"""
from __future__ import annotations
import os, csv, itertools
from pathlib import Path
from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mp_api.client import MPRester

RY = 13.605693122994
ROOT = Path(__file__).resolve().parent.parent
QE = ROOT / "outputs/qe/qe_summary.csv"
ELEM = ROOT / "outputs/qe/elemental_refs.csv"


def load_elem():
    return {r["element"]: float(r["E_per_atom_eV"]) for r in csv.DictReader(open(ELEM))}


def load_candidates():
    out = []
    for r in csv.DictReader(open(QE)):
        comp = Composition(r["formula"])
        out.append({"formula": r["formula"], "comp": comp,
                    "E_eV": float(r["E_Ry"]) * RY, "natoms": int(r["natoms"])})
    return out


def qe_formation_per_atom(cand, eref):
    e_elems = sum(cand["comp"][el] * eref[el.symbol] for el in cand["comp"].elements)
    return (cand["E_eV"] - e_elems) / cand["natoms"]


def main():
    key = os.environ["MP_API_KEY"]
    eref = load_elem()
    cands = load_candidates()
    rows = []
    # monty_decode=False -> plain dicts (avoids mp-api/pymatgen class-path mismatch).
    with MPRester(key, monty_decode=False) as mpr:
        for cand in cands:
            comp = cand["comp"]
            els = sorted(e.symbol for e in comp.elements)
            chemsys = "-".join(els)
            # all sub-chemsystems (binaries, ternaries, ...) so the hull includes
            # competing carbides (WC, ZrC, ...), not just the full 5-element system.
            subsys = ["-".join(sorted(c)) for k in range(1, len(els) + 1)
                      for c in itertools.combinations(els, k)]
            docs = mpr.materials.thermo.search(
                chemsys=subsys, thermo_types=["GGA_GGA+U"],
                fields=["formula_pretty", "formation_energy_per_atom"])
            # build a formation-energy phase diagram: MP compounds at their E_f,
            # elements pinned at 0 (formation-energy convention).
            pdentries = [PDEntry(Composition(e.symbol), 0.0) for e in comp.elements]
            for d in docs:
                ef = d.get("formation_energy_per_atom")
                if ef is None:
                    continue
                c = Composition(d["formula_pretty"])
                pdentries.append(PDEntry(c, ef * c.num_atoms))
            pd = PhaseDiagram(pdentries)
            hull_form = pd.get_hull_energy_per_atom(comp)  # already formation-E scale
            Ef = qe_formation_per_atom(cand, eref)
            e_above = Ef - hull_form
            # what MP says it decomposes into (competing phases at this composition)
            decomp = pd.get_decomposition(comp)
            products = ", ".join(f"{e.composition.reduced_formula}" for e in decomp)
            rows.append({"formula": cand["formula"], "chemsys": chemsys,
                         "Ef_QE_eV_atom": round(Ef, 4),
                         "hull_Ef_MP_eV_atom": round(hull_form, 4),
                         "e_above_hull_eV_atom": round(e_above, 4),
                         "n_MP_entries": len(pdentries) - len(comp.elements),
                         "MP_decomposition": products})
            print(f"{cand['formula']:14s} Ef(QE)={Ef:+.3f}  hull(MP)={hull_form:+.3f}  "
                  f"E_above_hull={e_above:+.3f} eV/atom  -> {products}")
    out = ROOT / "outputs/qe/hull_stability.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
