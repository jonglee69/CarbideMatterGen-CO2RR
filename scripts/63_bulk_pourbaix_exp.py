#!/usr/bin/env python
"""Bulk aqueous Pourbaix (EXPERIMENTAL thermodynamics) for the REE dissolution risk.

The MP Pourbaix DFT path (get_pourbaix_entries) is unusable with the available
mp-api/pymatgen/emmet stack (multi-layer deserialization + POTCAR-correction
version conflicts that would corrupt the energetics). Instead we build the Pourbaix
diagram from EXPERIMENTAL Gibbs formation energies -- the textbook method and the
same Wagman/NBS data MP's aqueous references are keyed to:
  * aqueous ions + reference oxide ΔGf from MP's ion-reference endpoint (works), and
  * solid hydroxides ΔGf from CRC/Wagman tables (below).
pymatgen's PourbaixDiagram then handles the Nernst pH/potential dependence.

Question: at CO2RR operating conditions (pH ~7; local cathode pH can rise to ~10;
U ~ 0..-1.1 V vs RHE) is each constituent metal IMMUNE (metal stable), PASSIVATED
(solid oxide/hydroxide), or CORRODING (dissolved ion -> leaching)?

    MP_API_KEY=... <pbx_v1>/bin/python scripts/63_bulk_pourbaix_exp.py
"""
from __future__ import annotations
import os, csv
from pathlib import Path
from pymatgen.core import Composition
from pymatgen.core.ion import Ion
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.pourbaix_diagram import PourbaixEntry, IonEntry, PourbaixDiagram
from mp_api.client import MPRester

KJ = 96.485  # kJ/mol per eV
ROOT = Path(__file__).resolve().parent.parent
ELEMENTS = ["La", "Ce", "Pr", "Y", "Zr", "W"]
# extra solid hydroxides/oxides not in the ion reference set (ΔGf°, kJ/mol; CRC/Wagman)
EXTRA_SOLIDS = {
    "La": [("La(OH)3", -1319.2)],
    "Ce": [("Ce(OH)3", -1306.0), ("Ce(OH)4", -1450.0)],
    "Pr": [("Pr(OH)3", -1285.0)],
    "Y":  [("Y(OH)3", -1291.0)],
    "Zr": [("Zr(OH)4", -1636.0)],
    "W":  [],  # WO3 (ref) + WO4^2- ion cover it
}
# operating points: (label, U vs RHE, pH)
CONDS = [("OCP pH7", 0.0, 7.0), ("-0.5V pH7", -0.5, 7.0),
         ("-0.8V pH7", -0.8, 7.0), ("-1.1V pH7", -1.1, 7.0),
         ("-0.8V pH10(local)", -0.8, 10.0)]


def classify(name: str) -> str:
    n = name.replace(" ", "")
    if "[" in n and ("+" in n or "-" in n):   # aqueous ion
        return "CORRODING (leach)"
    return "PASSIVATED (oxide/hydroxide)" if ("O" in n or "H" in n) else "IMMUNE (metal)"


def build(el, ion_data):
    entries = [PourbaixEntry(ComputedEntry(Composition(el), 0.0))]           # metal
    seen = set()
    d0 = [x for x in ion_data if x["data"]["MajElements"] == el]
    if d0:                                                                    # ref oxide
        rs, g = d0[0]["data"]["RefSolid"], d0[0]["data"]["ΔGᶠRefSolid"]["value"]
        entries.append(PourbaixEntry(ComputedEntry(Composition(rs), g / KJ))); seen.add(rs)
    for name, g in EXTRA_SOLIDS.get(el, []):
        entries.append(PourbaixEntry(ComputedEntry(Composition(name), g / KJ)))
    for x in d0:                                                             # ions
        entries.append(PourbaixEntry(IonEntry(Ion.from_formula(x["formula"]),
                                              x["data"]["ΔGᶠ"]["value"] / KJ)))
    return entries


def main():
    key = os.environ["MP_API_KEY"]
    rows = []
    with MPRester(key) as m:
        ion_all = {el: m.get_ion_reference_data_for_chemsys([el]) for el in ELEMENTS}
    print(f"{'metal':6s} " + " | ".join(l for l, _, _ in CONDS))
    for el in ELEMENTS:
        pbx = PourbaixDiagram(build(el, ion_all[el]), filter_solids=False)
        cells = []
        for label, Ur, pH in CONDS:
            Ush = Ur - 0.059 * pH
            s = pbx.get_stable_entry(pH, Ush)
            cat = classify(s.name)
            cells.append(f"{s.name}|{cat.split()[0]}")
            rows.append({"metal": el, "condition": label, "U_RHE": Ur, "pH": pH,
                         "U_SHE": round(Ush, 3), "stable_phase": s.name, "verdict": cat})
        print(f"{el:6s} " + "  ".join(cells))
    out = ROOT / "outputs/qe/bulk_pourbaix_exp.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
