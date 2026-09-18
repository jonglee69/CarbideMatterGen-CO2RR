#!/usr/bin/env python
"""Bulk Pourbaix (Materials Project) for the aqueous fate of each constituent
metal under CO2RR conditions -- the REE oxidation/dissolution risk test.

For each element we pull MP Pourbaix entries and read the thermodynamically stable
phase at (pH, E vs SHE): metal (protected), a solid oxide/hydroxide (passivating),
or a dissolved ion (leaching). We evaluate at open-circuit (E=0 V SHE) and across
the CO2RR window. CO2RR is usually run near-neutral; U_RHE -> U_SHE = U_RHE-0.059*pH.
CO2RR onset ~ -0.5..-1.1 V RHE at pH 7 -> ~ -0.91..-1.51 V SHE.

    MP_API_KEY=... <pinned-venv>/bin/python scripts/59_mp_pourbaix.py

STATUS: the MP Pourbaix ion-reference path (get_pourbaix_entries) is BLOCKED by an
mp-api/emmet/pydantic/monty version incompatibility -- emmet's ThermoDoc cannot
deserialise ComputedEntry ('@module' kwarg). A pinned pydantic-v1 stack
(pymatgen==2023.9.25, pydantic==1.10.13, emmet-core==0.55.4, mp-api==0.33.3,
numpy<2, +mpcontribs-client, +module-path shim above) gets furthest but still
fails ThermoDoc validation. The bulk-dissolution CONCLUSION is instead supported
qualitatively (thesis sec:ree-stability) by the +0.15..0.36 eV/atom hull
metastability (script 60) plus the very negative REE M3+/M standard potentials
(~-2.3 V). Full numeric diagram = TODO in a matched historical environment.
"""
from __future__ import annotations
import os, csv, sys
from pathlib import Path
# Shim: MP-serialized entries/ion data reference pre-2022 module paths. Run this
# script in the pinned pydantic-v1 venv (scratchpad/pbx_v1) -- see HANDOFF.
import pymatgen.entries, pymatgen.entries.computed_entries as _ce
sys.modules.setdefault("pymatgen.core.entries", pymatgen.entries)
sys.modules.setdefault("pymatgen.core.entries.computed_entries", _ce)
try:
    import pymatgen.entries.compatibility as _compat
    sys.modules.setdefault("pymatgen.analysis.compatibility", _compat)
except Exception:
    pass
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram
from mp_api.client import MPRester

ROOT = Path(__file__).resolve().parent.parent
ELEMENTS = ["La", "Ce", "Pr", "Y", "Zr", "W"]
PH = 7.0
# (label, U vs SHE) grid: OCP and the CO2RR window at pH 7
CONDS = [("OCP (0 V)", 0.0), ("-0.5 V_RHE", -0.5 - 0.059 * PH),
         ("-0.8 V_RHE", -0.8 - 0.059 * PH), ("-1.1 V_RHE", -1.1 - 0.059 * PH)]


def classify(name: str) -> str:
    n = name.replace(" ", "")
    if "(aq)" in n or "+" in n or "-" in n.rstrip(")"):
        return "DISSOLVED (ion)"
    if "O" in n or "H" in n:
        return "solid oxide/hydroxide"
    return "metal"


def main():
    key = os.environ["MP_API_KEY"]
    rows = []
    with MPRester(key) as mpr:
        for el in ELEMENTS:
            try:
                entries = mpr.get_pourbaix_entries([el])
            except Exception as e:
                print(f"[warn] {el}: {repr(e)[:120]}"); continue
            pbx = PourbaixDiagram(entries)
            print(f"\n=== {el} (pH {PH}) ===")
            for label, U in CONDS:
                stable = pbx.get_stable_entry(PH, U)
                cat = classify(stable.name)
                print(f"  {label:12s} (E={U:+.2f} V SHE): {stable.name:24s} -> {cat}")
                rows.append({"element": el, "condition": label, "E_SHE": round(U, 3),
                             "pH": PH, "stable_phase": stable.name, "category": cat})
    out = ROOT / "outputs/qe/mp_pourbaix_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
