#!/usr/bin/env python
"""Surface Pourbaix / resting-state coverage on a SINGLE consistent facet.

Fixes two issues in the first version of this analysis:
  (1) it omitted the CO2RR intermediate *COOH from the competing surface species,
      which for most candidates is actually the most stable adsorbate at operating
      potential -- so "all O/OH poisoned" was an artefact of leaving it out;
  (2) it mixed facets (O/OH from the strongest-*OH facet, CO/H from a different
      facet). Here every adsorbate is taken on the SAME facet -- the *OH facet from
      the QE O/OH run (dft_oh/), with CO/COOH/H read from the main QE run
      (dft/surface_dft_dG.csv) for that same facet. Candidates whose *OH facet is
      absent from the main run are reported as facet-inconsistent (need O/OH on the
      activity facet to complete rigorously).

CHE potential dependence (RHE; consume (H+ +e-) -> +U, release -> -U):
  *H   : dG_H(U)    = dG_H(0)    + U        (consume 1)
  *COOH: dG_COOH(U) = dG_COOH(0) + U        (consume 1)   <-- was omitted before
  *OH  : dG_OH(U)   = dG_OH(0)   - U        (release 1)
  *O   : dG_O(U)    = dG_O(0)    - 2U       (release 2)
  *CO  : dG_CO(U)   = dG_CO(0)              (chemical)
Resting state = min dG. COOH/CO => CO2RR working state; OH/O => oxidation/poisoning;
H => HER-competing coverage.

    /home/jonglee69/.venv_fairchem/bin/python scripts/61_surface_pourbaix.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DFT = ROOT / "outputs/qe_surface/dft/surface_dft_dG.csv"
DFT_OHO = ROOT / "outputs/qe_surface/dft_oh/surface_dft_dG.csv"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
US = [0.0, -0.3, -0.6, -1.0]

KIND = {"COOH": "CO2RR", "CO": "CO2RR", "OH": "oxidation", "O": "oxidation", "H": "HER"}


def dG(sp, g0, U):
    return {"H": g0 + U, "COOH": g0 + U, "OH": g0 - U, "O": g0 - 2 * U, "CO": g0}[sp]


def load():
    # O/OH DFT with facet
    oho = {}
    for r in csv.DictReader(open(DFT_OHO)):
        oho.setdefault(r["formula"], {})[r["adsorbate"]] = (float(r["dG_dft_eV"]), r["facet"])
    # main DFT per (formula, facet): CO/COOH/H
    main = {}
    for r in csv.DictReader(open(DFT)):
        main.setdefault((r["formula"], r["facet"]), {})[r["adsorbate"]] = float(r["dG_dft_eV"])
    return oho, main


def main_run():
    oho, main = load()
    rows, consistent = [], []
    for f in sorted(oho):
        ohv, ohf = oho[f]["OH"]; ov, _ = oho[f]["O"]
        same = main.get((f, ohf))
        if not same:
            print(f"[facet-inconsistent] {f}: *OH facet {ohf} absent from main DFT "
                  f"(need O/OH on an activity facet) -- skipped")
            continue
        consistent.append(f)
        g = {"OH": ohv, "O": ov, "CO": same["CO"], "H": same["H"], "COOH": same["COOH"]}
        for U in US:
            vals = {s: dG(s, g[s], U) for s in g}
            w = min(vals, key=vals.get)
            rows.append({"candidate": f, "facet": ohf, "U_RHE": U,
                         **{f"dG_{s}": round(vals[s], 3) for s in g},
                         "resting": w, "class": KIND[w]})
    out = ROOT / "outputs/qe_surface/surface_pourbaix_samefacet.csv"
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}\n")
    print(f"{'candidate':14s} facet   " + "  ".join(f"U={u:+.1f}" for u in US))
    for f in consistent:
        cells = []
        for U in US:
            r = next(x for x in rows if x["candidate"] == f and x["U_RHE"] == U)
            cells.append(f"{r['resting']:>4s}")
        fac = next(x["facet"] for x in rows if x["candidate"] == f)
        print(f"{f:14s} {fac:6s} " + "   ".join(cells))
    print("\nresting state: COOH/CO=CO2RR working  OH/O=oxidation  H=HER coverage")

    # figure: resting-state free energies vs U for the consistent candidates
    if consistent:
        n = len(consistent)
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
        if n == 1:
            axes = [axes]
        Ux = np.linspace(-1.05, 0.05, 40)
        colors = {"COOH": "#1b7837", "CO": "#5aae61", "OH": "#b2182b",
                  "O": "#d6604d", "H": "#4393c3"}
        for ax, f in zip(axes, consistent):
            ohv, ohf = oho[f]["OH"]; ov, _ = oho[f]["O"]; same = main[(f, ohf)]
            g = {"OH": ohv, "O": ov, "CO": same["CO"], "H": same["H"], "COOH": same["COOH"]}
            for s in g:
                ax.plot(Ux, [dG(s, g[s], U) for U in Ux], color=colors[s], lw=1.8, label=f"*{s}")
            ax.axvspan(-1.0, -0.3, color="grey", alpha=0.10)
            ax.set_title(f"{f}\n(facet {ohf})", fontsize=8)
            ax.set_xlabel("$U$ (V vs RHE)")
        axes[0].set_ylabel(r"$\Delta G$ (eV)")
        axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.suptitle("Surface Pourbaix (same facet, all species incl. *COOH; DFT)", fontsize=9)
        fig.tight_layout()
        p = FIG / "fig_co2rr_surface_pourbaix_dft.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main_run()
