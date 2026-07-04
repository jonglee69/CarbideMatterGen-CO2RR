#!/usr/bin/env python
"""Surface Pourbaix / active-site poisoning verdict (thesis REE-stability item).

Combines the UMA *OH/*O screen (oh_o_screen.jsonl, script co_surface_screen with
--adsorbates OH,O) with the UMA *CO/*H descriptors (surface_dft_dG.csv, dG_uma
column) on a single MLIP footing, and applies the CHE potential dependence to ask:
at CO2RR operating potential, is the strongest-binding site clean/*CO/*H, or is it
covered by *OH/*O (oxidation / poisoning)?

CHE potential dependence (RHE; n electrons released on adsorbate formation):
  *OH  (*+H2O -> *OH + (H+ +e-)):        dG_OH(U) = dG_OH(0) - 1 eU
  *O   (*+H2O -> *O + 2(H+ +e-)):        dG_O(U)  = dG_O(0)  - 2 eU
  *H   (* + (H+ +e-) -> *H):             dG_H(U)  = dG_H(0)  + 1 eU
  *CO  (chemical, from CO(g)):           dG_CO(U) = dG_CO(0)   (U-independent)
The most negative dG at a given U is the thermodynamically preferred coverage.
Strong, U-robust *OH/*O binding => predicted oxide/hydroxide poisoning.

    /home/jonglee69/.venv_fairchem/bin/python scripts/61_surface_pourbaix.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OHO = ROOT / "outputs/qe_surface/oh_o_screen.jsonl"
DFT = ROOT / "outputs/qe_surface/dft/surface_dft_dG.csv"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
US = [0.0, -0.3, -0.6, -1.0]  # V vs RHE (CO2RR window)


def load_oho():
    """min (strongest) dG_OH, dG_O per candidate (UMA, U=0)."""
    out = {}
    for line in open(OHO):
        d = json.loads(line)
        if "error" in d:
            continue
        name = d["source"].replace(".cif", "")
        oh = [v for f in d.get("facets", []) for v in f.get("dG_OH_values", [])]
        o = [v for f in d.get("facets", []) for v in f.get("dG_O_values", [])]
        out[name] = {"OH": min(oh) if oh else None, "O": min(o) if o else None}
    return out


def load_co_h():
    """UMA dG_CO (closest to 0) and min dG_H per candidate, from the DFT csv's UMA col."""
    co, h = {}, {}
    for r in csv.DictReader(open(DFT)):
        f = r["formula"]; g = float(r["dG_uma_eV"]); a = r["adsorbate"]
        if a == "CO":
            co.setdefault(f, []).append(g)
        elif a == "H":
            h.setdefault(f, []).append(g)
    co_best = {f: min(v, key=abs) for f, v in co.items()}
    h_best = {f: min(v) for f, v in h.items()}
    return co_best, h_best


def dG_at(species, g0, U):
    if species == "OH":
        return g0 - 1 * U
    if species == "O":
        return g0 - 2 * U
    if species == "H":
        return g0 + 1 * U
    return g0  # CO chemical


def main():
    oho = load_oho()
    co_best, h_best = load_co_h()
    rows = []
    for name in sorted(oho):
        g = {"OH": oho[name]["OH"], "O": oho[name]["O"],
             "CO": co_best.get(name), "H": h_best.get(name)}
        for U in US:
            vals = {s: dG_at(s, g[s], U) for s in g if g[s] is not None}
            winner = min(vals, key=vals.get)
            poisoned = winner in ("OH", "O")
            rows.append({"candidate": name, "U_RHE": U,
                         **{f"dG_{s}": round(vals[s], 3) for s in vals},
                         "dominant": winner, "poisoned": poisoned})
    # table
    out = ROOT / "outputs/qe_surface/surface_pourbaix.csv"
    cols = ["candidate", "U_RHE", "dG_OH", "dG_O", "dG_CO", "dG_H", "dominant", "poisoned"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}\n")
    print(f"{'candidate':14s} " + "  ".join(f"U={u:+.1f}" for u in US))
    for name in sorted(oho):
        cells = []
        for U in US:
            r = next(x for x in rows if x["candidate"] == name and x["U_RHE"] == U)
            cells.append(f"{r['dominant']:>4s}{'*' if r['poisoned'] else ' '}")
        print(f"{name:14s} " + "   ".join(cells))
    print("\n(* = *OH/*O dominant => predicted oxidation/poisoning; U in V vs RHE)")

    # figure: dG vs U for the strongest species, all candidates (OH & O only, +CO ref line)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    Ux = np.linspace(-1.1, 0.1, 50)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(oho)))
    for c, name in zip(cmap, sorted(oho)):
        oh0 = oho[name]["OH"]
        ax.plot(Ux, oh0 - Ux, color=c, lw=1.8, label=f"{name}")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axhspan(-9, 0, color="red", alpha=0.04)
    ax.text(-1.05, -0.6, "$*$OH favourable (poisoning)", color="darkred", fontsize=8)
    ax.axvspan(-1.0, -0.3, color="grey", alpha=0.10)
    ax.text(-0.85, ax.get_ylim()[1]*0.9 if False else -8.0, "CO$_2$RR window", fontsize=8, color="k")
    ax.set_xlabel("potential $U$ (V vs RHE)")
    ax.set_ylabel(r"strongest $\Delta G_{*\rm OH}(U)$ (eV)")
    ax.set_title("Surface Pourbaix: $*$OH binding vs potential (UMA, strongest site)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = FIG / "fig_co2rr_surface_pourbaix.pdf"; fig.savefig(p); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
