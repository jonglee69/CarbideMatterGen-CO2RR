#!/usr/bin/env python
"""Post-processing (no new DFT): CHE free-energy diagram + limiting potential U_L
(thesis \\tbc items 1 & 3) and transition-metal d-band centers (item 4).

Reads:
  * outputs/qe_surface/dft/surface_dft_dG.csv   (54 rows: formula,facet,adsorbate,
    dG_dft_eV,dG_uma_eV,...) -- the completed Phase-A surface DFT.
  * outputs/qe/<formula>/<formula>.pdos_atm#*(El)_wfc#*(d)  -- projwfc d-projected
    PDOS already on disk (53_qe_pdos.py), plus Fermi levels from
    outputs/qe/qe_pdos_Ef_contributions.csv.

CHE convention (identical to carbidemattergen/co_surface_screen.py:42-43):
  CO2 + * + (H+ +e-) -> *COOH        dG1 = dG_COOH
  *COOH + (H+ +e-)   -> *CO + H2O    dG2 = dG_CO - dG_COOH
  *CO                -> CO(g) + *     (chemical desorption)
  U_L = -max(dG1, dG2) / e     (two-electron limiting potential vs RHE)
CO2RR-vs-HER selectivity at a facet: dG_COOH < dG_H (CO2 activation beats H*).

d-band center: eps_d = \\int E rho_d dE / \\int rho_d dE, E referenced to E_F,
summed over the transition-metal d projections (W, Zr, Y, and REE 5d where present).

Outputs:
  outputs/qe_surface/dft/che_summary.csv         per-facet + per-candidate CHE/U_L
  outputs/qe/dband_centers.csv                   per-candidate/per-element eps_d
  figures/fig_co2rr_che_diagram.pdf              free-energy diagram (best facet)
  figures/fig_co2rr_dband_vs_dGCO.pdf            eps_d vs dG_CO (d-band model)

    /home/jonglee69/.venv_fairchem/bin/python scripts/57_co2rr_che_dband.py
"""
from __future__ import annotations
import csv, glob, os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DFTCSV = ROOT / "outputs/qe_surface/dft/surface_dft_dG.csv"
QEDIR = ROOT / "outputs/qe"
EFCSV = QEDIR / "qe_pdos_Ef_contributions.csv"
FIGDIR = ROOT / "figures"; FIGDIR.mkdir(exist_ok=True)

# Transition metals whose d-band we integrate (REE 4f treated as core; their 5d is
# included since it hybridises near E_F). Non-metals excluded.
TM_D = {"W", "Zr", "Y", "Ce", "La", "Pr", "Nd", "Re", "Mo", "Cr", "Ir", "Ru"}


# ------------------------------------------------------------------ CHE / U_L
def load_dG():
    """{(formula,facet): {'CO':..,'COOH':..,'H':..}} for DFT and UMA."""
    dft, uma = {}, {}
    for r in csv.DictReader(open(DFTCSV)):
        key = (r["formula"], r["facet"])
        dft.setdefault(key, {})[r["adsorbate"]] = float(r["dG_dft_eV"])
        uma.setdefault(key, {})[r["adsorbate"]] = float(r["dG_uma_eV"])
    return dft, uma


def che_row(g):
    """CHE descriptors from one facet's {CO,COOH,H} dict."""
    dG_COOH, dG_CO, dG_H = g["COOH"], g["CO"], g["H"]
    dG1 = dG_COOH
    dG2 = dG_CO - dG_COOH
    U_L = -max(dG1, dG2)                 # V vs RHE
    pds = "COOH" if dG1 >= dG2 else "CO"  # potential-determining step
    co_selective = dG_COOH < dG_H         # CO2RR beats HER at this facet
    return dict(dG_COOH=dG_COOH, dG_CO=dG_CO, dG_H=dG_H,
                dG1=dG1, dG2=dG2, U_L=U_L, pds=pds, co_selective=co_selective)


def analyse_che():
    dft, uma = load_dG()
    per_facet, best = [], {}
    for (formula, facet), g in sorted(dft.items()):
        row = {"formula": formula, "facet": facet, **che_row(g)}
        row["U_L_uma"] = che_row(uma[(formula, facet)])["U_L"]
        per_facet.append(row)
        # best facet per candidate: highest (least negative) U_L among CO-selective;
        # fall back to overall highest U_L if none selective.
        cur = best.get(formula)
        better = (cur is None
                  or (row["co_selective"], row["U_L"]) > (cur["co_selective"], cur["U_L"]))
        if better:
            best[formula] = row

    # write summary
    out = ROOT / "outputs/qe_surface/dft/che_summary.csv"
    cols = ["formula", "facet", "dG_COOH", "dG_CO", "dG_H", "dG1", "dG2",
            "U_L", "U_L_uma", "pds", "co_selective"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in per_facet:
            w.writerow({**r, "dG_COOH": round(r["dG_COOH"], 3),
                        "dG_CO": round(r["dG_CO"], 3), "dG_H": round(r["dG_H"], 3),
                        "dG1": round(r["dG1"], 3), "dG2": round(r["dG2"], 3),
                        "U_L": round(r["U_L"], 3), "U_L_uma": round(r["U_L_uma"], 3)})
    print(f"wrote {out}")
    return per_facet, best


def plot_che(best):
    """CHE free-energy diagram, best facet per candidate, at U=0."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    # path states: CO2+* (0) -> *COOH (dG1) -> *CO (dG_CO) -> CO(g)+* (dG_CO - dG_CO = ~0)
    xs = [0, 1, 2, 3]
    labels = [r"CO$_2$+*", r"*COOH", r"*CO", r"CO(g)+*"]
    order = sorted(best, key=lambda f: -best[f]["U_L"])  # best performer first
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(order)))
    for c, formula in zip(cmap, order):
        r = best[formula]
        # G at each state (eV), U=0, referenced to CO2+*+2(H++e-)=0
        G = [0.0, r["dG_COOH"], r["dG_CO"], 0.0]
        for i in range(4):
            ax.hlines(G[i], xs[i]-0.30, xs[i]+0.30, color=c, lw=2.4)
        for i in range(3):
            ax.plot([xs[i]+0.30, xs[i+1]-0.30], [G[i], G[i+1]], color=c, lw=1.0, ls=":")
        ax.plot([], [], color=c, lw=2.4,
                label=f"{formula} (f{r['facet']}, $U_L$={r['U_L']:+.2f} V)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta G$ at $U=0$ vs RHE (eV)")
    ax.set_title("CO$_2$RR-to-CO free-energy diagram (DFT, best facet per candidate)")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    # legend inside, lower-right (empty region: curves converge near G=0 at right)
    ax.legend(fontsize=7.4, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    p = FIGDIR / "fig_co2rr_che_diagram.pdf"; fig.savefig(p); plt.close(fig)
    print(f"wrote {p}")


# ------------------------------------------------------------------ d-band centers
def fermi_levels():
    return {r["formula"]: float(r["E_fermi"]) for r in csv.DictReader(open(EFCSV))}


def dband_center(formula, ef):
    """Return {element: eps_d, '_metal': overall} referenced to E_F (eV)."""
    cdir = QEDIR / formula
    files = glob.glob(f"{cdir}/{formula}.pdos_atm#*_wfc#*(d)")
    if not files:
        # brace/paren globbing can miss the '(d)'; match manually
        files = [str(p) for p in cdir.glob(f"{formula}.pdos_atm#*") if p.name.endswith("(d)")]
    per_el, Eref = {}, None
    for fp in files:
        m = re.search(r"pdos_atm#\d+\(([A-Za-z]+)\)", os.path.basename(fp))
        el = m.group(1) if m else None
        if el not in TM_D:
            continue
        d = np.loadtxt(fp, comments="#")
        if d.ndim != 2 or d.shape[0] < 3:
            continue
        E, rho = d[:, 0] - ef, d[:, 1]      # ldos column, referenced to E_F
        Eref = E
        per_el.setdefault(el, np.zeros_like(rho))
        per_el[el] = per_el[el] + rho
    if not per_el:
        return None
    def center(rho):
        num = np.trapz(Eref * rho, Eref); den = np.trapz(rho, Eref)
        return num / den if den else float("nan")
    res = {el: round(center(rho), 3) for el, rho in per_el.items()}
    total = np.sum(list(per_el.values()), axis=0)
    res["_metal"] = round(center(total), 3)
    return res


def analyse_dband(best):
    efs = fermi_levels()
    rows = []
    for formula, ef in efs.items():
        eps = dband_center(formula, ef)
        if eps is None:
            print(f"  [warn] no d-PDOS for {formula}"); continue
        rows.append({"formula": formula, "eps_d_metal": eps["_metal"],
                     **{f"eps_d_{el}": v for el, v in eps.items() if el != "_metal"},
                     "best_dG_CO": round(best[formula]["dG_CO"], 3)})
        print(f"  {formula:14s} eps_d(metal)={eps['_metal']:+.2f} eV  "
              f"per-el={ {k:v for k,v in eps.items() if k!='_metal'} }")
    # union of columns
    allcols = ["formula", "eps_d_metal", "best_dG_CO"]
    for r in rows:
        for k in r:
            if k not in allcols:
                allcols.append(k)
    out = QEDIR / "dband_centers.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}")

    # eps_d(metal) vs dG_CO  (d-band model: higher center -> stronger binding)
    x = [r["eps_d_metal"] for r in rows]; y = [r["best_dG_CO"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(x, y, c="crimson", s=60, zorder=3)
    for r in rows:
        ax.annotate(r["formula"], (r["eps_d_metal"], r["best_dG_CO"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xr = np.linspace(min(x), max(x), 50)
        ax.plot(xr, a + b*xr, "grey", ls="--", lw=1,
                label=f"slope={b:.2f} eV/eV, r={np.corrcoef(x,y)[0,1]:.2f}")
        ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(r"metal $d$-band center $\varepsilon_d - E_F$ (eV)")
    ax.set_ylabel(r"best-facet $\Delta G_{\rm CO}$ (eV)")
    ax.set_title("d-band model: CO binding vs $d$-band center")
    fig.tight_layout()
    p = FIGDIR / "fig_co2rr_dband_vs_dGCO.pdf"; fig.savefig(p); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    print("== CHE / limiting potential ==")
    per_facet, best = analyse_che()
    print("\nBest facet per candidate (DFT):")
    for f in sorted(best, key=lambda k: -best[k]["U_L"]):
        r = best[f]
        print(f"  {f:14s} f{r['facet']:5s} U_L={r['U_L']:+.2f} V  PDS={r['pds']:4s}  "
              f"dG_COOH={r['dG_COOH']:+.2f} dG_CO={r['dG_CO']:+.2f} "
              f"CO-selective={r['co_selective']}")
    plot_che(best)
    print("\n== d-band centers ==")
    analyse_dband(best)
