#!/usr/bin/env python
"""Step — DFT selectivity: CO2RR-to-CO limiting potential vs the competing HER,
per candidate, from the surface single-point energies.

Why this exists (user question)
-------------------------------
A catalyst with ΔG_CO ≈ 0 releases CO well, but the SAME early-TM carbides
(WC, Mo2C ...) are textbook HER catalysts. If HER is facile the current goes to
H2 gas and CO2RR starves. Selectivity is therefore a TWO-objective problem:

    want  U_L(CO2RR)  as HIGH (close to 0) as possible   -> low CO2RR overpotential
    want  U_L(HER)    as LOW  (very negative) as possible -> HER suppressed
    selectivity margin  ΔU = U_L(CO2RR) − U_L(HER)  > 0   -> CO2RR wins

Free energies (computational hydrogen electrode, U vs RHE, pH-independent at the
CHE level; Peterson 2010 / Nørskov 2004). All electronic energies are DFT
single points on the UMA-relaxed slab/adslab (DFT//UMA); the same gas references
(computed once at 60/480) enter both half-reactions, so systematic error cancels
in the difference.

    *  +  CO2(g) + (H+ + e-)  -> *COOH          ΔG1 = G(*COOH) − G(*) − G(CO2) − 1/2 G(H2)  + 0.41
    *COOH  + (H+ + e-)        -> *CO + H2O(g)    ΔG2 = [G(*CO)+G(H2O)] − G(*COOH) − 1/2 G(H2) ... (H2O ref optional)
    *CO                       -> * + CO(g)       ΔG3 = G(CO_g) + G(*) − G(*CO)  (+desorption corr)
      CO2RR-to-CO limiting potential:  U_L(CO2RR) = − max(ΔG1, ΔG2) / e

    *  +  (H+ + e-)  -> *H                        ΔG_H = G(*H) − G(*) − 1/2 G(H2) + 0.24

H SELECTIVITY IS ABOUT COVERAGE, NOT H2-EVOLUTION RATE (corrected).
An earlier version scored HER by U_L(HER) = -|ΔG_H|, i.e. distance from the
ΔG_H≈0 volcano top. That is the H2-evolution *rate* argument and it is the WRONG
descriptor for CO2RR selectivity, because it treats strong binding (ΔG_H<<0) as
"good" (HER slow). But strong H binding does its damage a different way: *H sits
on the surface at high coverage and BLOCKS the sites CO2/*COOH need. So strongly
negative ΔG_H poisons CO2RR even though it slows H2 evolution.

What CO2RR actually needs is a surface that REPELS H: ΔG_H > 0, so that even the
most favourable H site is unfavourable, *H coverage stays ~0, and the sites are
free for CO2 reduction. The descriptor is therefore ASYMMETRIC and one-sided:

    h_free = min_site ΔG_H        (we already condition-drive this positive)
    ΔG_H > 0  -> sites free for CO2RR (good, larger = more H-repelling)
    ΔG_H < 0  -> *H covers the surface, CO2RR poisoned (bad, magnitude = how bad)

We report ΔG_H directly and an H_POISON flag (ΔG_H < 0). We also flag the
*COOH/*H DISSOCIATION regime (|ΔG| beyond a physical bound) where the single
point sampled a broken intermediate and the number is not a real ΔG.

    python scripts/61_her_co2rr_selectivity.py --dft outputs/dft_ladder/dft
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

RY = 13.605693122994
DG = {"CO": 0.10, "COOH": 0.41, "H": 0.24}
# Beyond this, the "adsorption" is really dissociation/absorption — not a datum.
PHYS = 2.5  # eV


def final_E(out: Path):
    if not out.exists():
        return None
    t = out.read_text(errors="ignore")
    m = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)", t)
    return float(m[-1]) * RY if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dft", type=Path, default=Path("outputs/dft_ladder/dft"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.dft / "surface_manifest.csv")))
    E = {}
    for r in rows:
        E[r["prefix"]] = final_E(Path(r["dir"]) / (r.get("infile", "scf.in").replace(".in", ".out")))

    def g(name):  # gas
        return E.get(name)
    e_CO, e_CO2, e_H2 = g("CO"), g("CO2"), g("H2")
    ref = {"CO": e_CO,
           "COOH": (e_CO2 + 0.5 * e_H2) if (e_CO2 and e_H2) else None,
           "H": 0.5 * e_H2 if e_H2 else None}

    # group rows by (structure, facet)
    groups: dict[tuple, dict] = {}
    for r in rows:
        if r["formula"] == "_gas":
            continue
        key = (r["formula"], r["facet"])
        d = groups.setdefault(key, {})
        d[(r["adsorbate"], r["kind"])] = E.get(r["prefix"])

    out_rows = []
    for (formula, facet), d in sorted(groups.items()):
        rec = {"structure": formula.split("_f")[0], "facet": facet}
        dGs = {}
        for ads in ("CO", "COOH", "H"):
            e_ads, e_clean = d.get((ads, "ads")), d.get((ads, "clean"))
            if None in (e_ads, e_clean, ref[ads]):
                dGs[ads] = None
            else:
                dGs[ads] = round(e_ads - e_clean - ref[ads] + DG[ads], 3)
        rec.update({f"dG_{a}": dGs[a] for a in ("CO", "COOH", "H")})

        # CO2RR-to-CO limiting potential (2-electron path via *COOH then *CO)
        if dGs["COOH"] is not None and dGs["CO"] is not None:
            dG1 = dGs["COOH"]
            dG2 = dGs["CO"] - dGs["COOH"]
            rec["U_L_CO2RR"] = round(-max(dG1, dG2), 3)
        else:
            rec["U_L_CO2RR"] = None
        # H site-availability: ΔG_H > 0 => surface repels H, sites free for CO2RR.
        # ΔG_H < 0 => *H covers the surface and poisons CO2RR (the real problem,
        # independent of H2-evolution rate). One-sided, not |ΔG_H|.
        rec["h_free"] = dGs["H"]  # want > 0
        rec["H_poison"] = (dGs["H"] is not None and dGs["H"] < 0)
        # A candidate is CO2RR-selective only if it BOTH activates CO2 at low
        # overpotential AND keeps H off the surface.
        rec["co2rr_selective"] = (
            rec["U_L_CO2RR"] is not None and dGs["H"] is not None
            and dGs["CO"] is not None and abs(dGs["CO"]) < 0.3 and dGs["H"] > 0)
        rec["diss_flags"] = ",".join(a for a in ("CO", "COOH", "H")
                                     if dGs[a] is not None and abs(dGs[a]) > PHYS) or "-"
        out_rows.append(rec)

    out = args.out or (args.dft / "her_co2rr_selectivity.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    print(f"{'structure':22s}{'facet':6s}{'ΔG_CO':>7s}{'ΔG_H':>7s}"
          f"{'U_L(CO2RR)':>11s}{'H_free?':>8s}{'CO2RR-sel':>10s}  diss")
    for r in out_rows:
        def s(x): return f"{x:+.2f}" if isinstance(x, (int, float)) else "  -"
        hf = "REPELS-H" if (r["dG_H"] is not None and r["dG_H"] > 0) else "H-poison"
        sel = "YES" if r["co2rr_selective"] else "no"
        print(f"{r['structure']:22s}{r['facet']:6s}{s(r['dG_CO']):>7s}{s(r['dG_H']):>7s}"
              f"{s(r['U_L_CO2RR']):>11s}{hf:>8s}{sel:>10s}  {r['diss_flags']}")
    print(f"\nwrote {out}")
    print("CO2RR-selective = |ΔG_CO|<0.3 AND ΔG_H>0 (surface repels H so sites stay free).")
    print("H-poison = ΔG_H<0: *H covers the surface and blocks CO2 — kills CO2RR regardless of H2 rate.")


if __name__ == "__main__":
    main()
