#!/usr/bin/env python
"""Step 6 (Stage A') — build high-entropy carbide solid solutions from the
Stage-A HEC shortlist.

For each candidate composition in `top_HEC_candidates.csv`, this:
  1. identifies the parent framework (rock-salt MC / hcp M2C) from the carbon
     fraction;
  2. builds a metal-sublattice supercell of ~N target sites;
  3. occupies it near-equimolar with the candidate's metal palette and anneals
     the occupation to an SQS (Monte-Carlo pair-correlation matching);
  4. writes the SQS structure as a CIF and records the SQS quality
     (Phi_random -> Phi_SQS, shell radii, placed palette).

Outputs:
  outputs/sqs/<rank>_<reduced_formula>.cif     — the solid-solution supercells
  outputs/sqs/sqs_summary.csv                  — one row per candidate

These supercells are the input to Stage B (fairchem surface ΔG_H* screening).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from carbidemattergen.solid_solution import (
    Framework, build_framework_supercell, make_sqs, metal_palette,
)
from pymatgen.core import Composition


def build_one(formula: str, n_metal_target: int, steps: int, seed: int):
    comp = Composition(formula)
    xc = comp.get_atomic_fraction("C") if "C" in comp else 0.5
    fw = Framework("rocksalt_MC", 1.0, xc) if xc >= 0.40 else Framework("hcp_M2C", 2.0, xc)
    palette = metal_palette(formula)
    sc, midx = build_framework_supercell(fw, n_metal_target)
    res = make_sqs(sc, midx, palette, steps=steps, seed=seed)
    return fw, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path,
                    default=Path("outputs/gen/top_HEC_candidates.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sqs"))
    ap.add_argument("--top", type=int, default=20,
                    help="number of highest-entropy candidates to build")
    ap.add_argument("--n-metal-target", type=int, default=54)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    # rank by entropy then thermoneutral fraction (Stage A' input shortlist)
    df = df.sort_values(["S_mix", "frac_tn"], ascending=False).head(args.top)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        f = r["formula"]
        try:
            fw, res = build_one(f, args.n_metal_target, args.steps, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[{rank}] {f}: FAILED ({exc})")
            continue
        red = res.structure.composition.reduced_formula
        cif = args.out / f"{rank:02d}_{red}.cif"
        res.structure.to(filename=str(cif))
        rows.append({
            "rank": rank, "source_formula": f, "sqs_formula": red,
            "framework": fw.kind, "carbon_fraction": round(fw.carbon_fraction, 3),
            "n_metal_sites": res.n_metal_sites,
            "n_palette": len(res.palette),
            "phi_random": round(res.objective_initial, 5),
            "phi_sqs": round(res.objective, 5),
            "phi_reduction": round(res.objective_initial / res.objective, 2)
                              if res.objective > 0 else float("inf"),
            "shell_radii": ";".join(f"{x:.2f}" for x in res.shell_radii),
            "best_site": r.get("best_site"), "frac_tn": r.get("frac_tn"),
            "S_mix": round(r.get("S_mix", float("nan")), 3),
            "w": r.get("w"), "cif": cif.name,
        })
        print(f"[{rank}] {f} -> {red}  ({fw.kind}, {res.n_metal_sites} M-sites, "
              f"Phi {res.objective_initial:.4f}->{res.objective:.4f})")

    summ = args.out / "sqs_summary.csv"
    if rows:
        with open(summ, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\nbuilt {len(rows)} solid solutions -> {args.out}")
    print(f"summary -> {summ}")


if __name__ == "__main__":
    main()
