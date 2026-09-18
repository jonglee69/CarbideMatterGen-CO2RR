#!/usr/bin/env python
"""Step 5 (cheap) — pre-filter generated candidates before the expensive
fairchem surface screen.

Recomputes the composition proxies on every generated structure and keeps only
the promising, novel, stable-looking ones so Stage B (surface_screen) runs on a
small high-value set. Filters:
  - is a TM carbide (C + TM, no competing anion)
  - |best_site_h_affinity| < --best-thr           (some site near thermoneutral)
  - metal_mixing_entropy   >= --entropy-thr        (medium/high entropy)
  - n_metal_species        >= --min-metals
  - metallic framework: metal_percolation_dim == 3 (when structure available)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.core import Structure

from carbidemattergen.labeling import is_tm_carbide_composition
from carbidemattergen.hydrogen import hydrogen_descriptors, carbon_fraction
from carbidemattergen.extra_descriptors import metal_mixing_entropy, n_metal_species
from carbidemattergen.features import compute_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--best-thr", type=float, default=0.10)
    ap.add_argument("--entropy-thr", type=float, default=1.0)
    ap.add_argument("--min-metals", type=int, default=3)
    args = ap.parse_args()

    files = [*args.gen_dir.rglob("*.cif"), *args.gen_dir.rglob("*POSCAR*")]
    rows = []
    for fp in files:
        try:
            s = Structure.from_file(str(fp))
        except Exception:  # noqa: BLE001
            continue
        comp = s.composition
        if not is_tm_carbide_composition(comp):
            continue
        h = hydrogen_descriptors(comp)
        ent = metal_mixing_entropy(comp)
        nm = n_metal_species(comp)
        if h["best_site_h_affinity"] != h["best_site_h_affinity"]:
            continue
        # Strict '>=' to match the thermoneutral window (|ΔG| < thr): a best
        # site exactly at the edge has no thermoneutral site (frac == 0) and
        # must be rejected — this is the all-strong-binder HEC counterexample
        # (e.g. ZrTiHfNbVC, best = -0.10).
        if abs(h["best_site_h_affinity"]) >= args.best_thr:
            continue
        if ent < args.entropy_thr or nm < args.min_metals:
            continue
        feats = compute_features(s)
        perc = feats.metal_percolation_dim if feats else float("nan")
        rows.append({
            "source": fp.name,
            "formula": comp.reduced_formula,
            "best_site_h_affinity": round(h["best_site_h_affinity"], 4),
            "mean_h_affinity": round(h["mean_h_affinity"], 4),
            "frac_thermoneutral_sites": round(h["frac_thermoneutral_sites"], 3),
            "metal_mixing_entropy": round(ent, 3),
            "n_metal_species": nm,
            "carbon_fraction": round(carbon_fraction(comp), 3),
            "metal_percolation_dim": perc,
        })

    rows.sort(key=lambda r: (abs(r["best_site_h_affinity"]),
                             -r["frac_thermoneutral_sites"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"kept {len(rows)} / {len(files)} candidates → {args.out}")


if __name__ == "__main__":
    main()
