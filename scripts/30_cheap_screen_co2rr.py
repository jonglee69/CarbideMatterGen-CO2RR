#!/usr/bin/env python
"""Step 5 (cheap, CO2RR) — pre-filter generated candidates before the expensive
fairchem CO2RR surface screen (co_surface_screen).

CO2RR sibling of scripts/30_cheap_screen.py. Recomputes the CO2RR composition
proxies on every generated structure and keeps only the promising, novel,
affordable, metallic ones so Stage B runs on a small high-value set. Filters:
  - is a TM carbide (C + TM, no competing anion)
  - |best_site_co_affinity − CO_OPTIMUM| < --co-thr   (some site releases CO)
  - co2rr_selectivity      >= --sel-thr               (CO favoured over H2)
  - metal_mixing_entropy   >= --entropy-thr           (medium/high entropy)
  - n_metal_species        >= --min-metals
  - expensive_metal_fraction <= --max-expensive       (economic; PGM+heavy REE)
  - metallic framework: metal_percolation_dim == 3 (when structure available)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.core import Structure

from carbidemattergen.labeling import is_tm_carbide_composition
from carbidemattergen.hydrogen import carbon_fraction, expensive_metal_fraction
from carbidemattergen.co2rr import co2rr_descriptors, CO_OPTIMUM
from carbidemattergen.extra_descriptors import metal_mixing_entropy, n_metal_species
from carbidemattergen.features import compute_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--co-thr", type=float, default=0.20,
                    help="|best_site_co_affinity − CO_OPTIMUM| upper bound (eV)")
    ap.add_argument("--sel-thr", type=float, default=0.0,
                    help="min co2rr_selectivity (ΔG_H − ΔG_COOH); 0 = COOH ≥ H")
    ap.add_argument("--entropy-thr", type=float, default=1.0)
    ap.add_argument("--min-metals", type=int, default=3)
    ap.add_argument("--max-expensive", type=float, default=0.0,
                    help="max expensive_metal_fraction (PGM+heavy REE); 0 = clean")
    args = ap.parse_args()

    files = [*args.gen_dir.rglob("*.cif"), *args.gen_dir.rglob("*POSCAR*")]
    rows = []
    for fp in files:
        # Unique source id: generated files share bare names (gen_N.cif repeats
        # per chunk/weight), so record the path RELATIVE to gen_dir to keep the
        # candidate→file mapping unambiguous downstream.
        rel_source = str(fp.relative_to(args.gen_dir))
        try:
            s = Structure.from_file(str(fp))
        except Exception:  # noqa: BLE001
            continue
        comp = s.composition
        if not is_tm_carbide_composition(comp):
            continue
        co = co2rr_descriptors(comp)
        ent = metal_mixing_entropy(comp)
        nm = n_metal_species(comp)
        exp = expensive_metal_fraction(comp)
        best = co["best_site_co_affinity"]
        if best != best:  # NaN
            continue
        # Some site must sit in the CO-release window (|ΔG_CO − optimum| < thr).
        if abs(best - CO_OPTIMUM) >= args.co_thr:
            continue
        if co["co2rr_selectivity"] < args.sel_thr:
            continue
        if ent < args.entropy_thr or nm < args.min_metals:
            continue
        if exp == exp and exp > args.max_expensive:
            continue
        feats = compute_features(s)
        perc = feats.metal_percolation_dim if feats else float("nan")
        rows.append({
            "source": rel_source,
            "formula": comp.reduced_formula,
            "best_site_co_affinity": round(best, 4),
            "mean_co_affinity": round(co["mean_co_affinity"], 4),
            "frac_co_selective_sites": round(co["frac_co_selective_sites"], 3),
            "mean_cooh_affinity": round(co["mean_cooh_affinity"], 4),
            "co2rr_selectivity": round(co["co2rr_selectivity"], 4),
            "metal_mixing_entropy": round(ent, 3),
            "n_metal_species": nm,
            "expensive_metal_fraction": round(exp, 3) if exp == exp else None,
            "carbon_fraction": round(carbon_fraction(comp), 3),
            "metal_percolation_dim": perc,
        })

    # Rank: closest CO-release site first, then more selective sites, then
    # stronger CO2 activation (more negative *COOH).
    rows.sort(key=lambda r: (abs(r["best_site_co_affinity"] - CO_OPTIMUM),
                             -r["frac_co_selective_sites"],
                             r["mean_cooh_affinity"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"kept {len(rows)} / {len(files)} candidates → {args.out}")


if __name__ == "__main__":
    main()
