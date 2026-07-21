#!/usr/bin/env python
"""Rank surface-screened candidates on the DUAL objective: CO release AND H repulsion.

The DFT selectivity check (scripts/61) showed the binding constraint is not CO at
all — it is H coverage. Five of six DFT structures were H-poisoned (ΔG_H < 0, *H
covers the surface and blocks CO2); only the one that REPELLED H (ΔG_H > 0) was
CO2RR-selective. So a candidate is only interesting if BOTH hold:

    |ΔG_CO| < co_win      CO desorbs as product (not poisoned, not inert)
    min_dG_H > h_min      even the most favourable H site is unfavourable, so *H
                          coverage ~0 and the sites stay free for CO2 reduction

`min_dG_H` is the MOST NEGATIVE ΔG_H over all sites (the co_surface_screen
convention). Requiring it > 0 is the strong, correct condition: no site anywhere
binds H. That is exactly what we conditioned generation toward (min_dG_H:0.3).

Reads the *.jsonl produced by 54_co2rr_surface_expand and prints the candidates
that pass BOTH, ranked by how well they repel H (larger min_dG_H = better) among
those in the CO window.

    python scripts/62_dual_objective_rank.py --screen outputs/screen_co2rr_her/surface.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def load(fn: Path):
    out = []
    for line in Path(fn).read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("best_dG_CO") is None:
            continue
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, type=Path, help="surface screen jsonl")
    ap.add_argument("--co-win", type=float, default=0.15)
    ap.add_argument("--h-min", type=float, default=0.0, help="require min_dG_H above this (>0 = repels H)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows = load(args.screen)
    print(f"screened: {len(rows)}")

    def f(r):
        return re.split(r"__", r.get("formula", r.get("source", "?")))[0]

    co_ok = [r for r in rows if abs(r["best_dG_CO"]) < args.co_win]
    h_ok = [r for r in rows if r.get("min_dG_H") is not None and r["min_dG_H"] > args.h_min]
    both = [r for r in co_ok if r.get("min_dG_H") is not None and r["min_dG_H"] > args.h_min]

    print(f"  in CO window (|ΔG_CO|<{args.co_win}):        {len(co_ok)}")
    print(f"  repels H (min_dG_H>{args.h_min}):             {len(h_ok)}")
    print(f"  BOTH (the dual objective):              {len(both)}")

    if not both:
        print("\nno candidate satisfies both. Closest-to-H-repelling in the CO window:")
        near = sorted(co_ok, key=lambda r: -(r.get("min_dG_H") or -99))[:args.top]
        for r in near:
            print(f"  {f(r):22s} ΔG_CO={r['best_dG_CO']:+.3f}  min_dG_H={r.get('min_dG_H')}")
        return

    both.sort(key=lambda r: -r["min_dG_H"])   # most H-repelling first
    print(f"\n{'formula':24s}{'ΔG_CO':>8s}{'min_dG_H':>10s}{'min_dG_COOH':>12s}")
    for r in both[:args.top]:
        print(f"{f(r):24s}{r['best_dG_CO']:+8.3f}{r['min_dG_H']:+10.3f}"
              f"{(r.get('min_dG_COOH') or 0):+12.3f}")


if __name__ == "__main__":
    main()
