#!/usr/bin/env python
"""Build the DFT manifest for the dual-objective winners, selecting for each
structure the SPECIFIC facet that both releases CO and repels H.

The surface export (54, --save-facets 8) dumps geometries for several facets per
structure. For DFT we want exactly one facet per winner: the one the per-facet
analysis (63) flagged as the dual winner (|ΔG_CO|<co_win AND physical ΔG_H>0).
The CO-best facet is often NOT that facet, so we must pick deliberately.

This re-derives, from the freshly exported screen jsonl, each structure's best
dual facet, then keeps only that facet's rows (clean + *CO/*COOH/*H) from the
export manifest.

    python scripts/64_build_dual_dft_manifest.py \
        --screen outputs/dft_dual/surface.jsonl \
        --targets outputs/dft_dual/targets --out outputs/dft_dual/dft_targets.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path


def phys_min(vals, cut):
    p = [v for v in vals if abs(v) <= cut]
    return min(p) if p else None


def best_dual_facet(rec, co_win, diss_cut):
    """Return (miller_tuple, dG_CO, phys_dG_H) for the best dual facet, or None."""
    best = None
    for fac in rec.get("facets", []):
        co = fac.get("dG_CO_values") or []
        h = fac.get("dG_H_values") or []
        if not co or not h:
            continue
        co_best = min(co, key=abs)
        h_phys = phys_min(h, diss_cut)
        if h_phys is None:
            continue
        dual = abs(co_best) < co_win and h_phys > 0
        # rank: dual winners by ΔG_H, else fall back to CO-window facets by ΔG_H
        score = (dual, abs(co_best) < co_win, h_phys)
        if best is None or score > best[0]:
            best = (score, tuple(fac["miller"]), round(co_best, 3), round(h_phys, 3))
    return best[1:] if best else None


def facet_tag(miller):
    return "".join(str(i) for i in miller)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--co-win", type=float, default=0.15)
    ap.add_argument("--diss-cut", type=float, default=1.5)
    args = ap.parse_args()

    # 1) pick the dual facet per structure
    chosen = {}
    for line in args.screen.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("facets"):
            continue
        name = re.split(r"__", r.get("formula", r.get("source", "?")))[0]
        pick = best_dual_facet(r, args.co_win, args.diss_cut)
        if pick:
            chosen[name] = pick  # (miller, dG_CO, phys_dG_H)

    # 2) read the export manifest and keep only the chosen facet's rows
    man = args.targets / "dft_targets.csv"
    rows = list(csv.DictReader(open(man)))
    keep = []
    for r in rows:
        struct = re.split(r"__", r["formula"])[0]
        if struct not in chosen:
            continue
        tag = facet_tag(chosen[struct][0])
        if r["facet"] == tag:
            keep.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if keep:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
            w.writeheader(); w.writerows(keep)

    print(f"chose dual facet for {len(chosen)} structures; "
          f"{len(keep)} target rows (should be ~{len(chosen)*3})")
    covered = {re.split(r'__', r['formula'])[0] for r in keep}
    for name, (mi, co, h) in sorted(chosen.items()):
        mark = "OK" if name in covered else "!! facet not in export"
        print(f"  {name:20s} facet {str(list(mi)):12s} ΔG_CO={co:+.3f} ΔG_H={h:+.3f}  {mark}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
