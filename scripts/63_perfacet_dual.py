#!/usr/bin/env python
"""Per-FACET dual-objective analysis: CO release AND H repulsion, done right.

The global min_dG_H aggregation was wrong. co_surface_screen reports min_dG_H =
the single most-negative ΔG_H over ALL facets and sites, so one dissociative /
subsurface H site (ΔG_H ~ -6 to -9 eV, clearly unphysical) buries the fact that
an entire OTHER facet repels H at every site (ΔG_H ~ +1.2). Selectivity is a
per-facet property: a catalyst works if it exposes ONE facet that both releases
CO and keeps H off. So we go back to the stored per-site values and ask, facet by
facet:

    CO release:   some site with |ΔG_CO| < co_win
    H repulsion:  the most stable PHYSICAL H site is unfavourable (ΔG_H > 0)

"Physical" excludes |ΔG_H| > diss_cut (dissociation/subsurface — not an
adsorption datum). The H descriptor per facet is then min over the PHYSICAL H
sites (the site H would actually occupy); if that is > 0 the facet repels H.

    python scripts/63_perfacet_dual.py --screen outputs/ladder/sqs.jsonl outputs/ladder/candidates.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def phys_min(vals, cut):
    p = [v for v in vals if abs(v) <= cut]
    return min(p) if p else None


def analyze(files, co_win, diss_cut, h_thr):
    winners = []
    per_struct = {}
    for fn in files:
        for line in Path(fn).read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("facets"):
                continue
            name = re.split(r"__", r.get("formula", r.get("source", "?")))[0]
            best = None
            for fac in r["facets"]:
                co = fac.get("dG_CO_values") or []
                h = fac.get("dG_H_values") or []
                if not co or not h:
                    continue
                co_best = min((c for c in co), key=lambda c: abs(c), default=None)
                h_phys = phys_min(h, diss_cut)   # site H actually occupies
                if co_best is None or h_phys is None:
                    continue
                in_win = abs(co_best) < co_win
                repels = h_phys > h_thr
                score = (in_win and repels, h_phys if in_win else -99)
                cand = dict(struct=name, miller=fac["miller"], co=round(co_best, 3),
                            h_phys=round(h_phys, 3), in_win=in_win, repels=repels)
                if best is None or score > best[0]:
                    best = (score, cand)
            if best:
                per_struct[name] = best[1]
                if best[1]["in_win"] and best[1]["repels"]:
                    winners.append(best[1])
    return per_struct, winners


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", required=True, type=Path)
    ap.add_argument("--co-win", type=float, default=0.15)
    ap.add_argument("--diss-cut", type=float, default=1.5,
                    help="|ΔG_H| above this = dissociation/subsurface, not a datum")
    ap.add_argument("--h-thr", type=float, default=0.0)
    args = ap.parse_args()

    per, win = analyze(args.screen, args.co_win, args.diss_cut, args.h_thr)
    print(f"structures analyzed: {len(per)}")
    # how many have a facet that repels H at all (physical H site > 0)
    rep = [c for c in per.values() if c["repels"]]
    inw = [c for c in per.values() if c["in_win"]]
    print(f"  have an H-repelling facet (phys ΔG_H>{args.h_thr}):  {len(rep)}")
    print(f"  have a CO-window facet:                       {len(inw)}")
    print(f"  BOTH on the SAME facet (dual winners):        {len(win)}")

    print(f"\n=== dual winners (CO release + H repulsion on one facet) ===")
    for c in sorted(win, key=lambda c: -c["h_phys"]):
        print(f"  {c['struct']:22s} {str(c['miller']):12s} ΔG_CO={c['co']:+.3f}  ΔG_H(phys)={c['h_phys']:+.3f}")
    if not win:
        print("  none. Best H-repellers that are NOT in the CO window:")
        for c in sorted(rep, key=lambda c: -c["h_phys"])[:8]:
            print(f"  {c['struct']:22s} {str(c['miller']):12s} ΔG_CO={c['co']:+.3f}  ΔG_H(phys)={c['h_phys']:+.3f}")


if __name__ == "__main__":
    main()
