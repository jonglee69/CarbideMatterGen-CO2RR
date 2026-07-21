#!/usr/bin/env python
"""Step 5 (CO2RR, manufacturable palette) — pre-filter generated candidates.

Differs from 30_cheap_screen_co2rr.py in two deliberate ways.

1. HARD ELEMENT WHITELIST (the fix for the rare-earth failure).
   Every generated structure must pass `element_policy.is_allowed`: metals drawn
   only from the acid-leach-survivable group-4/5/6 set, non-metals from C/B/N.
   The training set is already filtered, but MatterGen's *base* prior can still
   emit a rare earth, so this gate is the belt-and-braces layer. Soft
   conditioning alone is exactly what failed last time.

2. THE COMPOSITION CO PROXY IS A RANKER, NOT A GATE.
   In the honest palette every metal's carbide-site ΔG_CO proxy is negative:
       W -0.20 | Mo -0.30 | Cr -0.40 | V/Nb/Ta -0.45 | Ti -0.55 | Zr/Hf -0.60
   so a composition-weighted "best site" can never exceed -0.20 (pure W). The old
   gate (|ΔG_CO| < 0.20) would therefore reject *everything* — not because the
   materials are bad, but because a mean-field descriptor cannot resolve the
   site distribution. The DFT already showed real sites reaching ΔG_CO = +0.22
   in a multi-metal carbide whose mean-field value is far more negative.

   That is precisely the hypothesis under test: mean field says impossible,
   high-entropy SITE diversity says otherwise. So we rank by the proxy and move
   the real gate downstream to the site-resolved UMA surface screen.

    python scripts/31_cheap_screen_manufacturable.py \
        --gen-dir outputs/gen_co2rr_mfg --out outputs/screen_co2rr_mfg/candidates.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from carbidemattergen.co2rr import co2rr_descriptors, CO_OPTIMUM
from carbidemattergen.extra_descriptors import metal_mixing_entropy, n_metal_species
from carbidemattergen.features import compute_features
from carbidemattergen.element_policy import is_allowed, reject_reason, leach_unstable_fraction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--entropy-thr", type=float, default=1.0)
    ap.add_argument("--min-metals", type=int, default=3)
    ap.add_argument("--co-rank-cap", type=float, default=1.0,
                    help="drop only absurd proxies (|ΔG_CO| > cap); NOT the CO window")
    ap.add_argument("--top", type=int, default=40, help="rows to keep after ranking")
    args = ap.parse_args()

    files = [*args.gen_dir.rglob("*.cif"), *args.gen_dir.rglob("*POSCAR*")]
    rows, rejects = [], {}
    for fp in files:
        rel = str(fp.relative_to(args.gen_dir))
        try:
            s = Structure.from_file(str(fp))
        except Exception:
            continue
        comp = s.composition

        # --- HARD GATE: manufacturability + scientific scope ------------------
        if not is_allowed(comp):
            r = reject_reason(comp) or "?"
            rejects[r] = rejects.get(r, 0) + 1
            continue

        co = co2rr_descriptors(comp)
        best = co["best_site_co_affinity"]
        if best != best or abs(best - CO_OPTIMUM) > args.co_rank_cap:
            continue

        ent = metal_mixing_entropy(comp)
        nm = n_metal_species(comp)
        if ent < args.entropy_thr or nm < args.min_metals:
            continue

        feats = compute_features(s)
        rows.append({
            "source": rel,
            "formula": comp.reduced_formula,
            # ranked on, not gated on:
            "best_site_co_affinity": round(best, 4),
            "abs_co_opt": round(abs(best - CO_OPTIMUM), 4),
            "mean_co_affinity": round(co["mean_co_affinity"], 4),
            "mean_cooh_affinity": round(co["mean_cooh_affinity"], 4),
            "co2rr_selectivity": round(co["co2rr_selectivity"], 4),
            "metal_mixing_entropy": round(ent, 3),
            "n_metal_species": nm,
            "carbon_fraction": round(comp.get_atomic_fraction("C"), 3),
            "leach_unstable_fraction": leach_unstable_fraction(comp),  # must be 0.0
            "metal_percolation_dim": (feats.metal_percolation_dim if feats else float("nan")),
        })

    df = pd.DataFrame(rows)
    print(f"scanned {len(files)} generated structures")
    if rejects:
        print("hard-gate rejections:")
        for r, n in sorted(rejects.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {n:5d}  {r}")
    if df.empty:
        print("no candidates passed the hard gate.")
        return

    # sanity: the whole point of the campaign
    bad = df[df.leach_unstable_fraction != 0.0]
    assert bad.empty, f"leach-unstable slipped through: {bad.formula.tolist()[:5]}"

    # DO NOT rank on the composition CO proxy. Measured against the UMA surface
    # screen it scores r = +0.035 over the 271 training carbides — it is noise,
    # so ordering by it would be an arbitrary shuffle dressed up as a ranking.
    # It is kept in the table as a column (to report the correlation), never as
    # a selector. Rank on what IS meaningful and cheap: more metals and higher
    # mixing entropy (the cocktail hypothesis under test), then C-rich (the
    # mechanism lever our DFT identified: C enrichment deepens eps_d).
    # The real gate is the site-resolved UMA surface screen downstream.
    df = df.sort_values(["n_metal_species", "metal_mixing_entropy", "carbon_fraction"],
                        ascending=[False, False, False]).head(args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nkept {len(df)} candidates (ALL rare-earth-free, acid-leach stable)")
    print(f"  best_site_co_affinity: best={df.best_site_co_affinity.max():.3f} "
          f"(mean-field ceiling is -0.20 = pure W — the surface screen is the real test)")
    print(f"  n_metal_species: {dict(df.n_metal_species.value_counts().sort_index())}")
    print(f"  carbon_fraction: median={df.carbon_fraction.median():.3f} max={df.carbon_fraction.max():.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
