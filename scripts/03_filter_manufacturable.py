#!/usr/bin/env python
"""Step 3 — filter the labelled carbide set to the MANUFACTURABLE palette.

The first CO2RR campaign produced six candidates that cannot be made: the user's
route is metallothermic reduction of oxides + a mandatory acid leach (to dissolve
the MgO/CaO reductant oxide), and that leach dissolves every rare earth. All 24
screened compositions contained one (Y was in all 24).

Root cause: the only compositional constraint in the pipeline was *cost*, and it
explicitly whitelisted the cheap light rare earths. Aqueous/acid stability was
never encoded, and Y/Sc slipped every "d-block only" check.

Fix, applied at the DATA level (the strongest lever — the model cannot generate
what it never saw). See :mod:`carbidemattergen.element_policy` for the palette
and the two distinct rationales (manufacturability vs. scientific scope).

Note on conditioning: once the training set is filtered, `expensive_metal_fraction`
and `leach_unstable_fraction` are identically 0 -> zero variance -> useless as
conditioning axes. The freed 9th slot is given to `carbon_fraction`, which has
real variance AND is the mechanism lever our own DFT identified
(corr(C/M, eps_d) = -0.53; deeper eps_d -> weaker CO binding -> CO release).

    python scripts/03_filter_manufacturable.py --in data --out data_mfg
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from carbidemattergen.element_policy import is_allowed, reject_reason


def filter_split(src: Path, dst: Path, allow_bn: bool, log=print) -> pd.DataFrame:
    df = pd.read_csv(src)
    col = "reduced_formula" if "reduced_formula" in df.columns else "pretty_formula"

    def keep(f):
        try:
            return is_allowed(Composition(f), allow_bn=allow_bn)
        except Exception:
            return False

    mask = df[col].apply(keep)
    out = df[mask].copy()
    out.to_csv(dst, index=False)

    # audit: why did we drop what we dropped
    from collections import Counter
    reasons = Counter()
    for f in df[~mask][col]:
        try:
            reasons[reject_reason(Composition(f), allow_bn=allow_bn) or "?"] += 1
        except Exception:
            reasons["unparseable"] += 1

    log(f"[{src.name}] {len(df):,} -> {len(out):,} kept ({100*mask.mean():.1f}%)")
    for r, n in reasons.most_common(6):
        log(f"    dropped {n:5d}  {r}")
    if len(out):
        cf = out["carbon_fraction"].describe()
        log(f"    carbon_fraction: mean={cf['mean']:.3f} p50={cf['50%']:.3f} p90={out['carbon_fraction'].quantile(0.9):.3f} max={cf['max']:.3f}")
        nm = out["n_metal_species"].value_counts().sort_index()
        log(f"    n_metal_species: {dict(nm)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=Path("data"))
    ap.add_argument("--out", dest="dst", type=Path, default=Path("data_mfg"))
    ap.add_argument("--no-bn", action="store_true",
                    help="C only (default also allows B/N: boro-carbides, carbonitrides)")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    allow_bn = not args.no_bn
    print(f"palette: group-4/5/6 TMs; non-metals = {'C,B,N' if allow_bn else 'C'}")
    for split in ("train", "val"):
        src = args.src / f"carbide_labeled_{split}.csv"
        dst = args.dst / f"carbide_labeled_{split}.csv"
        filter_split(src, dst, allow_bn)
    print(f"\nwrote -> {args.dst}")


if __name__ == "__main__":
    main()
