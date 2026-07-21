#!/usr/bin/env python
"""Step 5 — merge the COMPUTED UMA surface labels into the training CSVs,
replacing the four hand-table CO2RR descriptors.

The four conditioning targets that steered generation used to come from
`co2rr._CO_ADS_CARBIDE` / `_COOH_ADS_CARBIDE` — one hand-picked number per
element. Checked against the UMA surface screen of the nine binary carbides of
the allowed palette, those tables score r = +0.18 (and r = -0.53 over the
physical range, i.e. ANTI-correlated). They are replaced here by ΔG values that
UMA actually computed for each structure's surface:

    best_site_co_affinity    <- best (closest-to-0) site ΔG_CO
    frac_co_selective_sites  <- fraction of sites in the CO-release window
    mean_cooh_affinity       <- min ΔG_COOH  (CO2 activation onset)
    co2rr_selectivity        <- ΔG_H - ΔG_COOH  (CO2RR out-competing HER)

Also carries U_L (limiting potential) through as a diagnostic column.

Rows the screen could not label are dropped: a NaN conditioning value would make
MatterGen skip the structure anyway, and we would rather know how many we lost.

    python scripts/05_merge_uma_labels.py --labels data_mfg --uma outputs/uma_labels \
        --out data_uma
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

# the conditioning columns we overwrite with computed physics.
# min_dG_H is added so HER suppression can be conditioned DIRECTLY: the DFT
# selectivity check showed every over-binding carbide loses the current to HER
# (margin U_L(CO2RR)-U_L(HER) < 0), and the one selective structure was the one
# with WEAK H binding (ΔG_H > 0). Conditioning ΔG_H toward positive is the fix;
# the indirect co2rr_selectivity (=ΔG_H-ΔG_COOH) mixed two quantities and could
# not steer H alone.
REPLACED = ["best_site_co_affinity", "frac_co_selective_sites",
            "mean_cooh_affinity", "co2rr_selectivity", "min_dG_H"]


def merge(split: str, labels: Path, uma: Path, out: Path) -> None:
    df = pd.read_csv(labels / f"carbide_labeled_{split}.csv")
    recs = [json.loads(l) for l in (uma / f"uma_labels_{split}.jsonl").read_text().splitlines() if l.strip()]
    u = pd.DataFrame(recs).set_index("material_id")

    old = df.set_index("material_id")
    # correlation between the table we are discarding and the physics we computed
    common = old.index.intersection(u.index)
    both = old.loc[common, "best_site_co_affinity"].astype(float)
    new = pd.to_numeric(u.loc[common, "best_site_co_affinity"], errors="coerce")
    ok = both.notna() & new.notna()
    r = both[ok].corr(new[ok]) if ok.sum() > 2 else float("nan")

    for col in REPLACED:
        old[col] = pd.to_numeric(u[col], errors="coerce")
    old["limiting_potential_V"] = pd.to_numeric(u["limiting_potential_V"], errors="coerce")

    before = len(old)
    old = old.dropna(subset=REPLACED)
    out.mkdir(parents=True, exist_ok=True)
    old.reset_index().to_csv(out / f"carbide_labeled_{split}.csv", index=False)

    print(f"[{split}] {before} -> {len(old)} rows (dropped {before-len(old)} unlabelled)")
    print(f"   table-vs-UMA correlation on best_site_co_affinity: r = {r:+.3f}"
          f"   <- the reason we are replacing it")
    d = old["best_site_co_affinity"]
    print(f"   computed ΔG_CO: min={d.min():+.2f} p50={d.median():+.2f} max={d.max():+.2f}")
    win = (d.abs() < 0.20).sum()
    print(f"   in the CO-release window (|ΔG_CO|<0.2): {win}/{len(old)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data_mfg"))
    ap.add_argument("--uma", type=Path, default=Path("outputs/uma_labels"))
    ap.add_argument("--out", type=Path, default=Path("data_uma"))
    args = ap.parse_args()
    for split in ("train", "val"):
        merge(split, args.labels, args.uma, args.out)
    print(f"\nwrote -> {args.out}  (conditioning now runs on computed UMA physics)")


if __name__ == "__main__":
    main()
