#!/usr/bin/env python
"""Step A' — build the MIXING LADDER: explicit 3/4/5-metal solid-solution carbides.

Why this exists
---------------
The UMA-conditioned generator will not produce high-entropy carbides. Its
training set (the manufacturable palette in Materials Project) tops out at THREE
metals and a metal-mixing entropy of ~1.05 R, so conditioning on S_mix = 1.6 R is
an extrapolation it cannot make: at both guidance weights the 4-metal yield stays
near 1% and the model just piles on more Mo. That is a real, reportable limit of
generative extrapolation — not something to paper over.

But the scientific question needs 4- and 5-metal carbides to be answerable:

    binary   WC        ΔG_CO -0.035   U_L -1.92 V   (CO releases, but awful onset)
    ternary  HfMoC2    ΔG_CO -0.004   U_L -0.45 V   (mixing FIXES the onset)
    quaternary / quinary  = ???                     (does MORE mixing help further?)

So we construct the missing rungs directly, with the SQS machinery already in the
repo (carbidemattergen.solid_solution), restricted to the acid-leach-survivable
palette. The generator supplies the chemistry that works (Mo/W-rich, group-4/5
partners); this script pushes it up the mixing ladder.

Combination choice is physics-led, not exhaustive: every UMA-labelled winner so
far pairs a GROUP-6 metal (Mo, W — deep d-band, weak CO binding) with a group-4/5
partner (Ti/Zr/Hf/V/Nb/Ta). We therefore enumerate combos that contain at least
one of Mo/W, plus a control set with none (to test whether group-6 is actually
required) and the textbook equimolar TiZrHfNbTa HEC as an external reference.

    python scripts/36_build_hec_ladder.py --out outputs/sqs_hec --n-metal 20
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

from carbidemattergen.element_policy import ACID_STABLE_METALS

GROUP6 = ["Mo", "W"]
GROUP45 = ["Ti", "Zr", "Hf", "V", "Nb", "Ta"]
GROUP6_ALL = GROUP6 + ["Cr"]


def combos(n_metals: int, want: int, seed_pool=None) -> list[tuple[str, ...]]:
    """Physics-led combinations of `n_metals` from the palette.

    Every combo carries at least one group-6 metal (Mo/W/Cr): those are the
    deep-d-band CO-weakeners, and every UMA winner so far contains one. W matters
    as much as Mo (WC is the only binary in the CO window; Ta2W2C3 and Zr2W2C3
    are top ternaries), so Mo- and W-rooted combos are INTERLEAVED — taking the
    first N of a Mo-first enumeration would silently drop W entirely.
    """
    per_root: dict[str, list[tuple[str, ...]]] = {}
    for g6 in GROUP6_ALL:
        per_root[g6] = [tuple(sorted((g6,) + rest))
                        for rest in itertools.combinations(GROUP45, n_metals - 1)]
    # round-robin across Mo, W, Cr so all three roots are represented
    out: list[tuple[str, ...]] = []
    for i in range(max(len(v) for v in per_root.values())):
        for g6 in ("Mo", "W", "Cr"):
            if i < len(per_root[g6]):
                out.append(per_root[g6][i])
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq[:want]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/sqs_hec"))
    ap.add_argument("--n-metal", type=int, default=20,
                    help="metal sites in the SQS supercell (20 = 4 each for 5 metals)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--per-rung", type=int, default=8,
                    help="combinations per mixing rung (3, 4 and 5 metals)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from carbidemattergen.solid_solution import construct_solid_solution

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    plan: list[tuple[int, tuple[str, ...]]] = []
    for n in (3, 4, 5):
        for c in combos(n, args.per_rung):
            plan.append((n, c))
    # external reference: the textbook equimolar HEC (no group-6 at all)
    plan.append((5, ("Hf", "Nb", "Ta", "Ti", "Zr")))

    for n, metals in plan:
        assert set(metals) <= ACID_STABLE_METALS, metals
        # equimolar rock-salt-like MC stoichiometry
        formula = "".join(metals) + "C" + str(len(metals))
        try:
            s = construct_solid_solution(formula, n_metal_target=args.n_metal,
                                         steps=args.steps, seed=args.seed)
            struct = s.structure if hasattr(s, "structure") else s
            name = f"hec{n}_{''.join(metals)}"
            fp = args.out / f"{name}.cif"
            struct.to(filename=str(fp))
            rows.append((n, "-".join(metals), len(struct), str(fp)))
            print(f"  [{n}-metal] {'-'.join(metals):22s} nat={len(struct):3d} -> {fp.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{n}-metal] {'-'.join(metals):22s} FAILED: {exc}")

    print(f"\nbuilt {len(rows)} solid-solution carbides -> {args.out}")
    print("rungs:", {n: sum(1 for r in rows if r[0] == n) for n in (3, 4, 5)})


if __name__ == "__main__":
    main()
