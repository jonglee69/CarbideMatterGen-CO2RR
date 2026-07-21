#!/usr/bin/env python
"""Step 4 — replace the hand-made CO2RR proxy tables with COMPUTED fairchem/UMA
surface free energies, for every structure in the manufacturable training set.

WHY
---
Stage A (the generator's conditioning) was driven by hand-calibrated lookup
tables (`co2rr._CO_ADS_CARBIDE`, `_COOH_ADS_CARBIDE`): one number per element,
"monotonic/relative". We finally had a way to check them — the UMA surface screen
of the nine binary carbides of the allowed palette. The tables do not survive it:

    metal   table    UMA(best-site ΔG_CO)
    W       -0.20    -0.035
    Mo      -0.30    -0.253
    Cr      -0.40    -0.113
    Zr      -0.60    -0.148
    Hf      -0.60    +0.984
    Ti      -0.55    -8.414
    ...
    Pearson r = +0.18 over all nine; r = -0.53 over the physical ones.

So the descriptor steering the diffusion model was uncorrelated with — in the
physical range, ANTI-correlated with — the quantity it claims to predict. Any
"the generative model discovered X" claim built on that is unsound, and the
"mean-field ceiling of -0.20" that the table implied is an artefact of the table,
not physics (UMA puts WC at -0.035, i.e. already in the CO-release window).

This script recomputes the four CO2RR conditioning labels from actual UMA
surface physics — the same engine used for the Stage B screen and the binary
baselines — so the whole pipeline runs on computed energies end to end:

    best_site_co_affinity    -> best (closest-to-0) site ΔG_CO over all facets
    frac_co_selective_sites  -> fraction of sites inside the CO-release window
    mean_cooh_affinity       -> min ΔG_COOH (CO2 activation onset)
    co2rr_selectivity        -> ΔG_H - ΔG_COOH  (CO2RR out-competing HER)

Structures come from the MatterGen alex_mp_20 cache (the label CSV only carries
material_id), matched by structure_id.

Cheaper settings than the production screen (this is a *label*, and we need 279
of them, not 6): max_index=1, capped sites, fewer relaxation steps. Resumable.

    /home/jonglee69/.venv_fairchem/bin/python scripts/04_uma_relabel.py \
        --labels data_mfg --cache /home/jonglee69/mattergen/datasets/cache/alex_mp_20 \
        --out outputs/uma_labels --device cuda
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_cache_structures(cache: Path, split: str, wanted_ids: set[str]) -> dict:
    """Reconstruct pymatgen Structures for the wanted material_ids from the cache."""
    from pymatgen.core import Structure, Lattice

    d = cache / split
    sid = np.load(d / "structure_id.npy", allow_pickle=True)
    Z = np.load(d / "atomic_numbers.npy")
    cell = np.load(d / "cell.npy")
    pos = np.load(d / "pos.npy")
    nat = np.load(d / "num_atoms.npy")

    offs = np.concatenate([[0], np.cumsum(nat)])
    out = {}
    for i, s in enumerate(sid):
        s = str(s)
        if s not in wanted_ids:
            continue
        a, b = offs[i], offs[i + 1]
        z = Z[a:b]
        # cache stores FRACTIONAL coords
        st = Structure(Lattice(cell[i]), [int(x) for x in z], pos[a:b],
                       coords_are_cartesian=False)
        out[s] = st
    return out


def label_one(structure, calc, max_facets: int, site_cap: int, max_steps: int,
              max_slab_atoms: int) -> dict:
    """Lean UMA surface label for ONE structure.

    The production screen (co_surface_screen.screen_candidate) enumerates every
    facet and every site — ~10+ min per structure, i.e. >50 h for the training
    set. A *label* does not need that: it needs a physically-computed number
    produced by the same protocol for every structure. So we keep only the
    `max_facets` most stable terminations (the ones that actually dominate a real
    surface) and cap the sites. Same ΔG recipe, same UMA calculator, same gas
    references as the production screen — just a coarser, uniform sampling.
    """
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor
    from carbidemattergen.co_surface_screen import (
        _fix_bottom, _relax, _reference_energy, _adsorbate_molecule,
        select_terminations, DG_CORR, PHYS_DG_MAX, CO_OPTIMUM, CO_WINDOW, FIX_FRAC,
    )

    adsorbates = ("CO", "COOH", "H")
    refs = {a: _reference_energy(calc, a) for a in adsorbates}

    facets = select_terminations(structure, calc, max_index=1)
    # select_terminations already ranks by as-cut energy per atom; keep the most
    # stable few and skip pathologically large slabs.
    facets = [(mi, sl) for mi, sl in facets if len(sl) <= max_slab_atoms][:max_facets]

    agg = {a: [] for a in adsorbates}
    for mi, slab in facets:
        asf = AdsorbateSiteFinder(slab)
        slab_ref: dict[int, float] = {}
        for ads in adsorbates:
            structs = asf.generate_adsorption_structures(
                _adsorbate_molecule(ads), find_args={"distance": 1.8})[:site_cap]
            for st in structs:
                clean = st.copy()
                clean.remove_sites([i for i, s in enumerate(clean)
                                    if s.properties.get("surface_properties") == "adsorbate"])
                key = len(clean)
                if key not in slab_ref:
                    cs = AseAtomsAdaptor.get_atoms(clean)
                    _fix_bottom(cs, FIX_FRAC)
                    slab_ref[key], _, _ = _relax(cs, calc, 0.05, max_steps)
                a = AseAtomsAdaptor.get_atoms(st)
                _fix_bottom(a, FIX_FRAC)
                e_ads, _, _ = _relax(a, calc, 0.05, max_steps)
                dG = (e_ads - slab_ref[key] - refs[ads]) + DG_CORR[ads]
                if abs(dG) <= PHYS_DG_MAX:
                    agg[ads].append(dG)

    co, cooh, h = agg["CO"], agg["COOH"], agg["H"]
    if not co:
        return {"best_dG_CO": None, "frac_co_selective": None, "min_dG_COOH": None,
                "min_dG_H": None, "limiting_potential_V": None,
                "n_facets": len(facets), "n_sites": 0}
    best_co = min(co, key=lambda g: abs(g - CO_OPTIMUM))
    out = {
        "best_dG_CO": round(best_co, 4),
        "frac_co_selective": round(
            sum(1 for g in co if abs(g - CO_OPTIMUM) < CO_WINDOW) / len(co), 4),
        "min_dG_COOH": round(min(cooh), 4) if cooh else None,
        "min_dG_H": round(min(h), 4) if h else None,
        "n_facets": len(facets), "n_sites": len(co),
    }
    if cooh:
        out["limiting_potential_V"] = round(
            -max(min(cooh), best_co - min(cooh)), 4)
    else:
        out["limiting_potential_V"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data_mfg"))
    ap.add_argument("--cache", type=Path,
                    default=Path("/home/jonglee69/mattergen/datasets/cache/alex_mp_20"))
    ap.add_argument("--out", type=Path, default=Path("outputs/uma_labels"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--task", default="omat")
    ap.add_argument("--max-facets", type=int, default=1,
                    help="keep only the N most stable terminations (label-grade)")
    ap.add_argument("--site-cap", type=int, default=4,
                    help="sites per facet per adsorbate (label-grade, not production)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-slab-atoms", type=int, default=120)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from carbidemattergen.co_surface_screen import _load_fairchem_calc

    args.out.mkdir(parents=True, exist_ok=True)
    calc = _load_fairchem_calc(args.model, args.device, args.task)

    for split in ("train", "val"):
        csv = args.labels / f"carbide_labeled_{split}.csv"
        df = pd.read_csv(csv)
        ids = set(df["material_id"].astype(str))
        print(f"\n=== {split}: {len(ids)} structures to relabel ===", flush=True)

        structs = load_cache_structures(args.cache, split, ids)
        print(f"  reconstructed {len(structs)}/{len(ids)} from cache", flush=True)

        outfile = args.out / f"uma_labels_{split}.jsonl"
        done = set()
        if outfile.exists():
            for line in outfile.read_text().splitlines():
                try:
                    done.add(json.loads(line)["material_id"])
                except Exception:
                    pass
            print(f"  resuming: {len(done)} already done", flush=True)

        with open(outfile, "a") as f:
            for k, (mid, st) in enumerate(sorted(structs.items()), 1):
                if mid in done:
                    continue
                try:
                    r = label_one(st, calc, args.max_facets, args.site_cap,
                                  args.max_steps, args.max_slab_atoms)
                    rec = {
                        "material_id": mid,
                        "formula": st.composition.reduced_formula,
                        # the four CO2RR conditioning labels, now COMPUTED
                        "best_site_co_affinity": r.get("best_dG_CO"),
                        "frac_co_selective_sites": r.get("frac_co_selective"),
                        "mean_cooh_affinity": r.get("min_dG_COOH"),
                        "co2rr_selectivity": (
                            None if (r.get("min_dG_H") is None or r.get("min_dG_COOH") is None)
                            else round(r["min_dG_H"] - r["min_dG_COOH"], 4)),
                        "min_dG_H": r.get("min_dG_H"),
                        "limiting_potential_V": r.get("limiting_potential_V"),
                        "n_facets": r.get("n_facets"), "n_sites": r.get("n_sites"),
                    }
                except Exception as exc:  # noqa: BLE001
                    rec = {"material_id": mid,
                           "formula": st.composition.reduced_formula,
                           "error": str(exc)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  [{k}/{len(structs)}] {rec['formula']:12s} "
                      f"ΔG_CO={rec.get('best_site_co_affinity')} "
                      f"U_L={rec.get('limiting_potential_V')}", flush=True)

    print("\nUMA relabelling done ->", args.out)


if __name__ == "__main__":
    main()
