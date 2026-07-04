#!/usr/bin/env python
"""Phase B (GPU) — expanded CO2RR surface screen that ALSO exports geometries.

The sibling of :mod:`carbidemattergen.co_surface_screen`. That module aggregates
ΔG_CO/ΔG_COOH/ΔG_H descriptors but throws the winning slab/adslab structures
away. For DFT confirmation (Phase A) we need those geometries. This driver:

  1. enumerates terminations up to ``--max-index`` (expanded vs the max_index=1
     production screen), most-stable termination per Miller index,
  2. places *CO/*COOH/*H at EVERY distinct site (no site cap by default) and
     relaxes each adslab with UMA on the GPU,
  3. for the top ``--save-facets`` facets per candidate (ranked by CO-evolution
     activity |ΔG_CO − 0|), writes the relaxed clean slab + best *CO/*COOH/*H
     adslab to ``outputs/qe_surface/targets/<formula>/`` as CIFs, and
  4. emits an expanded per-facet JSONL + a ``dft_targets.csv`` manifest naming
     the facet, site fractional coords and ΔG for every structure handed to QE.

Reuses the validated helpers (terminations, bottom-fix relax, gas references,
ΔG recipe) from co_surface_screen so the numbers match the production screen.

Run inside the GPU fairchem env:
    /home/jonglee69/.venv_fairchem/bin/python scripts/54_co2rr_surface_expand.py \
        --candidates outputs/qe_surface/relaxed_bulk \
        --out outputs/qe_surface/surface_expanded.jsonl \
        --targets outputs/qe_surface/targets --device cuda --max-index 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carbidemattergen.co_surface_screen import (
    CO_OPTIMUM, CO_WINDOW, DG_CORR, PHYS_DG_MAX, FMAX, MAX_STEPS, FIX_FRAC,
    _load_fairchem_calc, _fix_bottom, _relax, _reference_energy,
    _adsorbate_molecule, select_terminations,
)

# Skip pathologically large slabs (high-index cuts of low-symmetry SQS cells can
# explode to hundreds of atoms; UMA still runs but wall-time/VRAM balloon and a
# DFT follow-up would be hopeless). Tunable from the CLI.
MAX_SLAB_ATOMS = 220


def _score_co(g: float) -> float:
    return abs(g - CO_OPTIMUM)


def screen_and_export(structure, calc, formula: str, targets_dir: Path,
                      max_index: int, fmax: float, max_steps: int,
                      site_cap, fix_frac: float, save_facets: int,
                      max_slab_atoms: int, progress) -> dict:
    """Expanded screen for one bulk; export top-facet geometries for DFT."""
    import numpy as np
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write as ase_write

    adsorbates = ("CO", "COOH", "H")
    refs = {a: _reference_energy(calc, a) for a in adsorbates}
    facets = select_terminations(structure, calc, max_index=max_index)
    progress(f"  {len(facets)} terminations: {[m for m, _ in facets]}")

    agg = {a: [] for a in adsorbates}
    facet_reports = []
    # Per-facet winning geometries: facet_best[mi] = {ads: (dG, adslab_atoms,
    # clean_atoms, site_frac)}; plus the facet's best ΔG_CO for ranking.
    facet_best: dict = {}

    for mi, slab in facets:
        if len(slab) > max_slab_atoms:
            progress(f"  facet {mi}: {len(slab)} atoms > cap {max_slab_atoms}, skip")
            continue
        m = slab.lattice.matrix
        area = float(np.linalg.norm(np.cross(m[0], m[1])))  # Å²
        asf = AdsorbateSiteFinder(slab)

        facet_g = {a: [] for a in adsorbates}
        slab_ref_cache: dict[int, tuple] = {}  # natoms -> (E, relaxed_atoms)
        best_here = {}  # ads -> (dG, adslab_atoms, clean_atoms, site_frac)
        slab_conv = False

        for ads in adsorbates:
            ads_structs = asf.generate_adsorption_structures(
                _adsorbate_molecule(ads), find_args={"distance": 1.8})
            if site_cap is not None:
                ads_structs = ads_structs[:site_cap]
            for st in ads_structs:
                clean = st.copy()
                ads_idx = [i for i, s in enumerate(clean)
                           if s.properties.get("surface_properties") == "adsorbate"]
                site_frac = [round(float(x), 4)
                             for x in clean[ads_idx[0]].frac_coords] if ads_idx else None
                clean.remove_sites(ads_idx)
                key = len(clean)
                if key not in slab_ref_cache:
                    cs = AseAtomsAdaptor.get_atoms(clean)
                    _fix_bottom(cs, fix_frac)
                    e_ref, conv_ref, _ = _relax(cs, calc, fmax, max_steps)
                    slab_ref_cache[key] = (e_ref, cs)
                    slab_conv = slab_conv or conv_ref
                e_slab, clean_atoms = slab_ref_cache[key]
                a = AseAtomsAdaptor.get_atoms(st)
                _fix_bottom(a, fix_frac)
                e_ads, _, _ = _relax(a, calc, fmax, max_steps)
                dG = (e_ads - e_slab - refs[ads]) + DG_CORR[ads]
                if abs(dG) > PHYS_DG_MAX:
                    continue
                facet_g[ads].append(dG)
                agg[ads].append(dG)
                better = (ads == "CO" and (ads not in best_here
                          or _score_co(dG) < _score_co(best_here[ads][0]))) or \
                         (ads != "CO" and (ads not in best_here
                          or dG < best_here[ads][0]))
                if better:
                    # ase .copy() drops the attached calculator (its torch
                    # tensors are non-leaf and can't be deepcopied) but keeps
                    # cell/positions/constraints — exactly the geometry we save.
                    best_here[ads] = (round(dG, 4), a.copy(),
                                      clean_atoms.copy(), site_frac)

        co = facet_g["CO"]
        n_sel = sum(1 for g in co if _score_co(g) < CO_WINDOW)
        facet_reports.append({
            "miller": list(mi), "n_sites": len(co), "area_A2": round(area, 2),
            "natoms": len(slab), "slab_converged": slab_conv,
            "best_dG_CO": round(min(co, key=_score_co), 4) if co else None,
            "n_co_selective": n_sel,
            "co_density_per_nm2": round(100.0 * n_sel / area, 4) if area else None,
            **{f"dG_{a}_values": [round(g, 4) for g in facet_g[a]] for a in adsorbates},
        })
        if best_here.get("CO"):
            facet_best[tuple(mi)] = best_here
        bb = round(min(co, key=_score_co), 3) if co else None
        progress(f"  facet {mi}: {len(co)} CO sites, bestΔG_CO={bb}, co_sel={n_sel}")

    # ---- aggregate descriptors (mirror co_surface_screen) ----
    co_all = agg["CO"]
    out = {"formula": formula, "n_facets": len(facet_reports),
           "n_sites": len(co_all), "facets": facet_reports}
    if co_all:
        best_co = min(co_all, key=_score_co)
        out["best_dG_CO"] = round(best_co, 4)
        out["min_abs_dG_CO_opt"] = round(_score_co(best_co), 4)
        out["frac_co_selective"] = round(
            sum(1 for g in co_all if _score_co(g) < CO_WINDOW) / len(co_all), 4)
        cooh_all, h_all = agg["COOH"], agg["H"]
        if cooh_all:
            best_cooh = min(cooh_all)
            out["min_dG_COOH"] = round(best_cooh, 4)
            out["limiting_potential_V"] = round(-max(best_cooh, best_co - best_cooh), 4)
        if cooh_all and h_all:
            out["min_dG_H"] = round(min(h_all), 4)
            out["co_selective_over_h"] = bool(min(cooh_all) < min(h_all))

    # ---- export top-facet geometries for DFT ----
    ranked = sorted(facet_best.items(),
                    key=lambda kv: _score_co(kv[1]["CO"][0]))[:save_facets]
    cdir = targets_dir / formula
    cdir.mkdir(parents=True, exist_ok=True)
    exported = []
    for mi, best in ranked:
        tag = "".join(str(i) for i in mi)
        for ads, (dG, adslab_atoms, clean_atoms, site_frac) in best.items():
            cl = cdir / f"facet{tag}_{ads}_clean.cif"
            ad = cdir / f"facet{tag}_{ads}_ads.cif"
            ase_write(str(cl), clean_atoms)
            ase_write(str(ad), adslab_atoms)
            exported.append({"formula": formula, "facet": tag, "adsorbate": ads,
                             "dG_uma": dG, "natoms_ads": len(adslab_atoms),
                             "natoms_clean": len(clean_atoms),
                             "site_frac": site_frac,
                             "clean_cif": str(cl), "ads_cif": str(ad)})
    out["exported_targets"] = exported
    progress(f"  exported {len(exported)} DFT target structures over "
             f"{len(ranked)} facets -> {cdir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded CO2RR surface screen + geometry export (GPU)")
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path, help="dir for exported DFT slab CIFs")
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--task", default="omat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-index", type=int, default=2)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--site-cap", type=int, default=None)
    ap.add_argument("--fix-frac", type=float, default=FIX_FRAC)
    ap.add_argument("--save-facets", type=int, default=2,
                    help="top facets (by |ΔG_CO|) to export geometries for")
    ap.add_argument("--max-slab-atoms", type=int, default=MAX_SLAB_ATOMS)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure

    done = set()
    stem = args.out.name.split(".")[0]
    for jf in args.out.parent.glob(f"{stem}*.jsonl"):
        for line in jf.read_text().splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:
                pass

    calc = _load_fairchem_calc(args.model, args.device, args.task)
    files = sorted([*args.candidates.glob("*.cif"), *args.candidates.glob("*POSCAR*")])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.targets.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.out, "a") as fout:
        for fp in files:
            if fp.name in done:
                print(f"[skip] {fp.name}", flush=True)
                continue
            formula = fp.stem
            print(f"[run ] {fp.name}", flush=True)
            try:
                s = Structure.from_file(str(fp))
                res = screen_and_export(
                    s, calc, formula, args.targets, args.max_index, args.fmax,
                    args.max_steps, args.site_cap, args.fix_frac, args.save_facets,
                    args.max_slab_atoms, progress=lambda m: print(m, flush=True))
            except Exception as exc:
                import traceback
                res = {"formula": formula, "error": str(exc),
                       "trace": traceback.format_exc()}
            res["source"] = fp.name
            fout.write(json.dumps(res) + "\n"); fout.flush()
            rows.extend(res.get("exported_targets", []))
            print(f"[done] {fp.name}: bestΔG_CO={res.get('best_dG_CO')} "
                  f"U_L={res.get('limiting_potential_V')} "
                  f"targets={len(res.get('exported_targets', []))}", flush=True)

    # write/update the DFT target manifest
    if rows:
        import csv
        man = args.targets / "dft_targets.csv"
        new = not man.exists()
        with open(man, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {len(rows)} DFT targets -> {man}", flush=True)


if __name__ == "__main__":
    main()
