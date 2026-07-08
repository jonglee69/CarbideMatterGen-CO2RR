#!/usr/bin/env python
"""Export *OH/*O adslab geometries for DFT confirmation of the poisoning screen.

Sibling of 54_co2rr_surface_expand but for the surface-Pourbaix adsorbates. For
each candidate it screens *OH and *O over the low-index facets (UMA), then exports
the single most-poisoning facet (strongest *OH binding) as clean-slab + best-*OH +
best-*O CIFs, plus a dft_targets_oh.csv manifest that 55_qe_surface_inputs.py
consumes unchanged. Kept deliberately small (1 facet/candidate) after the last
session's resource-pressure crash.

    /home/jonglee69/.venv_fairchem/bin/python scripts/62_oh_surface_expand.py \
        --candidates outputs/qe_surface/relaxed_bulk \
        --targets outputs/qe_surface/targets_oh --device cuda --max-index 1
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from carbidemattergen.co_surface_screen import (
    _load_fairchem_calc, select_terminations, _fix_bottom, _relax,
    _adsorbate_molecule, _reference_energy, DG_CORR, PHYS_DG_MAX, FMAX, MAX_STEPS)

ADS = ("OH", "O")

# activity (U_L) facet per candidate (tag = "".join(miller)); used with
# --activity-facets so *O/*OH land on the SAME facet as the CO/COOH/H main DFT,
# giving a fully consistent same-facet surface Pourbaix.
ACTIVITY_FACETS = {
    "Ce2Y2ZrWC6": "101", "CeY2ZrWC7": "011", "La3Y5(WC3)3": "101",
    "La5Y4W2C9": "111", "Pr3Y2WC6": "010", "Y2ZrWC8": "100",
}


def export_candidate(structure, calc, formula, targets_dir, max_index,
                     fmax, max_steps, site_cap, progress, target_tag=None):
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write as ase_write

    refs = {a: _reference_energy(calc, a) for a in ADS}
    facets = select_terminations(structure, calc, max_index=max_index)
    progress(f"  {len(facets)} terminations: {[m for m, _ in facets]}")

    if target_tag is not None:
        facets = [(mi, s) for mi, s in facets
                  if "".join(str(i) for i in mi) == target_tag]
        if not facets:
            progress(f"  [warn] target facet {target_tag} not found -- skip"); return []
        progress(f"  targeting activity facet {target_tag}")

    facet_best = {}  # mi -> {ads: (dG, adslab_atoms, clean_atoms, site_frac)}
    for mi, slab in facets:
        asf = AdsorbateSiteFinder(slab)
        slab_ref_cache = {}
        best_here = {}
        for ads in ADS:
            structs = asf.generate_adsorption_structures(
                _adsorbate_molecule(ads), find_args={"distance": 1.8})
            if site_cap:
                structs = structs[:site_cap]
            for st in structs:
                clean = st.copy()
                idx = [i for i, s in enumerate(clean)
                       if s.properties.get("surface_properties") == "adsorbate"]
                sfrac = [round(float(x), 4) for x in clean[idx[0]].frac_coords] if idx else None
                clean.remove_sites(idx)
                key = len(clean)
                if key not in slab_ref_cache:
                    cs = AseAtomsAdaptor.get_atoms(clean)
                    _fix_bottom(cs, 0.5)
                    e_ref, _, _ = _relax(cs, calc, fmax, max_steps)
                    slab_ref_cache[key] = (e_ref, cs)
                e_slab, clean_atoms = slab_ref_cache[key]
                a = AseAtomsAdaptor.get_atoms(st)
                _fix_bottom(a, 0.5)
                e_ads, _, _ = _relax(a, calc, fmax, max_steps)
                dG = (e_ads - e_slab - refs[ads]) + DG_CORR[ads]
                if abs(dG) > PHYS_DG_MAX:
                    continue
                if ads not in best_here or dG < best_here[ads][0]:  # strongest binding
                    best_here[ads] = (round(dG, 4), a.copy(), clean_atoms.copy(), sfrac)
        if "OH" in best_here:
            facet_best[tuple(mi)] = best_here
            progress(f"  facet {mi}: best dG_OH={best_here['OH'][0]} "
                     f"dG_O={best_here.get('O', [None])[0]}")

    if not facet_best:
        progress("  no facet produced *OH -- skip"); return []
    # most-poisoning facet = strongest *OH binding
    mi, best = min(facet_best.items(), key=lambda kv: kv[1]["OH"][0])
    tag = "".join(str(i) for i in mi)
    cdir = targets_dir / formula; cdir.mkdir(parents=True, exist_ok=True)
    exported = []
    for ads, (dG, adslab_atoms, clean_atoms, sfrac) in best.items():
        cl = cdir / f"facet{tag}_{ads}_clean.cif"
        ad = cdir / f"facet{tag}_{ads}_ads.cif"
        ase_write(str(cl), clean_atoms); ase_write(str(ad), adslab_atoms)
        exported.append({"formula": formula, "facet": tag, "adsorbate": ads,
                         "dG_uma": dG, "natoms_ads": len(adslab_atoms),
                         "natoms_clean": len(clean_atoms), "site_frac": sfrac,
                         "clean_cif": str(cl), "ads_cif": str(ad)})
    progress(f"  exported facet{tag}: {[e['adsorbate'] for e in exported]} -> {cdir}")
    return exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-index", type=int, default=1)
    ap.add_argument("--site-cap", type=int, default=None)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--activity-facets", action="store_true",
                    help="target each candidate's U_L (activity) facet (ACTIVITY_FACETS) "
                         "for a same-facet Pourbaix consistent with the main DFT")
    args = ap.parse_args()

    from pymatgen.core import Structure
    calc = _load_fairchem_calc(args.model, args.device, "omat")
    args.targets.mkdir(parents=True, exist_ok=True)
    files = sorted([*args.candidates.glob("*.cif"), *args.candidates.glob("*POSCAR*")])
    all_rows = []
    for fp in files:
        print(f"[run ] {fp.name}", flush=True)
        s = Structure.from_file(str(fp))
        tgt = ACTIVITY_FACETS.get(fp.stem) if args.activity_facets else None
        rows = export_candidate(s, calc, fp.stem, args.targets, args.max_index,
                                args.fmax, args.max_steps, args.site_cap,
                                progress=lambda m: print(m, flush=True), target_tag=tgt)
        all_rows.extend(rows)
    man = args.targets / "dft_targets_oh.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {man} ({len(all_rows)} targets)")


if __name__ == "__main__":
    main()
