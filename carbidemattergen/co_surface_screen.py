"""Stage B (CO2RR) — surface ΔG_CO* / ΔG_*COOH evaluation with a fairchem v2
(UMA) surrogate. The CO2RR-to-CO sibling of
:mod:`carbidemattergen.surface_screen`.

This closes the bulk→surface gap for the CO2RR pipeline. Stage A (the fine-tuned
MatterGen) generates bulk carbide crystals biased toward stable, metallic,
CO-releasing, economically-clean compositions; Stage A' relaxes the top
candidates into SQS solid-solution supercells. For each structure this module:

  1. enumerates low-index surface terminations (pymatgen SlabGenerator),
  2. selects the most stable termination per Miller index (single-point
     ranking — an SQS has no symmetry, so we keep the lowest-energy cut/plane),
  3. relaxes each chosen slab with the bottom layers fixed (frozen-bulk mimic),
  4. places each adsorbate (*CO, *COOH, *H) at every symmetry-distinct site,
  5. relaxes each adslab (same bottom-fix constraint),
  6. computes the CO2RR free-energy descriptors under the computational
     hydrogen electrode (CHE, U=0 vs RHE), and
  7. reports per-candidate CO-evolution activity AND CO-vs-H2 selectivity, with
     a per-facet breakdown — the real descriptors the composition proxies only
     approximate.

Free energies (computational hydrogen electrode; Peterson 2010, Nørskov 2004):

    ΔG_CO*   = E(*CO)   − E(*) − E(CO_g)                 + dG_CO
    ΔG_COOH* = E(*COOH) − E(*) − E(CO2_g) − 1/2 E(H2_g)  + dG_COOH
    ΔG_H*    = E(*H)    − E(*) − 1/2 E(H2_g)             + dG_H

with (H+ + e-) referenced to 1/2 H2(g) at U=0. Gas references E(CO_g), E(CO2_g),
E(H2_g) are computed once with the SAME UMA calculator (cancels systematic
error — the omat task head predicts total energies on one consistent reference).

Gas-phase free-energy corrections Δ(ZPE − TΔS) (eV; Peterson 2010 / Nørskov 2004):

    dG_CO   ≈ +0.10      (*CO vs CO(g))
    dG_COOH ≈ +0.41      (*COOH vs CO2(g)+1/2 H2)
    dG_H    ≈ +0.24      (*H  vs 1/2 H2; same as HER)

Interpretation
--------------
* CO production wants ΔG_CO ≈ 0 (slightly positive): CO desorbs as product
  rather than poisoning / over-reducing. ``CO_OPTIMUM`` / ``CO_WINDOW`` below.
* CO2 activation onset is set by ΔG_COOH (usually the limiting step): the
  two-step limiting potential is U_L = −max(ΔG_COOH, ΔG_CO − ΔG_COOH).
* CO-vs-H2 selectivity: a site is CO-selective when ΔG_COOH < ΔG_H (CO2RR
  thermodynamically out-competes H adsorption at that site).

Model weights are NOT bundled. This uses fairchem v2 (>=2.x) with the Universal
Model for Atoms (UMA). UMA checkpoints live in the gated HuggingFace repo
``facebook/UMA`` — request access there and authenticate (``hf auth login`` or
``HF_TOKEN``) before first use. We use the ``omat`` task head (slab+adsorbate
catalysis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# CO-evolution Sabatier optimum and selectivity window (eV); mirror co2rr.py.
CO_OPTIMUM = 0.0
CO_WINDOW = 0.20  # |ΔG_CO − CO_OPTIMUM| below this counts as a CO-releasing site.

# Gas-phase free-energy corrections (eV) added to the electronic ΔE.
DG_CORR = {"CO": 0.10, "COOH": 0.41, "H": 0.24, "OH": 0.30, "O": 0.05}

# Physical sanity bound on an adsorption ΔG (eV). Freshly-cut, low-symmetry
# carbide slabs (especially REE-bearing) sometimes fail to relax within the step
# cap and reconstruct, giving a divergent clean-slab reference that offsets every
# site on that facet by tens-to-hundreds of eV. Such values are unphysical for a
# single adsorbate; drop them from the aggregation so they cannot corrupt the
# best/min descriptors. Genuine physisorption/chemisorption sits well inside this.
PHYS_DG_MAX = 10.0

# Relaxation defaults (overridable from the CLI).
FMAX = 0.05          # eV/Å force convergence
MAX_STEPS = 150      # LBFGS step cap per relaxation
FIX_FRAC = 0.5       # fix the bottom this fraction of the slab thickness


def _load_fairchem_calc(model: str = "uma-s-1p1", device: str = "cpu",
                        task_name: str = "omat"):
    """Return a fairchem v2 ASE calculator (UMA model, given task head).

        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
        calc = FAIRChemCalculator(predictor, task_name="omat")
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _fix_bottom(atoms, frac: float = FIX_FRAC):
    """Constrain the bottom ``frac`` of the slab thickness (FixAtoms)."""
    from ase.constraints import FixAtoms
    import numpy as np

    z = atoms.positions[:, 2]
    z_lo, z_hi = z.min(), z.max()
    cutoff = z_lo + frac * (z_hi - z_lo)
    mask = z <= cutoff
    atoms.set_constraint(FixAtoms(mask=np.asarray(mask)))
    return int(mask.sum())


def _relax(atoms, calc, fmax: float = FMAX, max_steps: int = MAX_STEPS):
    """Relax ``atoms`` in place; return (energy, converged, nsteps)."""
    from ase.optimize import LBFGS

    atoms.calc = calc
    dyn = LBFGS(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=max_steps)
    return atoms.get_potential_energy(), bool(dyn.converged()), dyn.get_number_of_steps()


# Reference gas molecules (E computed once per process, cached on the calc).
_GAS_GEOM = {
    "H2":  ("H2",  [[0, 0, 0], [0, 0, 0.74]]),
    "CO":  ("CO",  [[0, 0, 0], [0, 0, 1.128]]),
    "CO2": ("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]),
    "H2O": ("H2O", [[0, 0, 0], [0.7575, 0.5865, 0], [-0.7575, 0.5865, 0]]),
}


def _gas_energy(calc, name: str) -> float:
    """Relaxed gas-phase energy of a reference molecule, cached on ``calc``."""
    from ase import Atoms

    cache = getattr(calc, "_gas_energies", None)
    if cache is None:
        cache = {}
        calc._gas_energies = cache
    if name in cache:
        return cache[name]
    sym, pos = _GAS_GEOM[name]
    atoms = Atoms(sym, positions=pos)
    atoms.center(vacuum=8.0)
    atoms.pbc = True
    e, _, _ = _relax(atoms, calc, fmax=0.01, max_steps=200)
    cache[name] = e
    return e


def _reference_energy(calc, adsorbate: str) -> float:
    """Gas-phase reference subtracted for one adsorbate (CHE, U=0)."""
    if adsorbate == "CO":
        return _gas_energy(calc, "CO")
    if adsorbate == "COOH":
        return _gas_energy(calc, "CO2") + 0.5 * _gas_energy(calc, "H2")
    if adsorbate == "H":
        return 0.5 * _gas_energy(calc, "H2")
    if adsorbate == "OH":  # *OH vs H2O(l≈g) - 1/2 H2 (CHE)
        return _gas_energy(calc, "H2O") - 0.5 * _gas_energy(calc, "H2")
    if adsorbate == "O":   # *O vs H2O - H2
        return _gas_energy(calc, "H2O") - _gas_energy(calc, "H2")
    raise ValueError(adsorbate)


def _adsorbate_molecule(name: str):
    """pymatgen Molecule for the adsorbate; surface-binding atom placed FIRST.

    The AdsorbateSiteFinder anchors the molecule by its lowest atom (C for
    *CO/*COOH, H for *H). Geometries are reasonable guesses; LBFGS relaxes them.
    """
    from pymatgen.core import Molecule
    if name == "H":
        return Molecule(["H"], [[0.0, 0.0, 0.0]])
    if name == "O":
        return Molecule(["O"], [[0.0, 0.0, 0.0]])
    if name == "OH":  # O binds to the surface, H points up
        return Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    if name == "CO":
        return Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
    if name == "COOH":
        return Molecule(
            ["C", "O", "O", "H"],
            [[0.00, 0.00, 0.00],
             [1.05, 0.00, 0.62],     # =O
             [-0.55, 0.95, 0.62],    # –O(H)
             [-0.20, 1.55, 1.25]],   # H on hydroxyl O
        )
    raise ValueError(f"unknown adsorbate: {name}")


def select_terminations(structure, calc, max_index: int = 1,
                        min_slab: float = 8.0, min_vac: float = 12.0):
    """Most stable termination per Miller index (as-cut single-point ranking)."""
    from pymatgen.core.surface import generate_all_slabs
    from pymatgen.io.ase import AseAtomsAdaptor

    slabs = generate_all_slabs(structure, max_index=max_index,
                               min_slab_size=min_slab, min_vacuum_size=min_vac,
                               center_slab=True)
    best: dict = {}
    for slab in slabs:
        atoms = AseAtomsAdaptor.get_atoms(slab)
        atoms.calc = calc
        e_per_atom = atoms.get_potential_energy() / len(atoms)
        mi = tuple(slab.miller_index)
        if mi not in best or e_per_atom < best[mi][0]:
            best[mi] = (e_per_atom, slab)
    return [(mi, slab) for mi, (_, slab) in sorted(best.items())]


def screen_candidate(structure, calc, max_index: int = 1,
                     fmax: float = FMAX, max_steps: int = MAX_STEPS,
                     site_cap: int | None = None, fix_frac: float = FIX_FRAC,
                     adsorbates: tuple[str, ...] = ("CO", "COOH", "H"),
                     progress=None) -> dict:
    """Compute the CO2RR-to-CO descriptors over all sites/terminations of one bulk.

    For every chosen termination each adsorbate is placed at the SAME
    enumerated sites, so per-site CO/COOH/H values are directly comparable. The
    clean slab is relaxed once per facet and reused for all three adsorbates.

    Returns a rich summary: best_dG_CO (closest to optimum), frac_co_selective,
    min_dG_COOH (CO2-activation onset proxy), limiting_potential_V,
    co_selective_over_h, plus a per-facet breakdown.
    """
    import numpy as np
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor

    refs = {a: _reference_energy(calc, a) for a in adsorbates}
    facets = select_terminations(structure, calc, max_index=max_index)
    if progress:
        progress(f"  {len(facets)} terminations: {[m for m, _ in facets]}")

    agg: dict[str, list[float]] = {a: [] for a in adsorbates}
    facet_reports = []
    for mi, slab in facets:
        m = slab.lattice.matrix
        area = float(np.linalg.norm(np.cross(m[0], m[1])))  # Å²

        asf = AdsorbateSiteFinder(slab)
        # The clean-slab reference MUST match EACH adslab's supercell. pymatgen
        # replicates the slab to space periodic adsorbate images, and the
        # replication can differ per adsorbate (a larger molecule like *COOH may
        # force a bigger supercell than *CO/*H). A bare-slab energy on the wrong
        # cell does NOT cancel — it leaves a slab-size-proportional offset (the
        # original bug: ΔG ~ -50..-400 eV scaling with slab size). So strip the
        # adsorbate from each adslab and relax THAT supercell as the reference,
        # caching by supercell atom count to avoid recomputation within a facet.
        facet_g: dict[str, list[float]] = {}
        slab_ref_cache: dict[int, float] = {}
        slab_conv = False
        for ads in adsorbates:
            ads_structs = asf.generate_adsorption_structures(
                _adsorbate_molecule(ads), find_args={"distance": 1.8})
            if site_cap is not None:
                ads_structs = ads_structs[:site_cap]
            vals: list[float] = []
            for st in ads_structs:
                clean = st.copy()
                clean.remove_sites([i for i, site in enumerate(clean)
                                    if site.properties.get("surface_properties") == "adsorbate"])
                key = len(clean)
                if key not in slab_ref_cache:
                    cs = AseAtomsAdaptor.get_atoms(clean)
                    _fix_bottom(cs, fix_frac)
                    e_ref, conv_ref, _ = _relax(cs, calc, fmax, max_steps)
                    slab_ref_cache[key] = e_ref
                    slab_conv = slab_conv or conv_ref
                e_slab = slab_ref_cache[key]
                a = AseAtomsAdaptor.get_atoms(st)
                _fix_bottom(a, fix_frac)
                e_ads, _, _ = _relax(a, calc, fmax, max_steps)
                dG = (e_ads - e_slab - refs[ads]) + DG_CORR[ads]
                if abs(dG) <= PHYS_DG_MAX:  # drop divergent/unphysical sites
                    vals.append(dG)
            facet_g[ads] = vals
            agg[ads].extend(vals)

        co = facet_g.get("CO", [])
        n_sel = sum(1 for g in co if abs(g - CO_OPTIMUM) < CO_WINDOW)
        facet_reports.append({
            "miller": list(mi),
            "n_sites": len(co),
            "area_A2": round(area, 2),
            "slab_converged": slab_conv,
            "best_dG_CO": round(min(co, key=lambda g: abs(g - CO_OPTIMUM)), 4) if co else None,
            "n_co_selective": n_sel,
            "co_density_per_nm2": round(100.0 * n_sel / area, 4) if area else None,
            **{f"dG_{a}_values": [round(g, 4) for g in facet_g.get(a, [])]
               for a in adsorbates},
        })
        if progress:
            bb = round(min(co, key=lambda g: abs(g - CO_OPTIMUM)), 3) if co else None
            progress(f"  facet {mi}: {len(co)} sites, bestΔG_CO={bb}, co_sel={n_sel}")

    co_all = agg.get("CO", [])
    out: dict = {"n_facets": len(facets), "n_sites": len(co_all),
                 "facets": facet_reports}
    if not co_all:
        out.update({"best_dG_CO": None, "frac_co_selective": None,
                    "min_dG_COOH": None, "limiting_potential_V": None,
                    "co_selective_over_h": None})
        return out

    best_co = min(co_all, key=lambda g: abs(g - CO_OPTIMUM))
    out["best_dG_CO"] = round(best_co, 4)
    out["min_abs_dG_CO_opt"] = round(abs(best_co - CO_OPTIMUM), 4)
    out["frac_co_selective"] = round(
        sum(1 for g in co_all if abs(g - CO_OPTIMUM) < CO_WINDOW) / len(co_all), 4)

    cooh_all = agg.get("COOH", [])
    if cooh_all:
        best_cooh = min(cooh_all)
        out["min_dG_COOH"] = round(best_cooh, 4)
        step2 = best_co - best_cooh
        out["limiting_potential_V"] = round(-max(best_cooh, step2), 4)
    else:
        out["min_dG_COOH"] = out["limiting_potential_V"] = None

    h_all = agg.get("H", [])
    if cooh_all and h_all:
        out["min_dG_H"] = round(min(h_all), 4)
        out["co_selective_over_h"] = bool(min(cooh_all) < min(h_all))
    else:
        out["co_selective_over_h"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="fairchem v2 (UMA) CO2RR ΔG_CO*/ΔG_COOH* screen")
    ap.add_argument("--candidates", required=True, type=Path,
                    help="dir of relaxed candidate structures (CIF/POSCAR)")
    ap.add_argument("--model", default="uma-s-1p1",
                    help="fairchem v2 UMA model name (gated HF repo facebook/UMA)")
    ap.add_argument("--task", default="omat",
                    help="UMA task head. Use 'omat' (total energies, default): the "
                         "ΔG recipe subtracts clean-slab + gas references, which only "
                         "cancel with a consistent total-energy head. 'oc20' returns "
                         "internally-referenced adsorption energies that do NOT cancel "
                         "here and give unphysical ΔG.")
    ap.add_argument("--out", required=True, type=Path, help="output JSONL")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-index", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--site-cap", type=int, default=None,
                    help="cap sites per facet (default: all distinct sites)")
    ap.add_argument("--adsorbates", default="CO,COOH,H",
                    help="comma list from CO,COOH,H,OH,O (default CO,COOH,H). "
                         "OH/O enable the surface-Pourbaix poisoning screen.")
    ap.add_argument("--shard", default=None,
                    help="process shard i of N, e.g. '0/4' (candidates split by "
                         "index % N == i for parallel workers)")
    args = ap.parse_args()

    from pymatgen.core import Structure

    # Resume: skip candidates already present in ANY sibling output file.
    done: set[str] = set()
    stem = args.out.name.split(".")[0]
    for jf in args.out.parent.glob(f"{stem}*.jsonl"):
        for line in jf.read_text().splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:  # noqa: BLE001
                pass

    calc = _load_fairchem_calc(args.model, args.device, args.task)
    adsorbates = tuple(a.strip() for a in args.adsorbates.split(",") if a.strip())

    files = sorted([*args.candidates.glob("*.cif"), *args.candidates.glob("*POSCAR*")])
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = [f for idx, f in enumerate(files) if idx % n == i]
        print(f"[shard {i}/{n}] {len(files)} candidates", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fout:
        for fp in files:
            if fp.name in done:
                print(f"[skip] {fp.name} (already done)", flush=True)
                continue
            print(f"[run ] {fp.name}", flush=True)
            try:
                s = Structure.from_file(str(fp))
                res = screen_candidate(s, calc, max_index=args.max_index,
                                       fmax=args.fmax, max_steps=args.max_steps,
                                       site_cap=args.site_cap, adsorbates=adsorbates,
                                       progress=lambda m: print(m, flush=True))
            except Exception as exc:  # noqa: BLE001
                import traceback
                res = {"error": str(exc), "trace": traceback.format_exc()}
            res["source"] = fp.name
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            print(f"[done] {fp.name}: bestΔG_CO={res.get('best_dG_CO')} "
                  f"co_sel={res.get('frac_co_selective')} "
                  f"U_L={res.get('limiting_potential_V')}", flush=True)


if __name__ == "__main__":
    main()
