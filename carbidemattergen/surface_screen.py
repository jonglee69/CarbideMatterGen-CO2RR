"""Stage B — surface ΔG_H* evaluation with a fairchem v2 (UMA) surrogate.

This is where the bulk→surface gap is closed. Stage A (the fine-tuned
MatterGen) generates bulk carbide crystals biased toward stable, metallic,
near-thermoneutral compositions; Stage A' relaxes the top candidates into SQS
solid-solution supercells. For each SQS structure this module:

  1. enumerates low-index surface terminations (pymatgen SlabGenerator),
  2. selects the most stable termination per Miller index (single-point
     ranking — an SQS has no symmetry, so SlabGenerator returns one slab per
     random surface decoration; we keep the lowest-energy cut per plane),
  3. relaxes each chosen slab with the bottom layers fixed (a standard
     surface-science constraint that mimics the frozen bulk),
  4. places a single H* at every symmetry-distinct adsorption site,
  5. relaxes each adslab (same bottom-fix constraint),
  6. computes ΔG_H* = E(slab+H) − E(slab) − 1/2 E(H2) + ΔE_ZPE − TΔS_H,
  7. reports per-candidate min |ΔG_H*|, the fraction and areal density of
     near-thermoneutral sites, and a per-facet breakdown — the real HER
     descriptors the composition proxies only approximate.

Model weights are NOT bundled. This uses fairchem v2 (>=2.x) with the Universal
Model for Atoms (UMA). The UMA checkpoints live in the gated HuggingFace repo
``facebook/UMA`` — request access there and authenticate (``hf auth login`` or
``HF_TOKEN``) before first use. We use the ``oc20`` task head (heterogeneous
catalysis: slab + adsorbate), which predicts RPBE-style total energies, so
E(slab+H), E(slab) and 1/2 E(H2) sit on a single reference and subtract
consistently.

Standard gas-phase corrections for atomic H adsorption (eV):
    ΔE_ZPE − TΔS_H ≈ +0.24    (Nørskov et al., J. Electrochem. Soc. 2005)
so ΔG_H* ≈ ΔE_H* + 0.24 .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Gas-phase free-energy correction added to the electronic adsorption energy.
DELTA_G_CORRECTION = 0.24  # eV
THERMONEUTRAL_WINDOW = 0.10  # eV; |ΔG_H*| below this counts as an active site.

# Relaxation defaults (overridable from the CLI).
FMAX = 0.05          # eV/Å force convergence
MAX_STEPS = 150      # LBFGS step cap per relaxation
FIX_FRAC = 0.5       # fix the bottom this fraction of the slab thickness


def _load_fairchem_calc(model: str = "uma-s-1p1", device: str = "cpu",
                        task_name: str = "oc20"):
    """Return a fairchem v2 ASE calculator (UMA model, given task head).

    v2 replaced the v1 ``OCPCalculator(checkpoint_path=...)`` with a two-step
    construction: a device-bound predict unit fetched by model name, wrapped in
    a ``FAIRChemCalculator`` pinned to one task. The model is pulled from the
    gated ``facebook/UMA`` HF repo on first use (auth required).

        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
        calc = FAIRChemCalculator(predictor, task_name="oc20")
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
    converged = bool(dyn.converged())
    return atoms.get_potential_energy(), converged, dyn.get_number_of_steps()


def _half_h2_energy(calc):
    """1/2 E(H2) gas reference, relaxed once and cached on the calc."""
    e = getattr(calc, "_e_half_h2", None)
    if e is not None:
        return e
    from ase import Atoms
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.center(vacuum=8.0)
    h2.pbc = True
    e_h2, _, _ = _relax(h2, calc, fmax=0.01, max_steps=200)
    calc._e_half_h2 = 0.5 * e_h2
    return calc._e_half_h2


def select_terminations(structure, calc, max_index: int = 1,
                        min_slab: float = 8.0, min_vac: float = 12.0):
    """Most stable termination per Miller index (as-cut single-point ranking).

    SlabGenerator on an SQS supercell returns one slab per surface decoration
    (no symmetry to fold them together). Relaxing every one is wasteful, so we
    rank the as-cut slabs by single-point energy/atom and keep the lowest per
    Miller plane — the cut the real surface would relax/segregate toward.
    """
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
                     progress=None) -> dict:
    """Compute ΔG_H* over all sites of the chosen terminations for one bulk.

    Returns a rich summary: per-candidate min |ΔG_H*|, thermoneutral fraction
    and areal density, plus a per-facet breakdown. ``progress`` (if given) is
    called with short status strings for live logging.
    """
    from pymatgen.core import Molecule
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor

    e_half_h2 = _half_h2_energy(calc)
    facets = select_terminations(structure, calc, max_index=max_index)
    if progress:
        progress(f"  {len(facets)} terminations: {[m for m, _ in facets]}")

    all_g: list[float] = []
    facet_reports = []
    for mi, slab in facets:
        # Surface area of the slab (a×b of the slab cell, ⟂ to c).
        m = slab.lattice.matrix
        import numpy as np
        area = float(np.linalg.norm(np.cross(m[0], m[1])))  # Å²

        slab_atoms = AseAtomsAdaptor.get_atoms(slab)
        _fix_bottom(slab_atoms, fix_frac)
        e_slab, slab_conv, _ = _relax(slab_atoms, calc, fmax, max_steps)

        ads_structs = AdsorbateSiteFinder(slab).generate_adsorption_structures(
            Molecule(["H"], [[0, 0, 0]]), find_args={"distance": 1.5})
        if site_cap is not None:
            ads_structs = ads_structs[:site_cap]

        g_facet: list[float] = []
        for ads in ads_structs:
            ads_atoms = AseAtomsAdaptor.get_atoms(ads)
            _fix_bottom(ads_atoms, fix_frac)
            e_ads, _, _ = _relax(ads_atoms, calc, fmax, max_steps)
            g = (e_ads - e_slab - e_half_h2) + DELTA_G_CORRECTION
            g_facet.append(g)
        all_g.extend(g_facet)

        n_thermo = sum(1 for g in g_facet if abs(g) < THERMONEUTRAL_WINDOW)
        facet_reports.append({
            "miller": list(mi),
            "n_sites": len(g_facet),
            "area_A2": round(area, 2),
            "slab_converged": slab_conv,
            "min_abs_dG_H": round(min(g_facet, key=abs), 4) if g_facet else None,
            "n_thermoneutral": n_thermo,
            "thermo_density_per_nm2": round(100.0 * n_thermo / area, 4) if area else None,
            "dG_H_values": [round(g, 4) for g in g_facet],
        })
        if progress:
            mm = round(min(g_facet, key=abs), 3) if g_facet else None
            progress(f"  facet {mi}: {len(g_facet)} sites, min|ΔG|={mm}, thermo={n_thermo}")

    if not all_g:
        return {"n_sites": 0, "min_abs_dG_H": None, "frac_thermoneutral": None,
                "facets": facet_reports}
    abs_min = min(all_g, key=abs)
    n_thermo = sum(1 for g in all_g if abs(g) < THERMONEUTRAL_WINDOW)
    return {
        "n_facets": len(facets),
        "n_sites": len(all_g),
        "min_abs_dG_H": round(abs_min, 4),
        "n_thermoneutral": n_thermo,
        "frac_thermoneutral": round(n_thermo / len(all_g), 4),
        "facets": facet_reports,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="fairchem v2 (UMA) ΔG_H* surface screen")
    ap.add_argument("--candidates", required=True, type=Path,
                    help="dir of relaxed candidate structures (CIF/POSCAR)")
    ap.add_argument("--model", default="uma-s-1p1",
                    help="fairchem v2 UMA model name (gated HF repo facebook/UMA)")
    ap.add_argument("--task", default="oc20",
                    help="UMA task head (oc20 = slab+adsorbate catalysis)")
    ap.add_argument("--out", required=True, type=Path, help="output JSONL")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-index", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--site-cap", type=int, default=None,
                    help="cap sites per facet (default: all distinct sites)")
    ap.add_argument("--shard", default=None,
                    help="process shard i of N, e.g. '0/4' (candidates split by "
                         "index % N == i for parallel workers)")
    args = ap.parse_args()

    from pymatgen.core import Structure

    # Resume: skip candidates already present in ANY sibling output file
    # (so parallel shards and earlier single-worker runs don't recompute).
    done: set[str] = set()
    stem = args.out.name.split(".")[0]
    for jf in args.out.parent.glob(f"{stem}*.jsonl"):
        for line in jf.read_text().splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:  # noqa: BLE001
                pass

    calc = _load_fairchem_calc(args.model, args.device, args.task)

    files = sorted([*args.candidates.glob("*.cif"), *args.candidates.glob("*POSCAR*")])
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = [f for idx, f in enumerate(files) if idx % n == i]
        print(f"[shard {i}/{n}] {len(files)} candidates: {[f.name for f in files]}", flush=True)
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
                                       site_cap=args.site_cap,
                                       progress=lambda m: print(m, flush=True))
            except Exception as exc:  # noqa: BLE001
                import traceback
                res = {"error": str(exc), "trace": traceback.format_exc()}
            res["source"] = fp.name
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            print(f"[done] {fp.name}: min|ΔG_H|={res.get('min_abs_dG_H')} "
                  f"thermo={res.get('n_thermoneutral')}/{res.get('n_sites')}", flush=True)


if __name__ == "__main__":
    main()
