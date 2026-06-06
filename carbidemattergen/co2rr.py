"""Composition-level descriptors that correlate with the electrochemical CO2
reduction reaction (CO2RR) toward **CO** on transition-metal carbides.

This is the CO2RR sibling of :mod:`carbidemattergen.hydrogen`. It reuses the
exact same machinery — cheap, composition-only *proxies* that recompute
identically for any generated candidate (no DFT, no surface slab) and serve as
conditioning labels during fine-tuning — but swaps the reaction descriptor from
ΔG_H* (HER) to the CO2RR-to-CO descriptors. The real values are evaluated later
in Stage B (:mod:`carbidemattergen.co_surface_screen`) with a fairchem v2 (UMA)
model and the proxy tables are recalibrated against those + DFT in the
active-learning loop.

The CO2RR → CO mechanism (Hori; Peterson 2010; Bagger/Nørskov 2017)
-------------------------------------------------------------------
    CO2(g) + * + (H+ + e-)  ->  *COOH                 (CO2 activation)
    *COOH    + (H+ + e-)    ->  *CO + H2O             (second PCET)
    *CO                     ->  CO(g) + *             (desorption)

Two things must be true for a selective CO producer:

  1. *CO must bind **weakly enough to desorb** as CO. If *CO binds too strongly
     the surface either poisons or over-reduces CO to hydrocarbons/oxygenates
     (the Cu pathway). The Sabatier optimum for *CO production therefore sits at
     ΔG_CO ≈ 0 (slightly positive) — exactly why Au/Ag/Zn are the canonical CO
     producers and Pt/Ni/Fe are not.
  2. CO2RR must out-compete HER. The selectivity-determining contrast is
     *COOH vs *H binding: a CO-selective surface stabilises *COOH more than *H
     (Bagger 2017). The descriptor ``co2rr_selectivity = ΔG_H − ΔG_COOH`` is
     positive when CO2RR is favoured over the parasitic H2 evolution.

Three primary descriptors mirror the proven "mean / extremal / favorable-
fraction" pattern (cf. hydrogen.py best_site_h_affinity):

  mean_co_affinity           Composition-weighted ΔG_CO proxy over metal sites.

  best_site_co_affinity      ΔG_CO proxy of the metal species CLOSEST to the CO-
                             evolution optimum (min |ΔG_CO − CO_OPTIMUM|). THIS
                             is the descriptor to drive toward CO_OPTIMUM (≈0):
                             a high-entropy carbide wins by *containing* sites
                             that release CO, even if its mean over-binds.

  frac_co_selective_sites    Fraction of distinct metal species whose
                             |ΔG_CO − CO_OPTIMUM| < CO_SELECTIVE_WINDOW. The
                             high-entropy "cocktail effect" for CO2RR: widen the
                             local-site distribution so more sites land in the
                             CO-releasing window.

Auxiliary descriptors:

  mean_cooh_affinity         Composition-weighted ΔG_*COOH proxy. CO2 activation
                             is usually the potential-limiting step; more
                             negative ⇒ easier CO2 activation (lower onset).

  co2rr_selectivity          ΔG_H − ΔG_COOH proxy (composition-weighted).
                             Positive ⇒ *COOH more stable than *H ⇒ CO2RR
                             favoured over HER. The CO-vs-H2 selectivity knob.

  d_band_proxy,              Reaction-agnostic electronic descriptors, reused
  valence_electron_conc      verbatim from hydrogen.py (Hammer–Nørskov / HEA).

Economic note (shared expensive_metal_fraction)
-----------------------------------------------
The benchmark CO producers Au/Ag are precious but are NOT platinum-group, so the
project's :func:`carbidemattergen.hydrogen.expensive_metal_fraction` policy (PGM
+ heavy REE) does not penalise them by default. The *point* of the carbide-HEC
route is to reach the CO-release window with **cheap** carbide-weakened d-block
metals (Mo/W carbide-weakened + Zn/Cu) instead of Au/Ag. If you want generation
to avoid precious coinage too, add ``{"Au", "Ag"}`` to ``_EXPENSIVE_METALS`` in
hydrogen.py (one line); see docs/co2rr_design.md.

Why pure-metal ΔG_CO is NOT used directly
-----------------------------------------
Early/mid transition metals (Ti, V, Mo, W, Fe, Co, Ni…) chemisorb CO far too
strongly as pure metals (CO poisoning), which is why pure-metal HEAs built only
from strong CO binders never reach Au/Ag-like CO selectivity. Carbide formation
shifts the metal d-band and weakens CO binding toward the desorption window —
the same "Pt-like carbide" d-band effect that weakens H binding (Levy & Boudart
1973), now exploited for CO release. Mo2C is the canonical CO2RR-active carbide.
The tables below store *effective carbide-site* ΔG_CO / ΔG_COOH, calibrated so
carbide-weakened mid-TMs are moderately negative and coinage/late metals sit
near the optimum — NOT the (much stronger) pure-metal values. They are
literature-informed, relative/monotonic estimates refined by the AL DFT +
fairchem loop (scripts/40_active_learning_loop.sh).
"""
from __future__ import annotations

from pymatgen.core.composition import Composition

# Reuse the shared, reaction-agnostic infrastructure from the HER module so the
# two reactions stay in sync (metal selection, d-band, VEC, H proxy table).
from .hydrogen import (
    _D_BAND_CENTER,
    _H_ADS_CARBIDE,
    _VALENCE_ELECTRONS,
    _metal_amounts,
)

# CO-evolution Sabatier optimum (eV). ΔG_CO at this value releases CO cleanly
# while *COOH still forms; Au/Ag/Zn cluster here. Driven toward this in
# generation (best_site_co_affinity -> CO_OPTIMUM).
CO_OPTIMUM: float = 0.0
# |ΔG_CO − CO_OPTIMUM| below this counts as a CO-selective (releasing) site.
# Wider than the HER thermoneutral window: CO binding is a coarser descriptor.
CO_SELECTIVE_WINDOW: float = 0.20  # eV

# ---------------------------------------------------------------------------
# Effective carbide-site ΔG_CO* proxy (eV) — CO chemisorption free energy on the
# METAL site WITHIN a carbide environment (not the pure-metal value).
#
#   0.0        = optimum: CO desorbs as product, COOH still forms (Au/Ag/Zn)
#   negative   = binds CO too strongly (poisons / over-reduces; Pt/Ni/Fe/Ru)
#   positive   = binds CO too weakly (CO2 never activates; Hg, far coinage)
#
# Calibrated so coinage/late metals sit near 0, carbide-weakened mid-TMs (Mo, W,
# Re) are moderately negative (recoverable by entropy mixing), and strong CO
# binders stay deeply negative. Monotonic/relative; refined by the AL loop.
# ---------------------------------------------------------------------------
_CO_ADS_CARBIDE: dict[str, float] = {
    # 3d
    "Sc": -0.70, "Ti": -0.55, "V": -0.45, "Cr": -0.40, "Mn": -0.45,
    "Fe": -0.80, "Co": -0.75, "Ni": -0.70, "Cu": -0.10, "Zn": +0.05,
    # 4d
    "Y": -0.75, "Zr": -0.60, "Nb": -0.45, "Mo": -0.30, "Tc": -0.70,
    "Ru": -0.85, "Rh": -0.80, "Pd": -0.65, "Ag": +0.15, "Cd": +0.20,
    # 5d
    "Hf": -0.60, "Ta": -0.45, "W": -0.20, "Re": -0.55, "Os": -0.90,
    "Ir": -0.80, "Pt": -0.65, "Au": +0.10, "Hg": +0.30,
}

# ---------------------------------------------------------------------------
# Effective carbide-site ΔG_*COOH proxy (eV) — first CO2RR intermediate.
#
#   negative   = stabilises *COOH ⇒ easy CO2 activation, low onset potential
#   positive   = *COOH formation is the limiting step (high onset; Au/Ag)
#
# Tracks oxophilicity/CO binding (scaling relation) with an oxophilic offset for
# early TMs. Used for mean_cooh_affinity and the CO2RR-vs-HER selectivity knob.
# ---------------------------------------------------------------------------
_COOH_ADS_CARBIDE: dict[str, float] = {
    # 3d
    "Sc": -0.30, "Ti": -0.15, "V": -0.05, "Cr": 0.00, "Mn": -0.05,
    "Fe": -0.10, "Co": -0.05, "Ni": 0.00, "Cu": +0.45, "Zn": +0.55,
    # 4d
    "Y": -0.35, "Zr": -0.20, "Nb": -0.05, "Mo": +0.10, "Tc": -0.10,
    "Ru": -0.15, "Rh": -0.10, "Pd": +0.05, "Ag": +0.70, "Cd": +0.65,
    # 5d
    "Hf": -0.20, "Ta": -0.05, "W": +0.15, "Re": -0.05, "Os": -0.15,
    "Ir": -0.05, "Pt": +0.05, "Au": +0.60, "Hg": +0.75,
}


def _weighted_mean(metals: dict[str, float], table: dict[str, float]) -> float:
    """Composition-weighted mean of ``table`` over present metal species."""
    acc = 0.0
    w = 0.0
    for sym, amt in metals.items():
        v = table.get(sym)
        if v is not None:
            acc += v * amt
            w += amt
    return acc / w if w > 0 else float("nan")


def co2rr_descriptors(comp: Composition) -> dict[str, float]:
    """Return the seven composition-only CO2RR-to-CO proxy descriptors.

    Keys:
      mean_co_affinity, best_site_co_affinity, frac_co_selective_sites,
      mean_cooh_affinity, co2rr_selectivity, d_band_proxy, valence_electron_conc
    """
    metals = _metal_amounts(comp)
    total_metal = sum(metals.values())

    nan = float("nan")
    if total_metal <= 0:
        return {
            "mean_co_affinity": nan,
            "best_site_co_affinity": nan,
            "frac_co_selective_sites": nan,
            "mean_cooh_affinity": nan,
            "co2rr_selectivity": nan,
            "d_band_proxy": nan,
            "valence_electron_conc": nan,
        }

    # --- ΔG_CO proxies over metal sites ----------------------------------
    per_species_co = [_CO_ADS_CARBIDE[s] for s in metals if s in _CO_ADS_CARBIDE]
    mean_co = _weighted_mean(metals, _CO_ADS_CARBIDE)
    # "best site" = species closest to the CO-evolution optimum (Sabatier).
    best_co = (min(per_species_co, key=lambda g: abs(g - CO_OPTIMUM))
               if per_species_co else nan)
    if per_species_co:
        n_sel = sum(1 for g in per_species_co
                    if abs(g - CO_OPTIMUM) < CO_SELECTIVE_WINDOW)
        frac_sel = n_sel / len(per_species_co)
    else:
        frac_sel = nan

    # --- *COOH proxy + CO2RR-vs-HER selectivity --------------------------
    mean_cooh = _weighted_mean(metals, _COOH_ADS_CARBIDE)
    mean_h = _weighted_mean(metals, _H_ADS_CARBIDE)
    # Positive ⇒ *COOH more stable than *H ⇒ CO2RR favoured over HER.
    selectivity = (mean_h - mean_cooh
                   if mean_h == mean_h and mean_cooh == mean_cooh else nan)

    # --- electronic descriptors (reused from HER) ------------------------
    d_band = _weighted_mean(metals, _D_BAND_CENTER)
    ve_acc = 0.0
    ve_w = 0.0
    for el, amt in comp.get_el_amt_dict().items():
        v = _VALENCE_ELECTRONS.get(el)
        if v is not None:
            ve_acc += v * float(amt)
            ve_w += float(amt)
    vec = ve_acc / ve_w if ve_w > 0 else nan

    return {
        "mean_co_affinity": mean_co,
        "best_site_co_affinity": best_co,
        "frac_co_selective_sites": frac_sel,
        "mean_cooh_affinity": mean_cooh,
        "co2rr_selectivity": selectivity,
        "d_band_proxy": d_band,
        "valence_electron_conc": vec,
    }


def descriptors_from_formula(formula: str) -> dict[str, float]:
    """CO2RR proxies + carbon_fraction for a formula string (screen-friendly)."""
    from .hydrogen import carbon_fraction
    try:
        comp = Composition(formula)
    except Exception:  # noqa: BLE001
        return co2rr_descriptors(Composition("Mo2C")) | {"carbon_fraction": float("nan")}
    out = co2rr_descriptors(comp)
    out["carbon_fraction"] = carbon_fraction(comp)
    return out


if __name__ == "__main__":
    # Sanity check against known CO2RR-to-CO behaviour.
    tests = [
        "Au",          # benchmark CO producer        -> best ~ +0.10, sel high
        "Ag",          # benchmark CO producer        -> best ~ +0.15
        "Zn",          # decent CO producer           -> best ~ +0.05
        "Cu",          # CO -> hydrocarbons (binds CO) -> best ~ -0.10
        "Pt",          # CO poisons                    -> best strongly negative
        "Mo2C",        # carbide-weakened CO binding
        "AuAgZnCuC",   # HEC mixing CO producers + carbide
        "MoWCuAgZnC",  # carbide formers + CO releasers -> best near 0, frac up
    ]
    print(f"{'formula':<14} {'meanCO':>7} {'bestCO':>7} {'fracSel':>7} "
          f"{'COOH':>7} {'sel':>6} {'dband':>7} {'VEC':>6} {'xC':>5}")
    for f in tests:
        d = descriptors_from_formula(f)
        print(f"{f:<14} {d['mean_co_affinity']:>7.2f} "
              f"{d['best_site_co_affinity']:>7.2f} "
              f"{d['frac_co_selective_sites']:>7.2f} "
              f"{d['mean_cooh_affinity']:>7.2f} {d['co2rr_selectivity']:>6.2f} "
              f"{d['d_band_proxy']:>7.2f} {d['valence_electron_conc']:>6.2f} "
              f"{d['carbon_fraction']:>5.2f}")
