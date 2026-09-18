"""Composition-level descriptors that correlate with the hydrogen evolution
reaction (HER) activity of transition-metal carbides.

The HER overpotential is governed by the Sabatier principle: the Gibbs free
energy of atomic-hydrogen adsorption, ΔG_H*, should be as close to zero as
possible (Pt ≈ -0.09 eV). The challenge for a *generative* model is that
ΔG_H* is a **surface** property, while MatterGen generates **bulk** crystals.
This module therefore exposes cheap, composition-only *proxies* for ΔG_H* that
can be recomputed identically for any generated candidate (no DFT, no surface
slab) and used as conditioning labels during fine-tuning. The real ΔG_H* is
evaluated later, in Stage B (`surface_screen`), with a fairchem (OC20/OC22)
model, and the proxy tables here are recalibrated against those + DFT in the
active-learning loop.

Three primary descriptors mirror the proven "mean / extremal / favorable-
fraction" pattern from OxideMatterGen (anodic_stability / min_cation_anodic /
frac_favorable_angle):

  mean_h_affinity            Composition-weighted ΔG_H proxy over the metal
                             sites. The *average* binding strength.

  best_site_h_affinity       The single metal species whose carbide-site ΔG_H
                             proxy is CLOSEST to thermoneutral (min |ΔG_H|).
                             HER proceeds on the best sites, not the average —
                             this is the "best-link" analogue of OxideMatterGen's
                             "weakest-link" min_cation_anodic_stability. THIS is
                             the descriptor to drive toward 0; a high-entropy
                             carbide wins by *containing* near-thermoneutral
                             sites even if its mean is off.

  frac_thermoneutral_sites   Fraction of distinct metal species with
                             |ΔG_H proxy| < THERMONEUTRAL_WINDOW. High-entropy
                             alloys win HER precisely by widening the local-site
                             distribution so that a larger fraction of sites sit
                             near 0. Conditioning on this directly rewards the
                             "cocktail effect".

Two auxiliary electronic descriptors (Hammer–Nørskov d-band theory / HEA
literature):

  d_band_proxy               Composition-weighted d-band-centre estimate (eV
                             rel. E_F). More negative ⇒ weaker H binding.

  valence_electron_conc      Mean valence-electron concentration (VEC) over ALL
                             atoms incl. C. A classic HEA-catalyst descriptor;
                             correlates monotonically with d-band filling.

Why pure-metal ΔG_H is NOT used directly
----------------------------------------
Early transition metals (Ti, Zr, Hf, V, Nb, Ta, Mo, W) bind H far too strongly
as pure metals (ΔG_H ≪ 0), which is exactly why ZrTiHfNbVC-type carbides built
only from strong binders never reach Pt-like activity — both their mean AND
their best site stay strongly negative. Carbide formation shifts the metal
d-band and weakens H binding toward thermoneutral (the classic "Pt-like
carbide" effect, Levy & Boudart, Science 1973; Mo2C being the canonical case).
The table below therefore stores *effective carbide-site* ΔG_H, calibrated so
Mo2C ≈ 0, not the pure-metal value. To approach Pt, a candidate generally needs
to mix strong binders (which carbide-formation pulls toward 0) with at least one
intrinsically weaker binder (late TM such as Ni/Cu, or noble-adjacent Mo/W).
"""
from __future__ import annotations

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element


# Anions / non-metal formers excluded from the *metal* averages.
_NONMETALS: set[str] = {"C", "N", "O", "B", "H", "F", "Cl", "Br", "I",
                        "S", "Se", "Te", "P", "As", "Si"}

THERMONEUTRAL_WINDOW: float = 0.10  # eV; |ΔG_H| below this counts as active.

# ---------------------------------------------------------------------------
# Effective carbide-site ΔG_H* proxy (eV).
#
# These are the free energy of atomic-H adsorption on the METAL site WITHIN a
# carbide environment, NOT the pure-metal value. They are literature-informed
# estimates calibrated against known carbide HER data (Mo2C ≈ 0, WC slightly
# H-weak, TiC/VC moderate binders) and the pure-metal HER volcano (Nørskov
# 2005). They are meant to be *relative / monotonic* and are refined by the
# active-learning DFT + fairchem loop (see scripts/40_active_learning_loop.sh).
#
#   0.0        = thermoneutral (Pt-like, ideal)
#   negative   = binds H too strongly (early TM, blocks desorption)
#   positive   = binds H too weakly (coinage / sp metals, slow Volmer)
# ---------------------------------------------------------------------------
_H_ADS_CARBIDE: dict[str, float] = {
    # 3d
    "Sc": -0.40, "Ti": -0.25, "V": -0.10, "Cr": -0.05, "Mn": -0.10,
    "Fe": -0.15, "Co": -0.10, "Ni": -0.20, "Cu": +0.20, "Zn": +0.40,
    # 4d
    "Y": -0.45, "Zr": -0.30, "Nb": -0.15, "Mo": 0.00, "Tc": -0.05,
    "Ru": -0.05, "Rh": -0.10, "Pd": -0.15, "Ag": +0.30, "Cd": +0.45,
    # 5d
    "Hf": -0.30, "Ta": -0.15, "W": +0.05, "Re": -0.05, "Os": -0.05,
    "Ir": -0.05, "Pt": -0.09, "Au": +0.30, "Hg": +0.50,
}

# Hammer–Nørskov d-band centre ε_d (eV rel. E_F) for clean close-packed
# surfaces. Approximate, representative values; missing entries are skipped in
# the weighted average (NaN propagation, same convention as OxideMatterGen).
_D_BAND_CENTER: dict[str, float] = {
    "Sc": -1.30, "Ti": -1.50, "V": -1.40, "Cr": -1.10, "Mn": -0.90,
    "Fe": -0.92, "Co": -1.17, "Ni": -1.29, "Cu": -2.67, "Zn": -6.00,
    "Y": -1.20, "Zr": -1.50, "Nb": -1.20, "Mo": -0.60, "Tc": -1.00,
    "Ru": -1.41, "Rh": -1.73, "Pd": -1.83, "Ag": -4.30, "Cd": -7.00,
    "Hf": -1.55, "Ta": -1.10, "W": -0.50, "Re": -0.90, "Os": -1.20,
    "Ir": -2.11, "Pt": -2.25, "Au": -3.56, "Hg": -8.00,
}

# Valence-electron count per element (group electrons). Used for VEC.
_VALENCE_ELECTRONS: dict[str, int] = {
    # TMs: group number (s+d)
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9,
    "Ni": 10, "Cu": 11, "Zn": 12,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9,
    "Pd": 10, "Ag": 11, "Cd": 12,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10,
    "Au": 11, "Hg": 12,
    # main-group formers commonly present in carbides / for completeness
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "Si": 4, "Al": 3,
}


def _metal_amounts(comp: Composition) -> dict[str, float]:
    """{metal_symbol: atoms in formula}, excluding C and other non-metals."""
    out: dict[str, float] = {}
    for el, amt in comp.get_el_amt_dict().items():
        if el in _NONMETALS:
            continue
        out[el] = float(amt)
    return out


# ---------------------------------------------------------------------------
# Economic conditioning descriptor.
#
# A 9th conditioning target that lets generation be steered toward *affordable*,
# non-platinum-group catalysts. The surface screen (Stage B) showed the proxy
# does not require expensive elements (clean d-block d-band centres match/beat
# the PGMs), yet the unconstrained generator's learned distribution favoured
# them. Conditioning ``expensive_metal_fraction → 0`` counteracts that bias.
#
# Policy (user-set): platinum-group metals and the expensive heavy rare earths
# are "expensive"; non-PGM d-block, the cheap light rare earths (La/Ce/Pr/Nd/Y)
# and Re (non-PGM, allowed despite price) are NOT penalised.
_PGM: set[str] = {"Ru", "Rh", "Pd", "Os", "Ir", "Pt"}
_EXPENSIVE_REE: set[str] = {"Sm", "Eu", "Gd", "Tb", "Dy",
                            "Ho", "Er", "Tm", "Yb", "Lu"}
_EXPENSIVE_METALS: set[str] = _PGM | _EXPENSIVE_REE


def expensive_metal_fraction(comp: Composition) -> float:
    """Fraction of metal-sublattice atoms that are expensive (PGM or heavy REE).

    0.0 for an economically clean composition (target at generation), up to 1.0.
    Re and the cheap light rare earths (La/Ce/Pr/Nd/Y) are deliberately NOT
    counted as expensive.
    """
    metals = _metal_amounts(comp)
    total = sum(metals.values())
    if total <= 0:
        return float("nan")
    expensive = sum(a for el, a in metals.items() if el in _EXPENSIVE_METALS)
    return expensive / total


def hydrogen_descriptors(comp: Composition) -> dict[str, float]:
    """Return the five composition-only HER proxy descriptors.

    Keys:
      mean_h_affinity, best_site_h_affinity, frac_thermoneutral_sites,
      d_band_proxy, valence_electron_conc
    """
    metals = _metal_amounts(comp)
    total_metal = sum(metals.values())

    nan = float("nan")
    if total_metal <= 0:
        return {
            "mean_h_affinity": nan,
            "best_site_h_affinity": nan,
            "frac_thermoneutral_sites": nan,
            "d_band_proxy": nan,
            "valence_electron_conc": nan,
        }

    # --- ΔG_H proxies over metal sites -----------------------------------
    h_acc = 0.0
    h_w = 0.0
    per_species_h: list[float] = []
    for sym, amt in metals.items():
        g = _H_ADS_CARBIDE.get(sym)
        if g is not None:
            h_acc += g * amt
            h_w += amt
            per_species_h.append(g)

    mean_h = h_acc / h_w if h_w > 0 else nan
    # "best site" = species closest to thermoneutral (Sabatier optimum).
    best_h = min(per_species_h, key=abs) if per_species_h else nan
    if per_species_h:
        n_active = sum(1 for g in per_species_h if abs(g) < THERMONEUTRAL_WINDOW)
        frac_active = n_active / len(per_species_h)
    else:
        frac_active = nan

    # --- d-band centre proxy (metal-weighted) ----------------------------
    d_acc = 0.0
    d_w = 0.0
    for sym, amt in metals.items():
        d = _D_BAND_CENTER.get(sym)
        if d is not None:
            d_acc += d * amt
            d_w += amt
    d_band = d_acc / d_w if d_w > 0 else nan

    # --- VEC over ALL atoms (incl. C) ------------------------------------
    ve_acc = 0.0
    ve_w = 0.0
    for el, amt in comp.get_el_amt_dict().items():
        v = _VALENCE_ELECTRONS.get(el)
        if v is not None:
            ve_acc += v * float(amt)
            ve_w += float(amt)
    vec = ve_acc / ve_w if ve_w > 0 else nan

    return {
        "mean_h_affinity": mean_h,
        "best_site_h_affinity": best_h,
        "frac_thermoneutral_sites": frac_active,
        "d_band_proxy": d_band,
        "valence_electron_conc": vec,
    }


def carbon_fraction(comp: Composition) -> float:
    """Atomic fraction of carbon — distinguishes MC, M2C, M3C, etc."""
    amts = comp.get_el_amt_dict()
    total = sum(amts.values())
    if total <= 0:
        return float("nan")
    return float(amts.get("C", 0.0)) / total


def descriptors_from_formula(formula: str) -> dict[str, float]:
    try:
        comp = Composition(formula)
    except Exception:  # noqa: BLE001
        return hydrogen_descriptors(Composition("Mo2C")) | {"carbon_fraction": float("nan")}
    out = hydrogen_descriptors(comp)
    out["carbon_fraction"] = carbon_fraction(comp)
    return out


if __name__ == "__main__":
    # Sanity check against known carbides / the user's tried compositions.
    tests = [
        "Mo2C",        # canonical Pt-like carbide   -> best ~ 0
        "WC",          # near Pt-like                 -> best ~ +0.05
        "TiC",         # binds H moderately strong
        "ZrTiHfNbVC",  # user's HEA: all strong binders -> best stays negative
        "WMoTaC",      # user's MEA: Mo pulls best toward 0
        "MoWNiVCrC",   # mixes weak (Ni) + Mo/W -> best near 0, frac up
        "Pt",          # reference
    ]
    print(f"{'formula':<14} {'mean':>7} {'best':>7} {'frac':>5} "
          f"{'dband':>7} {'VEC':>6} {'xC':>5}")
    for f in tests:
        d = descriptors_from_formula(f)
        print(f"{f:<14} {d['mean_h_affinity']:>7.2f} "
              f"{d['best_site_h_affinity']:>7.2f} "
              f"{d['frac_thermoneutral_sites']:>5.2f} "
              f"{d['d_band_proxy']:>7.2f} {d['valence_electron_conc']:>6.2f} "
              f"{d['carbon_fraction']:>5.2f}")
