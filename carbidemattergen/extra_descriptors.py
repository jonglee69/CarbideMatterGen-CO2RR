"""Composition-level diversity descriptors for medium/high-entropy carbides.

Fast, reduced-formula-only descriptors (no structure) so they recompute
identically for generated candidates during screening. The metal sublattice is
what defines a (medium/high)-entropy carbide, so carbon and other non-metals
are excluded from the entropy / species count.
"""
from __future__ import annotations

import math
from pymatgen.core.composition import Composition

_NONMETALS: set[str] = {"C", "N", "O", "B", "H", "F", "Cl", "Br", "I",
                        "S", "Se", "Te", "P", "As", "Si"}


def metal_mixing_entropy(comp: Composition) -> float:
    """Shannon entropy (units of R) of the metal-sublattice mole fractions.

    For an equimolar n-metal carbide this is ln(n): 5 metals → 1.609 R, the
    conventional ΔS_mix ≥ 1.5R threshold that defines a *high*-entropy phase.
    """
    metals: dict[str, float] = {}
    for el, amt in comp.get_el_amt_dict().items():
        if el in _NONMETALS:
            continue
        metals[el] = float(amt)
    total = sum(metals.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for amt in metals.values():
        p = amt / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def n_metal_species(comp: Composition) -> int:
    """Number of distinct metal species (excludes C and other non-metals)."""
    return sum(1 for el in comp.get_el_amt_dict() if el not in _NONMETALS)


if __name__ == "__main__":
    for f in ("Mo2C", "WMoTaC", "ZrTiHfNbVC", "MoWNiVCrC"):
        c = Composition(f)
        print(f"{f:<14} S_mix={metal_mixing_entropy(c):.3f}R  "
              f"n_metal={n_metal_species(c)}")
