"""Element policy — manufacturability (acid-leach survival) + scientific scope.

WHY THIS MODULE EXISTS
----------------------
The first CO2RR campaign generated six candidates that were *unmakeable*. Every
one of the 24 screened compositions contained a rare earth (Y appeared in all
24), and the user's synthesis route — metallothermic reduction of oxides,
followed by a MANDATORY acid leach to dissolve the reductant oxide (MgO/CaO) —
dissolves every rare earth away. The failure was in the objective, not the
chemistry: the only compositional constraint ever encoded was *cost*
(:func:`hydrogen.expensive_metal_fraction`), which explicitly whitelisted the
cheap light rare earths La/Ce/Pr/Nd/Y as "not expensive". Aqueous / acid
stability was never a constraint anywhere in the pipeline.

Two bugs compounded it:
  * Y and Sc are formally d-block, so any "d-block only" purity check passes
    them. Chemically they are rare earths and they leach. They are classified
    as REE here — that is the fix.
  * Conditioning on stability pulls toward REE carbides (very negative
    formation energies), so the generator was actively rewarded for them.

WHAT THE DFT SAID (why dropping REE is safe)
--------------------------------------------
On the six DFT-validated candidates the rare earths are spectators:
  corr(REE fraction, ΔG_CO)  = +0.03   -> no contribution to CO binding
  corr(ε_d, ΔG_CO)           = -0.63   -> the real lever is the d-band centre
  corr(REE fraction, ε_d)    = +0.59   -> REE push ε_d UP = *stronger* CO binding
  corr(C/M, ε_d)             = -0.53   -> carbon enrichment deepens ε_d
The Fermi level is C-dominated in 4 of 6. So the rare earths were not providing
the active site; if anything they were driving CO binding the wrong way.

SCOPE OF THE EXCLUSIONS (two different rationales — keep them straight)
----------------------------------------------------------------------
1. MANUFACTURABILITY: anything that dissolves in the dilute non-oxidizing acid
   leach (HCl) is excluded. Rare earths (incl. Y/Sc), alkali, alkaline earth,
   Al, Mn, Fe, Co, Ni, Zn, and the post-transition metals.
2. SCIENTIFIC SCOPE: Cu, Ag, Au, Zn are *also* excluded even though Cu/Ag would
   survive non-oxidizing HCl. They are the textbook CO2RR-to-CO metals. Handing
   them to the generator would answer the research question by fiat — the whole
   point is whether a carbide can reach the CO-release window on its OWN d-band,
   with no coinage/noble metal. Excluding them protects the contribution.
3. ECONOMIC: the platinum-group metals stay excluded (original constraint).

The surviving palette is exactly the refractory, metallic, acid-resistant
carbide formers — group 4/5/6 transition metals.
"""
from __future__ import annotations

from pymatgen.core import Composition

# --- the allowed metal sublattice ------------------------------------------
# Group 4/5/6 TMs: their carbides (TiC, ZrC, HfC, VC, NbC, TaC, Cr3C2, Mo2C, WC)
# are metallic AND famously resistant to non-oxidizing acids — they survive the
# leach that removes MgO/CaO.
ACID_STABLE_METALS: frozenset[str] = frozenset({
    "Ti", "Zr", "Hf",   # group 4
    "V", "Nb", "Ta",    # group 5
    "Cr", "Mo", "W",    # group 6
})

# Light elements permitted on the non-metal sublattice. C is the carbide anion;
# B and N are legitimate *mechanistic* levers (boro-carbides / carbonitrides
# tune ε_d) and are NOT known CO2RR-to-CO actives, so they do not give the
# answer away the way Cu/Ag would.
ALLOWED_NONMETALS: frozenset[str] = frozenset({"C", "B", "N"})

# --- exclusions -------------------------------------------------------------
# Rare earths INCLUDING Y and Sc. This is the bug fix: Y/Sc are d-block by
# electron configuration but rare-earth by chemistry, and they leach.
RARE_EARTHS: frozenset[str] = frozenset({
    "Sc", "Y",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
})

# Dissolve in dilute non-oxidizing acid (the leach step).
LEACH_SOLUBLE: frozenset[str] = RARE_EARTHS | frozenset({
    "Li", "Na", "K", "Rb", "Cs",              # alkali
    "Be", "Mg", "Ca", "Sr", "Ba",             # alkaline earth
    "Al", "Ga", "In", "Sn", "Pb", "Bi",       # post-transition
    "Mn", "Fe", "Co", "Ni", "Zn", "Cd",       # above H in the activity series
    "Th", "U",                                 # actinides
})

PGM: frozenset[str] = frozenset({"Ru", "Rh", "Pd", "Os", "Ir", "Pt"})

# Excluded to protect the research question, not for stability reasons.
KNOWN_CO2RR_METALS: frozenset[str] = frozenset({"Cu", "Ag", "Au", "Zn"})

EXCLUDED: frozenset[str] = LEACH_SOLUBLE | PGM | KNOWN_CO2RR_METALS


# Non-metals never counted in the metal sublattice (matches the project-wide
# _NONMETALS convention in hydrogen/extra_descriptors/solid_solution).
_NONMETALS: frozenset[str] = frozenset({
    "C", "N", "O", "B", "H", "F", "Cl", "Br", "I", "S", "Se", "P", "Si",
})


def _metal_amounts(comp: Composition) -> dict[str, float]:
    return {sym: amt for sym, amt in comp.get_el_amt_dict().items()
            if sym not in _NONMETALS}


def leach_unstable_fraction(comp: Composition) -> float:
    """Fraction of the metal sublattice that dissolves in the acid leach.

    0.0 = fully manufacturable by metallothermic reduction + acid leach.
    Kept as a *label* (diagnostics / post-hoc filtering). It is deliberately NOT
    used as a conditioning target when the training set is already filtered —
    a filtered set has zero variance in it, so it would carry no signal.
    """
    metals = _metal_amounts(comp)
    total = sum(metals.values())
    if total <= 0:
        return float("nan")
    bad = sum(a for sym, a in metals.items() if sym in LEACH_SOLUBLE)
    return bad / total


def is_allowed(comp: Composition, allow_bn: bool = True) -> bool:
    """Hard whitelist: every element must be an allowed metal or non-metal.

    This is the belt-and-braces gate. Soft conditioning is what let the rare
    earths through last time; a hard filter is applied to the training set AND
    to every generated structure.
    """
    nonmetals = ALLOWED_NONMETALS if allow_bn else frozenset({"C"})
    syms = set(comp.get_el_amt_dict())
    if not (syms & {"C"}):          # must actually be a carbide
        return False
    for s in syms:
        if s in nonmetals:
            continue
        if s not in ACID_STABLE_METALS:
            return False
    return True


def reject_reason(comp: Composition, allow_bn: bool = True) -> str | None:
    """Why a composition is rejected (None if allowed). For audit logs."""
    syms = set(comp.get_el_amt_dict())
    if "C" not in syms:
        return "not a carbide (no C)"
    nonmetals = ALLOWED_NONMETALS if allow_bn else frozenset({"C"})
    for s in sorted(syms):
        if s in nonmetals or s in ACID_STABLE_METALS:
            continue
        if s in RARE_EARTHS:
            return f"{s}: rare earth — dissolves in the acid leach"
        if s in LEACH_SOLUBLE:
            return f"{s}: dissolves in the acid leach"
        if s in PGM:
            return f"{s}: platinum-group (cost)"
        if s in KNOWN_CO2RR_METALS:
            return f"{s}: known CO2RR-to-CO metal — excluded to keep the search honest"
        return f"{s}: not in the allowed palette"
    return None
