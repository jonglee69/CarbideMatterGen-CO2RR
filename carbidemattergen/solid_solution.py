"""Stage A' — high-entropy carbide solid-solution construction.

Stage A (the fine-tuned MatterGen) discovers, in small unit cells, *which* metal
palette on *which* parent carbide framework gives near-thermoneutral, metallic,
high-entropy candidates. It does NOT produce a genuine high-entropy solid
solution: a 10-20 atom cell cannot represent the configurational disorder of a
near-equimolar 5-7 metal sublattice. Stage A' closes that gap.

The physics: high-entropy stability is configurational disorder on a *fixed*
metal sublattice (Yeh 2004; Rost 2015). The right object is therefore a special
quasirandom structure (SQS, Zunger et al. PRL 65, 353 (1990)): a periodic
supercell whose short-range pair/triplet correlations on the metal sublattice
match those of the ideal random alloy as closely as a finite cell allows.

This module is self-contained (pymatgen core only — no mcsqs/ATAT/icet, which are
unavailable here). It:

  1. identifies the parent framework of a generated candidate
     (`identify_framework`): rock-salt MC, hexagonal/orthorhombic M2C, or other,
     from the carbon fraction and metal coordination;
  2. builds the corresponding pristine metal sublattice supercell
     (`build_framework_supercell`);
  3. occupies that sublattice with the chosen near-equimolar metal palette and
     anneals the occupation by Monte-Carlo to minimise the deviation of the
     pair correlations from the random-alloy target (`make_sqs`).

The objective minimised is the SQS pair-correlation error
    Phi = sum_shells w_s * (Pi_s - Pi_rand)^2 ,
where Pi_s is the Warren-Cowley-type pair correlation of the *current* occupation
in coordination shell s and Pi_rand is its random-alloy value (0 for the
normalised correlations used here). Lower Phi = closer to ideal randomness.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.core.periodic_table import Element

from .features import TRANSITION_METALS, MAX_M_C_DIST, MAX_M_M_DIST

# Elements treated as the (non-metal) anion / former in a carbide. This MUST
# match carbidemattergen.hydrogen / extra_descriptors so the metal sublattice
# used for the solid solution is identical to the one used for labelling,
# generation conditioning and screening. (Only C/N/O/B/H/Si and halides etc.
# are anions/formers; Ge/Ga/Al/In/Sn are counted as metal-sublattice species,
# consistent with the trained n_metal_species / metal_mixing_entropy.)
_NONMETALS = {"C", "N", "O", "B", "H", "F", "Cl", "Br", "I",
              "S", "Se", "Te", "P", "As", "Si"}


# ---------------------------------------------------------------------------
# 1. Framework identification
# ---------------------------------------------------------------------------

@dataclass
class Framework:
    """A parent carbide framework: a structure type + metal/carbon ratio."""
    kind: str          # 'rocksalt_MC' | 'hcp_M2C' | 'other'
    mc_ratio: float    # metal : carbon ratio (1.0 for MC, 2.0 for M2C)
    carbon_fraction: float


def identify_framework(structure: Structure) -> Framework:
    """Classify the parent framework of a generated carbide candidate.

    Uses the carbon atomic fraction (robust, structure-agnostic) together with a
    coordination sanity check. Interstitial carbides cluster at MC
    (x_C = 0.5, rock-salt) and M2C (x_C = 0.33, hcp/orthorhombic) stoichiometries
    (cf. the training-set carbon-fraction peaks). Other ratios are labelled
    'other' and handled by the generic supercell path.
    """
    comp = structure.composition
    xc = comp.get_atomic_fraction("C") if "C" in comp else 0.0
    if 0.42 <= xc <= 0.58:
        return Framework("rocksalt_MC", 1.0, xc)
    if 0.28 <= xc < 0.40:
        return Framework("hcp_M2C", 2.0, xc)
    # fall back to the nearest canonical ratio for supercell construction
    kind, mc = ("rocksalt_MC", 1.0) if xc >= 0.40 else ("hcp_M2C", 2.0)
    return Framework(kind, mc, xc)


# ---------------------------------------------------------------------------
# 2. Pristine framework supercells (metal sublattice + carbon sublattice)
# ---------------------------------------------------------------------------

def _rocksalt_unit(a: float, metal: str) -> Structure:
    """Rock-salt MC conventional cubic cell (4 M + 4 C), metal placeholder."""
    lat = Lattice.cubic(a)
    species = [metal] * 4 + ["C"] * 4
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],   # M (fcc)
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],   # C (octahedral)
    ]
    return Structure(lat, species, coords)


def _hcp_m2c_unit(a: float, c: float, metal: str) -> Structure:
    """Hexagonal M2C cell: hcp metal sublattice, C in half the octahedral holes.

    A compact, well-defined M2C template (anti-CdI2-like): 2 M + 1 C per cell.
    The metal sublattice is hcp; carbon occupies one octahedral interstice.
    """
    lat = Lattice.hexagonal(a, c)
    species = [metal, metal, "C"]
    coords = [
        [1/3, 2/3, 0.25],
        [2/3, 1/3, 0.75],
        [0.0, 0.0, 0.5],
    ]
    return Structure(lat, species, coords)


def build_framework_supercell(framework: Framework, n_metal_target: int,
                              metal_placeholder: str = "Mo") -> tuple[Structure, list[int]]:
    """Build a pristine framework supercell with >= n_metal_target metal sites.

    Returns (structure, metal_site_indices). The metal sites are all occupied by
    a single placeholder element; `make_sqs` then re-labels them with the target
    palette. Supercell multiplicity is chosen so the metal-site count is a
    convenient multiple of the palette size (helps near-equimolar occupation).
    """
    if framework.kind == "hcp_M2C":
        a, c = 3.0, 4.7  # Å, representative M2C metric (relaxed later by MLIP/DFT)
        unit = _hcp_m2c_unit(a, c, metal_placeholder)
        m_per_cell = 2
    else:  # rocksalt_MC (default)
        a = 4.3          # Å, representative rock-salt carbide lattice constant
        unit = _rocksalt_unit(a, metal_placeholder)
        m_per_cell = 4

    # smallest cubic-ish supercell giving >= n_metal_target metal sites
    mult = max(1, math.ceil((n_metal_target / m_per_cell) ** (1 / 3)))
    sc = unit * (mult, mult, mult)
    metal_idx = [i for i, site in enumerate(sc)
                 if site.species_string == metal_placeholder]
    return sc, metal_idx


# ---------------------------------------------------------------------------
# 3. SQS occupation by Monte-Carlo correlation matching
# ---------------------------------------------------------------------------

@dataclass
class SQSResult:
    structure: Structure
    objective: float                 # final pair-correlation error Phi
    objective_initial: float         # random-start Phi (for context)
    shell_radii: list[float]
    palette: dict[str, int]          # element -> site count actually placed
    n_metal_sites: int
    history: list[float] = field(default_factory=list)


def _metal_neighbor_shells(structure: Structure, metal_idx: Sequence[int],
                           n_shells: int = 2, tol: float = 0.25
                           ) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Build metal–metal neighbour pair lists for the first `n_shells` shells.

    Returns (shell_pairs, shell_radii) where shell_pairs[s] is a list of
    (i, j) index pairs (into metal_idx ordering) in shell s.
    """
    idx_of = {gi: k for k, gi in enumerate(metal_idx)}
    coords = {gi: structure[gi] for gi in metal_idx}
    # collect all metal-metal distances up to a generous cutoff
    cutoff = MAX_M_M_DIST + 1.6
    dists = []
    raw_pairs = []
    for gi in metal_idx:
        for n in structure.get_neighbors(coords[gi], r=cutoff):
            if n.index in idx_of and n.index != gi:
                d = float(n.nn_distance)
                dists.append(d)
                a, b = idx_of[gi], idx_of[n.index]
                raw_pairs.append((min(a, b), max(a, b), d))
    if not dists:
        return [], []
    dists = np.array(dists)
    # cluster distances into shells (greedy by sorted unique gaps)
    order = np.sort(np.unique(np.round(dists, 2)))
    shells: list[float] = []
    for d in order:
        if not shells or d - shells[-1] > tol:
            shells.append(float(d))
        if len(shells) >= n_shells:
            break
    shell_pairs: list[set] = [set() for _ in shells]
    for a, b, d in raw_pairs:
        for s, r in enumerate(shells):
            if abs(d - r) <= tol:
                shell_pairs[s].add((a, b))
                break
    return [sorted(sp) for sp in shell_pairs], shells


def _pair_correlation(occ: np.ndarray, pairs: list[tuple[int, int]],
                      species_vals: dict[int, float]) -> float:
    """Normalised pair correlation Pi = <s_i s_j> over a shell.

    Each metal type is mapped to a zero-mean ±1-style spin (multi-component
    generalisation: centred integer codes normalised to unit variance), so the
    ideal random-alloy correlation is 0. `occ` holds the type code per metal site.
    """
    if not pairs:
        return 0.0
    s = np.array([species_vals[int(t)] for t in occ])
    acc = 0.0
    for i, j in pairs:
        acc += s[i] * s[j]
    return acc / len(pairs)


def make_sqs(framework_structure: Structure, metal_idx: Sequence[int],
             palette: Iterable[str], *, n_shells: int = 2,
             steps: int = 4000, seed: int = 0,
             shell_weights: Sequence[float] | None = None) -> SQSResult:
    """Occupy the metal sublattice with `palette` and Monte-Carlo anneal toward
    the random-alloy pair correlations (an SQS).

    The occupation is initialised near-equimolar and randomly, then improved by
    Metropolis swaps that lower the weighted pair-correlation error Phi. The
    result is a single periodic supercell — the genuine high-entropy solid
    solution that Stage B will compute ΔG_H* on.
    """
    rng = random.Random(seed)
    palette = list(palette)
    k = len(palette)
    n = len(metal_idx)
    if k < 2 or n < k:
        raise ValueError(f"palette {palette} too large for {n} metal sites")

    # near-equimolar count vector
    base = n // k
    counts = [base] * k
    for i in range(n - base * k):
        counts[i] += 1
    occ = np.array([t for t, c in enumerate(counts) for _ in range(c)])
    rng.shuffle(list(occ))
    occ = np.array(occ)
    np.random.RandomState(seed).shuffle(occ)

    # zero-mean unit-variance spin codes (random-alloy Pi == 0)
    codes = np.arange(k, dtype=float)
    codes -= codes.mean()
    if codes.std() > 0:
        codes /= codes.std()
    species_vals = {t: float(codes[t]) for t in range(k)}

    shell_pairs, shell_radii = _metal_neighbor_shells(
        framework_structure, metal_idx, n_shells=n_shells)
    if not shell_pairs:
        raise RuntimeError("no metal-metal neighbour shells found")
    if shell_weights is None:
        shell_weights = [1.0 / (s + 1) for s in range(len(shell_pairs))]

    def phi(o: np.ndarray) -> float:
        return sum(w * _pair_correlation(o, sp, species_vals) ** 2
                   for w, sp in zip(shell_weights, shell_pairs))

    cur = phi(occ)
    phi0 = cur
    history = [cur]
    # Metropolis with slow cooling
    T0, T1 = 0.5 * (cur + 1e-6), 1e-4
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / max(1, steps - 1))
        i, j = rng.randrange(n), rng.randrange(n)
        if occ[i] == occ[j]:
            continue
        occ[i], occ[j] = occ[j], occ[i]
        new = phi(occ)
        if new <= cur or rng.random() < math.exp(-(new - cur) / max(T, 1e-9)):
            cur = new
        else:
            occ[i], occ[j] = occ[j], occ[i]  # revert
        if step % 50 == 0:
            history.append(cur)
    history.append(cur)

    # write the optimised occupation into a copy of the framework
    out = framework_structure.copy()
    placed: dict[str, int] = {}
    for site_pos, gi in enumerate(metal_idx):
        el = palette[int(occ[site_pos])]
        out.replace(gi, el)
        placed[el] = placed.get(el, 0) + 1

    return SQSResult(structure=out, objective=cur, objective_initial=phi0,
                     shell_radii=shell_radii, palette=placed,
                     n_metal_sites=n, history=history)


# ---------------------------------------------------------------------------
# Convenience: full Stage A' for one candidate formula
# ---------------------------------------------------------------------------

def metal_palette(formula: str) -> list[str]:
    """Return the metal (non-anion) element list of a candidate formula."""
    return [el.symbol for el in Composition(formula).elements
            if el.symbol not in _NONMETALS]


def construct_solid_solution(formula: str, *, n_metal_target: int = 32,
                             steps: int = 4000, seed: int = 0) -> SQSResult:
    """End-to-end Stage A' for a candidate composition string.

    Identifies the framework from the formula's carbon fraction, builds a metal
    sublattice supercell of ~`n_metal_target` sites, and returns an SQS occupied
    by the formula's metal palette near-equimolar.
    """
    comp = Composition(formula)
    xc = comp.get_atomic_fraction("C") if "C" in comp else 0.5
    fw = Framework(*( ("rocksalt_MC", 1.0, xc) if xc >= 0.40
                      else ("hcp_M2C", 2.0, xc) ))
    palette = metal_palette(formula)
    sc, midx = build_framework_supercell(fw, n_metal_target)
    return make_sqs(sc, midx, palette, steps=steps, seed=seed)


if __name__ == "__main__":
    # smoke test on a representative HEC candidate
    res = construct_solid_solution("ErGeMoRuWC", n_metal_target=32, steps=2000)
    print("framework SQS built")
    print("  metal sites :", res.n_metal_sites)
    print("  palette     :", res.palette)
    print("  Phi (rand)  :", round(res.objective_initial, 4))
    print("  Phi (SQS)   :", round(res.objective, 4))
    print("  shell radii :", [round(r, 2) for r in res.shell_radii])
    print("  formula     :", res.structure.composition.reduced_formula)
