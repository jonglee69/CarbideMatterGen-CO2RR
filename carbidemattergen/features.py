"""Structure-level descriptors for transition-metal carbides relevant to
electrical conductivity and to the metal–carbon framework that hosts HER.

Carbides are predominantly metallic conductors, so unlike the oxide case the
goal is NOT a super-exchange angle window but rather a dense, well-connected
metal framework (good electron transport) with a metal–carbon coordination
characteristic of interstitial carbides (rock-salt MC, hexagonal M2C, …).

All descriptors are post-hoc, computed from the relaxed structure; they do not
modify any input file. They are used (a) as optional conditioning labels and
(b) as cheap pre-filters before the expensive fairchem surface screen.

  metal_cn_mean          Mean metal coordination number (CrystalNN). High,
                         uniform CN ⇒ close-packed metallic framework.
  mc_coordination        Mean number of C neighbours per metal site (interstitial
                         carbon filling) within MAX_M_C_DIST.
  metal_packing_density  Metal atoms per Å³ — a conductivity / framework-density
                         proxy (analogue of OxideMatterGen's path_density).
  metal_percolation_dim  Dimensionality (0..3) of the metal–metal contact network
                         under PBC; 3 ⇒ isotropic metallic conduction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from pymatgen.core import Structure

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

MAX_M_C_DIST = 2.6   # Å, typical interstitial M–C bond ceiling
MAX_M_M_DIST = 3.2   # Å, metal–metal contact ceiling


@dataclass
class CarbideFeatures:
    n_metal_sites: int
    n_carbon: int
    metal_cn_mean: float
    mc_coordination: float
    metal_packing_density: float
    metal_percolation_dim: float

    def to_dict(self) -> dict:
        return asdict(self)


def _metal_indices(s: Structure) -> list[int]:
    return [i for i, site in enumerate(s)
            for sp in site.species.elements if sp.symbol in TRANSITION_METALS]


def _carbon_indices(s: Structure) -> list[int]:
    return [i for i, site in enumerate(s)
            if any(sp.symbol == "C" for sp in site.species.elements)]


def _metal_percolation(s: Structure, m_idx: list[int]) -> float:
    """Dimensionality of the M–M contact network via a 2x2x2 union-find."""
    if len(m_idx) < 2:
        return float("nan")
    shifts = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
    node = {(mi, sh): k for k, (mi, sh) in
            enumerate((mi, sh) for mi in m_idx for sh in shifts)}
    parent = list(range(len(node)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    m_set = set(m_idx)
    fc = {mi: s[mi].frac_coords for mi in m_idx}
    found_edge = False
    for mi in m_idx:
        for n in s.get_neighbors(s[mi], r=MAX_M_M_DIST, include_index=True):
            if n.index in m_set and n.index != mi:
                found_edge = True
                dfrac = fc[n.index] - fc[mi]
                off = tuple(int(np.round(x)) for x in
                            dfrac - (dfrac - np.floor(dfrac + 0.5)))
                for sh in shifts:
                    sh2 = tuple((sh[k] + off[k]) % 2 for k in range(3))
                    union(node[(mi, sh)], node[(n.index, sh2)])
    if not found_edge:
        return float("nan")

    inv = {v: k for k, v in node.items()}
    comp: dict[int, list] = {}
    for k in range(len(node)):
        comp.setdefault(find(k), []).append(inv[k])
    largest = max(comp.values(), key=len)
    present = {tuple(sh) for (_mi, sh) in largest}
    dim = sum(1 for axis in range(3) if len({sh[axis] for sh in present}) >= 2)
    return float(dim)


def compute_features(s: Structure) -> Optional[CarbideFeatures]:
    m_idx = _metal_indices(s)
    c_idx = _carbon_indices(s)
    if not m_idx or not c_idx:
        return None

    cns: list[int] = []
    mc_counts: list[int] = []
    c_set = set(c_idx)
    for mi in m_idx:
        neigh = s.get_neighbors(s[mi], r=MAX_M_M_DIST, include_index=True)
        cns.append(len(neigh))
        cneigh = s.get_neighbors(s[mi], r=MAX_M_C_DIST, include_index=True)
        mc_counts.append(sum(1 for n in cneigh if n.index in c_set))

    vol = float(s.volume) if s.volume and s.volume > 0 else float("nan")
    return CarbideFeatures(
        n_metal_sites=len(m_idx),
        n_carbon=len(c_idx),
        metal_cn_mean=float(np.mean(cns)) if cns else float("nan"),
        mc_coordination=float(np.mean(mc_counts)) if mc_counts else float("nan"),
        metal_packing_density=(len(m_idx) / vol) if vol == vol else float("nan"),
        metal_percolation_dim=_metal_percolation(s, m_idx),
    )
