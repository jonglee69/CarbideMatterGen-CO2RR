#!/usr/bin/env python
"""Step 2 — build small ORDERED rocksalt proxy bulks for the DFT-confirmed winners.

The three dual-objective winners (hec3_MoTiZr / CrTiV / CrTiZr) were validated
only as DFT//UMA *single points* on 64-atom SQS slabs — the geometry was UMA-
relaxed, DFT gave energies but never relaxed the ions, and U_L therefore carries
±0.3-0.5 eV. Full DFT relaxation of a 64-atom disordered slab is impractical here
(GPU USPP force routine segfaults on large slabs; CPU relax with k-points is days
per state). So we confirm U_L on a small ORDERED rocksalt proxy of the same metal
combination that CAN be relaxed at DFT with a real k-mesh.

Proxy construction (deterministic, documented):
  * rocksalt (Fm-3m) conventional cell, 4 metal + 4 C sites,
  * metal sublattice = 2 Ti + 1 M2 + 1 M3 (Ti-rich: matches that the winning
    H-repelling chemistry is Ti-based / group-4-rich; also keeps it to ONE
    ordered cell = 8 atoms so the (100) slab stays ~16-24 atoms),
  * lattice constant = Vegard average of the binary MC rocksalt constants,
    weighted by the 2:1:1 composition.

The proxy is an idealisation (ordered, not the equimolar SQS) and is used only to
check that the RELAXED (100) rocksalt surface of this metal set still gives
ΔG_CO ~ 0 and ΔG_H > 0. Feed the CIFs through 37 (UMA bulk relax) -> 54 (facet /
adsorbate export, --max-index 1 --save-facets 1) -> 55 (--calc relax --kpts auto)
-> 56 (GPU QE) -> 61 (relaxed U_L).

    python scripts/65_build_ordered_proxies.py --out outputs/dft_proxy/bulk
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Structure, Lattice

# Binary rocksalt lattice constants (Å). Group-4/5/6 metal carbides; the
# metastable/high-pressure rocksalt polymorph value is used for Cr/Mo/W.
A_MC = {
    "Ti": 4.328, "Zr": 4.698, "Hf": 4.638,
    "V": 4.166, "Nb": 4.470, "Ta": 4.456,
    "Cr": 4.030, "Mo": 4.270, "W": 4.260,
}

# Winners as (M2, M3); Ti is always the 2x majority metal.
WINNERS = {
    "proxy_MoTiZr": ("Mo", "Zr"),
    "proxy_CrTiV":  ("Cr", "V"),
    "proxy_CrTiZr": ("Cr", "Zr"),
}

# Fractional coordinates of the rocksalt conventional cell.
METAL_SITES = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
C_SITES = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]


def build(m2: str, m3: str) -> Structure:
    a = (2 * A_MC["Ti"] + A_MC[m2] + A_MC[m3]) / 4.0
    # metal assignment: Ti at the two z=0 sites, M2/M3 at the two z=1/2 sites
    metals = ["Ti", "Ti", m2, m3]
    species = metals + ["C"] * 4
    coords = METAL_SITES + C_SITES
    return Structure(Lattice.cubic(a), species, coords), a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (m2, m3) in WINNERS.items():
        s, a = build(m2, m3)
        fp = args.out / f"{name}.cif"
        s.to(filename=str(fp))
        print(f"{name:16s} a={a:.3f} Å  {s.composition.reduced_formula:10s} "
              f"({len(s)} atoms) -> {fp}")


if __name__ == "__main__":
    main()
