"""Build an overlay cache that extends the MatterGen training cache with
physics-aware property labels, *without modifying the original cache*.

Strategy
--------
- Every file in the source cache (atomic_numbers.npy, cell.npy, pos.npy,
  num_atoms.npy, structure_id.npy, chemical_system.json, existing
  *.json property labels) is SYMLINKED into the overlay directory.
- For each new physics label, a fresh *.json file is written next to the
  symlinks. Its order is aligned with ``structure_id.npy``; rows whose
  ``material_id`` is absent from the labeled CSV get NaN so MatterGen's
  ``filter_sparse_properties`` transform restricts training to the
  oxide-with-physics subset automatically when these properties are
  requested.
- The JSON schema matches MatterGen's expected format:
  ``{"values": [...], "property_source_doc_id": "<name>", "origins": None}``.

Running the command twice is idempotent: only the new JSONs are
regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Map output property name -> column in labeling CSV
NEW_PROPS: dict[str, str] = {
    # HER composition proxies (HER Stage A conditioning targets)
    "mean_h_affinity":          "mean_h_affinity",
    "best_site_h_affinity":     "best_site_h_affinity",
    "frac_thermoneutral_sites": "frac_thermoneutral_sites",
    "d_band_proxy":             "d_band_proxy",
    "valence_electron_conc":    "valence_electron_conc",
    "carbon_fraction":          "carbon_fraction",
    # CO2RR-to-CO composition proxies (CO2RR Stage A conditioning targets)
    "mean_co_affinity":         "mean_co_affinity",
    "best_site_co_affinity":    "best_site_co_affinity",
    "frac_co_selective_sites":  "frac_co_selective_sites",
    "mean_cooh_affinity":       "mean_cooh_affinity",
    "co2rr_selectivity":        "co2rr_selectivity",
    # Entropy / diversity
    "metal_mixing_entropy":     "metal_mixing_entropy",
    "n_metal_species":          "n_metal_species",
    # Economic constraint (PGM + heavy-REE fraction)
    "expensive_metal_fraction": "expensive_metal_fraction",
    # Carbide framework / conductivity
    "metal_cn_mean":            "metal_cn_mean",
    "mc_coordination":          "mc_coordination",
    "metal_packing_density":    "metal_packing_density",
    "metal_percolation_dim":    "metal_percolation_dim",
}


def _build_split(
    source_cache: Path,
    overlay_cache: Path,
    labeled_csv: Path,
    log=print,
) -> None:
    overlay_cache.mkdir(parents=True, exist_ok=True)
    new_files = {f"{p}.json" for p in NEW_PROPS}

    for f in source_cache.iterdir():
        tgt = overlay_cache / f.name
        if f.name in new_files:
            continue  # will be written fresh below
        if tgt.is_symlink():
            tgt.unlink()
        elif tgt.exists():
            continue  # don't clobber real file
        tgt.symlink_to(f.resolve())

    sid = np.load(source_cache / "structure_id.npy", allow_pickle=True)
    n = len(sid)
    sid_list = [str(x) for x in sid]
    log(f"[{overlay_cache.name}] cache has {n:,} structures")

    if not labeled_csv.exists():
        log(f"[{overlay_cache.name}] MISSING labeled CSV: {labeled_csv}")
        return
    df = pd.read_csv(labeled_csv)
    if "material_id" not in df.columns:
        log(f"[{overlay_cache.name}] missing material_id in {labeled_csv}")
        return
    df = df.set_index("material_id")
    log(f"[{overlay_cache.name}] labeled oxides: {len(df):,}")

    aligned = {prop: np.full(n, np.nan, dtype=np.float64) for prop in NEW_PROPS}
    hits = 0
    for i, s_id in enumerate(sid_list):
        if s_id in df.index:
            hits += 1
            row = df.loc[s_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for out_prop, col in NEW_PROPS.items():
                v = row.get(col, np.nan)
                try:
                    aligned[out_prop][i] = float(v)
                except (TypeError, ValueError):
                    pass
    log(f"[{overlay_cache.name}] matched material_ids: {hits:,} / {n:,}")

    for prop, arr in aligned.items():
        payload = {
            "values": arr.tolist(),
            "property_source_doc_id": prop,
            "origins": None,
        }
        out_path = overlay_cache / f"{prop}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)
        n_valid = int(np.isfinite(arr).sum())
        log(f"[{overlay_cache.name}]   wrote {out_path.name}: {n_valid:,} valid")


def build_overlay(
    source_cache_root: Path,
    overlay_root: Path,
    labeled_data_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    log=print,
) -> Path:
    """Create an overlay cache containing symlinks + new property JSONs."""
    source_cache_root = Path(source_cache_root)
    overlay_root = Path(overlay_root)
    labeled_data_dir = Path(labeled_data_dir)
    for split in splits:
        _build_split(
            source_cache=source_cache_root / split,
            overlay_cache=overlay_root / split,
            labeled_csv=labeled_data_dir / f"carbide_labeled_{split}.csv",
            log=log,
        )
    log("overlay build done.")
    return overlay_root


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", required=True, type=Path,
                    help="e.g. /path/to/mattergen/datasets/cache/alex_mp_20")
    ap.add_argument("--overlay", required=True, type=Path,
                    help="output overlay root, e.g. ./cache/alex_mp_20_oxide")
    ap.add_argument("--labels", required=True, type=Path,
                    help="directory with oxide_labeled_{train,val}.csv")
    args = ap.parse_args()
    build_overlay(args.source_cache, args.overlay, args.labels)
