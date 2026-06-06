"""Build a physics-labeled transition-metal-carbide subset from alex_mp_20.

For every TM carbide in the source CSVs, computes the HER composition proxies
(`hydrogen`), the diversity descriptors (`extra_descriptors`) and the carbide
framework descriptors (`features`), and joins them onto the existing MatterGen
labels (formula, energy_above_hull, formation_energy_per_atom, dft_band_gap…).
The output CSVs feed `cache_overlay`, which injects them into the MatterGen
training cache without rewriting the structure cache.

Mirrors OxideMatterGen's labeling pipeline; the only domain change is the
anion: we keep C-containing TM compounds and exclude O/N/other anions so the
model learns the carbide chemistry rather than oxycarbides/carbonitrides.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymatgen.core import Structure

from .features import TRANSITION_METALS, compute_features
from .hydrogen import hydrogen_descriptors, carbon_fraction, expensive_metal_fraction
from .co2rr import co2rr_descriptors
from .extra_descriptors import metal_mixing_entropy, n_metal_species

# Anions that would make a structure NOT a (pure) carbide.
NON_CARBIDE_ANIONS: set[str] = {"O", "N", "F", "Cl", "Br", "I",
                               "S", "Se", "Te", "P", "As"}


def _elements(cs: str) -> set[str]:
    return {s for s in cs.split("-") if s} if isinstance(cs, str) else set()


def is_tm_carbide(chem_sys: str) -> bool:
    """Require C + at least one TM, and NO competing anion."""
    els = _elements(chem_sys)
    if "C" not in els:
        return False
    if not (els & TRANSITION_METALS):
        return False
    if els & NON_CARBIDE_ANIONS:
        return False
    return True


def is_tm_carbide_composition(comp) -> bool:
    """Strict carbide test on a pymatgen Composition (element-symbol based).

    Same logic as ``is_tm_carbide`` but operates on a Composition so it can be
    applied to generated structures during screening. Requires elemental carbon,
    at least one transition metal, and NO competing anion. Unlike
    ``is_tm_carbide_by_formula`` it does NOT misread the "C" in element symbols
    such as Cr, Co, Cu, Ca, Cs, Cd, Ce, Cl as carbon.
    """
    els = {el.symbol for el in comp.elements}
    if "C" not in els:
        return False
    if not (els & TRANSITION_METALS):
        return False
    if els & NON_CARBIDE_ANIONS:
        return False
    return True


def is_tm_carbide_by_formula(formula: str) -> bool:
    """DEPRECATED substring pre-filter — buggy for Cr/Co/Cu/etc. (the "C" in the
    symbol is read as carbon) and ignores competing anions. Use
    ``is_tm_carbide_composition`` for screening."""
    if not isinstance(formula, str) or "C" not in formula:
        return False
    return any(tm in formula for tm in TRANSITION_METALS)


def _process_row(row: dict) -> dict:
    cif = row.get("cif")
    if not isinstance(cif, str) or len(cif) < 50:
        return {"_status": "skip_no_cif", **row}
    try:
        s = Structure.from_str(cif, fmt="cif")
    except Exception as exc:  # noqa: BLE001
        return {"_status": f"parse_fail: {exc}", **row}

    syms = {sp.symbol for site in s for sp in site.species.elements}
    if "C" not in syms or not (syms & TRANSITION_METALS):
        return {"_status": "not_carbide", **row}

    comp = s.composition
    h = hydrogen_descriptors(comp)
    co = co2rr_descriptors(comp)
    feats = compute_features(s)
    fd = feats.to_dict() if feats is not None else {}

    out = dict(row)
    out.update({
        # --- HER composition proxies (HER fine-tune conditioning targets) -
        "mean_h_affinity":          h["mean_h_affinity"],
        "best_site_h_affinity":     h["best_site_h_affinity"],
        "frac_thermoneutral_sites": h["frac_thermoneutral_sites"],
        "d_band_proxy":             h["d_band_proxy"],
        "valence_electron_conc":    h["valence_electron_conc"],
        "carbon_fraction":          carbon_fraction(comp),
        # --- CO2RR-to-CO composition proxies (CO2RR fine-tune targets) ----
        "mean_co_affinity":         co["mean_co_affinity"],
        "best_site_co_affinity":    co["best_site_co_affinity"],
        "frac_co_selective_sites":  co["frac_co_selective_sites"],
        "mean_cooh_affinity":       co["mean_cooh_affinity"],
        "co2rr_selectivity":        co["co2rr_selectivity"],
        # --- entropy / diversity -----------------------------------------
        "metal_mixing_entropy":     metal_mixing_entropy(comp),
        "n_metal_species":          n_metal_species(comp),
        # --- economic constraint (PGM + heavy-REE fraction) --------------
        "expensive_metal_fraction": expensive_metal_fraction(comp),
        # --- carbide framework / conductivity ----------------------------
        "metal_cn_mean":            fd.get("metal_cn_mean", float("nan")),
        "mc_coordination":          fd.get("mc_coordination", float("nan")),
        "metal_packing_density":    fd.get("metal_packing_density", float("nan")),
        "metal_percolation_dim":    fd.get("metal_percolation_dim", float("nan")),
        "_status": "ok",
    })
    return out


def build_split(csv_path: Path, out_path: Path, n_workers: int = 32,
                max_rows: int = 0, log=print) -> pd.DataFrame:
    log(f"[{csv_path.name}] reading…")
    df = pd.read_csv(csv_path)
    log(f"[{csv_path.name}] loaded {len(df):,} rows")

    if "chemical_system" in df.columns:
        mask = df["chemical_system"].apply(is_tm_carbide)
        log(f"[{csv_path.name}] strict TM-carbide filter via chemical_system")
    else:
        col = "reduced_formula" if "reduced_formula" in df.columns else "pretty_formula"
        mask = df[col].apply(is_tm_carbide_by_formula)
        log(f"[{csv_path.name}] permissive formula filter (no chemical_system)")
    df_c = df[mask].reset_index(drop=True)
    log(f"[{csv_path.name}] accepted carbides: {len(df_c):,}")
    if max_rows > 0:
        df_c = df_c.head(max_rows)

    records = df_c.to_dict(orient="records")
    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_row, r) for r in records]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r.get("_status") == "ok":
                r.pop("cif", None)
                r.pop("_status", None)
                results.append(r)
            if i % 1000 == 0:
                log(f"[{csv_path.name}] {i}/{len(records)} kept={len(results)} "
                    f"{i/(time.time()-t0):.0f}/s")

    out_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log(f"[{csv_path.name}] saved {len(out_df):,} labeled carbides → {out_path}")
    return out_df


def build_all(source_dir: Path, output_dir: Path,
              splits: Iterable[str] = ("train", "val"),
              n_workers: int | None = None, max_rows: int = 0) -> dict[str, Path]:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = min(48, os.cpu_count() or 4)

    log_path = output_dir / "label_build.log"
    outputs: dict[str, Path] = {}
    with open(log_path, "w") as lf:
        def log(msg: str) -> None:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            print(line); lf.write(line + "\n"); lf.flush()
        log(f"source_dir = {source_dir}")
        log(f"output_dir = {output_dir} workers = {n_workers} max_rows = {max_rows}")
        for split in splits:
            csv = source_dir / f"{split}.csv"
            if not csv.exists():
                log(f"MISSING: {csv}"); continue
            out = output_dir / f"carbide_labeled_{split}.csv"
            build_split(csv, out, n_workers=n_workers, max_rows=max_rows, log=log)
            outputs[split] = out
    return outputs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    build_all(a.source_dir, a.output_dir, n_workers=a.workers, max_rows=a.max_rows)
