# One-line patch to MatterGen for new property names

MatterGen validates every requested property against
`PROPERTY_SOURCE_IDS` in
[`mattergen/common/utils/globals.py`](https://github.com/microsoft/mattergen/blob/main/mattergen/common/utils/globals.py).
Our five new physics labels must be added to that list. The patch is
purely additive — no existing entries are touched.

```python
# in mattergen/common/utils/globals.py
PROPERTY_SOURCE_IDS = [
    "dft_mag_density",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "energy_above_hull",
    "formation_energy_per_atom",
    "space_group",
    "hhi_score",
    "ml_bulk_modulus",
    "chemical_system",
    "dft_band_gap",
    # --- CarbideMatterGen additions (HER) ---
    "mean_h_affinity",
    "best_site_h_affinity",
    "frac_thermoneutral_sites",
    "d_band_proxy",
    "valence_electron_conc",
    "carbon_fraction",
    "metal_mixing_entropy",
    "n_metal_species",
    "metal_cn_mean",
    "mc_coordination",
    "metal_packing_density",
    "metal_percolation_dim",
    # --- CarbideMatterGen additions (economic) ---
    "expensive_metal_fraction",
    # --- CarbideMatterGen additions (CO2RR-to-CO) ---
    "mean_co_affinity",
    "best_site_co_affinity",
    "frac_co_selective_sites",
    "mean_cooh_affinity",
    "co2rr_selectivity",
]
```

> The CO2RR run reuses `d_band_proxy`, `valence_electron_conc`,
> `metal_mixing_entropy`, `metal_packing_density`, `expensive_metal_fraction`
> from the lists above — only the five `*_co_*` / `*cooh*` / `co2rr_*` names are
> new. `deploy/setup_wsl_co2rr.sh` applies this whole block idempotently to your
> installed mattergen.

If you forget this patch, MatterGen will raise:

```
AssertionError: property_source_doc_id best_site_h_affinity not found in the database.
```

A cleaner alternative would be to register the names via a plugin hook,
but MatterGen as of the current pin (v1.0.3) does not expose one.

## Reverting

Delete the added lines. That fully restores the original file.
