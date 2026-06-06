# Label QC — alex_mp_20 carbide subset (v0.1)

Built with `carbidemattergen.labeling` on the OxideMatterGen-vendored
alex_mp_20 source CSVs (`mattergen/datasets/alex_mp_20/{train,val}.csv`).

## Counts
| split | source rows | labeled carbides |
|---|---|---|
| train | 607,683 | 5,275 |
| val | 67,521 | 577 |

Filter: structure contains C + ≥1 transition metal, no competing anion
(O/N/F/Cl/Br/I/S/Se/Te/P/As).

## Label health (train, 5,275 rows)
| label | mean | median | min | max | NaN% |
|---|---|---|---|---|---|
| mean_h_affinity | −0.110 | −0.117 | −0.45 | +0.50 | 0.0 |
| best_site_h_affinity | −0.112 | −0.100 | −0.45 | +0.50 | 0.0 |
| frac_thermoneutral_sites | 0.208 | 0.000 | 0 | 1 | 0.0 |
| d_band_proxy | −1.978 | −1.300 | −8.0 | −0.5 | 0.0 |
| valence_electron_conc | 5.538 | 5.400 | 2.29 | 10.86 | 0.0 |
| carbon_fraction | 0.248 | 0.200 | 0.05 | 0.667 | 0.0 |
| metal_mixing_entropy | 0.796 | 0.693 | 0 | 1.099 | 0.0 |
| metal_cn_mean | 8.11 | 9.0 | 0 | 18 | 0.0 |
| mc_coordination | 1.98 | 2.0 | 0 | 6.67 | 0.0 |
| metal_packing_density | 0.028 | 0.024 | 0.003 | 0.086 | 0.0 |
| metal_percolation_dim | 2.215 | 2.0 | 0 | 3 | **38.8** |

All composition proxies are fully populated. **`metal_percolation_dim` is NaN
for 38.8%** of rows — single-metal-site cells or carbides whose M–M contacts
exceed the 3.2 Å cutoff. It is therefore an *optional* conditioning label, not
a core target; use it only as a soft conductivity pre-filter where present.

## Entropy / diversity coverage
| distinct metals | count |
|---|---|
| 1 | 231 |
| 2 | 2,524 |
| 3 | 2,518 |
| 4 | 2 |

- Structures with **≥3 metals: 2,520**; of those **727 have
  |best_site_h_affinity| < 0.1 eV** — the primary conditioning target is
  well-populated, so fine-tuning has in-domain examples.
- The dataset tops out at 4 distinct metals — there are essentially **no true
  5-component HEA carbides** in alex_mp_20. Reaching ZrTiHfNbVC-style 5-metal
  space is exactly what classifier-free guidance toward high
  `metal_mixing_entropy` must *extrapolate* to; the model will not have seen it.

## Caveat — what counts as a "metal"
`n_metal_species` / `metal_mixing_entropy` count every non-(C,N,O,B,H,Si,
halogen,…) element as a metal, so **lanthanides and main-group metals (La, Pr,
Nd, Sm, Dy, Y, Al, Ga, Ge, Zn, …) are included**. Many ≥3-metal hits are thus
rare-earth carbides whose only table-listed (and genuinely H-active) site is a
late TM — e.g. La2Zn2Ru2C, DyGa2Ru2C, Tb2Al2Os2C, where `best_site` = Ru/Os
(−0.05). `_H_ADS_CARBIDE` has no entry for the RE / main-group elements, so
`best_site_h_affinity` is correctly evaluated over the catalytically relevant
TM sites only — but it means the "entropy" of these rows is partly spectator
chemistry. A future refinement: weight/restrict entropy to the d-block, or add
an explicit "active-metal entropy" descriptor.

## Verdict
Labeling pipeline runs end-to-end on real data with clean composition proxies
and a well-populated target region. Two things to carry forward: (1) treat
`metal_percolation_dim` as optional, and (2) the genuine 5-metal HEA region is
out-of-distribution and must come from guided generation, not the training set.
