# CarbideMatterGen — design notes

## Goal
Inverse-design **non-platinum-group HER electrocatalysts**: medium/high-entropy
transition-metal **carbides** whose hydrogen adsorption free energy ΔG_H* is as
close to thermoneutral (Pt ≈ −0.09 eV) as possible, while remaining
**electrically conductive (metallic)** and **thermodynamically stable**
(entropy-stabilized, hence durable against long-term degradation).

## The central problem: bulk → surface
MatterGen generates **bulk periodic crystals** and conditions on bulk/
composition-derived properties. But ΔG_H* is a **surface adsorption** property
(facet-, site-, and coordination-dependent). The whole design is organized
around bridging that gap:

| Stage | What it does | Sees ΔG_H*? |
|---|---|---|
| **A — generative** | fine-tuned MatterGen biased toward stable, metallic, near-thermoneutral high-entropy carbides via cheap composition proxies | proxy only |
| **B — surface screen** | fairchem (OC20/OC22) ΔG_H* on enumerated slabs + sites | real (ML) |
| **C — DFT validate** | QE: metallicity (DOS@E_F) + ΔG_H* on best facet | real (DFT) |
| **AL — active loop** | recalibrate the proxy tables against B/C, regenerate | closes the loop |

## Why the previous HEA carbides failed (and the key insight)
The user synthesized WMoTaC (medium-entropy) and ZrTiHfNbVC (high-entropy) and
never reached Pt-like ΔG_H. The composition proxy reproduces *why*:

```
formula        mean    best  frac
Mo2C            0.00    0.00  1.00   <- canonical Pt-like carbide
ZrTiHfNbVC     -0.19   -0.10  0.20   <- all strong binders: best stays negative
WMoTaC          0.03   -0.15  0.50   <- Mo helps but Ta/W mix off-target
MoWNiVCrC       0.03    0.00  0.75   <- weak binder (Ni) + Mo -> best≈0, many active sites
```

HER does **not** require the *average* ΔG_H to be 0 — it requires the existence
and **density of near-thermoneutral active sites**. High entropy is valuable
precisely because it widens the distribution of local site environments, raising
the chance that some sites sit at ΔG_H ≈ 0 (the "cocktail effect"). A carbide
built only from strong-binding early TMs (Zr/Ti/Hf/Nb/V) has its *entire*
distribution shifted negative — no amount of mixing rescues it. The fix is to
mix strong binders (which carbide formation already pulls toward 0) with at
least one intrinsically weaker binder (late TM such as Ni/Cu, or Mo/W).

This is why the primary conditioning target is **`best_site_h_affinity`**
(closest-to-thermoneutral site, the "best-link" analogue of OxideMatterGen's
weakest-link `min_cation_anodic_stability`) together with
**`frac_thermoneutral_sites`**, not the mean.

## Conditioning properties
Computed in `carbidemattergen/`:

| property | module | role |
|---|---|---|
| `mean_h_affinity` | hydrogen | average ΔG_H proxy |
| `best_site_h_affinity` | hydrogen | **primary** — drive → 0 |
| `frac_thermoneutral_sites` | hydrogen | **primary** — drive → high |
| `d_band_proxy` | hydrogen | Hammer–Nørskov electronic descriptor |
| `valence_electron_conc` | hydrogen | HEA VEC descriptor |
| `carbon_fraction` | hydrogen | MC vs M2C vs M3C stoichiometry |
| `metal_mixing_entropy` | extra_descriptors | drive → high (≥1.5R = HEA) |
| `n_metal_species` | extra_descriptors | companion to entropy |
| `metal_cn_mean`, `mc_coordination` | features | framework character |
| `metal_packing_density`, `metal_percolation_dim` | features | conductivity proxy |
| `energy_above_hull`, `formation_energy_per_atom`, `dft_band_gap` | MatterGen base | stability + metallicity (gap→0) |

## The proxy tables are hypotheses, not ground truth
`_H_ADS_CARBIDE` (effective carbide-site ΔG_H per metal) and `_D_BAND_CENTER`
in `hydrogen.py` are literature-calibrated estimates (Mo2C≈0, Nørskov volcano,
Levy–Boudart carbide d-band effect). They are **deliberately refined** by the
active-learning loop: every DFT/fairchem ΔG_H* measurement is a new
(composition → ΔG_H*) datapoint to fit the table against. Early iterations bias
the search direction; later iterations sharpen it.

## Pipeline
```
01_build_labels.sh      carbide subset of alex_mp_20 -> physics labels
02_inject_labels.sh     overlay labels onto MatterGen cache (symlink + JSON)
   (docs/patch_mattergen.md: whitelist new property names)
10_finetune_carbide.sh  fine-tune MatterGen on the property set
20_generate.sh          CFG generation, w=1.0 & 2.5, targets best≈0 / entropy↑ / gap→0
30_cheap_screen.py      recompute proxies, keep top-K promising/novel/stable
40_active_learning_loop.sh  Stage B fairchem ΔG_H* on top-K  ->  recalibrate -> repeat
50_dft_validate.sh      QE: metallicity + ΔG_H* on best facet (reuse OxideMatterGen QE)
```

## Open questions / next refinements
- Replace the per-element `_H_ADS_CARBIDE` lookup with a small learned surrogate
  (composition + local-coordination → ΔG_H*) trained on Catalysis-Hub / OC data,
  once the AL loop has accumulated enough in-domain points.
- Add a surface-segregation check: the active site must be *exposed*; condition
  or post-filter on predicted surface composition, not just bulk.
- Consider an explicit metallicity label (DOS@E_F proxy) beyond band_gap→0.
