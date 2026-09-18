# CarbideMatterGen — CO₂RR-to-CO variant

> **Generative discovery of *economical* high-entropy carbide (HEC) electrocatalysts for the
> electrochemical CO₂-to-CO reduction reaction (CO₂RR-to-CO).**

⚠️ **This is the CO₂RR sibling, not the water-electrolysis / HER version.**
This repository targets the **CO₂-to-CO reduction reaction**. It is a distinct sibling of the
original CarbideMatterGen project, which targets the **hydrogen-evolution / water-electrolysis
reaction (HER)**. The two share the same generative machinery but condition on different reaction
descriptors and screen with different surface free energies:

| | HER variant (original) | **CO₂RR variant (this repo)** |
|---|---|---|
| Reaction | H₂ evolution (water electrolysis) | **CO₂ → CO** |
| Primary descriptor | ΔG\_H\* → 0 (thermoneutral H) | **ΔG\_CO\* → 0, ΔG\_COOH\* < ΔG\_H\*** |
| Surface screen | \*H adsorption | **\*CO / \*COOH / \*H** |
| Conditioning set | HER carbide properties | **CO₂RR-to-CO + economic** |

If you are looking for the HER/water-splitting version, this is **not** it.

---

## What this does

An end-to-end **generative → screen → validate** pipeline that designs affordable, platinum-group-
metal-free high-entropy carbides predicted to reduce CO₂ to CO:

```
Stage A   economic-conditioned generation   (fine-tuned MatterGen, 9 properties)
Stage B   cheap composition screen          (CO-release + economic + HEC filters)
Stage C   MLIP surface ΔG screen            (fairchem / UMA: ΔG_CO, ΔG_COOH, ΔG_H)
Stage D   DFT validation                    (Quantum ESPRESSO: vc-relax → scf → DOS → PDOS)
Stage A'  solid-solution supercells (SQS)   (explicit-disorder representation)
```

The decisive feature is that **affordability is a *generation* objective**: the diffusion model is
fine-tuned on an `expensive_metal_fraction` label (PGM + heavy rare-earth) driven to zero, so the
economical subspace is sampled by construction rather than filtered post-hoc.

## Headline result

From **1024** generated structures → **1008** transition-metal carbides → **104** economical
CO-releasing structures spanning **90 unique compositions** → **24** taken into the MLIP surface
screen → **6** DFT-validated candidates. All six are DFT-metallic (a conductivity
prerequisite) with a hybridized C-2p / metal-d Fermi-level character. The screening tables behind
these numbers are in [`results/`](results/).

| Candidate | best ΔG\_CO (eV) | CO>H | DOS(E\_F) | dominant |
|---|---|---|---|---|
| Pr₃Y₂WC₆ | −0.001 | yes | 7.4 | C |
| La₅Y₄W₂C₉ | −0.001 | yes | 9.8 | C |
| Ce₂Y₂ZrWC₆ | +0.007 | yes | 6.7 | C |
| La₃Y₅(WC₃)₃ | −0.008 | yes | 12.4 | Y |
| Y₂ZrWC₈ | −0.010 | yes | 1.7 | W |
| CeY₂ZrWC₇ | −0.011 | yes | 3.3 | C |

## Repository layout

```
carbidemattergen/   core package (co2rr.py, co_surface_screen.py, labeling.py, ...)
scripts/            pipeline drivers (01–02 labels, 11 finetune, 21 generate,
                    30 cheap screen, 35 SQS, 51–53 QE input/run/PDOS)
configs/            hydra overlays (9 CO2RR + economic property embeddings)
notebooks/          re-runnable analysis: 17/18 surface screen · 19 generation stats ·
                    20 QE validation · 21 pipeline summary
docs/               design + run guide (docs/co2rr_design.md)
deploy/             WSL setup (setup_wsl_co2rr.sh)
data/               the generated La5Y4W2C9 structure (CIF)
results/            screening tables (top_CO, top6, surface screen, QE validation)
```

Large artifacts (the 16 GB `outputs/`, the overlay `cache/`, datasets and checkpoints) are
**not** tracked — they are regenerable from the scripts.

## Reproducing

Requires a working MatterGen install (CUDA), `fairchem`/UMA with gated `facebook/UMA` access for
the surface screen, and Quantum ESPRESSO + SSSP / pseudo-dojo pseudopotentials for the DFT stage.
See `docs/co2rr_design.md` and `deploy/setup_wsl_co2rr.sh`. End-to-end:

```bash
bash scripts/01_build_labels.sh ; bash scripts/02_inject_labels.sh     # labels + overlay
bash scripts/11_finetune_co2rr_economic.sh                             # Stage A (GPU)
bash scripts/21_generate_co2rr_economic.sh                             # Stage A generation
python scripts/30_cheap_screen_co2rr.py --gen-dir outputs/gen_co2rr_economic \
       --out outputs/gen_co2rr_economic/top_CO.csv                     # Stage B
python -m carbidemattergen.co_surface_screen --candidates <dir> --task omat \
       --device cuda --out outputs/screen_co2rr/co2rr.jsonl --site-cap 4  # Stage C (UMA)
python scripts/51_qe_gen_inputs.py ... ; python scripts/52_qe_run.py --qe-dir outputs/qe
python scripts/53_qe_pdos.py --qe-dir outputs/qe                        # Stage D (DFT + PDOS)
```

> Note: UMA's surface-screen energies use the **`omat` total-energy task** (not `oc20`), and the
> rare-earth pseudopotentials are norm-conserving (4f-in-core) because the SSSP REE PAWs are
> incompatible with the QE build used here — see `docs/co2rr_design.md`.

## From design to the experimental catalyst

The top candidate **La₅Y₄W₂C₉** was carried into the laboratory, and the experimental outcome is
*not* the predicted bulk phase. Mg-assisted heat treatment followed by HCl leaching removes the
La/Y-containing constituents, leaving an **ε-W₂C-rich catalyst** with exposed carbide regions,
dispersed residual carbon and a hydrated WOₓ surface. That reconstructed material converts CO₂ to
**ethanol (77.2% FE)** and **acetone (up to 19.0% FE)** with no detectable CO or C₁ liquid
products — i.e. the designed composition acts as a *sacrificial precursor* rather than as the
catalyst itself. The conditioning target used during generation was CO release (ΔG\_CO\* → 0);
the multicarbon selectivity is a consequence of the post-leaching surface, and is rationalised by
DFT in the paper below.

## Status

Computational pipeline complete. Experimental validation (synthesis, XRD, TEM/EDS, XPS,
electrochemical CO₂RR, GC-BID gas analysis, ¹H NMR) is **complete** and is reported in the
manuscript below.

## Citing this work

If you use this code or the generated structures, please cite:

> H. Lee, H. H. Nersisyan, J. H. Lee. *Rare-earth extraction from a generatively designed carbide
> yields an ε-W₂C/WOₓ catalyst that converts CO₂ into ethanol and acetone* (under review, 2026).

and the underlying generative model:

> C. Zeni *et al.* *A generative model for inorganic materials design*. **Nature** 639, 624–632
> (2025). https://doi.org/10.1038/s41586-025-08628-5

`CarbideMatterGen` is MatterGen fine-tuned on 5,275 transition-metal carbides with nine
conditioning properties; it is not a separate model architecture.

## License

See [`LICENSE`](LICENSE).
