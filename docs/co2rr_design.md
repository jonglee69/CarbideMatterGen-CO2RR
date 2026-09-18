# CO2RR-to-CO pipeline (sibling of the HER pipeline)

CarbideMatterGen was built to inverse-design **non-PGM HER** carbide catalysts
(ΔG_H\* ≈ 0). This document describes the **CO2RR-to-CO** variant that reuses the
*entire* machinery — labels → overlay → property-conditioned fine-tune → guided
generation → cheap screen → fairchem surface screen → SQS template → AL loop —
and swaps only the **reaction descriptor**: from atomic-H adsorption (HER) to the
CO2-reduction-to-CO descriptors. The economic constraint
(`expensive_metal_fraction`) and the high-entropy "cocktail" design are carried
over unchanged.

## Why carbides for CO2RR → CO

CO2RR to CO proceeds (Hori; Peterson 2010; Bagger/Nørskov 2017):

```
CO2(g) + * + (H+ + e-)  ->  *COOH       (CO2 activation)
*COOH    + (H+ + e-)    ->  *CO + H2O   (second PCET)
*CO                     ->  CO(g) + *   (desorption)
```

A selective CO producer needs two things:

1. **\*CO binds weakly enough to desorb** as CO (ΔG_CO ≈ 0, slightly positive).
   Too strong ⇒ poisoning / over-reduction to hydrocarbons (the Cu pathway).
   Au/Ag/Zn are the canonical CO producers precisely because they bind \*CO weakly.
2. **CO2RR out-competes HER.** The selectivity contrast is \*COOH vs \*H binding:
   a CO-selective surface stabilises \*COOH more than \*H (Bagger 2017).

The carbide hypothesis is the same "Pt-like carbide" d-band effect (Levy &
Boudart 1973): carbide formation weakens the over-strong CO binding of cheap
mid-transition metals (Mo, W, Fe…) **toward the CO-release window**, so a
high-entropy carbide can reach Au/Ag-like CO selectivity **without** precious
metals — the whole point of the economic constraint. Mo2C is the canonical
CO2RR-active carbide.

## Descriptor mapping (HER → CO2RR)

| role | HER (`hydrogen.py`) | CO2RR (`co2rr.py`) |
|---|---|---|
| **primary, extremal** | `best_site_h_affinity` → 0 | `best_site_co_affinity` → 0 (CO releases) |
| **primary, fraction** | `frac_thermoneutral_sites` → high | `frac_co_selective_sites` → high |
| mean binding | `mean_h_affinity` | `mean_co_affinity` |
| activation / 2nd descriptor | — | `mean_cooh_affinity` (CO2 activation, onset) |
| selectivity | — | `co2rr_selectivity` = ΔG_H − ΔG_COOH (CO over H2) |
| electronic | `d_band_proxy`, `valence_electron_conc` | reused verbatim |
| entropy / framework / economic | `metal_mixing_entropy`, `metal_packing_density`, `expensive_metal_fraction` | reused verbatim |

`CO_OPTIMUM = 0.0 eV`, `CO_SELECTIVE_WINDOW = 0.20 eV` (CO binding is a coarser
descriptor than H, hence the wider window than the HER ±0.10 eV).

The composition proxy tables `_CO_ADS_CARBIDE` / `_COOH_ADS_CARBIDE` are
**literature-informed, monotonic/relative estimates** (same status as the HER
`_H_ADS_CARBIDE` table), calibrated so coinage/late metals sit near the optimum
and carbide-weakened mid-TMs are recoverable by entropy mixing. They are
**refined by the active-learning DFT + fairchem loop** — they are not DFT
accurate on their own.

### Honest caveat on `co2rr_selectivity`

The descriptor `ΔG_H − ΔG_COOH` is positive when \*COOH is more stable than \*H
(CO2RR thermodynamically favoured). Note that Au/Ag come out *negative* here:
their real CO selectivity is driven by **weak \*CO binding (facile desorption)**
plus **kinetically slow HER**, not by thermodynamically out-binding \*H. So treat
`co2rr_selectivity` as a secondary steering knob; the **primary** signal is
`best_site_co_affinity` / `frac_co_selective_sites`, validated by the Stage B
ΔG_CO\* / ΔG_COOH\* numbers.

## Economic policy and precious coinage (Au/Ag)

`expensive_metal_fraction` (in `hydrogen.py`) penalises **PGM + heavy REE**. The
benchmark CO producers **Au/Ag are precious but not PGM**, so they are *not*
penalised by default. The carbide-HEC route is meant to reach the CO-release
window with cheap carbide-weakened d-block metals instead — but if you want
generation to avoid precious coinage too, add one line:

```python
# carbidemattergen/hydrogen.py
_EXPENSIVE_METALS = _PGM | _EXPENSIVE_REE | {"Au", "Ag"}   # CO2RR precious-coinage option
```

Then rebuild the overlay (step 2) so the relabeled `expensive_metal_fraction`
takes effect. Left as a deliberate, documented choice rather than silently
changing your validated HER economic policy.

## Stage B — surface ΔG_CO\* / ΔG_COOH\* (`co_surface_screen.py`)

Mirrors `surface_screen.py` exactly (fairchem **v2 / UMA**, `omat` task head,
most-stable-termination-per-Miller selection, bottom-fixed relaxations, per-facet
reporting, `--shard`/resume), but places **\*CO, \*COOH, \*H** at each site and
computes, under the computational hydrogen electrode (U=0 vs RHE):

```
ΔG_CO*   = E(*CO)   − E(*) − E(CO_g)                 + 0.10
ΔG_COOH* = E(*COOH) − E(*) − E(CO2_g) − 1/2 E(H2_g)  + 0.41
ΔG_H*    = E(*H)    − E(*) − 1/2 E(H2_g)             + 0.24
```

Per candidate it reports `best_dG_CO` (closest to optimum), `frac_co_selective`,
`min_dG_COOH` (activation onset), `limiting_potential_V`
(`U_L = −max(ΔG_COOH, ΔG_CO − ΔG_COOH)`), and `co_selective_over_h`.

## End-to-end on WSL (RTX 5070)

The CO2RR run shares steps 1–2 with HER (labels are additive) and the SQS
template stage (`scripts/35_build_solid_solutions.py`, reaction-agnostic).

```bash
export VENV=...                    # your mattergen virtualenv (or leave unset if active)
export BASE_MODEL=/path/to/mattergen_base   # the base checkpoint dir

# 0. one-time: whitelist the new property names in your installed mattergen
bash deploy/setup_wsl_co2rr.sh     # idempotent globals.py patch + editable install

# 1-2. labels + overlay (reuse if you already built them for HER; rebuild so the
#      *_co_* JSONs exist):
bash scripts/01_build_labels.sh
bash scripts/02_inject_labels.sh

# 3. fine-tune the 9-property CO2RR economic adapter
bash scripts/11_finetune_co2rr_economic.sh

# 4. guided generation (RTX-5070 defaults; raise N_* on bigger GPUs)
bash scripts/21_generate_co2rr_economic.sh

# 5. cheap composition pre-filter (CO-release + selectivity + economic)
python scripts/30_cheap_screen_co2rr.py \
  --gen-dir outputs/gen_co2rr_economic --out outputs/gen_co2rr_economic/top_CO.csv

# 5'. (optional) expand top candidates into real HEC supercells
python scripts/35_build_solid_solutions.py ...

# 6. fairchem v2 (UMA) CO2RR surface screen (needs facebook/UMA access)
python -m carbidemattergen.co_surface_screen \
  --candidates outputs/sqs --model uma-s-1p1 --task omat --device cuda \
  --out outputs/screen_co2rr/co2rr.jsonl --site-cap 6
```

RTX 5070 (12 GB) notes: fine-tune/generate batch defaults are 16 (vs 64/50 on an
A100); raise `BATCH_SIZE`/`N_CHUNKS`/`NUM_BATCHES` if VRAM allows. Stage B is the
slow part — use `--site-cap` and `--shard i/N` across runs.
