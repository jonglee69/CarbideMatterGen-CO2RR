# HANDOFF — CO2RR-to-CO pipeline, run locally on WSL (RTX 5070)

This bundle is **CarbideMatterGen** with a new **CO2RR-to-CO** variant added on
top of the existing HER + economic pipeline. It was prepared on a cloud box and
is meant to be unzipped on your WSL machine and continued with a local Claude
Code session. Everything below is what a fresh local session needs to know.

## What this is
The HER pipeline (ΔG_H\* → 0) now has a CO2RR sibling that reuses the *whole*
machinery (labels → overlay → 9-property economic fine-tune → guided generation
→ cheap screen → fairchem v2/UMA surface screen → SQS template → AL loop) and
only swaps the reaction descriptor to CO2-reduction-to-CO. Read
`docs/co2rr_design.md` first — it has the science, the descriptor mapping, and
the end-to-end commands.

## What was added (new/changed files)
```
carbidemattergen/co2rr.py                 NEW  CO2RR composition proxies (sibling of hydrogen.py)
carbidemattergen/co_surface_screen.py     NEW  Stage B: ΔG_CO*/ΔG_COOH*/ΔG_H* (fairchem v2 UMA)
carbidemattergen/labeling.py              EDIT now also emits the 5 CO2RR labels
carbidemattergen/cache_overlay.py         EDIT NEW_PROPS includes the 5 CO2RR labels
carbidemattergen/__init__.py              EDIT imports co2rr
scripts/11_finetune_co2rr_economic.sh     NEW  9-property CO2RR economic fine-tune
scripts/21_generate_co2rr_economic.sh     NEW  guided generation (RTX-5070 defaults)
scripts/30_cheap_screen_co2rr.py          NEW  cheap CO-release + selectivity + economic filter
configs/.../property_embeddings/*.yaml     NEW  5 CO2RR property-embedding configs
docs/co2rr_design.md                      NEW  full design + run guide
docs/patch_mattergen.md                   EDIT whitelist now lists CO2RR + economic names
deploy/setup_wsl_co2rr.sh                  NEW  editable install + idempotent globals.py patch
```
The 5 CO2RR conditioning properties: `best_site_co_affinity`,
`frac_co_selective_sites`, `mean_cooh_affinity`, `co2rr_selectivity`,
`mean_co_affinity` (the last is labeled but not in the 9-property set; available
if you want to swap it in).

## Prerequisites on WSL (you said these are ready)
- mattergen installed and working on CUDA (RTX 5070). ✔
- `mattergen_base` checkpoint dir (point `BASE_MODEL` at it).
- For steps 1–2 (labeling): the `alex_mp_20` source CSVs + the MatterGen training
  cache you used for the HER run (same inputs — labels are additive).
- For step 6 (Stage B): fairchem **v2** (>=2.x) + access to the gated HF repo
  `facebook/UMA` (`hf auth login` or `HF_TOKEN`). No checkpoint is bundled.

## First commands (run with local Claude)
```bash
cd <unzip-dir>/CarbideMatterGen
export VENV=/path/to/your/mattergen/venv      # or leave unset if already active
export BASE_MODEL=/path/to/mattergen_base

bash deploy/setup_wsl_co2rr.sh                # editable install + whitelist patch + import check
python -m carbidemattergen.co2rr             # sanity: proxy table (Au/Ag/Zn near optimum)
```
Then follow the 7-step run in `docs/co2rr_design.md` (§ "End-to-end on WSL").
If you already built the HER overlay cache, just re-run `scripts/02_inject_labels.sh`
so the new `*_co_*` property JSONs are written, then go straight to
`scripts/11_finetune_co2rr_economic.sh`.

## RTX 5070 (12 GB) notes
- Fine-tune / generate batch defaults are **16** (A100 used 64 / 50). Raise
  `BATCH_SIZE`, `N_CHUNKS`, `NUM_BATCHES` if VRAM allows; lower if you OOM.
- Stage B is the slow part: use `--site-cap 6` and `--shard i/N` across runs.

## Open decisions for you (documented, not silently chosen)
1. **Precious coinage (Au/Ag).** The economic filter penalises PGM + heavy REE
   only; Au/Ag are *not* penalised by default. The carbide-HEC route is meant to
   reach the CO-release window with cheap metals instead. To also avoid Au/Ag,
   add `{"Au","Ag"}` to `_EXPENSIVE_METALS` in `hydrogen.py` and rebuild the
   overlay. See `docs/co2rr_design.md` § "Economic policy".
2. **Proxy calibration.** `_CO_ADS_CARBIDE` / `_COOH_ADS_CARBIDE` are
   literature-informed, monotonic estimates (same status as the HER table) — they
   are meant to be recalibrated against your Stage B (UMA) + DFT numbers via the
   active-learning loop, not trusted as DFT-accurate out of the box.
3. **`co2rr_selectivity` sign caveat** for Au/Ag — see the design doc; rely on
   `best_site_co_affinity` / `frac_co_selective_sites` as the primary signal.

## Validation done before shipping (on CPU, no GPU/UMA needed)
- `python -m carbidemattergen.co2rr` → Au/Ag/Zn at CO optimum, Pt strongly
  bound, `MoWCuAgZnC` HEC: best ≈ 0, frac ≈ 0.6 (cocktail effect reproduced).
- `co_surface_screen.screen_candidate` end-to-end with a dummy ASE calculator:
  slab/termination enumeration, *CO/*COOH/*H placement, ΔG aggregation,
  limiting-potential and per-facet bookkeeping all run.
- globals.py patcher is idempotent and produces a valid, de-duplicated list.
- All imports succeed; overlay exposes 18 properties.

Not run here (needs your GPU / gated UMA weights): the actual fine-tune,
generation, and the real fairchem v2 surface screen.
