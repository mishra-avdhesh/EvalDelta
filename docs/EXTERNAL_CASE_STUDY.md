# Independent-source case study: MAGIC Gamma Telescope

## Frozen plan (recorded before downloading or inspecting source outcomes)

This is an **exploratory external-source robustness check**, separate from the v1/v2 DeltaBench claims. The source is [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), 19,020 Monte Carlo simulated particle events with 10 numeric features and binary labels, licensed CC BY 4.0. It is a new task and dataset for this project, but its simulated events are not production observations. The model's existing four-source historical training and `configs/frozen.yaml` are not modified. This plan is fixed before downloading the dataset or viewing candidate outcomes. Do not treat this as an additional independent pre-registered confirmation of v1/v2, whose results were already seen.

- Deterministic split: NumPy `default_rng(20260923).permutation(N)`; first 10,000 rows form the evaluation pool, remaining rows train the versions. No evaluation-pool label is used to fit models or choose settings. All versions use the same train/pool split.
- Release ladder: `v01` StandardScaler + LogisticRegression (`C=1`, `max_iter=1000`); `v02` HistGradientBoosting (`max_iter=100`, `max_leaf_nodes=31`, `random_state=0`); `v03` compressed HistGradientBoosting (`max_iter=50`, `max_leaf_nodes=7`, `random_state=0`); `v04` `v02` architecture trained and evaluated without the **first raw feature** (`fLength`). Fixed pairs: `v01→v02`, `v02→v03`, `v02→v04`. No pair is discarded based on direction or difficulty.
- Paired 0/1 loss; positive new-minus-old loss means regression. A pre-existing candidate outcome is stored only in the replay oracle; selectors see old outcome, permitted historical outcomes from versions preceding the candidate, and public metadata. Use the repository's leakage and budget checks.
- Global comparison: budgets 200 and 500 paid candidate calls, 20 deterministic replay seeds per pair/method/budget. Methods: B1 fixed exact McNemar (one look, i.i.d. superpopulation assumption), B8 sequential finite-pool betting, adaptive IPW uniform, and frozen `OURS2_paired_shift_ipw`. Settings and historical model are the current four-source freeze, never tuned on MAGIC. Main descriptive quantity: confirmed-regression frequency on pairs with true pool Δ > 0, plus paid calls. Show each pair separately; do not bootstrap a confidence interval across one source family.
- Boundary-null check: at budgets 200 and 500, B8 and OURS2 each get 200 deterministic replay seeds per pair, with regression margin set to `max(0, true pool Δ)` as in `phase_nullreal`. This margin is for calibration scoring only, not the primary detection comparison. Report numerator/denominator per method and budget; repeated seeds of the same pair are Monte Carlo replicates, not independent external datasets.
- The run records source, code, model and frozen-config hashes before replay and refuses to overwrite outputs. If no pair has positive Δ, report the external check as non-informative for detection power. Observed null alarms are descriptive; values below 5% do not prove a general type-I guarantee.

Dataset credit: Bock, R. (2004), *MAGIC Gamma Telescope*, UCI Machine Learning Repository, [DOI 10.24432/C52C8B](https://doi.org/10.24432/C52C8B), CC BY 4.0.

The official dataset and raw/derived row-level matrices are **not** part of the code release. Users regenerate them using `benchmarks/prepare_magic.py`, then run `benchmarks/run_magic_case_study.py`. Aggregate findings and limitations will be appended below only after the frozen plan has a Git commit predating first access to the new source outcomes.

## Results (recorded after the frozen run)

The plan was committed as `3283511` and the implementation as `67aa6a6` **before** the official archive was downloaded. Its SHA-256 is `252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d`. Archived run manifests are `docs/run_manifests/magic_external_{test,null}_v1.json` (local runs also have a `manifest.json`); aggregate CSVs are `results/frozen/analysis/magic_external_{real,null}.csv`. All **480** detection and **2,400** null trials completed with no evaluation errors or budget violations. The evaluation pool contains 10,000 rows; model training used the separate 9,020 rows.

| Fixed pair | True pool Δ | B200 confirmations (B1 / B8 / IPW uniform / OURS2), each of 20 | B500 confirmations, each of 20 |
|---|---:|---:|---:|
| Logistic `v01` → HGB `v02` | −0.0887 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| HGB `v02` → compressed `v03` | +0.0183 | 3 / 1 / 3 / 0 | 7 / 4 / 11 / 10 |
| HGB `v02` → feature-drop `v04` | +0.0090 | 3 / 0 / 0 / 0 | 5 / 4 / 1 / 3 |

On the two genuinely regressive pairs combined, OURS2 confirmed **0/40** at B200 versus B1 **6/40**. At B500 it confirmed **13/40** versus B1 **12/40**, while IPW uniform also confirmed **12/40**. This is a mixed result on **one source family**; a source-level confidence interval or broad transfer claim would be unjustified. Each method's paid-call counts, including early stops, are in the aggregate CSV.

For the two positive-Δ pairs at the boundary-null margin, B8 false alarms were **2/400** at B200 and **10/400** at B500; OURS2 had **0/400** and **2/400**, respectively. The improving `v01→v02` pair is an interior null (`Δ < 0`, margin 0) and had zero false alarms in 200 seeds per method/budget. These are Monte Carlo replays of three fixed pairs, not 2,400 independent datasets. Observed rates do not prove a universal guarantee.

**Interpretation:** the independent source demonstrates that the packaged replay/test workflow can run on a task absent from historical training. It does **not** establish an adaptive efficiency win. The simulated source and narrow pair ladder further limit generalization; see the [UCI source card](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).
