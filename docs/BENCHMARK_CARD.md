# DeltaBench benchmark card

## Intended use

DeltaBench is an offline replay benchmark for comparing old and new classifier versions under a fixed number of paid candidate evaluations. Full candidate outcomes are stored in an oracle. Selectors receive cached old outcomes, metadata and permissible historical-version signals; they cannot read the target candidate's unqueried outcomes. The primary comparisons are paired 0/1 loss changes. Larger `Delta = new_loss - old_loss` means harm.

## Sources and split

| Source | Task | Public pool | Version pairs in frozen experiment | Provenance |
|---|---|---:|---:|---|
| Adult | tabular classification | 10,000 | 12 train, 9 validation, 9 held-out test | OpenML/UCI Adult, derived outcomes from local scikit-learn model versions |
| Covertype | tabular classification | 10,000 | 12 train, 9 validation, 9 held-out test | scikit-learn/Covertype, derived outcomes from local model versions |
| AG News | text classification | 7,600 | 12 train, 9 validation, 9 held-out test | AG News test data, derived outcomes from local text classifier versions |
| Bank Marketing | tabular classification | 10,000 | 12 train, 9 validation, 9 held-out test | OpenML/UCI Bank Marketing, derived outcomes from local scikit-learn model versions (added in results revision 2 to tighten source-family uncertainty, see docs/RESULTS.md) |
| CIFAR-10 | image classification | 10,000 | 27 held-out transfer pairs across 11 cached versions | CIFAR-10 test split; chenyaofo PyTorch checkpoints and labelled pipeline changes |

The exact pool sizes and licenses are read from `benchmarks/data/*/versions.json` and verified by `benchmarks/prepare_real.py` / `benchmarks/prepare_vision.py`; do not treat this table as a substitute for those manifests. Vision models v12–v15 are omitted because the v12 checkpoint was corrupt. The `ladder_complete: false` flag is included in the CIFAR-10 manifest. Results for v04→v10 and earlier cached versions remain reproducible.

Pairs are split by **candidate version**, not random item rows. Shared source pools, upstream data, prior checkpoints and repeated replay seeds mean pairs are not fully independent. The transfer family was withheld from selector training and validation. Injection scenarios are marked separately and never described as real model upgrades.

## Separate external-source case study

UCI MAGIC Gamma Telescope is a **sixth dataset**, evaluated only in the separately committed protocol at [EXTERNAL_CASE_STUDY.md](EXTERNAL_CASE_STUDY.md). Its 19,020 records are Monte Carlo simulated physics events, not production observations. A 10,000-row evaluation pool and 9,020-row training split are deterministic; three fixed model-version pairs are tested without retraining PairedShift. The source is [CC BY 4.0](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). MAGIC is **not** included in the v1/v2 DeltaBench bootstrap or used to tune the four-source frozen settings.

## Scenarios

Real version changes are evaluated without injection. Additional controlled `null`, `improvement`, `global_degradation`, `slice_only`, `rare_severe`, and `compensating` outcome modifications probe behavior; they share a real background pair. A pipeline perturbation such as fp16 or JPEG ingestion is labelled as a pipeline change, not a natural architecture upgrade.

## Licensing and distribution

The EvalDelta **code** uses Apache-2.0. That license does not grant rights to upstream datasets or model checkpoints. The repository does not redistribute raw Adult, AG News, CIFAR-10 or model checkpoint files. Derived per-item prediction matrices require an owner review of each upstream source's redistribution terms before publication; they are excluded from Git. The Git release includes source inventories and aggregate result tables, with attribution. The `benchmarks/data/raw/` directory is ignored by Git. The small CI example generates synthetic tables locally; no example table is stored in Git.

## Measurement limits

The suite contains only a few task families and uses the same benchmark pools repeatedly. Accuracy changes are not a proxy for fairness, clinical safety or arbitrary LLM output quality. `mcnemar_exact` needs its documented superpopulation sampling assumption; finite-pool conclusions use the betting confidence sequence. Injection stress scenarios help calibration but do not establish effectiveness in production deployment. See `docs/STATISTICAL_PROTOCOL.md` and `docs/RESULTS.md`.
