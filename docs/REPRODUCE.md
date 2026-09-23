# Reproduce EvalDelta experiments

## Environment

Use Python 3.11 or 3.12. `pip install -e '.[dev,plots]'` is CPU-only. The frozen policy is in `configs/frozen.yaml`; its model hash is checked before held-out runs. The revision-1 frozen settings were committed in `f2c0067` before access to held-out outcomes; the revision-2 re-freeze was recorded in file timestamps and run manifests before the rerun, then committed afterward (see `docs/RESULTS.md` §5b). Repeated seeds are Monte Carlo replicates of a version pair, not independent real datasets.

## Data generation

`python benchmarks/prepare_real.py --sources covtype adult agnews bank` rebuilds the CPU matrices from upstream data (from UCI/OpenML and a Hugging Face parquet mirror; check each source's terms before downloading or redistributing). This may download data and takes longer than the default tests. The CIFAR-10 checkpoint cache is optional: `python benchmarks/prepare_vision.py --require-cuda`. Current partial cache covers v01–v11; `--through v11` assembles that labelled subset when later checkpoints are unavailable (checkpoints v12+ are blocked on a slow/corrupt upstream GitHub release download — see `docs/COMPUTE_LOG.md`). The research replay itself is CPU-only.

## Validation and frozen tests (revision 2: 4 source families)

When generated locally, validation artifacts are in `results/frozen/validate/` and `validate2/` (the prior 3-source revision-1 files are retained only in this working environment; the v1 freeze itself is in commit `f2c0067`). Row-level trial files are excluded from the public Git repository. Archived run manifests are in `docs/run_manifests/`, and the aggregate published tables are in `results/frozen/analysis/`. They were used to select settings. Do not retune the policy from held-out artifacts. The frozen CPU experiment was run with:

```bash
python benchmarks/prepare_real.py --sources bank                          # 4th source family
python benchmarks/train_paired_shift.py                                   # retrain on 4-source historical_train
python benchmarks/replay_sweep.py validate --reps 10 --workers 32
python benchmarks/replay_sweep.py validate2 --reps 10 --workers 32        # slice-ranking revision check
python benchmarks/replay_sweep.py freeze                                  # writes configs/frozen.yaml
python benchmarks/replay_sweep.py test --splits test --reps 20 --workers 32 --output results/frozen/test_cpu_v2
python benchmarks/replay_sweep.py nullreal --splits test --reps 200 --workers 32 --output results/frozen/nullreal_cpu_v2
python benchmarks/replay_sweep.py ablate --reps 20 --workers 32 --output results/frozen/ablate_cpu_v2
python benchmarks/replay_sweep.py test --splits test_transfer --reps 20 --workers 32 --output results/frozen/test_transfer_v2
python benchmarks/replay_sweep.py nullreal --splits test_transfer --reps 200 --workers 32 --output results/frozen/nullreal_transfer_v2
python benchmarks/analyze_results.py --suffix v2 --out-suffix v2
```

Each run writes a `manifest.json` with hashes of source, data and settings; existing outputs cannot be overwritten silently. The analysis script reads only complete runs, writes aggregate CSVs to `results/frozen/analysis/` and figures to `docs/figures/`. `--suffix` selects which `test_cpu_<suffix>`/etc. run directories to read (default `v1`); `--out-suffix` tags the output files so revisions don't overwrite each other. A fresh analysis can use `--allow-partial` to describe CPU results while transfer is still running; such an interim report must not be labelled final.

## Statistical cautions

Compare methods at the same maximum paid-call budget and report actual calls spent. `confirmed_regression` is an error-controlled decision only under the protocol's sampling assumptions. Source-level bootstrap intervals resample source families, not individual replay seeds; the small number of source families leaves wide uncertainty. All raw matrices and data-source rights need review before third-party redistribution. The software package itself contains no raw datasets or pretrained weights.

## Separate MAGIC external-source check

The sixth dataset has a [precommitted protocol](EXTERNAL_CASE_STUDY.md) and is **not** pooled into DeltaBench v2. Its source is UCI MAGIC Gamma Telescope (CC BY 4.0); the data are Monte Carlo simulated physics events. The download and row-level outcome matrix remain local. The plan commit `3283511` and implementation commit `67aa6a6` predate the first download. Reproduce with:

```bash
python benchmarks/prepare_magic.py
python benchmarks/run_magic_case_study.py test --workers 16
python benchmarks/run_magic_case_study.py null --workers 16
python benchmarks/analyze_magic.py
```

Run directories refuse overwrite. If an existing run directory is present, inspect its `manifest.json`; use a fresh checkout/output only for an intentional new replication. The aggregate CSVs are `results/frozen/analysis/magic_external_{real,null}.csv`.
