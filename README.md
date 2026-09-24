# EvalDelta

EvalDelta checks whether a new model version performs worse than the version it replaces. It compares both versions on the same items, counts each candidate evaluation against a fixed budget, and keeps exploratory item selection separate from the data used for a statistical decision.

A run ends with `confirmed_regression`, `evidence_of_noninferiority`, `inconclusive`, or `evaluation_error`. Inconclusive is not approval to deploy. The default installation runs on CPU and needs no model API.

## Try it

```bash
pip install -e .
evaldelta demo --scenario slice-regression --budget 100 --seed 42
```

The demo generates a synthetic episode and writes a JSON report, an event log, and a readable report under `runs/`. The command returns exit code 2 for a confirmed regression, 3 for an inconclusive result, 0 for evidence of non-inferiority, and 4 for an evaluation error.

For an example that calls a candidate Python function:

```bash
python examples/ci_demo/make_demo_data.py
evaldelta compare --config examples/ci_demo/evaldelta.yaml --output runs/ci-demo
```

The script creates its synthetic tables locally. The [example workflow](.github/workflows/example_regression.yml) generates the same tables before running the [GitHub Action](action.yml). None of the tables are stored in Git.

## Method

EvalDelta fixes a discovery pool and separate random confirmation pools before querying candidate outcomes. A policy can inspect old-version results and the candidate outcomes it has paid for; it cannot inspect unqueried outcomes. Confirmation uses fresh items and a predeclared test. Each attempted candidate call, including retries, is charged to the budget. See the [statistical protocol](docs/STATISTICAL_PROTOCOL.md) for assumptions and decision rules.

The package includes uniform and stratified selection, a historical-signal heuristic (`PairedShift`), exact paired tests, and finite-population betting confidence sequences. A propensity-weighted adaptive global method is available for research use, but its efficiency advantage is unconfirmed.

## What the experiments found

The four-source benchmark and CIFAR-10 transfer study contain 153,540 replay trials. In the tested boundary-null settings, observed false-alarm rates ranged from 0.04% to 2.35% across methods and budgets, below the 5% target. Repeated seeds of the same version pair are not independent datasets.

At a budget of 500 candidate calls, the adaptive global method confirmed 42.7% of tested regressions versus 40.2% for fixed-sample McNemar. The source-level 95% interval for the difference was −7.3 to +14.0 percentage points, so this is not evidence of an efficiency gain. `PairedShift` slice discovery did not outperform uniform or stratified selection. A separate, predeclared [MAGIC Gamma Telescope case study](docs/EXTERNAL_CASE_STUDY.md) gave mixed results.

Read the [results](docs/RESULTS.md) for denominators, settings, plots, and limitations. The [benchmark card](docs/BENCHMARK_CARD.md) identifies the source datasets and version pairs; [reproduction steps](docs/REPRODUCE.md) show how to rebuild the experiments.

## Repository contents

- `src/evaldelta/`: replay engine, policies, tests, providers, reports, and CLI.
- `benchmarks/`: scripts to prepare source data and run the experiments.
- `tests/`: statistical, budget, leakage, integration, and unit checks.
- `results/frozen/analysis/` and `docs/figures/`: aggregate outputs and plots. The [run manifests](docs/run_manifests/) record settings and file hashes.
- `apps/static_space/`: a [free static demo](docs/SPACE_DEPLOY.md) built from saved synthetic replays. `apps/hf_space/` holds the optional live Gradio version.

Raw source data, model checkpoints, item-level prediction matrices, and trial-level records are not included. The example tables are generated when needed. Dataset and checkpoint terms remain with their original publishers; the [Apache-2.0 license](LICENSE) covers this repository's code.
