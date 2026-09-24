# EvalDelta

[![ci](https://github.com/mishra-avdhesh/EvalDelta/actions/workflows/ci.yml/badge.svg)](https://github.com/mishra-avdhesh/EvalDelta/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

**Statistically controlled, budget-aware regression testing for ML model versions.** Release v0.1.0.

## What it solves

When a model, prompt, or pipeline changes, teams need to know whether the new version got worse, but
they can only afford a limited number of expensive evaluations. EvalDelta compares the old and new
version on the same items, counts every candidate evaluation (including retries) against a fixed
budget. Its decisions control false-alarm rates under documented sampling assumptions. The default
install runs on CPU and needs no model API.

### Why this is different from ordinary evaluation

- **Paired and directional.** Ordinary evaluation estimates one model's score. Regression testing
  asks about the *difference* between two versions on the same items, and whether the new one is
  worse by more than a tolerance.
- **No detection is not approval.** Seeing no problem in a small sample is not evidence of none.
  EvalDelta reports that as `inconclusive` and supports approval only through an explicit
  non-inferiority test.
- **Item selection can bias results.** Picking "interesting" items and then testing on them
  overstates regressions. EvalDelta keeps exploratory selection separate from the fresh items used
  for the decision.

## Outcomes

| Outcome | Meaning | Exit code |
|---|---|---:|
| `confirmed_regression` | Statistical evidence that the new version exceeds the predeclared regression margin | 2 |
| `evidence_of_noninferiority` | Statistical evidence that harm is below a stated tolerance | 0 |
| `inconclusive` | Not enough evidence either way. **Not approval to deploy** | 3 |
| `evaluation_error` | Bad configuration, or too many failed candidate calls | 4 |

## Install

Requires Python 3.11 or newer. EvalDelta is not on PyPI yet; install from GitHub:

```bash
pip install git+https://github.com/mishra-avdhesh/EvalDelta.git
# or, for development
git clone https://github.com/mishra-avdhesh/EvalDelta.git && cd EvalDelta && pip install -e '.[dev]'
```

## Quick start

**A 30-second replay demo** (synthetic, deterministic; writes `report.json`, `events.jsonl`, `report.md`):

```bash
evaldelta demo --scenario slice-regression --budget 100 --seed 42
```

**A real old-vs-new model example.** Two scikit-learn versions of a digit classifier, CPU only, no
downloads. The new version is evaluated live, at most 300 times:

```bash
python examples/sklearn_versions/run.py           # new version trained on too little data
python examples/sklearn_versions/run.py --same    # routine retrain of the same model
```

| Run | Decision | Paid evaluations | Difference (new − old loss) | Full-evaluation truth |
|---|---|---:|---|---:|
| weakened new version | `confirmed_regression` | 140 of 300 | +0.200, interval [+0.004, +0.399] | +0.098 |
| routine retrain | `inconclusive` | 240 of 300 | +0.007, interval [−0.044, +0.064] | +0.003 |

Stopping early makes the point estimate noisy; read the interval. The last column needs a full
evaluation and is shown only to check the verdict.

**From Python:**

```python
from evaldelta import Budget, CallableProvider, EvalSession, ItemTable, RunConfig, zero_one_loss
from evaldelta import UniformPolicy

session = EvalSession(ItemTable(items_df), CallableProvider(call_new_model, zero_one_loss))
result = session.compare(RunConfig(run_id="pr-123", budget=Budget(max_candidate_calls=300)),
                         UniformPolicy())
print(result.report.decision, result.report.exit_code)
result.save("runs/pr-123")
```

`items_df` has a `sample_id`, the `old_loss` (0 or 1) from the cached old version, and whatever
your candidate needs. The full runnable version is `examples/sklearn_versions/run.py`.

## CLI

```bash
evaldelta demo --scenario slice-regression --budget 100 --seed 42
evaldelta compare --config examples/ci_demo/evaldelta.yaml --output runs/ci-demo
evaldelta report runs/ci-demo --format html          # md | html | json
evaldelta benchmark --config my_sweep.yaml           # replay sweeps
```

`compare --mode` sets how results map to the exit code: `strict` (default),
`allow-inconclusive`, or `report-only` (always 0 unless there is an evaluation error).

## GitHub Action

Gate a pull request on a statistically supported regression. This example generates its own data and
uses no secrets or paid API ([workflow](.github/workflows/example_regression.yml)):

```yaml
jobs:
  evaldelta:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: mishra-avdhesh/EvalDelta@v0.1.0      # or @main until the tag exists
        with:
          config: evaldelta.yaml                   # items, provider, budget, plan
          mode: strict                             # strict | allow-inconclusive | report-only
```

It uploads `report.json` and `report.html` as an artifact and writes a summary. It installs from
its own checkout, so it does not need a PyPI release.

## How it works

```
cached old results ─┐
candidate provider ─┼─► seeded split into three pools ─┬─► discovery      (bounded budget, no hidden outcomes)
budget + test plan ─┘                                  ├─► global confirm (random sample, predeclared test)
                                                       └─► slice confirm  (fresh items, Holm across slices)
                                                                   │
                            report.json / report.md / report.html  ◄┘  → decision + exit code
```

**Statistical design.** The split into discovery, global-confirmation and slice-confirmation pools
is fixed by a seed before any candidate outcome is queried. A selection policy sees old-version
results and only the candidate outcomes it has already paid for. Confirmation uses fresh items and a
predeclared test: exact McNemar for 0/1 loss, or finite-population betting confidence sequences,
with Bonferroni over predeclared looks and Holm across slices. Every attempted candidate call is
charged to the budget before it runs. See the [statistical protocol](docs/STATISTICAL_PROTOCOL.md).

**Providers:** offline replay, Python callable, shell command, HTTP (host allow-list, optional token
from an environment variable). No paid API is required.

## Benchmark (DeltaBench)

Version-pair episodes built from public datasets. Each main source has 12 historical-train,
9 validation and 9 held-out test pairs built from model or pipeline changes. Pairs are split
by candidate version, not by item row.

| Source | Role |
|---|---|
| Adult, Covertype, AG News, Bank Marketing | main benchmark (9 held-out test pairs each) |
| CIFAR-10 | held-out transfer family (27 test pairs, 11 of 15 planned versions) |
| MAGIC Gamma Telescope | separate predeclared external case study |

Raw data, checkpoints, and item-level matrices are not distributed. See the
[benchmark card](docs/BENCHMARK_CARD.md) and [reproduction steps](docs/REPRODUCE.md).

## Results

Read [docs/RESULTS.md](docs/RESULTS.md) for denominators, settings, and plots.

- **Calibration in the tested settings.** Revision 2 contains 153,540 replay trials across
  detection, boundary-null, ablation and transfer runs. Observed false-alarm rates in the
  boundary-null settings ranged from 0.04% to 2.35%, below the 5% target. These are repeated
  seeds of fixed version pairs, so they are evidence for these scenarios, not a general guarantee.
- **The adaptive sampling strategy did not demonstrate a reliable advantage.** The optional
  `PairedShift` policy and the propensity-weighted adaptive global method did not improve efficiency
  in this benchmark. At a budget of 500 calls the adaptive method confirmed 42.7% of tested
  regressions versus 40.2% for fixed-sample McNemar; the source-level 95% interval for the difference
  was −7.3 to +14.0 percentage points. `PairedShift` slice discovery did not beat uniform or
  stratified selection, and the method did not transfer to a held-out model family. A predeclared
  [MAGIC case study](docs/EXTERNAL_CASE_STUDY.md) gave mixed results.
- **The contribution is the framework and its validation**, not an efficiency gain. `uniform` and
  `stratified` remain the default selection policies.

## Limitations

- Four main source families, one transfer family and a separate MAGIC case study form a small
  evidence base. Repeated seeds of one version pair are not independent datasets.
- The second benchmark revision reused test pairs seen in the first; see
  [RESULTS.md](docs/RESULTS.md) for the provenance caveats.
- Losses are 0/1 accuracy-style. F1, AUROC, and judged free-text outputs need their own estimand and
  inference, and are not supported yet.
- Rare slices below the minimum confirmation size cannot be confirmed at any budget.
- `inconclusive` means more evidence is needed, not that the model is safe.

## Demo

[Browser demo](https://huggingface.co/spaces/avdheshmishra/evaldelta-demo): compare saved synthetic
replays and inspect selected benchmark aggregates. It shows precomputed results, not a live
evaluation. Build instructions: [docs/SPACE_DEPLOY.md](docs/SPACE_DEPLOY.md).

## Repository contents

- `src/evaldelta/`: replay engine, policies, statistics, providers, reports, and CLI.
- `benchmarks/`, `scripts/`: data preparation, experiment sweeps, and the demo build.
- `tests/`: statistical, budget, leakage, integration, and unit checks.
- `results/frozen/analysis/`, `docs/figures/`, `docs/run_manifests/`: aggregate outputs and run manifests.
- `examples/`: the scikit-learn old-vs-new example and a CI example.

## Citation and license

Cite via [CITATION.cff](CITATION.cff). Version history is in [CHANGELOG.md](CHANGELOG.md). The code is
licensed under [Apache-2.0](LICENSE); dataset and checkpoint terms remain with their original
publishers.
