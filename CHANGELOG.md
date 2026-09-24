# Changelog

## [0.1.0] - 2026-09-24

First release. EvalDelta tests whether a new model version is worse than the one it replaces,
under a fixed evaluation budget, with error rates controlled under the stated assumptions.

### Added
- **Statistically controlled paired regression testing.** Old and new versions are compared on the
  same items. Decisions are `confirmed_regression`, `evidence_of_noninferiority`, `inconclusive`
  (never approval) and `evaluation_error`, mapped to exit codes 2, 0, 3 and 4.
- **Sealed discovery/confirmation design.** A seeded 60/20/20 split into discovery, global
  confirmation and slice confirmation pools is fixed before any candidate outcome is queried.
  Selection policies cannot read unqueried outcomes; leakage tests check this.
- **Explicit evaluation-budget accounting.** Every attempted candidate call, including retries and
  failures, is charged before it runs. Call and total-cost ceilings are never exceeded.
- **Sequential and fixed-sample tests.** Exact McNemar, finite-population betting confidence
  sequences and a Hoeffding reference; Bonferroni over predeclared looks.
- **Slice-level testing** with Holm correction across slices, plus separate confirmation items.
- **Providers:** replay (offline), Python callable, shell command and HTTP, with retries charged
  to the budget. No paid API is required.
- **CLI** (`demo`, `compare`, `report`, `benchmark`) and JSON, Markdown and HTML reports.
- **GitHub Action** with `strict`, `allow-inconclusive` and `report-only` modes.
- **Browser demo**, a free static Hugging Face Space built from saved replays and benchmark
  aggregates. It does not run a live evaluation.
- **DeltaBench validation.** 153,540 revision-2 replay trials across detection, boundary-null,
  ablation and transfer runs on Adult, Covertype, AG News, Bank Marketing and CIFAR-10, plus a
  separate MAGIC case study. Run manifests with settings and file hashes are in
  `docs/run_manifests/`.
- **Examples:** a real scikit-learn old-vs-new model example and a CI example.

### Results
- In boundary-null scenarios the observed false-alarm rate was 0.04% to 2.35%, below the 5% target.
- **Negative result:** the adaptive sampling strategy (`PairedShift` and the propensity-weighted
  adaptive global method) did **not** demonstrate a reliable advantage. At a budget of 500 it
  confirmed 42.7% of regressions versus 40.2% for fixed-sample McNemar; the source-level 95%
  interval for the difference was -7.3 to +14.0 percentage points. `PairedShift` slice discovery
  did not outperform uniform or stratified selection, and the method did not transfer to the
  held-out CIFAR-10 family. The contribution of this release is the framework and its validation,
  not an efficiency gain.

### Known limitations
- Four main source families, one transfer family and a separate MAGIC case study are a small
  evidence base; repeated seeds are not independent datasets. The second benchmark revision
  reused test pairs already seen in the first.
- Accuracy-style 0/1 losses only; other metrics need their own estimand and inference.
- CIFAR-10 covers 11 of 15 planned versions.
- Not published on PyPI; install from GitHub.
