# EvalDelta

Budget-constrained, statistically valid **paired regression testing** for ML model versions.

EvalDelta compares an old (production) and a new (candidate) model/prompt/pipeline on the same
items under a fixed evaluation budget, then reports one of four decisions: `confirmed_regression`,
`evidence_of_noninferiority`, `inconclusive` (never a pass), or `evaluation_error`.

> **Status: v0.1.0 pre-release.** The statistical core, replay engine, benchmark, CLI, GitHub
> Action and Gradio demo are implemented and tested (`pytest`: 144 passed; `ruff`/`mypy --strict`:
> clean). Held-out results are in **[docs/RESULTS.md](docs/RESULTS.md)** — read it before quoting
> any number from this project. Revision 2 (4 real source families, up from 3) is the current
> honest summary: observed null false-alarm rates in these benchmark scenarios were
> at most ~2.4% against a 5% target, subject to the protocol assumptions; the adaptive global method's apparent efficiency gain
> **more than halved** when a 4th dataset was added and its confidence intervals cross zero, so it
> should not be advertised as a validated win; it does not transfer to a held-out model family;
> and the PairedShift slice-triage heuristic does not beat plain random sampling in either
> revision.

## 30-second demo (CPU only)

```bash
pip install -e '.[dev]'
evaldelta demo --scenario slice-regression --budget 100 --seed 42
```

This replays a deterministic synthetic episode. It writes `report.json`, `events.jsonl` and
`report.md`, and exits 0 (non-inferiority evidence), 2 (confirmed regression), 3 (inconclusive) or
4 (evaluation error). See [examples/ci_demo](examples/ci_demo) for a live, free (no API key)
worked example wired into a GitHub Action via [action.yml](action.yml), and
[apps/hf_space](apps/hf_space) for the CPU-only Gradio demo.

## Documents

| | |
|---|---|
| [docs/STATISTICAL_PROTOCOL.md](docs/STATISTICAL_PROTOCOL.md) | Exact estimand, hypotheses, tests and validity arguments |
| [docs/NOVELTY_AUDIT.md](docs/NOVELTY_AUDIT.md) | Related-work matrix and scope boundary |
| [docs/RESULTS.md](docs/RESULTS.md) | Held-out experimental results (honest, incl. negative results) |
| [docs/BENCHMARK_CARD.md](docs/BENCHMARK_CARD.md) | DeltaBench sources, splits and licensing |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Synthetic null-calibration simulation report |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Commands to regenerate every result and figure |
| [docs/API.md](docs/API.md) | Python API reference |
| [docs/COMPUTE_LOG.md](docs/COMPUTE_LOG.md) | GPU-hour accounting against the 10-hour budget |
| [docs/EXTERNAL_CASE_STUDY.md](docs/EXTERNAL_CASE_STUDY.md) | Separately predeclared sixth-source check; mixed result, no efficiency claim |
| [docs/SPACE_DEPLOY.md](docs/SPACE_DEPLOY.md) | Build a data-free Gradio Space bundle and deploy it from your account |
| [docs/DEMO_WALKTHROUGH.md](docs/DEMO_WALKTHROUGH.md) | Two-minute recruiter demo script |
| [docs/PUBLISH.md](docs/PUBLISH.md) | Owner commands for GitHub and separate Space publication |

## Architecture

```
old system + candidate --> sealed 60/20/20 discovery/global-confirm/slice-confirm split
  --> discovery policy (bounded budget, no hidden-outcome access)
  --> predeclared confirmation test (exact McNemar / betting confidence sequence)
  --> report.json / report.html: confirmed_regression | evidence_of_noninferiority | inconclusive
```

## License

Apache-2.0 for the code (see [LICENSE](LICENSE)). This does **not** cover third-party datasets or
model checkpoints used by DeltaBench — see [docs/BENCHMARK_CARD.md](docs/BENCHMARK_CARD.md).
