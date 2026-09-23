# EvalDelta API and CLI

`pip install -e .` installs the CPU-only package. The library compares paired bounded losses: positive new minus old means degradation. A run reports `confirmed_regression`, `evidence_of_noninferiority`, `inconclusive`, or `evaluation_error`.

```python
from evaldelta import Budget, EvalSession, RunConfig, UniformPolicy, generate_episode

episode = generate_episode("slice_regression", n_items=5000, seed=42)
config = RunConfig(run_id="quickstart", seed=42, budget=Budget(max_candidate_calls=300))
result = EvalSession(episode.item_table(), episode.oracle()).compare(config, UniformPolicy())
result.save("runs/quickstart")
print(result.report.decision, result.report.total_paid_calls)
```

An application can supply a `CallableProvider(fn, scorer)` for a live candidate. The callable receives **public item data** and returns an output; the scorer returns a loss in `[0,1]`. `EvalSession.from_files(items_path, provider)` loads a public CSV/Parquet table. This table must contain stable `sample_id` and `old_loss` columns. Optional `slice`, `task`, `estimated_candidate_cost`, and `f_*`/`hist_*` columns support selection. Candidate losses must remain in the provider until a paid call occurs.

The CLI accepts `evaldelta demo`, `evaldelta compare --config path.yaml --output runs/name`, `evaldelta benchmark --config path.yaml`, and `evaldelta report runs/name --format html`. `examples/ci_demo/evaldelta.yaml` is a local example with a callable provider; `examples/ci_demo/evaldelta_replay.yaml` is an offline replay. See `evaldelta --help` for options.

Strict exit codes: `0` means an actual non-inferiority decision; `2` means confirmed regression; `3` means inconclusive; `4` means evaluation/configuration error. `report-only` changes only CI gating, not the report decision. A result of `inconclusive` is never an approval.

A GitHub Action at `action.yml` wraps the same CLI. Example workflow: `.github/workflows/example_regression.yml`. The demo is offline replay, even if an upload includes both old/new outcomes. The demo does not execute uploaded Python code.
