# EvalDelta

Budget-constrained, statistically valid **paired regression testing** for ML model versions.

> Status: under active development (milestone M1: CPU replay core). No efficacy claims yet.

```bash
pip install -e '.[dev]'
evaldelta demo --scenario slice-regression --budget 100 --seed 42
```

The demo replays a deterministic synthetic episode on CPU. It writes `report.json`,
`events.jsonl` and `report.md`, and exits with 0 (evidence of non-inferiority),
2 (confirmed regression), 3 (inconclusive) or 4 (evaluation error).

See `docs/STATISTICAL_PROTOCOL.md` for the exact estimand and guarantees, and
`docs/NOVELTY_AUDIT.md` for related work.
