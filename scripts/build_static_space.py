"""Build a static Space from deterministic, synthetic EvalDelta replays."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from evaldelta.demo import run_replay

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (100, 300, 600)
SCENARIOS = {
    "slice_regression": "A regression concentrated in one slice",
    "global_regression": "A regression across the pool",
    "null_identical": "Identical old and candidate outcomes",
    "improvement": "A candidate that improves overall",
    "compensating": "Slice harm offset by gains elsewhere",
    "rare_severe": "A small slice with a severe regression",
}


def build(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a fresh output path")
    destination.mkdir(parents=True)
    runs: dict[str, dict[str, object]] = {}
    for scenario in SCENARIOS:
        runs[scenario] = {}
        for budget in BUDGETS:
            table, _, details = run_replay(scenario=scenario, budget=budget, seed=42)
            runs[scenario][str(budget)] = {
                "rows": json.loads(table.to_json(orient="records")),
                "true_delta": details["full_pool_delta_after_run"],
            }
    payload = {
        "seed": 42,
        "n_items": 5000,
        "budgets": BUDGETS,
        "scenarios": SCENARIOS,
        "runs": runs,
    }
    (destination / "data.json").write_text(json.dumps(payload, indent=2) + "\n")
    shutil.copy2(ROOT / "apps" / "static_space" / "index.html", destination / "index.html")
    shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
    (destination / "README.md").write_text(
        """---
title: EvalDelta Replay Demo
emoji: 📊
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
---

# EvalDelta replay demo

This static page shows deterministic, precomputed runs of the EvalDelta Python engine
on synthetic data. Changing a control selects a saved run; it does not execute Python
in the browser or evaluate an uploaded model. Source code and benchmark limitations:
https://github.com/mishra-avdhesh/EvalDelta

No third-party datasets, model checkpoints, private inputs, or trial-level records are
included. The software code is licensed under Apache-2.0; see `LICENSE`.
"""
    )
    print(f"Built {destination} ({len(SCENARIOS) * len(BUDGETS)} saved comparisons)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "static-space")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
