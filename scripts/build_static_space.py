"""Build a static Space from deterministic, synthetic EvalDelta replays."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from evaldelta.demo import run_replay

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (100, 300, 600)
REAL_BUDGETS = (200, 500)
REAL_SPLITS = ("test", "test_transfer")
REAL_METHODS = ("B1_fixed_mcnemar", "OURS2_paired_shift_ipw")
SCENARIOS = {
    "slice_regression": "A regression concentrated in one slice",
    "global_regression": "A regression across the pool",
    "null_identical": "Identical old and candidate outcomes",
    "improvement": "A candidate that improves overall",
    "compensating": "Slice harm offset by gains elsewhere",
    "rare_severe": "A small slice with a severe regression",
}


def real_benchmark_summary() -> dict[str, object]:
    """Publish selected aggregate cells only; never bundle item or trial records."""
    analysis = ROOT / "results" / "frozen" / "analysis"
    with (analysis / "global_real_v2.csv").open(newline="") as stream:
        rates = {
            (row["split"], int(row["budget"]), row["method"]): row
            for row in csv.DictReader(stream)
            if row["split"] in REAL_SPLITS
            and int(row["budget"]) in REAL_BUDGETS
            and row["method"] in REAL_METHODS
        }
    comparisons = json.loads((analysis / "comparisons_v2.json").read_text())
    matched = {
        (row["split"], row["budget"]): row
        for row in comparisons
        if row["split"] in REAL_SPLITS
        and row["budget"] in REAL_BUDGETS
        and row["method"] == REAL_METHODS[1]
        and row["baseline"] == REAL_METHODS[0]
    }
    cells: dict[str, dict[str, object]] = {}
    for split in REAL_SPLITS:
        cells[split] = {}
        for budget in REAL_BUDGETS:
            baseline = rates[(split, budget, REAL_METHODS[0])]
            adaptive = rates[(split, budget, REAL_METHODS[1])]
            comparison = matched[(split, budget)]
            assert baseline["trials"] == adaptive["trials"] == str(comparison["trials"])
            assert baseline["pairs"] == adaptive["pairs"] == str(comparison["pairs"])
            assert baseline["sources"] == adaptive["sources"] == str(comparison["sources"])
            assert (
                abs(
                    float(adaptive["detection_rate"])
                    - float(baseline["detection_rate"])
                    - comparison["difference"]
                )
                < 1e-9
            )
            cells[split][str(budget)] = {
                "sources": comparison["sources"],
                "pairs": comparison["pairs"],
                "trials_per_method": comparison["trials"],
                "baseline_rate": float(baseline["detection_rate"]),
                "adaptive_rate": float(adaptive["detection_rate"]),
                "difference": comparison["difference"],
                "source_bootstrap_95": comparison["source_bootstrap_95"],
            }
    return {"revision": 2, "budgets": REAL_BUDGETS, "cells": cells}


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
    (destination / "benchmark.json").write_text(
        json.dumps(real_benchmark_summary(), indent=2) + "\n"
    )
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

This static page shows deterministic synthetic replays and selected aggregate results
from the revision-2 real-data benchmark. Changing a control selects saved results; it
does not execute Python in the browser or evaluate an uploaded model. Source code and
benchmark limitations:
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
