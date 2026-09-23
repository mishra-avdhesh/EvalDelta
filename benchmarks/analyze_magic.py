"""Descriptive per-pair audit of the frozen MAGIC external-source case study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "frozen"
OUT = BASE / "analysis"


def read_complete(name: str) -> pd.DataFrame:
    run = BASE / name
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"{name} is not complete")
    trials = pd.read_parquet(run / "trials.parquet")
    if len(trials) != manifest["trials_planned"]:
        raise ValueError(f"{name}: missing trials")
    if (trials.paid_calls > trials.budget).any() or trials.decision.eq("evaluation_error").any():
        raise ValueError(f"{name}: budget violation or evaluation error")
    return trials


def main() -> None:
    real = read_complete("magic_external_test_v1")
    null = read_complete("magic_external_null_v1")
    real_rows = []
    for (pair, budget, method), group in real.groupby(["episode_id", "tag_budget", "tag_method"]):
        real_rows.append(
            {
                "pair": pair,
                "budget": budget,
                "method": method,
                "true_pool_delta": group.true_delta.iloc[0],
                "trials": len(group),
                "confirmed_regressions": int(
                    group.global_decision.eq("confirmed_regression").sum()
                ),
                "mean_paid_calls": group.paid_calls.mean(),
            }
        )
    null_rows = []
    for (pair, budget, method), group in null.groupby(["episode_id", "tag_budget", "tag_method"]):
        alarms = int(group.false_regression.sum())
        null_rows.append(
            {
                "pair": pair,
                "budget": budget,
                "method": method,
                "true_pool_delta": group.true_delta.iloc[0],
                "regression_margin": group.tag_margin.iloc[0],
                "trials": len(group),
                "false_alarms": alarms,
                "observed_rate": alarms / len(group),
                "mean_paid_calls": group.paid_calls.mean(),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(real_rows).to_csv(OUT / "magic_external_real.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "magic_external_null.csv", index=False)
    print(f"MAGIC: {len(real)} detection and {len(null)} null trials; aggregates saved to {OUT}")


if __name__ == "__main__":
    main()
