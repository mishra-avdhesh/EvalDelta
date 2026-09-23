"""CPU-only, offline replay core for the public demo.

An uploaded table has full candidate outcomes, so this is explicitly a replay study.
The selection policies receive only an ItemTable without the new_loss column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaldelta.bench.episode import Episode
from evaldelta.bench.synthetic import SCENARIOS, generate_episode
from evaldelta.data.io import ItemTable
from evaldelta.policies.registry import make_policy
from evaldelta.replay.session import EvalSession
from evaldelta.schemas import Budget, PolicyConfig, RunConfig

MAX_BYTES = 20_000_000
MAX_ITEMS = 10_000


def _uploaded_episode(path: str) -> Episode:
    p = Path(path)
    if p.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Upload a CSV or Parquet outcome table")
    if p.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds the 20 MB demo limit")
    raw = (
        pd.read_csv(p, dtype={"sample_id": str})
        if p.suffix.lower() == ".csv"
        else pd.read_parquet(p)
    )
    if len(raw) < 100 or len(raw) > MAX_ITEMS:
        raise ValueError("Demo tables need 100 to 10,000 rows")
    required = {"sample_id", "old_loss", "new_loss"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(raw.columns))}")
    if not raw.sample_id.notna().all() or raw.sample_id.astype(str).duplicated().any():
        raise ValueError("sample_id must be present and unique")
    old = pd.to_numeric(raw.old_loss, errors="coerce").to_numpy(dtype=float)
    new = pd.to_numeric(raw.new_loss, errors="coerce").to_numpy(dtype=float)
    if not np.isin(old, (0.0, 1.0)).all() or not np.isin(new, (0.0, 1.0)).all():
        raise ValueError("This demo accepts binary 0/1 losses only")
    public = raw.drop(columns=["new_loss"])
    items = ItemTable(public)
    hidden = dict(zip(items.ids, new.tolist(), strict=True))
    return Episode(
        "uploaded_offline_replay",
        items.frame,
        hidden,
        {"is_synthetic": False, "source": "user_uploaded_offline_replay"},
    )


def run_replay(
    upload_path: str | None = None,
    scenario: str = "slice_regression",
    budget: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Compare uniform and PairedShift under the same budget and split seed."""
    if not 1 <= budget <= 2000:
        raise ValueError("Budget must be between 1 and 2,000 paid candidate evaluations")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown bundled scenario: {scenario}")
    ep = (
        _uploaded_episode(upload_path)
        if upload_path
        else generate_episode(scenario, n_items=5000, seed=seed)
    )
    items = ep.item_table()
    if budget > len(items):
        raise ValueError("Budget cannot exceed the number of items")
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name in ("uniform", "paired_shift"):
        config = RunConfig(
            run_id=f"space-{name}-{seed}",
            seed=seed,
            budget=Budget(max_candidate_calls=budget),
            policy=PolicyConfig(name=name),
        )
        result = EvalSession(items, ep.oracle(), episode_meta=ep.meta).compare(
            config, make_policy(config.policy, items)
        )
        report = result.report
        global_ev = report.global_result
        rows.append(
            {
                "selector": name,
                "decision": report.decision.value,
                "paid_calls": report.total_paid_calls,
                "budget": budget,
                "global_sample": 0 if global_ev is None else global_ev.n,
                "estimated_change": None if global_ev is None else global_ev.effect,
                "lower_bound": None if global_ev is None else global_ev.lower,
                "upper_bound": None if global_ev is None else global_ev.upper,
                "confirmed_slices": ", ".join(
                    e.scope.removeprefix("slice:")
                    for e in report.slice_results
                    if e.decision.value == "confirmed_regression"
                ),
            }
        )
        reports[name] = {
            "decision": report.decision.value,
            "calls": report.total_paid_calls,
            "split_hash": report.split_hash,
        }
    truth = ep.truth()
    summary = (
        f"**Offline replay** of {len(items):,} items. Full-pool paired loss change: "
        f"{truth['delta']:+.4f} (shown for auditing after the replay). "
        "Positive means the candidate is worse. Discovery-selected examples are not used "
        "as a population estimate. Confirmation uses separate randomized items. "
        "Inconclusive is never deployment approval."
    )
    return (
        pd.DataFrame(rows),
        summary,
        {
            "replay_only": True,
            "source": ep.episode_id,
            "full_pool_delta_after_run": truth["delta"],
            "aggregate_reports": reports,
        },
    )
