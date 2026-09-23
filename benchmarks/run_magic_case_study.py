"""Run the frozen MAGIC external-source check from docs/EXTERNAL_CASE_STUDY.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from replay_sweep import adaptive, global_methods, global_only, load_frozen, specs_for

from evaldelta.bench.deltabench import load_source
from evaldelta.experiments.runner import TrialSpec, run_trials

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "data" / "magic"
PAIRS = [("v01", "v02"), ("v02", "v03"), ("v02", "v04")]
BUDGETS = (200, 500)
TEST_METHODS = (
    "B1_fixed_mcnemar",
    "B8_seq_betting_wor",
    "IPW_uniform",
    "OURS2_paired_shift_ipw",
)


def episode(old: str, new: str) -> dict[str, str]:
    return {"type": "deltabench", "source": "magic", "old": old, "new": new}


def test_specs(frozen: dict[str, Any]) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    episodes = [episode(old, new) for old, new in PAIRS]
    for budget in BUDGETS:
        methods = global_methods(budget, frozen["ours2"])
        for name in TEST_METHODS:
            specs += specs_for(
                f"magic_{name}_B{budget}",
                episodes,
                methods[name],
                20,
                kind="global",
                method=name,
                budget=budget,
                split="external_magic",
            )
    return specs


def null_specs(frozen: dict[str, Any]) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    source = load_source(str(SOURCE))
    for old, new in PAIRS:
        true_delta = float((source.loss(new) - source.loss(old)).mean())
        margin = max(0.0, true_delta)
        ep = episode(old, new)
        for budget in BUDGETS:
            fixed = global_only(budget, "betting_wor", 10)
            fixed["plan"]["regression_margin"] = margin
            ours = adaptive(
                budget, "paired_shift", **{**frozen["ours2"], "regression_margin": margin}
            )
            for name, plan in (
                ("B8_seq_betting_wor", fixed),
                ("OURS2_paired_shift_ipw", ours),
            ):
                specs += specs_for(
                    f"magic_null_{name}_B{budget}",
                    [ep],
                    plan,
                    200,
                    kind="null_real",
                    method=name,
                    budget=budget,
                    split="external_magic",
                    margin=margin,
                )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["test", "null"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not (SOURCE / "matrix.parquet").exists():
        parser.error("MAGIC matrix missing; run benchmarks/prepare_magic.py first")
    frozen = load_frozen()
    specs = test_specs(frozen) if args.phase == "test" else null_specs(frozen)
    out = ROOT / "results" / "frozen" / f"magic_external_{args.phase}_v1"
    if out.exists():
        parser.error(f"{out} exists; results are never overwritten")
    paths = [
        ROOT / "benchmarks" / "prepare_magic.py",
        Path(__file__).resolve(),
        ROOT / "benchmarks" / "replay_sweep.py",
        ROOT / "docs" / "EXTERNAL_CASE_STUDY.md",
        ROOT / "docs" / "STATISTICAL_PROTOCOL.md",
        ROOT / "configs" / "frozen.yaml",
        ROOT / "configs" / "paired_shift_model.json",
        ROOT / "benchmarks" / "data" / "raw" / "magic_gamma_telescope.zip",
        SOURCE / "matrix.parquet",
        SOURCE / "versions.json",
        *sorted((ROOT / "src" / "evaldelta").rglob("*.py")),
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    out.mkdir(parents=True)
    manifest = {
        "status": "started",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "source": "magic",
        "pairs": PAIRS,
        "budgets": BUDGETS,
        "trials_planned": len(specs),
        "seeds": [0, 19] if args.phase == "test" else [0, 199],
        "workers": args.workers,
        "sha256": hashes,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    trials = run_trials(specs, args.workers)
    if len(trials) != len(specs):
        raise RuntimeError("Missing trial rows")
    if (trials.paid_calls > trials.budget).any() or trials.decision.eq("evaluation_error").any():
        raise RuntimeError("Budget violation or evaluation error; refusing complete status")
    trials.to_parquet(out / "trials.parquet", index=False)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(UTC).isoformat()
    manifest["seconds"] = round(time.monotonic() - started, 3)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.phase}: {len(trials)} complete trials in {manifest['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
