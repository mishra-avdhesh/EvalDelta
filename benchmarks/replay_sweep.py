"""DeltaBench replay sweeps (M4/M5).

    python benchmarks/replay_sweep.py validate   # hyper-parameter selection on VALIDATION episodes only
    python benchmarks/replay_sweep.py freeze     # writes configs/frozen.yaml from validation results
    python benchmarks/replay_sweep.py test       # locked held-out test (refuses without frozen.yaml)
    python benchmarks/replay_sweep.py ablate     # F7 ablations on held-out test episodes, frozen settings
    python benchmarks/replay_sweep.py nullreal   # boundary-null calibration on real test episodes

Outputs: results/frozen/<phase>/trials.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaldelta.bench.deltabench import registry
from evaldelta.experiments.runner import TrialSpec, run_trials

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frozen"
MODEL = "configs/paired_shift_model.json"
FROZEN = ROOT / "configs" / "frozen.yaml"
BUDGETS = [25, 50, 100, 200, 500]

# Near-null real pairs per split, used as realistic backgrounds for injected stress episodes.
BASES = {
    "validation": [
        ("covtype", "v06", "v07"),
        ("adult", "v06", "v09"),
        ("agnews", "v06", "v08"),
        ("bank", "v06", "v08"),
    ],
    "test": [
        ("covtype", "v09", "v11"),
        ("adult", "v09", "v11"),
        ("agnews", "v09", "v11"),
        ("bank", "v09", "v11"),
    ],
    "test_transfer": [("cifar10", "v04", "v10")],
}

# ------------------------------------------------------------------------------------------------
# Method definitions (identical information and paid-call accounting for all methods)
# ------------------------------------------------------------------------------------------------


def global_only(budget: int, method: str, looks: int) -> dict[str, Any]:
    return {
        "split": {"discovery": 0.0, "global_confirm": 1.0, "slice_confirm": 0.0},
        "budget": {
            "max_candidate_calls": budget,
            "discovery_fraction": 0.0,
            "global_fraction": 1.0,
            "slice_fraction": 0.0,
        },
        "plan": {"method": method, "looks": looks, "max_slices": 0},
        "policy": {"name": "uniform"},
    }


def adaptive(budget: int, acquisition: str, **plan: Any) -> dict[str, Any]:
    return {
        "mode": "adaptive",
        "budget_calls": budget,
        "adaptive": {"acquisition": acquisition, **plan},
        "policy_params": {"model_path": MODEL},
    }


def global_methods(budget: int, ours2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "B1_fixed_mcnemar": global_only(budget, "auto", 1),
        "B8_seq_mcnemar_bonf5": global_only(budget, "auto", 5),
        "B8_seq_betting_wor": global_only(budget, "betting_wor", 10),
        "IPW_uniform": adaptive(budget, "uniform", epsilon=1.0),
        "B6_active_testing_ipw": adaptive(
            budget, "active_testing", epsilon=ours2.get("epsilon", 0.2)
        ),
        "OURS2_paired_shift_ipw": adaptive(budget, "paired_shift", **ours2),
    }


def full_design(
    budget: int,
    policy: str,
    params: dict[str, Any] | None = None,
    fractions: tuple[float, float, float] = (0.3, 0.4, 0.3),
    max_slices: int = 2,
) -> dict[str, Any]:
    d, g, s = fractions
    pol: dict[str, Any] = {"name": policy, "params": dict(params or {})}
    if policy == "paired_shift":
        pol["params"].setdefault("model_path", MODEL)
    return {
        "budget": {
            "max_candidate_calls": budget,
            "discovery_fraction": d,
            "global_fraction": g,
            "slice_fraction": s,
        },
        "plan": {"method": "auto", "looks": 3, "max_slices": max_slices, "min_slice_confirm": 30},
        "policy": pol,
    }


DISCOVERY_POLICIES = [
    "uniform",
    "stratified",
    "historical_cohort",
    "old_uncertainty",
    "diversity",
    "paired_shift",
]

# ------------------------------------------------------------------------------------------------
# Episode sets
# ------------------------------------------------------------------------------------------------


def real_episodes(split: str) -> list[dict[str, Any]]:
    reg = registry()
    sub = reg[reg.split == split]
    return [
        {"type": "deltabench", "source": r.source, "old": r.old, "new": r.new}
        for r in sub.itertuples()
    ]


def injected_global(split: str) -> list[dict[str, Any]]:
    out = []
    for src, a, b in BASES[split]:
        base = {"type": "deltabench", "source": src, "old": a, "new": b}
        out.append({**base, "inject": {"pattern": "null"}})
        out.append({**base, "inject": {"pattern": "improvement", "severity": 0.02}})
        for sev in (0.01, 0.02, 0.04):
            for w in ("unstable", "uniform"):
                out.append(
                    {
                        **base,
                        "inject": {
                            "pattern": "global_degradation",
                            "severity": sev,
                            "weighting": w,
                        },
                    }
                )
    return out


def injected_slices(split: str) -> list[dict[str, Any]]:
    out = []
    for src, a, b in BASES[split]:
        base = {"type": "deltabench", "source": src, "old": a, "new": b}
        for sev in (0.1, 0.2):
            for w in ("unstable", "uniform"):
                out.append(
                    {
                        **base,
                        "inject": {
                            "pattern": "slice_only",
                            "slice_pick": "common",
                            "slice_severity": sev,
                            "weighting": w,
                        },
                    }
                )
        out.append(
            {
                **base,
                "inject": {
                    "pattern": "rare_severe",
                    "slice_pick": "rare",
                    "slice_severity": 0.3,
                    "weighting": "unstable",
                },
            }
        )
        out.append(
            {
                **base,
                "inject": {
                    "pattern": "compensating",
                    "slice_pick": "common",
                    "slice_severity": 0.2,
                    "weighting": "unstable",
                },
            }
        )
        out.append({**base, "inject": {"pattern": "null"}})
    return out


def ep_key(ep: dict[str, Any]) -> str:
    return json.dumps(ep, sort_keys=True)


def specs_for(
    config: str, episodes: Iterable[dict[str, Any]], run: dict[str, Any], reps: int, **tags: Any
) -> list[TrialSpec]:
    out = []
    for ep in episodes:
        for t in range(reps):
            out.append(TrialSpec(config, ep, run, t, tags={**tags, "episode_spec": ep_key(ep)}))
    return out


# ------------------------------------------------------------------------------------------------
# Phases
# ------------------------------------------------------------------------------------------------

OURS2_GRID = [{"epsilon": e, "variance_floor": v} for e in (0.1, 0.3) for v in (0.003, 0.02)]
PS_GRID = [{"lam": lam, "explore_frac": ex} for lam in (0.25, 1.0) for ex in (0.1, 0.3)] + [
    {"lam": 0.5, "explore_frac": 0.2}
]
SPLIT_GRID = [(0.2, 0.4, 0.4), (0.3, 0.4, 0.3), (0.4, 0.3, 0.3)]


def phase_validate(reps: int) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    eps_g = real_episodes("validation") + injected_global("validation")
    for i, grid in enumerate(OURS2_GRID):
        for b in (50, 200, 500):
            specs += specs_for(
                f"val_ours2_{i}",
                eps_g,
                adaptive(b, "paired_shift", **grid),
                reps,
                kind="global",
                method="OURS2",
                grid=json.dumps(grid),
                budget=b,
            )
    for b in (50, 200, 500):  # references on the same validation episodes
        for name in ("B8_seq_betting_wor", "IPW_uniform"):
            specs += specs_for(
                f"val_{name}",
                eps_g,
                global_methods(b, {})[name],
                reps,
                kind="global",
                method=name,
                grid="",
                budget=b,
            )
    eps_s = injected_slices("validation")
    for i, grid in enumerate(PS_GRID):
        for fr in SPLIT_GRID:
            specs += specs_for(
                f"val_ps_{i}_{fr}",
                eps_s,
                full_design(600, "paired_shift", grid, fr),
                reps,
                kind="slice",
                method="OURS1",
                grid=json.dumps(grid),
                fractions=str(fr),
                budget=600,
            )
    for fr in SPLIT_GRID:
        specs += specs_for(
            f"val_uniform_{fr}",
            eps_s,
            full_design(600, "uniform", None, fr),
            reps,
            kind="slice",
            method="uniform",
            grid="",
            fractions=str(fr),
            budget=600,
        )
    return specs


def phase_freeze() -> dict[str, Any]:
    df = pd.read_parquet(OUT / "validate" / "trials.parquet")
    g = df[(df.tag_kind == "global") & (df.tag_method == "OURS2")]
    reg = g[g.true_delta >= 0.005]
    score = reg.groupby("tag_grid").apply(
        lambda x: (x.decision == "confirmed_regression").mean(), include_groups=False
    )
    best_ours2 = json.loads(score.idxmax())
    s = df[(df.tag_kind == "slice") & (df.tag_method == "OURS1")]
    v2 = OUT / "validate2" / "trials.parquet"
    if v2.exists():  # validation-only revision (slice ranking); see docs/RESULTS.md
        s = pd.concat([s, pd.read_parquet(v2)], ignore_index=True)
    s = s[s.target_slice.notna()]
    s_score = s.groupby(["tag_grid", "tag_fractions"]).apply(
        lambda x: x.target_confirmed.mean(), include_groups=False
    )
    best_ps, best_fr = s_score.idxmax()
    u = df[(df.tag_kind == "slice") & (df.tag_method == "uniform") & df.target_slice.notna()]
    u_score = u.groupby("tag_fractions").apply(
        lambda x: x.target_confirmed.mean(), include_groups=False
    )
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_on": "validation episodes only (results/frozen/validate)",
        "ours2": best_ours2,
        "ours2_validation_score": {k: round(float(v), 4) for k, v in score.items()},
        "paired_shift": json.loads(best_ps),
        "budget_fractions": [float(x) for x in best_fr.strip("()").split(",")],
        "paired_shift_validation_score": {
            f"{k[0]} {k[1]}": round(float(v), 4) for k, v in s_score.items()
        },
        "uniform_validation_score": {k: round(float(v), 4) for k, v in u_score.items()},
        "uniform_best_fractions": [float(x) for x in u_score.idxmax().strip("()").split(",")],
        "model": MODEL,
        "model_sha256": hashlib.sha256((ROOT / MODEL).read_bytes()).hexdigest(),
    }
    FROZEN.write_text(yaml.safe_dump(frozen, sort_keys=False))
    return frozen


def load_frozen() -> dict[str, Any]:
    if not FROZEN.exists():
        sys.exit(
            "configs/frozen.yaml missing: run `validate` and `freeze` before touching test data"
        )
    fz = yaml.safe_load(FROZEN.read_text())
    sha = hashlib.sha256((ROOT / fz["model"]).read_bytes()).hexdigest()
    if sha != fz["model_sha256"]:
        sys.exit("PairedShift model changed after freezing; refusing to run the locked test")
    return fz


def phase_test(reps: int, splits: tuple[str, ...] = ("test", "test_transfer")) -> list[TrialSpec]:
    fz = load_frozen()
    specs: list[TrialSpec] = []
    for split in splits:
        eps_real = real_episodes(split) if split == "test" else real_episodes("test_transfer")
        eps_g = eps_real + injected_global(split)
        for b in BUDGETS:
            for name, run in global_methods(b, fz["ours2"]).items():
                specs += specs_for(
                    f"test_{name}_B{b}",
                    eps_g,
                    run,
                    reps,
                    kind="global",
                    method=name,
                    budget=b,
                    split=split,
                )
        eps_s = injected_slices(split)
        fr = tuple(fz["budget_fractions"])
        for b in (300, 600, 1000):
            for pol in DISCOVERY_POLICIES:
                params = fz["paired_shift"] if pol == "paired_shift" else None
                specs += specs_for(
                    f"test_slice_{pol}_B{b}",
                    eps_s,
                    full_design(b, pol, params, fr),
                    reps,
                    kind="slice",
                    method=pol,
                    budget=b,
                    split=split,
                )
            # uniform discovery at ITS best validation split (fairness check)
            specs += specs_for(
                f"test_slice_uniform_tuned_B{b}",
                eps_s,
                full_design(b, "uniform", None, tuple(fz["uniform_best_fractions"])),
                reps,
                kind="slice",
                method="uniform_tuned_split",
                budget=b,
                split=split,
            )
    return specs


def phase_ablate(reps: int) -> list[TrialSpec]:
    fz = load_frozen()
    specs: list[TrialSpec] = []
    eps_g = real_episodes("test") + injected_global("test")
    b = 200
    variants = {
        "frozen": fz["ours2"],
        "eps_0.05": {**fz["ours2"], "epsilon": 0.05},
        "eps_0.5": {**fz["ours2"], "epsilon": 0.5},
        "no_control_variate": {**fz["ours2"], "use_control_variate": False},
        "batch_100": {**fz["ours2"], "batch_size": 100},
    }
    for name, plan in variants.items():
        specs += specs_for(
            f"abl_ours2_{name}",
            eps_g,
            adaptive(b, "paired_shift", **plan),
            reps,
            kind="ablation_global",
            variant=name,
            budget=b,
        )
    # prior-only (no historical training) = stale/absent history
    no_hist = adaptive(b, "paired_shift", **fz["ours2"])
    no_hist["policy_params"] = {}
    specs += specs_for(
        "abl_ours2_default_prior",
        eps_g,
        no_hist,
        reps,
        kind="ablation_global",
        variant="default_prior_no_history_model",
        budget=b,
    )
    eps_s = injected_slices("test")
    fr = tuple(fz["budget_fractions"])
    ps = fz["paired_shift"]
    for name, params in {
        "frozen": ps,
        "lam_0": {**ps, "lam": 0.0},
        "lam_2": {**ps, "lam": 2.0},
        "eta_0": {**ps, "eta": 0.0},
        "eta_0.5": {**ps, "eta": 0.5},
        "explore_0": {**ps, "explore_frac": 0.0},
        "explore_0.5": {**ps, "explore_frac": 0.5},
        "explore_1.0(=uniform)": {**ps, "explore_frac": 1.0},
    }.items():
        specs += specs_for(
            f"abl_ps_{name}",
            eps_s,
            full_design(600, "paired_shift", params, fr),
            reps,
            kind="ablation_slice",
            variant=name,
            budget=600,
        )
    for frac in SPLIT_GRID + [(0.5, 0.25, 0.25), (0.1, 0.5, 0.4)]:
        specs += specs_for(
            f"abl_split_{frac}",
            eps_s,
            full_design(600, "paired_shift", ps, frac),
            reps,
            kind="ablation_split",
            variant=str(frac),
            budget=600,
        )
    return specs


def phase_nullreal(
    reps: int, splits: tuple[str, ...] = ("test", "test_transfer")
) -> list[TrialSpec]:
    """Boundary nulls on REAL test pairs: margin set to the episode's true Delta."""
    specs: list[TrialSpec] = []
    fz = load_frozen()
    reg = registry()
    sub = reg[reg.split.isin(splits)]
    for r in sub.itertuples():
        ep = {"type": "deltabench", "source": r.source, "old": r.old, "new": r.new}
        margin = max(0.0, float(r.true_delta))
        for b in (200, 1000):
            run_b = global_only(b, "betting_wor", 10)
            run_b["plan"]["regression_margin"] = margin
            specs += specs_for(
                f"null_betting_B{b}",
                [ep],
                run_b,
                reps,
                kind="null_real",
                method="B8_seq_betting_wor",
                budget=b,
                margin=margin,
            )
            run_o = adaptive(b, "paired_shift", **{**fz["ours2"], "regression_margin": margin})
            specs += specs_for(
                f"null_ours2_B{b}",
                [ep],
                run_o,
                reps,
                kind="null_real",
                method="OURS2_paired_shift_ipw",
                budget=b,
                margin=margin,
            )
            if margin == 0.0:
                specs += specs_for(
                    f"null_mcnemar_B{b}",
                    [ep],
                    global_only(b, "auto", 5),
                    reps,
                    kind="null_real",
                    method="B8_seq_mcnemar_bonf5",
                    budget=b,
                    margin=0.0,
                )
    return specs


def phase_validate2(reps: int) -> list[TrialSpec]:
    """Validation-only revision check for slice ranking (added after the first validation run)."""
    specs: list[TrialSpec] = []
    eps_s = injected_slices("validation")
    for rank in ("model_assisted", "residual_z"):
        for grid in (
            {"lam": 0.25, "explore_frac": 0.1},
            {"lam": 0.5, "explore_frac": 0.2},
            {"lam": 0.5, "explore_frac": 0.4},
        ):
            for fr in SPLIT_GRID:
                params = {**grid, "slice_rank": rank}
                specs += specs_for(
                    f"val2_{rank}_{json.dumps(grid)}_{fr}",
                    eps_s,
                    full_design(600, "paired_shift", params, fr),
                    reps,
                    kind="slice",
                    method="OURS1",
                    grid=json.dumps(params),
                    fractions=str(fr),
                    budget=600,
                )
    return specs


PHASES = {
    "validate": phase_validate,
    "validate2": phase_validate2,
    "test": phase_test,
    "ablate": phase_ablate,
    "nullreal": phase_nullreal,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=[*PHASES, "freeze"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--splits",
        nargs="+",
        choices=["test", "test_transfer"],
        default=["test", "test_transfer"],
        help="Explicit scope for test/nullreal; omitted splits are not evaluated",
    )
    ap.add_argument(
        "--output", type=Path, help="New output directory; existing trials are never overwritten"
    )
    args = ap.parse_args()
    if args.phase == "freeze":
        print(yaml.safe_dump(phase_freeze(), sort_keys=False))
        return
    if args.reps <= 0 or (args.workers is not None and args.workers <= 0):
        ap.error("reps and workers must be positive")
    out = args.output or OUT / args.phase
    if (out / "trials.parquet").exists() or (out / "manifest.json").exists():
        ap.error(f"{out} already contains a run; choose a new --output directory")
    if args.phase in {"test", "nullreal"}:
        required = {source for split in args.splits for source, _, _ in BASES[split]}
        missing = sorted(
            source
            for source in required
            if not (ROOT / "benchmarks" / "data" / source / "matrix.parquet").exists()
        )
        if missing:
            ap.error(
                f"missing prediction caches: {missing}; prepare them or explicitly select --splits test"
            )
        factory = phase_test if args.phase == "test" else phase_nullreal
        specs = factory(args.reps, tuple(args.splits))
    else:
        specs = PHASES[args.phase](args.reps)
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__).resolve(),
        ROOT / "configs" / "frozen.yaml",
        ROOT / MODEL,
        ROOT / "docs" / "STATISTICAL_PROTOCOL.md",
    ]
    for source in sorted(
        {str(spec.episode["source"]) for spec in specs if "source" in spec.episode}
    ):
        paths += [
            ROOT / "benchmarks" / "data" / source / name
            for name in ("matrix.parquet", "versions.json")
        ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "reps": args.reps,
        "workers": args.workers,
        "trials_planned": len(specs),
        "splits": args.splits if args.phase in {"test", "nullreal"} else None,
        "sha256": hashes,
        "seed_range": [0, args.reps - 1],
        "status": "started",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.phase}: {len(specs)} trials", flush=True)
    t0 = time.time()
    df = run_trials(specs, args.workers)
    df.to_parquet(out / "trials.parquet", index=False)
    (out / "meta.json").write_text(
        json.dumps({"trials": len(df), "seconds": time.time() - t0, "reps": args.reps}, indent=2)
    )
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(UTC).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
