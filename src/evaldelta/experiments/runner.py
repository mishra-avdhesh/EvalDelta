"""Replay experiment runner: one trial = one episode x one method x one seed.

Trials are independent and run in parallel processes. Ground truth is read from the episode
**after** the run finishes, and only for scoring.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from evaldelta.bench.episode import Episode
from evaldelta.bench.synthetic import generate_episode
from evaldelta.policies.registry import make_policy
from evaldelta.replay.session import EvalSession
from evaldelta.schemas import Decision, PolicyConfig, RunConfig

EpisodeLoader = Callable[[Mapping[str, Any], int], Episode]


def load_episode(source: Mapping[str, Any], trial: int) -> Episode:
    """Build or load the episode for a trial.

    ``{"type": "synthetic", ...kwargs}`` generates a fresh episode with ``seed = trial`` unless the
    source pins ``seed``. ``{"type": "deltabench", "path": ...}`` loads a cached real episode.
    """
    kind = source.get("type", "synthetic")
    if kind == "synthetic":
        kw = {k: v for k, v in source.items() if k not in {"type"}}
        kw.setdefault("seed", trial)
        return generate_episode(**kw)
    if kind == "deltabench":
        from evaldelta.bench.deltabench import episode_from_spec

        return episode_from_spec(source, trial)
    raise ValueError(f"unknown episode source {kind!r}")


@dataclass(frozen=True)
class TrialSpec:
    config_name: str
    episode: Mapping[str, Any]
    run: Mapping[str, Any]  # RunConfig fields except run_id/seed
    trial: int
    randomize_split: bool = True
    tags: Mapping[str, Any] = field(default_factory=dict)


def _resolve_policy_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(params)
    mp = out.get("model_path")
    if mp and not Path(mp).is_absolute():
        out["model_path"] = str(Path(__file__).resolve().parents[3] / mp)
    return out


def run_adaptive(spec: TrialSpec, ep: Episode) -> dict[str, Any]:
    from evaldelta.policies.paired_shift import PairedShift
    from evaldelta.replay.adaptive import AdaptiveGlobalPlan, AdaptiveGlobalSession

    run = dict(spec.run)
    plan = AdaptiveGlobalPlan(**run.get("adaptive", {}))
    policy = PairedShift(**_resolve_policy_params(run.get("policy_params", {})))
    sess = AdaptiveGlobalSession(ep.item_table(), ep.oracle(), policy=policy)
    budget = int(run["budget_calls"])
    res = sess.run(plan, budget, seed=spec.trial)
    truth = ep.truth()["delta"]
    return {
        "config": spec.config_name,
        "trial": spec.trial,
        "episode_id": ep.episode_id,
        "policy": f"adaptive:{plan.acquisition}",
        "budget": budget,
        "true_delta": truth,
        "decision": res.decision.value,
        "global_decision": res.decision.value,
        "global_method": "ipw_betting",
        "paid_calls": res.paid_calls,
        "paid_cost": res.paid_cost,
        "global_n": res.n,
        "global_lower": res.lower,
        "global_upper": res.upper,
        "global_effect": res.effect,
        "covered": bool(res.lower <= truth <= res.upper),
        "false_regression": res.decision == Decision.CONFIRMED_REGRESSION
        and truth <= plan.regression_margin + 1e-12,
        "false_noninferiority": res.decision == Decision.EVIDENCE_OF_NONINFERIORITY
        and plan.noninferiority_margin is not None
        and truth >= plan.noninferiority_margin - 1e-12,
        "stop_reason": res.stop_reason,
        "wall_time_s": res.wall_time_s,
        **{f"tag_{k}": v for k, v in spec.tags.items()},
    }


def run_trial(spec: TrialSpec) -> dict[str, Any]:
    ep = load_episode(spec.episode, spec.trial)
    if spec.run.get("mode") == "adaptive":
        return run_adaptive(spec, ep)
    run = dict(spec.run)
    run.pop("mode", None)
    split = dict(run.pop("split", {}))
    if spec.randomize_split:
        split["seed"] = 1_000_003 * spec.trial + 17
    cfg = RunConfig.model_validate(
        {
            **run,
            "split": split,
            "run_id": f"{spec.config_name}-t{spec.trial}",
            "seed": spec.trial,
        }
    )
    items = ep.item_table()
    pol_cfg = dict(run.get("policy", {}))
    pol_cfg["params"] = _resolve_policy_params(pol_cfg.get("params", {}))
    policy = make_policy(PolicyConfig.model_validate(pol_cfg), items)
    oracle = ep.oracle()
    res = EvalSession(items, oracle, episode_meta=ep.meta).compare(cfg, policy)
    truth = ep.truth()
    r = res.report
    g = r.global_result
    plan = cfg.plan
    rec: dict[str, Any] = {
        "config": spec.config_name,
        "trial": spec.trial,
        "episode_id": ep.episode_id,
        "policy": cfg.policy.name,
        "budget": cfg.budget.max_candidate_calls,
        "true_delta": truth["delta"],
        "decision": r.decision.value,
        "paid_calls": r.total_paid_calls,
        "paid_cost": r.total_paid_cost,
        "disc_calls": r.spend["discovery"].paid_calls if "discovery" in r.spend else 0,
        "global_calls": (
            r.spend["global_confirmation"].paid_calls if "global_confirmation" in r.spend else 0
        ),
        "slice_calls": (
            r.spend["slice_confirmation"].paid_calls if "slice_confirmation" in r.spend else 0
        ),
        "wall_time_s": r.wall_time_s,
        **{f"tag_{k}": v for k, v in spec.tags.items()},
    }
    if g is not None:
        rec.update(
            global_decision=g.decision.value,
            global_method=g.method,
            global_n=g.n,
            global_lower=g.lower,
            global_upper=g.upper,
            global_effect=g.effect,
            global_looks=g.looks_used,
            global_p=g.p_value,
            covered=(
                g.lower is not None and g.upper is not None and g.lower <= truth["delta"] <= g.upper
            ),
            false_regression=(
                g.decision == Decision.CONFIRMED_REGRESSION
                and truth["delta"] <= plan.regression_margin + 1e-12
            ),
            false_noninferiority=(
                g.decision == Decision.EVIDENCE_OF_NONINFERIORITY
                and plan.noninferiority_margin is not None
                and truth["delta"] >= plan.noninferiority_margin - 1e-12
            ),
        )
    # slices
    target = ep.meta.get("target_slice")
    slice_conf = [
        e.scope.split(":", 1)[1]
        for e in r.slice_results
        if e.decision == Decision.CONFIRMED_REGRESSION
    ]
    tested = [e.scope.split(":", 1)[1] for e in r.slice_results]
    sd = truth["slice_delta"]
    rec.update(
        target_slice=target,
        target_slice_delta=sd.get(target) if target else None,
        slices_tested=json.dumps(tested),
        slices_confirmed=json.dumps(slice_conf),
        target_tested=bool(target in tested) if target else False,
        target_confirmed=bool(target in slice_conf) if target else False,
        target_found_exploratory=bool(target in r.exploratory.get("suspicious_slices", []))
        if target
        else False,
        false_slice_confirmations=sum(
            1 for s in slice_conf if sd.get(s, 0.0) <= plan.slice_margin + 1e-12
        ),
        n_down_found=r.exploratory.get("n_down_found", 0),
        n_up_found=r.exploratory.get("n_up_found", 0),
    )
    return rec


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_trials(specs: Iterable[TrialSpec], workers: int | None = None) -> pd.DataFrame:
    specs = list(specs)
    workers = workers or max(1, (os.cpu_count() or 2) - 4)
    if workers == 1 or len(specs) < 8:
        rows = [run_trial(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            rows = list(ex.map(run_trial, specs, chunksize=max(1, len(specs) // (workers * 8))))
    return pd.DataFrame(rows)


def expand(
    config_name: str,
    episode: Mapping[str, Any],
    run: Mapping[str, Any],
    trials: int,
    start: int = 0,
    **tags: Any,
) -> list[TrialSpec]:
    return [
        TrialSpec(config_name, episode, run, t, tags=tags) for t in range(start, start + trials)
    ]


def run_sweep_config(cfg: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """YAML-driven sweep used by ``evaldelta benchmark``."""
    specs: list[TrialSpec] = []
    for entry in cfg["configs"]:
        specs += expand(entry["name"], entry["episode"], entry["run"], int(entry.get("trials", 50)))
    df = run_trials(specs, cfg.get("workers"))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "trials.parquet", index=False)
    summary = (
        df.groupby("config")
        .agg(
            trials=("trial", "size"),
            regression_rate=("decision", lambda s: float((s == "confirmed_regression").mean())),
            mean_paid_calls=("paid_calls", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"output": str(out), "configs": summary}
