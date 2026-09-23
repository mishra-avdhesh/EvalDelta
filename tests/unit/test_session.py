from __future__ import annotations

import json

import numpy as np
import pytest

from evaldelta import (
    Budget,
    ConfirmPlan,
    Decision,
    EvalSession,
    PolicyConfig,
    RunConfig,
    RunResult,
    StratifiedPolicy,
    UniformPolicy,
    generate_episode,
)
from evaldelta.policies.base import PolicyView


def _run(ep, budget=200, seed=0, policy=None, **plan):
    cfg = RunConfig(
        run_id="t",
        seed=seed,
        budget=Budget(max_candidate_calls=budget),
        plan=ConfirmPlan(**plan),
    )
    return EvalSession(ep.item_table(), ep.oracle(), episode_meta=ep.meta).compare(
        cfg, policy or UniformPolicy()
    )


def test_same_seed_reproduces_ids_and_summary(slice_episode):
    a = _run(slice_episode, seed=4)
    b = _run(slice_episode, seed=4)
    assert a.report.selected_ids == b.report.selected_ids
    assert a.report.global_result == b.report.global_result
    assert a.report.decision == b.report.decision
    c = _run(slice_episode, seed=5)
    assert c.report.selected_ids != a.report.selected_ids


@pytest.mark.parametrize("budget", [1, 7, 50, 333])
def test_budget_never_exceeded_and_no_duplicates(slice_episode, budget):
    r = _run(slice_episode, budget=budget)
    assert r.report.total_paid_calls <= budget
    ids = r.events["sample_id"].tolist() if len(r.events) else []
    assert len(ids) == len(set(ids))
    for phase, spend in r.report.spend.items():
        assert spend.paid_calls <= spend.planned_calls, phase


def test_phase_membership_matches_partition(slice_episode):
    from evaldelta.data.splits import make_partition
    from evaldelta.schemas import SplitConfig

    r = _run(slice_episode, budget=300)
    part = make_partition(slice_episode.items["sample_id"], SplitConfig())
    mem = part.membership()
    names = {
        "discovery": "discovery",
        "global_confirmation": "global_confirm",
        "slice_confirmation": "slice_confirm",
    }
    for _, ev in r.events.iterrows():
        assert mem[ev["sample_id"]] == names[ev["phase"]]


def test_identical_models_never_confirm_regression(null_episode):
    for seed in range(5):
        r = _run(null_episode, budget=300, seed=seed)
        assert r.report.decision != Decision.CONFIRMED_REGRESSION
        g = r.report.global_result
        assert g is not None and g.n_down == 0 and g.n_up == 0 and g.effect == 0.0


def test_null_and_inconclusive_reported_separately(null_episode):
    r = _run(null_episode, budget=20, seed=1)
    # tiny budget: cannot establish non-inferiority -> inconclusive, never a pass
    assert r.report.decision == Decision.INCONCLUSIVE
    assert r.report.exit_code == 3


def test_large_regression_confirmed():
    ep = generate_episode("global_regression", n_items=4000, seed=2, severity=0.15)
    r = _run(ep, budget=300, seed=0)
    assert r.report.decision == Decision.CONFIRMED_REGRESSION
    assert r.report.global_result.decision == Decision.CONFIRMED_REGRESSION
    # early stopping at a predeclared look leaves budget unspent
    assert r.report.spend["global_confirmation"].paid_calls <= 150


def test_serialization_roundtrip(tmp_path, slice_episode):
    r = _run(slice_episode, budget=120)
    out = r.save(tmp_path / "run")
    loaded = RunResult.load(out)
    assert loaded.report == r.report
    assert len(loaded.events) == len(r.events)
    data = json.loads((out / "report.json").read_text())
    assert data["decision"] in {d.value for d in Decision}
    assert "selection_probability" in loaded.events.columns


def test_stratified_policy_is_proportional(slice_episode):
    items = slice_episode.items
    view = PolicyView.build(items, {}, 100)
    ids = StratifiedPolicy().select(view, 200, np.random.default_rng(0))
    assert len(ids) == len(set(ids)) == 200
    got = items.set_index("sample_id").loc[ids, "slice"].value_counts(normalize=True)
    want = items["slice"].value_counts(normalize=True)
    for s in want.index:
        assert abs(got.get(s, 0.0) - want[s]) < 0.03


def test_failed_calls_are_errors_not_wrong_answers(slice_episode):
    ids = slice_episode.items["sample_id"].tolist()
    fails = tuple(ids[::3])  # ~33% failure rate
    cfg = RunConfig(run_id="f", budget=Budget(max_candidate_calls=200))
    s = EvalSession(slice_episode.item_table(), slice_episode.oracle(fail_ids=fails))
    r = s.compare(cfg, UniformPolicy())
    assert r.report.decision == Decision.EVALUATION_ERROR
    assert r.report.exit_code == 4
    failed = r.events[r.events["error"].notna()]
    assert len(failed) > 0 and failed["new_loss"].isna().all()


def test_policy_config_default():
    assert PolicyConfig().name == "uniform"
