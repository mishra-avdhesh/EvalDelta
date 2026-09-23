"""Counterfactual leakage test for every selector.

Run once, then flip the hidden candidate outcome of every item the run did NOT pay for, and run
again with the same seeds. A policy that never reads unrevealed outcomes must make exactly the
same choices. Any difference proves leakage.
"""

from __future__ import annotations

import pytest

from evaldelta import Budget, EvalSession, PolicyConfig, RunConfig
from evaldelta.bench.episode import Episode
from evaldelta.bench.synthetic import generate_episode
from evaldelta.policies.paired_shift import PairedShift
from evaldelta.policies.registry import make_policy
from evaldelta.replay.adaptive import AdaptiveGlobalPlan, AdaptiveGlobalSession

POLICIES = [
    "uniform",
    "stratified",
    "historical_cohort",
    "old_uncertainty",
    "diversity",
    "paired_shift",
]


def _flip_unpaid(ep: Episode, paid: set[str]) -> Episode:
    hidden = {k: (v if k in paid else 1.0 - v) for k, v in ep._hidden.items()}
    return Episode(ep.episode_id + "-cf", ep.items, hidden, ep.meta)


@pytest.mark.parametrize("policy", POLICIES)
def test_session_choices_invariant_to_unpaid_outcomes(policy):
    ep = generate_episode("slice_regression", n_items=2000, seed=31)
    cfg = RunConfig(
        run_id="cf",
        seed=5,
        budget=Budget(max_candidate_calls=300),
        policy=PolicyConfig(name=policy),
    )
    r1 = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, make_policy(cfg.policy))
    paid = set(r1.events["sample_id"])
    ep2 = _flip_unpaid(ep, paid)
    r2 = EvalSession(ep2.item_table(), ep2.oracle()).compare(cfg, make_policy(cfg.policy))
    assert r1.report.selected_ids == r2.report.selected_ids
    assert r1.report.global_result == r2.report.global_result


@pytest.mark.parametrize("acq", ["uniform", "active_testing", "paired_shift"])
def test_adaptive_choices_invariant_to_unpaid_outcomes(acq):
    ep = generate_episode("global_regression", n_items=1500, seed=32, severity=0.03)
    plan = AdaptiveGlobalPlan(acquisition=acq)

    def run(e):
        s = AdaptiveGlobalSession(e.item_table(), e.oracle(), policy=PairedShift())
        return s.run(plan, 200, seed=9)

    a = run(ep)
    paid = set(a.events["sample_id"])
    b = run(_flip_unpaid(ep, paid))
    assert list(a.events["sample_id"]) == list(b.events["sample_id"])
    assert list(a.events["selection_probability"]) == list(b.events["selection_probability"])
    assert (a.lower, a.upper, a.decision) == (b.lower, b.upper, b.decision)
