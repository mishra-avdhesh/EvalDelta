"""OURS-2 propensity-weighted inference: unbiasedness and anytime coverage (protocol §7)."""

from __future__ import annotations

import numpy as np
import pytest

from evaldelta.bench.synthetic import generate_episode
from evaldelta.policies.paired_shift import LogisticModel, PairedShift, PairedShiftModel
from evaldelta.replay.adaptive import AdaptiveGlobalPlan, AdaptiveGlobalSession


def _run(ep, plan, budget, seed, policy=None):
    s = AdaptiveGlobalSession(ep.item_table(), ep.oracle(), policy=policy or PairedShift())
    return s.run(plan, budget, seed)


def test_pseudo_outcomes_are_unbiased():
    ep = generate_episode("global_regression", n_items=800, seed=51, severity=0.04)
    delta = ep.truth()["delta"]
    plan = AdaptiveGlobalPlan(
        acquisition="paired_shift", noninferiority_margin=None, regression_margin=0.99
    )  # never stops early
    first = [
        _run(ep, plan, 30, seed).events["pseudo_outcome"].iloc[:30].mean() for seed in range(300)
    ]
    se = np.std(first) / np.sqrt(len(first))
    assert abs(np.mean(first) - delta) < 4 * se


def _inverted_model() -> PairedShift:
    """A deliberately wrong predictor: flips are predicted where they are least likely."""
    m = PairedShiftModel.default()
    bad = PairedShiftModel(
        LogisticModel([-c for c in m.down.coef], m.down.intercept, m.down.mean, m.down.scale),
        LogisticModel([-c for c in m.up.coef], m.up.intercept, m.up.mean, m.up.scale),
    )
    ps = PairedShift()
    ps.model = bad
    return ps


@pytest.mark.parametrize("acq", ["uniform", "paired_shift", "active_testing"])
@pytest.mark.parametrize("miscalibrated", [False, True])
def test_null_false_alarm_and_coverage(acq, miscalibrated):
    reps, alpha = 60, 0.05
    false, covered = 0, 0
    for seed in range(reps):
        ep = generate_episode("null_noisy", n_items=1200, seed=100 + seed, flip_rate=0.1)
        pol = _inverted_model() if miscalibrated else None
        r = _run(
            ep, AdaptiveGlobalPlan(acquisition=acq, noninferiority_margin=None), 250, seed, pol
        )
        false += r.decision.value == "confirmed_regression"
        covered += r.lower <= ep.truth()["delta"] <= r.upper
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert false / reps <= alpha + 3 * se
    assert covered / reps >= 0.9 - 3 * np.sqrt(0.09 / reps)


def test_propensities_logged_and_positive():
    ep = generate_episode("slice_regression", n_items=1000, seed=52)
    r = _run(ep, AdaptiveGlobalPlan(), 120, 0)
    q = r.events["selection_probability"]
    assert len(q) == r.n and (q > 0).all() and (q <= 1).all()
    assert r.events["sample_id"].is_unique  # every draw is a new paid call


def test_budget_respected_and_exhaustive_pool_is_exact():
    ep = generate_episode("global_regression", n_items=150, seed=53, severity=0.1)
    plan = AdaptiveGlobalPlan(noninferiority_margin=None, regression_margin=0.99)
    r = _run(ep, plan, 1000, 0)
    assert r.n == 150 and r.stop_reason == "pool_exhausted"
    assert r.lower <= ep.truth()["delta"] <= r.upper
