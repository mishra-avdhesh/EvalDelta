from __future__ import annotations

import numpy as np
import pytest

from evaldelta import Budget, ConfirmPlan, Decision, EvalSession, RunConfig, UniformPolicy
from evaldelta.bench.synthetic import generate_episode
from evaldelta.statistics.confirmation import ConfirmationTest, look_schedule, resolve_method


def _test(method="mcnemar_exact", margin=0.0, ni=0.02, n=300, looks=3, pop=5000):
    return ConfirmationTest("global", "Delta", method, margin, ni, 0.05, n, looks, pop)


def test_look_schedule():
    assert look_schedule(300, 3) == [100, 200, 300]
    assert look_schedule(2, 3) == [1, 2]
    assert look_schedule(0, 3) == []


def test_bonferroni_only_for_fixed_sample_methods():
    assert _test("mcnemar_exact").per_look_alpha == pytest.approx(0.05 / 3)
    assert _test("hoeffding_wor").per_look_alpha == pytest.approx(0.05 / 3)
    assert _test("betting_wor").per_look_alpha == pytest.approx(0.05)


def test_zero_discordants_is_not_regression():
    ev = _test().evaluate(np.zeros(300), np.zeros(300), 2, final=True)
    assert ev.decision != Decision.CONFIRMED_REGRESSION
    assert ev.p_value == 1.0 and ev.n_down == 0 and ev.n_up == 0


def test_all_down_is_regression_and_swap_is_not():
    old = np.zeros(100)
    new = np.r_[np.ones(20), np.zeros(80)]
    reg = _test(n=100).evaluate(old, new, 0, final=False)
    assert reg.decision == Decision.CONFIRMED_REGRESSION
    swapped = _test(n=100).evaluate(new, old, 0, final=False)
    assert swapped.decision != Decision.CONFIRMED_REGRESSION
    assert swapped.effect == -reg.effect


def test_noninferiority_requires_tight_upper_bound():
    # 2000 identical results: upper bound on Delta well below 0.02 -> NI evidence
    t = _test("betting_wor", ni=0.02, n=2000, looks=1, pop=10000)
    ev = t.evaluate(np.zeros(2000), np.zeros(2000), 0, final=True)
    assert ev.decision == Decision.EVIDENCE_OF_NONINFERIORITY
    assert ev.upper < 0.02
    # 30 identical results are NOT enough evidence
    ev2 = _test("betting_wor", ni=0.02, n=30, looks=1).evaluate(np.zeros(30), np.zeros(30), 0, True)
    assert ev2.decision == Decision.INCONCLUSIVE


def test_unsupported_methods_refused():
    with pytest.raises(ValueError):
        ConfirmPlan(method="mcnemar_exact", regression_margin=0.01)
    with pytest.raises(ValueError):
        resolve_method(ConfirmPlan(method="mcnemar_exact"), binary=False, margin=0.0)
    assert resolve_method(ConfirmPlan(), binary=True, margin=0.0) == "mcnemar_exact"
    assert resolve_method(ConfirmPlan(), binary=True, margin=0.01) == "betting_wor"
    assert resolve_method(ConfirmPlan(), binary=False, margin=0.0) == "betting_wor"
    ev = _test("hoeffding_wor").evaluate(np.zeros(10), np.ones(10), 0, final=False)
    assert ev.p_value is None  # hoeffding reports an interval, never an invented p-value


def test_small_slices_are_inconclusive_not_confirmed():
    ep = generate_episode("rare_severe", n_items=3000, seed=4, rare_prevalence=0.01)
    cfg = RunConfig(
        run_id="s",
        budget=Budget(max_candidate_calls=400),
        plan=ConfirmPlan(predeclared_slices=("rare",), min_slice_confirm=30),
    )
    r = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, UniformPolicy())
    rare = [e for e in r.report.slice_results if e.scope == "slice:rare"][0]
    assert rare.decision == Decision.INCONCLUSIVE
    assert "insufficient_fresh_examples" in rare.reason
    assert rare.n == 0


def test_budget_exhaustion_gives_final_inconclusive():
    ep = generate_episode("null_noisy", n_items=3000, seed=5)
    cfg = RunConfig(run_id="b", budget=Budget(max_candidate_calls=30))
    r = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, UniformPolicy())
    g = r.report.global_result
    assert g.decision == Decision.INCONCLUSIVE and g.reason == "boundaries_not_crossed"
    assert r.report.total_paid_calls <= 30


def test_pool_smaller_than_plan():
    ep = generate_episode("global_regression", n_items=200, seed=6, severity=0.2)
    cfg = RunConfig(run_id="p", budget=Budget(max_candidate_calls=1000))
    r = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, UniformPolicy())
    assert r.report.spend["global_confirmation"].paid_calls <= 40  # |C_global| = 40
