"""Property-based invariants (hypothesis)."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evaldelta import Budget, ConfirmPlan, Decision, EvalSession, RunConfig, UniformPolicy
from evaldelta.bench.synthetic import generate_episode
from evaldelta.policies.uniform import StratifiedPolicy

EP = generate_episode("slice_regression", n_items=1200, seed=21)
NULL = generate_episode("null_identical", n_items=1200, seed=22)
SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(
    budget=st.integers(1, 400),
    fr=st.tuples(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1)),
    seed=st.integers(0, 10_000),
    batch=st.integers(1, 60),
    strat=st.booleans(),
)
@SETTINGS
def test_budget_and_phase_ceilings_hold(budget, fr, seed, batch, strat):
    tot = sum(fr) or 1.0
    d, g, s = (x / tot * 0.999 for x in fr)
    cfg = RunConfig(
        run_id="p",
        seed=seed,
        budget=Budget(
            max_candidate_calls=budget, discovery_fraction=d, global_fraction=g, slice_fraction=s
        ),
        policy={"name": "uniform", "batch_size": batch},
    )
    r = EvalSession(EP.item_table(), EP.oracle()).compare(
        cfg, StratifiedPolicy() if strat else UniformPolicy()
    )
    assert r.report.total_paid_calls <= budget
    for spend in r.report.spend.values():
        assert spend.paid_calls <= spend.planned_calls
    ids = list(r.events["sample_id"]) if len(r.events) else []
    assert len(ids) == len(set(ids))
    # per-phase denominators equal the number of successful paid calls
    if r.report.global_result is not None:
        n_glob = int((r.events["phase"] == "global_confirmation").sum()) if len(ids) else 0
        assert r.report.global_result.n == n_glob


@given(
    seed=st.integers(0, 10_000),
    budget=st.integers(10, 400),
    method=st.sampled_from(["auto", "betting_wor", "hoeffding_wor"]),
)
@SETTINGS
def test_identical_models_never_confirm_regression(seed, budget, method):
    cfg = RunConfig(
        run_id="n",
        seed=seed,
        budget=Budget(max_candidate_calls=budget),
        plan=ConfirmPlan(method=method),
    )
    r = EvalSession(NULL.item_table(), NULL.oracle()).compare(cfg, UniformPolicy())
    assert r.report.decision != Decision.CONFIRMED_REGRESSION


@given(seed=st.integers(0, 1000))
@settings(max_examples=10, deadline=None)
def test_swap_negates_observed_effects(seed):
    cfg = RunConfig(
        run_id="s",
        seed=seed,
        budget=Budget(max_candidate_calls=200),
        plan=ConfirmPlan(method="betting_wor"),
    )
    a = EvalSession(EP.item_table(), EP.oracle()).compare(cfg, UniformPolicy())
    sw = EP.swapped()
    b = EvalSession(sw.item_table(), sw.oracle()).compare(cfg, UniformPolicy())
    ga, gb = a.report.global_result, b.report.global_result
    # same IDs are drawn from confirmation (discovery is uniform & seeded identically)
    assert a.report.selected_ids.get("global_confirmation") == b.report.selected_ids.get(
        "global_confirmation"
    )
    assert ga.effect == -gb.effect
    assert ga.n_down == gb.n_up and ga.n_up == gb.n_down
    assert abs(ga.lower + gb.upper) < 1e-4 and abs(ga.upper + gb.lower) < 1e-4
