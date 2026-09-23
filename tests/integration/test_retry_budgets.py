"""Provider retries must never bypass paid-call, cost, or confirmation contracts."""

import subprocess
import urllib.error
from collections import Counter

import pytest

from evaldelta import (
    Budget,
    ConfirmPlan,
    Decision,
    EvalSession,
    RunConfig,
    UniformPolicy,
    generate_episode,
)
from evaldelta.config import _import_object
from evaldelta.data.io import ItemTable
from evaldelta.providers.base import evaluate_paid
from evaldelta.providers.callable import CallableProvider
from evaldelta.providers.command import CommandProvider
from evaldelta.providers.http import HTTPProvider
from evaldelta.replay.adaptive import AdaptiveGlobalPlan, AdaptiveGlobalSession
from evaldelta.replay.budget import BudgetMeter
from evaldelta.schemas import SplitConfig


def _config(calls, **budget):
    return RunConfig(
        run_id="retry-test",
        split=SplitConfig(discovery=0, global_confirm=1, slice_confirm=0),
        budget=Budget(
            max_candidate_calls=calls,
            discovery_fraction=0,
            global_fraction=1,
            slice_fraction=0,
            **budget,
        ),
        plan=ConfirmPlan(looks=1, noninferiority_margin=None),
    )


def test_module_and_file_callable_imports(tmp_path):
    assert _import_object("math:sqrt")(9) == 3
    (tmp_path / "candidate.py").write_text("def predict(item):\n    return item['value']\n")
    assert _import_object("candidate.py:predict", tmp_path)({"value": 7}) == 7


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("cost_cap", [None, 5.0])
def test_retries_charged_before_calls_and_failed_attempt_logged(adaptive, cost_cap):
    ep = generate_episode("null_identical", n_items=100, seed=7)
    frame = ep.items.copy()
    frame["estimated_candidate_cost"] = 2.0
    items = ItemTable(frame)
    calls = Counter()

    def transient(item):
        sid = item["sample_id"]
        calls[sid] += 1
        if calls[sid] % 2:
            raise RuntimeError("transient")
        return item["old_loss"]

    provider = CallableProvider(transient, lambda out, item: out, retries=5)
    if adaptive:
        result = AdaptiveGlobalSession(items, provider).run(
            AdaptiveGlobalPlan(regression_margin=0.99, noninferiority_margin=None),
            5,
            max_cost=cost_cap,
        )
        paid, cost, events = result.paid_calls, result.paid_cost, result.events
        decision = result.decision
    else:
        result = EvalSession(items, provider).compare(
            _config(5, cost_unit="units", max_cost=cost_cap), UniformPolicy()
        )
        paid, cost, events = (
            result.report.total_paid_calls,
            result.report.total_paid_cost,
            result.events,
        )
        decision = result.decision
    assert paid == sum(calls.values()) == (5 if cost_cap is None else 2)
    assert cost == 2 * paid
    assert events.candidate_calls.sum() == paid
    assert events.cost_actual.sum() == cost
    if cost_cap is None:
        assert decision == Decision.EVALUATION_ERROR
        assert events.iloc[-1].new_loss != events.iloc[-1].new_loss  # missing, not a wrong answer
    else:
        assert decision == Decision.INCONCLUSIVE


def test_no_unscheduled_fixed_sample_decision_after_retry_exhausts_budget():
    ep = generate_episode("null_identical", n_items=100, seed=3)
    calls = Counter()

    def candidate(item):
        calls[item["sample_id"]] += 1
        if calls[item["sample_id"]] == 1:
            raise RuntimeError("retry")
        return 1.0

    result = EvalSession(
        ep.item_table(), CallableProvider(candidate, lambda out, item: out)
    ).compare(_config(20), UniformPolicy())
    assert sum(calls.values()) == result.report.total_paid_calls == 20
    assert result.report.global_result.n == 10
    assert result.decision == Decision.INCONCLUSIVE
    assert result.report.global_result.p_value is None
    assert result.report.global_result.reason == "budget_exhausted_before_checkpoint"


def test_one_missing_confirmation_outcome_blocks_pass():
    ep = generate_episode("null_identical", n_items=300, seed=2)
    provider = CallableProvider(lambda item: 1 / 0, lambda out, item: 0, retries=0)
    result = EvalSession(ep.item_table(), provider).compare(_config(200), UniformPolicy())
    assert result.decision == Decision.EVALUATION_ERROR
    assert result.report.total_paid_calls == 1


@pytest.mark.parametrize("kind", ["http", "command"])
def test_external_provider_does_not_retry_past_reservation(kind, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if kind == "http":
            raise urllib.error.URLError("offline")
        raise subprocess.TimeoutExpired("candidate", 1)

    if kind == "http":
        provider = HTTPProvider(
            "http://localhost/test", lambda out, item: 0, retries=4, backoff_s=0, opener=fail
        )
    else:
        monkeypatch.setattr(subprocess, "run", fail)
        provider = CommandProvider("candidate", lambda out, item: 0, retries=4)
    meter = BudgetMeter(2)

    def reserve():
        if not meter.can_afford("test", 1, 1):
            return False
        meter.charge("test", 1, 1)
        return True

    outcome = evaluate_paid(provider, {"sample_id": "one"}, reserve)
    assert outcome.loss is None
    assert outcome.attempts == meter.calls == len(calls) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0},
        {"epsilon": 1.5},
        {"batch_size": 0},
        {"check_every": 0},
        {"alpha": 1},
        {"variance_floor": float("nan")},
    ],
)
def test_invalid_adaptive_plan_rejected(kwargs):
    with pytest.raises(ValueError):
        AdaptiveGlobalPlan(**kwargs)
