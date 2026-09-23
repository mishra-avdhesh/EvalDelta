from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evaldelta.data.splits import make_partition
from evaldelta.replay.budget import BudgetExceeded, BudgetMeter
from evaldelta.schemas import Budget, SplitConfig


def test_partition_disjoint_complete_and_sized():
    ids = [f"i{k}" for k in range(1000)]
    p = make_partition(ids, SplitConfig(seed=3))
    d, g, s = set(p.discovery), set(p.global_confirm), set(p.slice_confirm)
    assert not (d & g) and not (d & s) and not (g & s)
    assert d | g | s == set(ids)
    assert (len(d), len(g), len(s)) == (600, 200, 200)


def test_partition_deterministic_and_seed_sensitive():
    ids = [f"i{k}" for k in range(300)]
    a = make_partition(ids, SplitConfig(seed=1))
    b = make_partition(ids, SplitConfig(seed=1))
    c = make_partition(ids, SplitConfig(seed=2))
    assert a == b
    assert a.split_hash != c.split_hash


@given(st.permutations([f"id{k}" for k in range(60)]), st.integers(0, 10_000))
@settings(max_examples=40, deadline=None)
def test_partition_invariant_to_row_order(perm, seed):
    base = make_partition(sorted(perm), SplitConfig(seed=seed))
    assert make_partition(perm, SplitConfig(seed=seed)) == base


def test_partition_rejects_duplicates():
    with pytest.raises(ValueError):
        make_partition(["a", "a", "b"], SplitConfig())


def test_split_fractions_validated():
    with pytest.raises(ValueError):
        SplitConfig(discovery=0.5, global_confirm=0.2, slice_confirm=0.2)


def test_budget_meter_raises_before_charging():
    m = BudgetMeter(10, phase_caps={"a": 4, "b": 6})
    m.charge("a", 4, 4.0)
    with pytest.raises(BudgetExceeded):
        m.charge("a", 1, 1.0)
    assert m.calls == 4  # nothing charged by the failed attempt
    m.charge("b", 6, 6.0)
    assert m.remaining_calls("b") == 0


def test_budget_cost_ceiling():
    m = BudgetMeter(100, max_cost=5.0, phase_caps={"a": 100})
    m.charge("a", 2, 4.5)
    assert not m.can_afford("a", 1, 1.0)
    with pytest.raises(BudgetExceeded):
        m.charge("a", 1, 1.0)


def test_budget_config_validation():
    with pytest.raises(ValueError):
        Budget(max_candidate_calls=10, discovery_fraction=0.6, global_fraction=0.6)
    with pytest.raises(ValueError):
        Budget(max_candidate_calls=10, max_cost=3.0)  # cost ceiling needs a declared unit
    assert Budget(max_candidate_calls=100).phase_calls() == {
        "discovery": 30,
        "global_confirmation": 50,
        "slice_confirmation": 20,
    }
