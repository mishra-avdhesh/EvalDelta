from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaldelta.bench.synthetic import SCENARIOS, generate_episode
from evaldelta.data.io import DataValidationError, ItemTable, load_hidden_outcomes
from evaldelta.statistics.paired_tests import discordance


def _items(**over):
    df = pd.DataFrame(
        {"sample_id": ["a", "b", "c"], "slice": ["x", "x", "y"], "old_loss": [0.0, 1.0, 0.0]}
    )
    for k, v in over.items():
        df[k] = v
    return df


def test_discordance_orientation():
    old = np.array([0, 0, 1, 1, 0.0])
    new = np.array([1, 0, 0, 1, 1.0])
    d = discordance(old, new)
    assert (d.n_down, d.n_up) == (2, 1)  # old-correct->new-wrong counted as down
    assert d.effect == pytest.approx((2 - 1) / 5)
    assert np.mean(new - old) == pytest.approx(d.effect)  # positive = degradation


def test_swap_negates_delta():
    ep = generate_episode("global_regression", n_items=1500, seed=1, severity=0.04)
    t, ts = ep.truth(), ep.swapped().truth()
    assert t["delta"] > 0
    assert ts["delta"] == pytest.approx(-t["delta"])
    assert (ts["n_down"], ts["n_up"]) == (t["n_up"], t["n_down"])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_have_intended_sign(scenario):
    ep = generate_episode(scenario, n_items=4000, seed=11)
    tr = ep.truth()
    if scenario == "null_identical":
        assert tr["n_down"] == tr["n_up"] == 0
    elif scenario in {"null_noisy", "compensating"}:
        assert tr["delta"] == pytest.approx(0.0, abs=1e-12)
    elif scenario == "improvement":
        assert tr["delta"] < 0
    else:
        assert tr["delta"] > 0
    if scenario in {"slice_regression", "compensating", "rare_severe"}:
        target = ep.meta["target_slice"]
        assert tr["slice_delta"][target] > 0.1


def test_generator_is_deterministic():
    a = generate_episode("slice_regression", n_items=800, seed=5)
    b = generate_episode("slice_regression", n_items=800, seed=5)
    pd.testing.assert_frame_equal(a.items, b.items)
    assert a.truth() == b.truth()


def test_duplicate_ids_rejected():
    df = _items()
    df.loc[2, "sample_id"] = "a"
    with pytest.raises(DataValidationError, match="duplicate"):
        ItemTable(df)


@pytest.mark.parametrize("col", ["new_loss", "candidate_output", "delta"])
def test_candidate_columns_rejected_from_public_table(col):
    with pytest.raises(DataValidationError, match="candidate-outcome-like"):
        ItemTable(_items(**{col: [0.0, 1.0, 0.0]}))


def test_bad_losses_rejected():
    with pytest.raises(DataValidationError, match="NaN"):
        ItemTable(_items(old_loss=[0.0, np.nan, 1.0]))
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        ItemTable(_items(old_loss=[0.0, 1.5, 1.0]))
    with pytest.raises(DataValidationError, match="old_loss"):
        ItemTable(_items().drop(columns=["old_loss"]))


def test_hidden_outcome_join_is_strict():
    items = ItemTable(_items())
    ok = pd.DataFrame({"sample_id": ["c", "a", "b"], "new_loss": [1.0, 0.0, 0.0]})
    m = load_hidden_outcomes(ok, items)
    assert m == {"c": 1.0, "a": 0.0, "b": 0.0}
    with pytest.raises(DataValidationError, match="missing"):
        load_hidden_outcomes(ok.iloc[:2], items)
    extra = pd.concat([ok, pd.DataFrame({"sample_id": ["z"], "new_loss": [0.0]})])
    with pytest.raises(DataValidationError, match="unknown"):
        load_hidden_outcomes(extra, items)
    dup = pd.concat([ok, ok.iloc[:1]])
    with pytest.raises(DataValidationError, match="duplicate"):
        load_hidden_outcomes(dup, items)
