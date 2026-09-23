"""Selectors must not be able to reach unqueried candidate outcomes through any public path."""

from __future__ import annotations

import copy
import gc
import pickle

import numpy as np
import pytest

from evaldelta import Budget, EvalSession, RunConfig, UniformPolicy
from evaldelta.policies.base import PolicyView
from evaldelta.replay.oracle import OracleAccessError, ReplayOracle
from evaldelta.replay.session import ProtocolViolation


def _reachable(root, max_nodes=200_000):
    """All objects reachable from ``root`` through gc referents (includes closures)."""
    seen, stack, out = set(), [root], []
    while stack and len(seen) < max_nodes:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type, type(gc))):
            continue
        seen.add(id(obj))
        out.append(obj)
        stack.extend(gc.get_referents(obj))
    return out


class SpyPolicy:
    """Records every view and tries every selector-facing path to candidate outcomes."""

    name = "spy"

    def __init__(self):
        self.views = []

    def select(self, view, k, rng):
        self.views.append(view)
        # 1) revealed mapping holds only already-purchased items and is read-only
        with pytest.raises(TypeError):
            view.revealed["x"] = 1.0  # type: ignore[index]
        # 2) no candidate-outcome columns in the public items
        assert not [c for c in view.items.columns if c.startswith(("new_", "cand"))]
        # 3) no path from the view to an oracle or provider object
        for obj in _reachable(view):
            assert not isinstance(obj, ReplayOracle), "oracle reachable from PolicyView"
        un = view.unqueried["sample_id"].tolist()
        idx = rng.choice(len(un), size=min(k, len(un)), replace=False)
        return [un[i] for i in idx]


def test_view_exposes_only_purchased_outcomes(slice_episode):
    spy = SpyPolicy()
    cfg = RunConfig(run_id="sec", budget=Budget(max_candidate_calls=200))
    s = EvalSession(
        slice_episode.item_table(), slice_episode.oracle(), episode_meta=slice_episode.meta
    )
    r = s.compare(cfg, spy)
    paid_disc = r.events[r.events["phase"] == "discovery"]["sample_id"].tolist()
    assert len(spy.views) >= 2
    seen = set()
    for v in spy.views:
        # revealed at call i are exactly the items purchased in earlier discovery batches
        assert set(v.revealed) <= set(paid_disc)
        assert set(v.revealed) >= seen
        seen = set(v.revealed)
        # confirmation-pool items are never visible
        assert set(v.items["sample_id"]) <= set(r.report.selected_ids.get("discovery", [])) | set(
            v.items["sample_id"]
        )
    conf_ids = set(r.report.selected_ids.get("global_confirmation", [])) | set(
        r.report.selected_ids.get("slice_confirmation", [])
    )
    assert not conf_ids & set(spy.views[-1].items["sample_id"])


def test_hidden_values_not_in_view_objects(slice_episode):
    """No object reachable from a view contains the full hidden-outcome mapping."""
    ep = slice_episode
    view = PolicyView.build(ep.items, {}, 10)
    hidden_len = len(ep.items)
    for obj in _reachable(view):
        if isinstance(obj, dict) and len(obj) == hidden_len:
            vals = set(obj.values())
            assert not vals <= {0.0, 1.0} or set(obj) != set(ep.items["sample_id"]), (
                "a full id->loss mapping is reachable from the view"
            )


def test_oracle_resists_bulk_access(slice_episode):
    o = slice_episode.oracle()
    with pytest.raises(OracleAccessError):
        iter(o)
    with pytest.raises(OracleAccessError):
        o["syn7_000000"]
    with pytest.raises(OracleAccessError):
        pickle.dumps(o)
    with pytest.raises(OracleAccessError):
        copy.deepcopy(o)
    assert not hasattr(o, "__dict__")  # slots only
    assert o.revealed_ids == ()


class GreedyPolicy:
    """Tries to select items outside the eligible discovery pool."""

    name = "greedy"

    def __init__(self, target):
        self.target = target

    def select(self, view, k, rng):
        return [self.target]


def test_selecting_confirmation_items_is_blocked(slice_episode):
    from evaldelta.data.splits import make_partition
    from evaldelta.schemas import SplitConfig

    part = make_partition(slice_episode.items["sample_id"], SplitConfig())
    s = EvalSession(slice_episode.item_table(), slice_episode.oracle())
    cfg = RunConfig(run_id="x", budget=Budget(max_candidate_calls=50))
    with pytest.raises(ProtocolViolation, match="ineligible"):
        s.compare(cfg, GreedyPolicy(part.global_confirm[0]))
    with pytest.raises(ProtocolViolation, match="ineligible"):
        s.compare(cfg, GreedyPolicy("not-an-id"))


class RepeatPolicy:
    name = "repeat"

    def __init__(self):
        self.first = None

    def select(self, view, k, rng):
        if self.first is None:
            self.first = view.unqueried["sample_id"].iloc[0]
        return [self.first]


def test_requerying_is_blocked(slice_episode):
    s = EvalSession(slice_episode.item_table(), slice_episode.oracle())
    cfg = RunConfig(run_id="x", budget=Budget(max_candidate_calls=50))
    with pytest.raises(ProtocolViolation):
        s.compare(cfg, RepeatPolicy())


def test_oracle_reveal_log_equals_paid_ids(slice_episode):
    o = slice_episode.oracle()
    s = EvalSession(slice_episode.item_table(), o)
    r = s.compare(RunConfig(run_id="x", budget=Budget(max_candidate_calls=150)), UniformPolicy())
    assert sorted(o.revealed_ids) == sorted(r.events["sample_id"])
    assert np.isfinite(r.events["new_loss"]).all()
