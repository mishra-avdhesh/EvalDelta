from __future__ import annotations

import numpy as np
import pytest

from evaldelta.bench.synthetic import generate_episode
from evaldelta.policies.base import PolicyView
from evaldelta.policies.paired_shift import FEATURES, PairedShift, PairedShiftModel, feature_matrix
from evaldelta.policies.static import STATIC_POLICIES

EP = generate_episode("slice_regression", n_items=3000, seed=41)


def _view(revealed=None):
    return PolicyView.build(EP.items, revealed or {}, 100)


@pytest.mark.parametrize("name", sorted(STATIC_POLICIES) + ["paired_shift"])
def test_policies_return_unique_unqueried_ids(name):
    pol = PairedShift() if name == "paired_shift" else STATIC_POLICIES[name]()
    ids = EP.items["sample_id"].tolist()
    revealed = {i: 0.0 for i in ids[:50]}
    out = pol.select(_view(revealed), 40, np.random.default_rng(0))
    assert len(out) == len(set(out)) == 40
    assert not set(out) & set(revealed)


def test_paired_shift_exploration_and_cap():
    ps = PairedShift(explore_frac=0.0, slice_cap=0.25)
    out = ps.select(_view(), 40, np.random.default_rng(1))
    counts = EP.items.set_index("sample_id").loc[out, "slice"].value_counts()
    assert counts.max() <= 10  # ceil(0.25 * 40)


def test_paired_shift_refits_only_at_fixed_increments():
    ps = PairedShift(refit_every=25)
    ids = EP.items["sample_id"].tolist()
    base = {i: 1.0 for i in ids[:25]}
    p1 = ps.probabilities(_view(base))
    # 24 more reveals (< next increment) must not change probabilities
    more = dict(base) | {i: 1.0 for i in ids[25:49]}
    p2 = ps.probabilities(_view(more))
    np.testing.assert_allclose(p1["p_down"], p2["p_down"])
    # crossing the increment triggers a refit
    more2 = dict(more) | {ids[49]: 1.0}
    p3 = ps.probabilities(_view(more2))
    assert not np.allclose(p1["p_down"], p3["p_down"])


def test_model_roundtrip_and_training(tmp_path):
    items = EP.items
    new = items["sample_id"].map(EP._hidden).to_numpy(dtype=float)
    m = PairedShiftModel.train([("e", items, new)])
    path = tmp_path / "m.json"
    m.save(path)
    m2 = PairedShiftModel.load(path)
    x = feature_matrix(items)
    np.testing.assert_allclose(m.down.logits(x), m2.down.logits(x))
    assert x.shape[1] == len(FEATURES)


def test_suspicious_slices_finds_injected_slice_with_enough_data():
    target = EP.meta["target_slice"]
    ids = EP.items["sample_id"].tolist()
    revealed = {i: EP._hidden[i] for i in ids[:1500]}  # lots of purchased discovery data
    found = PairedShift(min_slice_size=50).suspicious_slices(_view(revealed), 3)
    assert target in found
