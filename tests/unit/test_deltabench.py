from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaldelta.bench.deltabench import (
    build_pair_episode,
    inject,
    leakage_check,
    load_source,
    pick_slice,
)


@pytest.fixture()
def toy_source(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    m = {
        "sample_id": [f"t{i}" for i in range(n)],
        "slice": rng.choice(["a", "b", "c"], n),
        "label": "0",
        "estimated_candidate_cost": 1.0,
        "f_meta": rng.normal(size=n),
    }
    for v in ("v1", "v2", "v3", "v4"):
        loss = (rng.random(n) < 0.2).astype(float)
        m[f"loss__{v}"] = loss
        m[f"conf__{v}"] = rng.uniform(0.5, 1, n)
        m[f"ptrue__{v}"] = np.where(loss == 0, 0.8, 0.2)
        m[f"pred__{v}"] = "x"
    pd.DataFrame(m).to_parquet(tmp_path / "matrix.parquet")
    (tmp_path / "versions.json").write_text(
        json.dumps(
            {
                "source": "toy",
                "meta": {"task": "toy"},
                "versions": [
                    {"id": v, "release_index": i, "change_type": "seed"}
                    for i, v in enumerate(["v1", "v2", "v3", "v4"], 1)
                ],
            }
        )
    )
    load_source.cache_clear()
    return str(tmp_path)


def test_public_table_independent_of_new_version(toy_source):
    src = load_source(toy_source)
    ep = build_pair_episode(src, "v2", "v3")
    # tamper with the NEW version's outcomes: public items must not change
    src.matrix["loss__v3"] = 1.0 - src.matrix["loss__v3"]
    src.matrix["conf__v3"] = 0.0
    ep2 = build_pair_episode(src, "v2", "v3")
    pd.testing.assert_frame_equal(ep.items, ep2.items)
    assert ep.truth()["delta"] != ep2.truth()["delta"]
    leakage_check(ep)


def test_history_uses_only_prior_versions(toy_source):
    src = load_source(toy_source)
    ep = build_pair_episode(src, "v1", "v3")
    assert (ep.items["hist_n_versions"] == 2).all()  # v1, v2 only (never v3 or v4)
    expected = (src.loss("v1") + src.loss("v2")) / 2
    np.testing.assert_allclose(ep.items["hist_err_rate"], expected)
    with pytest.raises(ValueError):
        build_pair_episode(src, "v3", "v2")


def test_injections_have_intended_effect(toy_source):
    src = load_source(toy_source)
    base = build_pair_episode(src, "v2", "v3")
    d0 = base.truth()["delta"]
    assert inject(base, pattern="null", seed=0).truth()["delta"] == 0.0
    g = inject(base, pattern="global_degradation", seed=0, severity=0.05)
    assert g.truth()["delta"] == pytest.approx(d0 + 0.05, abs=1e-9)
    assert g.meta["is_synthetic"] is True
    s = inject(base, pattern="slice_only", seed=0, slice_name="a", slice_severity=0.2)
    assert s.truth()["slice_delta"]["a"] > base.truth()["slice_delta"]["a"] + 0.15
    assert s.truth()["slice_delta"]["b"] == base.truth()["slice_delta"]["b"]
    c = inject(base, pattern="compensating", seed=0, slice_name="a", slice_severity=0.2)
    assert c.truth()["delta"] == pytest.approx(d0, abs=1e-9)
    assert pick_slice(src, "common", 3) == pick_slice(src, "common", 3)
