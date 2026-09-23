"""Coverage and monotonicity of the betting confidence sequence engine."""

from __future__ import annotations

import numpy as np
import pytest

from evaldelta.statistics import sequential as seq


def _pool(rng, n, p_down, p_up):
    return rng.choice([1.0, -1.0, 0.0], size=n, p=[p_down, p_up, 1 - p_down - p_up])


@pytest.mark.parametrize("p_down,p_up", [(0.03, 0.03), (0.06, 0.02), (0.2, 0.25)])
def test_wor_anytime_coverage(p_down, p_up):
    rng = np.random.default_rng(1)
    pool = _pool(rng, 1500, p_down, p_up)
    mu = pool.mean()
    reps, alpha, miss_lo, miss_up = 300, 0.05, 0, 0
    for _ in range(reps):
        s = rng.permutation(pool)[:400]
        data_lo = seq.wor_data(s, len(pool), alpha, 400)
        data_up = seq.wor_data(-s, len(pool), alpha, 400)
        # anytime: miscoverage at ANY time counts
        miss_lo += bool(np.max(seq.log_capital_path(data_lo, mu)) >= np.log(1 / alpha))
        miss_up += bool(np.max(seq.log_capital_path(data_up, -mu)) >= np.log(1 / alpha))
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert miss_lo / reps <= alpha + 3 * se
    assert miss_up / reps <= alpha + 3 * se


def test_iid_coverage():
    rng = np.random.default_rng(2)
    reps, alpha, miss = 300, 0.05, 0
    for _ in range(reps):
        s = rng.choice([1.0, -1.0, 0.0], size=300, p=[0.05, 0.03, 0.92])
        cs = seq.confidence_bounds(s, alpha, 300)
        miss += cs.lower > 0.02
    assert miss / reps <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)


def test_capital_is_monotone_in_m():
    rng = np.random.default_rng(3)
    for design in ("iid", "wor"):
        s = rng.choice([1.0, -1.0, 0.0], size=200, p=[0.1, 0.05, 0.85])
        data = seq.iid_data(s, 0.05, 200) if design == "iid" else seq.wor_data(s, 500, 0.05, 200)
        grid = np.linspace(-0.3, 0.3, 121)
        caps = [np.max(seq.log_capital_path(data, m)) for m in grid]
        assert all(a >= b - 1e-9 for a, b in zip(caps, caps[1:], strict=False))


def test_exhaustive_wor_pins_the_mean():
    rng = np.random.default_rng(4)
    pool = rng.choice([1.0, -1.0, 0.0], size=120, p=[0.1, 0.05, 0.85])
    cs = seq.confidence_bounds(rng.permutation(pool), 0.05, 120, population_size=120)
    assert cs.lower <= pool.mean() + 1e-6 <= cs.upper + 2e-6
    assert cs.upper - cs.lower < 0.02  # sampling the whole pool leaves ~no uncertainty


def test_all_zero_differences():
    cs = seq.confidence_bounds(np.zeros(200), 0.05, 200, population_size=1000)
    assert cs.lower < 0 < cs.upper
    assert cs.p_regression == 1.0


def test_hoeffding_references():
    assert seq.hoeffding_radius(0, 0.05) == float("inf")
    r = seq.hoeffding_radius(100, 0.05)
    assert r == pytest.approx(2 * np.sqrt(np.log(20) / 200))
    assert seq.hoeffding_ipw_radius(10, 0.05, 0.1) > seq.hoeffding_ipw_radius(1000, 0.05, 0.1)
