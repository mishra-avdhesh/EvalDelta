"""Exact size of one-sided McNemar under finite-pool sampling without replacement (protocol §4).

This is a numerical check on a grid, not a theorem: at the null boundary n_down_pool == n_up_pool,
P(reject) is computed exactly from the multivariate hypergeometric distribution.
"""

from __future__ import annotations

import pytest
from scipy import stats

from evaldelta.statistics.paired_tests import mcnemar_exact_one_sided


def exact_size(pool_n: int, d: int, n: int, alpha: float, up: int | None = None) -> float:
    up = d if up is None else up
    total = 0.0
    for xd in range(0, min(d, n) + 1):
        for xu in range(0, min(up, n - xd) + 1):
            if mcnemar_exact_one_sided(xd, xu) <= alpha:
                total += stats.multivariate_hypergeom.pmf(
                    [xd, xu, n - xd - xu], [d, up, pool_n - d - up], n
                )
    return total


@pytest.mark.parametrize("pool_n", [40, 200, 1000])
@pytest.mark.parametrize("frac", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("alpha", [0.05, 0.05 / 3])
def test_mcnemar_size_at_boundary(pool_n, frac, alpha):
    d = max(1, int(pool_n * frac / 2))
    for n in sorted({10, 30, pool_n // 4, pool_n // 2, pool_n}):
        if n <= pool_n:
            assert exact_size(pool_n, d, n, alpha) <= alpha + 1e-12


def test_mcnemar_size_inside_null():
    # strictly inside H0 (more ups than downs) the test is even more conservative
    assert exact_size(300, 10, 150, 0.05, up=20) <= exact_size(300, 15, 150, 0.05)
