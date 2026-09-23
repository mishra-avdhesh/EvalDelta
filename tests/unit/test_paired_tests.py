from __future__ import annotations

import pytest
from scipy import stats

from evaldelta.statistics.multiplicity import holm, holm_adjusted
from evaldelta.statistics.paired_tests import (
    clopper_pearson_lower,
    mcnemar_exact_one_sided,
)


def test_mcnemar_trivial_values():
    assert mcnemar_exact_one_sided(0, 0) == 1.0  # zero discordants -> no evidence
    assert mcnemar_exact_one_sided(1, 0) == pytest.approx(0.5)
    assert mcnemar_exact_one_sided(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_exact_one_sided(0, 5) == pytest.approx(1.0)
    # reference: P(Bin(10, .5) >= 8)
    ref = sum(stats.binom.pmf(k, 10, 0.5) for k in range(8, 11))
    assert mcnemar_exact_one_sided(8, 2) == pytest.approx(ref)


def test_mcnemar_direction():
    # more ups than downs = improvement -> never significant as regression
    assert mcnemar_exact_one_sided(2, 30) > 0.99


def test_clopper_pearson_matches_mcnemar_decision():
    alpha = 0.05
    for down in range(0, 25):
        for up in range(0, 25):
            k = down + up
            if k == 0:
                continue
            p = mcnemar_exact_one_sided(down, up)
            lb = clopper_pearson_lower(down, k, alpha)
            assert (p <= alpha) == (lb > 0.5 - 1e-12) or abs(p - alpha) < 1e-9


def test_holm():
    assert holm([0.01, 0.04, 0.03], 0.05) == [True, False, False]
    assert holm([0.01, 0.02, 0.03], 0.05) == [True, True, True]
    assert holm([], 0.05) == []
    adj = holm_adjusted([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
