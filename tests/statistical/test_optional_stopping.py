"""Naive repeated peeking inflates type-I error. The declared plans do not."""

from __future__ import annotations

import numpy as np

from evaldelta.statistics import sequential as seq
from evaldelta.statistics.paired_tests import mcnemar_exact_one_sided

ALPHA = 0.05


def _null_stream(rng, n):
    # symmetric discordance: Delta = 0 exactly in expectation (i.i.d.)
    return rng.choice([1.0, -1.0, 0.0], size=n, p=[0.15, 0.15, 0.7])


def _reject_mcnemar(d, looks, alpha_look):
    for n in looks:
        x = d[:n]
        if mcnemar_exact_one_sided(int((x == 1).sum()), int((x == -1).sum())) <= alpha_look:
            return True
    return False


def test_peeking_inflates_but_plans_hold():
    rng = np.random.default_rng(0)
    reps = 600
    naive = bonf = betting = 0
    peek_looks = list(range(10, 401, 10))  # 40 uncorrected looks
    plan_looks = [134, 267, 400]
    for _ in range(reps):
        d = _null_stream(rng, 400)
        naive += _reject_mcnemar(d, peek_looks, ALPHA)
        bonf += _reject_mcnemar(d, plan_looks, ALPHA / len(plan_looks))
        data = seq.iid_data(d, ALPHA, 400)
        betting += bool(np.max(seq.log_capital_path(data, 0.0)) >= np.log(1 / ALPHA))
    se = np.sqrt(ALPHA * (1 - ALPHA) / reps)
    assert naive / reps > ALPHA + 3 * se, "expected naive peeking to inflate the error"
    assert bonf / reps <= ALPHA + 3 * se
    assert betting / reps <= ALPHA + 3 * se
