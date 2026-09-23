"""Fixed-sample paired tests for 0/1 losses (protocol §4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Discordance:
    n: int
    n_down: int  # old correct, new wrong  (d = +1)
    n_up: int  # old wrong,  new correct (d = -1)

    @property
    def n_discordant(self) -> int:
        return self.n_down + self.n_up

    @property
    def effect(self) -> float:
        return (self.n_down - self.n_up) / self.n if self.n else 0.0


def discordance(old_loss: np.ndarray, new_loss: np.ndarray) -> Discordance:
    """Count paired discordances with explicit orientation (old -> new)."""
    old = np.asarray(old_loss, dtype=float)
    new = np.asarray(new_loss, dtype=float)
    if old.shape != new.shape:
        raise ValueError("old and new losses must be aligned")
    if not (np.isin(old, (0.0, 1.0)).all() and np.isin(new, (0.0, 1.0)).all()):
        raise ValueError("discordance counts require 0/1 losses")
    down = int(np.sum((old == 0.0) & (new == 1.0)))
    up = int(np.sum((old == 1.0) & (new == 0.0)))
    return Discordance(len(old), down, up)


def mcnemar_exact_one_sided(n_down: int, n_up: int) -> float:
    """One-sided exact McNemar p-value for H0: P(down) <= P(up) (i.e. Delta <= 0).

    ``p = P(Bin(n_down + n_up, 1/2) >= n_down)``. Zero discordant pairs gives p = 1.
    """
    if n_down < 0 or n_up < 0:
        raise ValueError("counts must be non-negative")
    k = n_down + n_up
    if k == 0:
        return 1.0
    return float(stats.binom.sf(n_down - 1, k, 0.5))


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided (1 - alpha) Clopper-Pearson lower bound for a binomial proportion."""
    if trials == 0 or successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if trials == 0 or successes == trials:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, successes + 1, trials - successes))
