"""Betting confidence sequences with predictable bounds (protocol §6).

The engine handles three sampling designs through the affine conditional-mean map
``mu_s(m) = a_s + b_s * m``:

* i.i.d. / with replacement:        a_s = 0, b_s = 1
* without replacement, pool size N:  a_s = -S_{s-1}/(N-s+1),  b_s = N/(N-s+1)
* importance-weighted (propensity):  a_s = 0, b_s = 1 with predictable lower bounds L_s

Capital for a lower bound at hypothesised value ``m``::

    K_t(m) = prod_{s<=t} (1 + lam_s(m) * (Y_s - mu_s(m))),
    lam_s(m) = min(lam_tilde_s, c / (mu_s(m) - L_s)),  0 < c < 1.

Monotonicity (needed for bisection): write g = Y - mu. If g >= 0 the factor 1 + lam*g decreases in
m because mu increases and lam is nonincreasing. If g < 0 the factor is 1 - lam*|g|. Untruncated,
|g| grows with m. Truncated, lam*|g| = c*(mu - Y)/(mu - L), whose derivative in mu is
c*(Y - L)/(mu - L)^2 >= 0. Either way each factor is nonincreasing in m, so the rejection set
{m : max_t K_t(m) >= 1/alpha} is a lower half-line. Factors are >= 1 - c > 0.

At the true mean, K_t is a nonnegative martingale. Ville's inequality then gives time-uniform
(anytime-valid) coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

C_TRUNC = 0.5
_EPS = 1e-12


def plugin_lambdas(
    y: np.ndarray, alpha: float, n_plan: int, prior_mean: float = 0.0, prior_var: float = 0.25
) -> np.ndarray:
    """Predictable plug-in bet sizes lam_s = sqrt(2 log(1/alpha) / (n_plan * var_{s-1})).

    ``var_{s-1}`` is a regularised running variance computed from Y_1..Y_{s-1} only.
    """
    y = np.asarray(y, dtype=float)
    t = len(y)
    if t == 0:
        return np.empty(0)
    # running regularised mean after each observation: (prior + sum_{i<=j} y_i) / (j + 2)
    j = np.arange(1, t + 1, dtype=float)
    means = (prior_mean + np.cumsum(y)) / (j + 1.0)
    ss = prior_var + np.cumsum((y - means) ** 2)
    # variance available *before* step s uses observations 1..s-1 (predictable)
    var_prev = np.empty(t)
    var_prev[0] = prior_var
    var_prev[1:] = ss[:-1] / (j[:-1] + 1.0)
    log_term = 2.0 * math.log(1.0 / alpha)
    lam: np.ndarray = np.sqrt(log_term / (max(n_plan, 1) * np.maximum(var_prev, 1e-6)))
    return lam


@dataclass(frozen=True)
class BettingData:
    """Observed sequence plus predictable quantities for a lower-bound capital process."""

    y: np.ndarray
    lower: np.ndarray  # predictable lower bound L_s of Y_s
    a: np.ndarray
    b: np.ndarray
    lam: np.ndarray  # predictable, m-independent bet sizes


def log_capital_path(data: BettingData, m: float, c: float = C_TRUNC) -> np.ndarray:
    """log K_s(m) for s = 1..t; +inf once m is logically impossible."""
    mu = data.a + data.b * m
    gap = mu - data.lower
    if np.any(gap < -1e-9):
        first = int(np.argmax(gap < -1e-9))
        path = np.full(len(data.y), np.inf)
        if first > 0:
            path[:first] = log_capital_path(_truncate(data, first), m, c)
        return path
    with np.errstate(divide="ignore"):
        cap = np.where(gap > _EPS, c / np.maximum(gap, _EPS), np.inf)
    lam = np.minimum(data.lam, cap)
    lam = np.where(np.isfinite(lam), lam, 0.0)
    factors = 1.0 + lam * (data.y - mu)
    factors = np.maximum(factors, 1e-300)
    return np.cumsum(np.log(factors))


def _truncate(data: BettingData, t: int) -> BettingData:
    return BettingData(data.y[:t], data.lower[:t], data.a[:t], data.b[:t], data.lam[:t])


def rejects(data: BettingData, m: float, alpha: float, c: float = C_TRUNC) -> bool:
    if len(data.y) == 0:
        return False
    return bool(np.max(log_capital_path(data, m, c)) >= math.log(1.0 / alpha))


def anytime_p_value(data: BettingData, m: float, c: float = C_TRUNC) -> float:
    """Anytime-valid p-value for H0: mean <= m, equal to 1 / max_t K_t(m) (capped at 1)."""
    if len(data.y) == 0:
        return 1.0
    mx = float(np.max(log_capital_path(data, m, c)))
    return 1.0 if mx <= 0 else float(min(1.0, math.exp(-mx)))


def lower_bound(data: BettingData, alpha: float, lo: float, hi: float, tol: float = 1e-6) -> float:
    """Largest m (to tolerance, rounded down) rejected by the level-alpha capital process."""
    if len(data.y) == 0 or not rejects(data, lo, alpha):
        return lo
    if rejects(data, hi, alpha):
        return hi
    a, b = lo, hi  # invariant: rejects(a) and not rejects(b)
    while b - a > tol:
        mid = 0.5 * (a + b)
        if rejects(data, mid, alpha):
            a = mid
        else:
            b = mid
    return a


# ----------------------------------------------------------------------------------------------
# Convenience constructors for the three designs
# ----------------------------------------------------------------------------------------------


def iid_data(y: np.ndarray, alpha: float, n_plan: int, y_min: float = -1.0) -> BettingData:
    y = np.asarray(y, dtype=float)
    t = len(y)
    return BettingData(
        y, np.full(t, y_min), np.zeros(t), np.ones(t), plugin_lambdas(y, alpha, n_plan)
    )


def wor_data(
    y: np.ndarray, population_size: int, alpha: float, n_plan: int, y_min: float = -1.0
) -> BettingData:
    """Sampling without replacement from a finite population of size N (Waudby-Smith & Ramdas)."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    n_pop = population_size
    if t > n_pop:
        raise ValueError("more WoR draws than population members")
    s_idx = np.arange(1, t + 1, dtype=float)
    prev_sum = np.concatenate([[0.0], np.cumsum(y)[:-1]])
    denom = n_pop - s_idx + 1.0
    return BettingData(
        y, np.full(t, y_min), -prev_sum / denom, n_pop / denom, plugin_lambdas(y, alpha, n_plan)
    )


def negate(data: BettingData, upper: np.ndarray) -> BettingData:
    """Data for the lower bound of -Y (used to obtain upper bounds of Y)."""
    return BettingData(-data.y, -upper, -data.a, data.b, data.lam)


@dataclass(frozen=True)
class CSResult:
    lower: float
    upper: float
    p_regression: float  # anytime p-value for H0: mean <= margin
    n: int


def confidence_bounds(
    y: np.ndarray,
    alpha: float,
    n_plan: int,
    *,
    population_size: int | None = None,
    margin: float = 0.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
) -> CSResult:
    """One-sided (1-alpha) lower and upper anytime-valid bounds for the mean of ``y``."""
    y = np.asarray(y, dtype=float)
    if population_size is None:
        lo_data = iid_data(y, alpha, n_plan, y_min)
        up_data = iid_data(-y, alpha, n_plan, -y_max)
    else:
        lo_data = wor_data(y, population_size, alpha, n_plan, y_min)
        up_data = wor_data(-y, population_size, alpha, n_plan, -y_max)
    lb = lower_bound(lo_data, alpha, y_min, y_max)
    ub = -lower_bound(up_data, alpha, -y_max, -y_min)
    return CSResult(lb, ub, anytime_p_value(lo_data, margin), len(y))


# ----------------------------------------------------------------------------------------------
# Hoeffding references
# ----------------------------------------------------------------------------------------------


def hoeffding_radius(n: int, alpha: float, value_range: float = 2.0) -> float:
    """One-sided Hoeffding radius. Valid for i.i.d. and for SRS without replacement."""
    if n == 0:
        return math.inf
    return value_range * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))


def hoeffding_ipw_radius(t: int, alpha: float, epsilon: float) -> float:
    """Spec §4 conservative anytime radius for IPW draws in [-1/eps, 1/eps] (union over t)."""
    if t == 0:
        return math.inf
    alpha_t = 6.0 * alpha / (math.pi**2 * t**2)
    return (1.0 / epsilon) * math.sqrt(2.0 * math.log(2.0 / alpha_t) / t)
