"""Confirmation tests: predeclared looks, alpha allocation and three-way decisions (protocol §2-5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evaldelta.schemas import ConfirmPlan, Decision, TestEvidence
from evaldelta.statistics import sequential as seq
from evaldelta.statistics.multiplicity import bonferroni_alpha
from evaldelta.statistics.paired_tests import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    discordance,
    mcnemar_exact_one_sided,
)

FIXED_SAMPLE_METHODS = {"mcnemar_exact", "hoeffding_wor"}


def resolve_method(plan: ConfirmPlan, binary: bool, margin: float) -> str:
    if plan.method != "auto":
        if plan.method == "mcnemar_exact" and not binary:
            raise ValueError("mcnemar_exact requires 0/1 losses")
        return plan.method
    return "mcnemar_exact" if (binary and margin == 0.0) else "betting_wor"


def look_schedule(n_planned: int, looks: int) -> list[int]:
    """Predeclared cumulative sample counts ceil(n*k/K), deduplicated, strictly increasing."""
    if n_planned <= 0:
        return []
    sched = sorted({max(1, math.ceil(n_planned * k / looks)) for k in range(1, looks + 1)})
    return sched


@dataclass
class ConfirmationTest:
    """One confirmation hypothesis (global or one slice) evaluated at predeclared looks.

    Parameters
    ----------
    scope: "global" or "slice:<name>"
    margin: regression margin delta (H0: Delta <= delta)
    ni_margin: non-inferiority margin (None disables the NI decision)
    alpha: family-level alpha available to this hypothesis
    n_planned: planned confirmation sample size (fixed before any call)
    population_size: size of the finite population sampled without replacement
    """

    scope: str
    estimand: str
    method: str
    margin: float
    ni_margin: float | None
    alpha: float
    n_planned: int
    looks: int
    population_size: int

    def __post_init__(self) -> None:
        self.schedule = look_schedule(self.n_planned, self.looks)
        self.k_looks = max(1, len(self.schedule))

    @property
    def per_look_alpha(self) -> float:
        if self.method in FIXED_SAMPLE_METHODS:
            return bonferroni_alpha(self.alpha, self.k_looks)
        return self.alpha  # anytime-valid (Ville): no split across looks

    def evaluate(
        self, old_loss: np.ndarray, new_loss: np.ndarray, look_index: int, final: bool
    ) -> TestEvidence:
        """Evaluate at look ``look_index`` (0-based) on all confirmation data so far."""
        old = np.asarray(old_loss, dtype=float)
        new = np.asarray(new_loss, dtype=float)
        d = new - old
        n = len(d)
        a_look = self.per_look_alpha
        common = dict(
            scope=self.scope,
            estimand=self.estimand,
            method=self.method,
            n=n,
            alpha_allocated=self.alpha,
            looks_used=look_index + 1,
            looks_planned=self.k_looks,
            population_size=self.population_size,
        )
        if n == 0:
            return TestEvidence(
                **common,
                decision=Decision.INCONCLUSIVE,
                reason="no_confirmation_data",
                interval_level=1 - 2 * a_look,
            )
        effect = float(d.mean())
        n_down = n_up = None
        if np.isin(old, (0.0, 1.0)).all() and np.isin(new, (0.0, 1.0)).all():
            disc = discordance(old, new)
            n_down, n_up = disc.n_down, disc.n_up

        # --- interval for Delta (betting WoR is always computed; it is valid on its own) ---
        cs_alpha = a_look if self.method != "betting_wor" else self.alpha
        cs = seq.confidence_bounds(
            d,
            cs_alpha,
            self.n_planned,
            population_size=self.population_size,
            margin=self.margin,
        )
        lower, upper = cs.lower, cs.upper
        interval_method = "betting_wor_cs"
        p_value: float | None = cs.p_regression

        if self.method == "mcnemar_exact":
            assert n_down is not None and n_up is not None
            p_value = mcnemar_exact_one_sided(n_down, n_up)
            regression = p_value <= a_look
        elif self.method == "hoeffding_wor":
            r = seq.hoeffding_radius(n, a_look)
            lower, upper = max(-1.0, effect - r), min(1.0, effect + r)
            interval_method = "hoeffding_wor"
            p_value = None
            regression = lower > self.margin
        else:  # betting_wor
            regression = lower > self.margin

        noninferior = self.ni_margin is not None and upper < self.ni_margin
        if regression:
            decision, reason = Decision.CONFIRMED_REGRESSION, "lower_bound_exceeds_margin"
            if self.method == "mcnemar_exact":
                reason = "mcnemar_rejects_H0"
        elif noninferior:
            decision, reason = Decision.EVIDENCE_OF_NONINFERIORITY, "upper_bound_below_ni_margin"
        else:
            decision = Decision.INCONCLUSIVE
            reason = "boundaries_not_crossed" if final else "continue"
        ev = TestEvidence(
            **common,
            decision=decision,
            reason=reason,
            n_down=n_down,
            n_up=n_up,
            effect=effect,
            lower=lower,
            upper=upper,
            interval_method=interval_method,
            interval_level=1 - 2 * (cs_alpha if interval_method == "betting_wor_cs" else a_look),
            p_value=p_value,
        )
        return ev


def slice_p_value(ev: TestEvidence, k_looks: int) -> float:
    """Per-slice p-value for Holm: Bonferroni over looks for fixed-sample, anytime otherwise."""
    if ev.p_value is None:
        return 0.0 if ev.decision == Decision.CONFIRMED_REGRESSION else 1.0
    if ev.method in FIXED_SAMPLE_METHODS:
        return min(1.0, ev.p_value * k_looks)
    return ev.p_value


def mcnemar_compatible_interval(n_down: int, n_up: int, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson bounds for pi = P(down | discordant); McNemar rejects iff lower > 1/2."""
    k = n_down + n_up
    return clopper_pearson_lower(n_down, k, alpha), clopper_pearson_upper(n_down, k, alpha)
