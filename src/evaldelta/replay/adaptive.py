"""OURS-2: propensity-weighted adaptive global inference (protocol §7, research extension).

At each step the acquisition rule sees only already-revealed outcomes. It outputs a predictor
g_t(i) of d_i and sampling probabilities q_t(i) over unrevealed eligible items, with
q_t(i) >= eps/|R_t|. Revealed items are known exactly (g = d, q = 0). Every draw is a new paid
evaluation. The pseudo-outcome

    Y_t = (1/N) sum_i g_t(i) + (d_I - g_t(I)) / (N q_t(I))

is conditionally unbiased for the pool Delta, with a predictable lower bound. The betting CS in
``statistics.sequential`` then gives anytime-valid bounds under arbitrary predictable q_t and g_t.

Acquisition rules
-----------------
* ``uniform``:        q uniform over unrevealed items, g = 0 (propensity baseline)
* ``active_testing``: q ∝ sqrt(predicted candidate loss), g = 0 (Kossen-style, single-model risk;
  the adapted B6 baseline)
* ``paired_shift``:   q ∝ sqrt(Var[d_i]) / sqrt(cost) with g = E[d_i] from PairedShift's
  p_down / p_up (paired Neyman allocation + control variate)
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from evaldelta.data.io import COST_COL, ID_COL, OLD_LOSS_COL, ItemTable
from evaldelta.policies.base import PolicyView
from evaldelta.policies.paired_shift import PairedShift
from evaldelta.providers.base import CandidateProvider, evaluate_paid
from evaldelta.replay.budget import BudgetMeter
from evaldelta.schemas import EXIT_CODES, Decision
from evaldelta.statistics import sequential as seq

ACQUISITIONS = ("uniform", "active_testing", "paired_shift")


@dataclass(frozen=True)
class AdaptiveGlobalPlan:
    alpha: float = 0.05
    regression_margin: float = 0.0
    noninferiority_margin: float | None = 0.02
    epsilon: float = 0.2  # uniform mixing weight: q >= eps/|R|
    acquisition: str = "paired_shift"
    batch_size: int = 25  # g/q recomputed at fixed increments
    check_every: int = 1  # anytime-valid: may check after every draw
    variance_floor: float = 0.01
    use_control_variate: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or not 0 < self.epsilon <= 1:
            raise ValueError("alpha must be in (0, 1) and epsilon in (0, 1]")
        if self.batch_size <= 0 or self.check_every <= 0:
            raise ValueError("batch_size and check_every must be positive")
        if not math.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        if not math.isfinite(self.regression_margin) or self.regression_margin < 0:
            raise ValueError("regression_margin must be finite and non-negative")
        if self.noninferiority_margin is not None and (
            not math.isfinite(self.noninferiority_margin) or self.noninferiority_margin <= 0
        ):
            raise ValueError("noninferiority_margin must be finite and positive")


@dataclass
class AdaptiveResult:
    decision: Decision
    n: int
    lower: float
    upper: float
    effect: float  # running mean of pseudo-outcomes
    paid_cost: float
    paid_calls: int
    stop_reason: str
    events: pd.DataFrame
    wall_time_s: float

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.decision]


class AdaptiveGlobalSession:
    """Global-only adaptive comparison over an eligible pool (default: the whole pool)."""

    def __init__(
        self,
        items: ItemTable,
        provider: CandidateProvider,
        *,
        policy: PairedShift | None = None,
        eligible: list[str] | None = None,
    ) -> None:
        self.items = items
        self._provider = provider
        self.policy = policy or PairedShift()
        self.eligible = items.ids if eligible is None else eligible
        if not self.eligible or len(set(self.eligible)) != len(self.eligible):
            raise ValueError("eligible items must be nonempty and unique")

    def _predict(self, view: PolicyView, plan: AdaptiveGlobalPlan) -> tuple[np.ndarray, np.ndarray]:
        """Return (g, spread) for every eligible item. The spread proxy drives the propensities."""
        probs = self.policy.probabilities(view).set_index(ID_COL).loc[self.eligible]
        pd_, pu = probs["p_down"].to_numpy(), probs["p_up"].to_numpy()
        g = pd_ - pu
        var = pd_ + pu - g**2
        if plan.acquisition == "paired_shift":
            spread = np.sqrt(np.maximum(var, 0) + plan.variance_floor)
        elif plan.acquisition == "active_testing":
            old = probs[OLD_LOSS_COL].to_numpy()
            p_new_wrong = np.where(old == 0, pd_, 1 - pu)
            spread = np.sqrt(p_new_wrong + plan.variance_floor)
        else:
            spread = np.ones(len(g))
        if not plan.use_control_variate or plan.acquisition != "paired_shift":
            g = np.zeros(len(g))
        return g, spread

    def run(
        self,
        plan: AdaptiveGlobalPlan,
        budget_calls: int,
        seed: int = 0,
        max_cost: float | None = None,
    ) -> AdaptiveResult:
        if plan.acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {plan.acquisition!r}")
        t0 = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0xADA9]))
        public = self.items.subset(self.eligible)
        ids = np.array(self.eligible)
        n_pool = len(ids)
        old = public[OLD_LOSS_COL].to_numpy(dtype=float)
        cost = public[COST_COL].to_numpy(dtype=float)
        lo_d, hi_d = -old, 1.0 - old
        known = np.zeros(n_pool, dtype=bool)
        d_known = np.zeros(n_pool)
        revealed: dict[str, float] = {}
        ys: list[float] = []
        lows: list[float] = []
        ups: list[float] = []
        events: list[dict[str, Any]] = []
        g = np.zeros(n_pool)
        spread = np.ones(n_pool)
        meter = BudgetMeter(budget_calls, max_cost)
        decision, reason = Decision.INCONCLUSIVE, "budget_exhausted"
        lb, ub = -1.0, 1.0
        log_thr = math.log(1 / plan.alpha)
        for t in range(budget_calls):
            if known.all():
                reason = "pool_exhausted"
                break
            if t % plan.batch_size == 0:
                view = PolicyView.build(public, revealed, budget_calls - t)
                g, spread = self._predict(view, plan)
            unk = ~known
            cost_w = spread / np.sqrt(np.maximum(cost, 1e-3))
            q = np.zeros(n_pool)
            q[unk] = cost_w[unk] / cost_w[unk].sum()
            q[unk] = (1 - plan.epsilon) * q[unk] + plan.epsilon / unk.sum()
            g_eff = np.where(known, d_known, g)
            gbar = g_eff.mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_terms = np.where(unk, (lo_d - g_eff) / (n_pool * q), np.inf)
                hi_terms = np.where(unk, (hi_d - g_eff) / (n_pool * q), -np.inf)
            L_t = gbar + lo_terms.min()
            U_t = gbar + hi_terms.max()
            i = int(rng.choice(n_pool, p=q))
            sid = str(ids[i])

            def reserve(unit_cost: float = float(cost[i])) -> bool:
                if not meter.can_afford("adaptive_global", 1, unit_cost):
                    return False
                meter.charge("adaptive_global", 1, unit_cost)
                return True

            row = cast(dict[str, Any], public.iloc[i].to_dict())
            outcome = evaluate_paid(self._provider, row, reserve)
            if outcome.attempts == 0:
                reason = (
                    "budget_exhausted" if meter.calls >= budget_calls else "cost_budget_exhausted"
                )
                break
            if outcome.loss is None:
                # A failed call is treated as an evaluation error: skipping it would bias Y.
                decision, reason = Decision.EVALUATION_ERROR, f"candidate call failed for {sid}"
                events.append(
                    {
                        "sample_id": sid,
                        "phase": "adaptive_global",
                        "draw_index": t + 1,
                        "selection_probability": float(q[i]),
                        "old_loss": old[i],
                        "new_loss": None,
                        "delta": None,
                        "pseudo_outcome": None,
                        "candidate_calls": outcome.attempts,
                        "error": outcome.error,
                        "cost_actual": float(cost[i]) * outcome.attempts,
                    }
                )
                break
            d_i = float(outcome.loss) - old[i]
            y = gbar + (d_i - g_eff[i]) / (n_pool * q[i])
            ys.append(y)
            lows.append(L_t)
            ups.append(U_t)
            known[i] = True
            d_known[i] = d_i
            revealed[sid] = float(outcome.loss)
            events.append(
                {
                    "sample_id": sid,
                    "phase": "adaptive_global",
                    "draw_index": t + 1,
                    "selection_probability": float(q[i]),
                    "old_loss": old[i],
                    "new_loss": float(outcome.loss),
                    "delta": d_i,
                    "pseudo_outcome": y,
                    "cost_actual": float(cost[i]) * outcome.attempts,
                    "candidate_calls": outcome.attempts,
                    "error": None,
                }
            )
            if (t + 1) % plan.check_every == 0 or t + 1 == budget_calls:
                yarr = np.array(ys)
                lam = seq.plugin_lambdas(yarr, plan.alpha, budget_calls)
                n = len(yarr)
                lo_data = seq.BettingData(yarr, np.array(lows), np.zeros(n), np.ones(n), lam)
                up_data = seq.BettingData(
                    -yarr,
                    -np.array(ups),
                    np.zeros(n),
                    np.ones(n),
                    seq.plugin_lambdas(-yarr, plan.alpha, budget_calls),
                )
                reg = (
                    float(np.max(seq.log_capital_path(lo_data, plan.regression_margin))) >= log_thr
                )
                ni = (
                    plan.noninferiority_margin is not None
                    and float(np.max(seq.log_capital_path(up_data, -plan.noninferiority_margin)))
                    >= log_thr
                )
                if reg or ni:
                    lb = seq.lower_bound(lo_data, plan.alpha, -1.0, 1.0)
                    ub = -seq.lower_bound(up_data, plan.alpha, -1.0, 1.0)
                    if reg:
                        decision, reason = (
                            Decision.CONFIRMED_REGRESSION,
                            "lower_bound_exceeds_margin",
                        )
                    else:
                        decision, reason = (
                            Decision.EVIDENCE_OF_NONINFERIORITY,
                            "upper_bound_below_ni_margin",
                        )
                    break
        if ys and decision == Decision.INCONCLUSIVE:
            yarr = np.array(ys)
            n = len(yarr)
            lo_data = seq.BettingData(
                yarr,
                np.array(lows),
                np.zeros(n),
                np.ones(n),
                seq.plugin_lambdas(yarr, plan.alpha, budget_calls),
            )
            up_data = seq.BettingData(
                -yarr,
                -np.array(ups),
                np.zeros(n),
                np.ones(n),
                seq.plugin_lambdas(-yarr, plan.alpha, budget_calls),
            )
            lb = seq.lower_bound(lo_data, plan.alpha, -1.0, 1.0)
            ub = -seq.lower_bound(up_data, plan.alpha, -1.0, 1.0)
        return AdaptiveResult(
            decision=decision,
            n=len(ys),
            lower=lb,
            upper=ub,
            effect=float(np.mean(ys)) if ys else 0.0,
            paid_cost=meter.cost,
            paid_calls=meter.calls,
            stop_reason=reason,
            events=pd.DataFrame(events),
            wall_time_s=time.perf_counter() - t0,
        )


def run_adaptive_trial(
    ep: Any,
    plan: AdaptiveGlobalPlan,
    budget: int,
    seed: int,
    policy_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for experiments: returns a flat record with ground truth."""
    items = ep.item_table()
    sess = AdaptiveGlobalSession(items, ep.oracle(), policy=PairedShift(**(policy_kwargs or {})))
    res = sess.run(plan, budget, seed)
    truth = ep.truth()["delta"]
    return {
        "episode_id": ep.episode_id,
        "acquisition": plan.acquisition,
        "budget": budget,
        "true_delta": truth,
        "decision": res.decision.value,
        "n": res.n,
        "lower": res.lower,
        "upper": res.upper,
        "effect": res.effect,
        "covered": res.lower <= truth <= res.upper,
        "false_regression": res.decision == Decision.CONFIRMED_REGRESSION
        and truth <= plan.regression_margin + 1e-12,
        "false_noninferiority": res.decision == Decision.EVIDENCE_OF_NONINFERIORITY
        and plan.noninferiority_margin is not None
        and truth >= plan.noninferiority_margin - 1e-12,
        "stop_reason": res.stop_reason,
        "wall_time_s": res.wall_time_s,
    }
