"""EvalSession: budgeted paired comparison with sealed discovery/confirmation (protocol §3-5, §8)."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from evaldelta.data.io import COST_COL, OLD_LOSS_COL, SLICE_COL, ItemTable
from evaldelta.data.splits import Partition, make_partition
from evaldelta.policies.base import Policy, PolicyView, suspicious_slices
from evaldelta.providers.base import CandidateProvider, EvalOutcome, evaluate_paid
from evaldelta.replay.budget import BudgetMeter
from evaldelta.schemas import (
    EXIT_CODES,
    Decision,
    PhaseSpend,
    RunConfig,
    RunReport,
    TestEvidence,
)
from evaldelta.statistics.confirmation import (
    FIXED_SAMPLE_METHODS,
    ConfirmationTest,
    resolve_method,
    slice_p_value,
)
from evaldelta.statistics.multiplicity import holm

MAX_FAILURE_RATE = 0.10

LIMITATIONS = [
    "Absence of a confirmed regression is not evidence of safety; only "
    "'evidence_of_noninferiority' against the declared margin supports approval.",
    "Global inference concerns the declared item pool / target population; it does not cover "
    "inputs outside that population.",
    "Slices with too few fresh slice-confirmation items cannot be confirmed at any budget.",
    "Exploratory (discovery) findings are not confirmed results.",
]


class ProtocolViolation(RuntimeError):
    """A policy or provider broke the access/budget contract."""


@dataclass
class RunResult:
    report: RunReport
    events: pd.DataFrame
    extra_files: dict[str, str] = field(default_factory=dict)

    @property
    def decision(self) -> Decision:
        return self.report.decision

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(self.report.model_dump_json(indent=2))
        self.events.to_json(out / "events.jsonl", orient="records", lines=True)
        from evaldelta.reporting.markdown_report import render_markdown

        (out / "report.md").write_text(render_markdown(self.report))
        return out

    @staticmethod
    def load(out_dir: str | Path) -> RunResult:
        out = Path(out_dir)
        report = RunReport.model_validate_json((out / "report.json").read_text())
        events = pd.read_json(out / "events.jsonl", orient="records", lines=True, dtype=False)
        return RunResult(report, events)


class EvalSession:
    """Runs one budgeted paired comparison between a cached old system and a candidate."""

    def __init__(
        self,
        items: ItemTable,
        provider: CandidateProvider,
        *,
        history: pd.DataFrame | None = None,
        episode_meta: Mapping[str, Any] | None = None,
        log_outputs: bool = False,
    ) -> None:
        self.items = items
        self._provider = provider
        self.history = history
        self.meta = dict(episode_meta or {})
        self.log_outputs = log_outputs

    @classmethod
    def from_files(
        cls,
        items_path: str | Path,
        provider: CandidateProvider,
        history_path: str | Path | None = None,
        **kw: Any,
    ) -> EvalSession:
        items = ItemTable.from_path(items_path)
        history = None if history_path is None else pd.read_parquet(history_path)
        return cls(items, provider, history=history, **kw)

    # ------------------------------------------------------------------------------------------
    def compare(self, config: RunConfig, policy: Policy) -> RunResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0xD15C]))
        part = make_partition(self.items.ids, config.split)
        budget = config.budget
        caps = budget.phase_calls()
        meter = BudgetMeter(budget.max_candidate_calls, budget.max_cost, caps)
        run = _Run(self, config, part, meter, policy)

        run.discovery(rng)
        slices, exploratory = run.freeze_slices()
        global_ev = run.global_confirmation()
        slice_evs = run.slice_confirmation(slices)

        decision = _overall(global_ev, slice_evs, run.evaluation_error)
        spend = {
            name: PhaseSpend(
                planned_calls=caps.get(name, 0),
                paid_calls=int(led["calls"]),
                paid_cost=float(led["cost"]),
                failed_attempts=int(led["failed"]),
            )
            for name, led in meter.summary().items()
        }
        report = RunReport(
            run_id=config.run_id,
            episode_id=self.meta.get("episode_id"),
            old_version_id=self.meta.get("old_version_id"),
            new_version_id=self.meta.get("new_version_id"),
            config=config.model_dump(mode="json"),
            config_hash=config.config_hash(),
            split_hash=part.split_hash,
            pool_size=len(self.items),
            partition_sizes=part.sizes(),
            decision=decision,
            exit_code=EXIT_CODES[decision],
            global_result=global_ev,
            slice_results=slice_evs,
            exploratory=exploratory,
            spend=spend,
            total_paid_calls=meter.calls,
            total_paid_cost=meter.cost,
            cost_unit=budget.cost_unit,
            selected_ids=run.selected_ids(),
            limitations=LIMITATIONS + run.extra_limitations,
            wall_time_s=time.perf_counter() - t0,
            extra={"policy": getattr(policy, "name", type(policy).__name__)},
        )
        events = pd.DataFrame(run.events)
        run.audit()
        return RunResult(report, events)


def _overall(
    global_ev: TestEvidence | None, slice_evs: Sequence[TestEvidence], error: str | None
) -> Decision:
    if error:
        return Decision.EVALUATION_ERROR
    if global_ev is not None and global_ev.decision == Decision.CONFIRMED_REGRESSION:
        return Decision.CONFIRMED_REGRESSION
    if any(e.decision == Decision.CONFIRMED_REGRESSION for e in slice_evs):
        return Decision.CONFIRMED_REGRESSION
    if global_ev is not None and global_ev.decision == Decision.EVIDENCE_OF_NONINFERIORITY:
        return Decision.EVIDENCE_OF_NONINFERIORITY
    return Decision.INCONCLUSIVE


class _Run:
    """Mutable state of one comparison. Not exposed to policies."""

    def __init__(
        self,
        session: EvalSession,
        config: RunConfig,
        part: Partition,
        meter: BudgetMeter,
        policy: Policy,
    ) -> None:
        self.s = session
        self.cfg = config
        self.part = part
        self.meter = meter
        self.policy = policy
        self.events: list[dict[str, Any]] = []
        self.revealed: dict[str, float] = {}  # all purchased candidate losses (any phase)
        self.phase_of: dict[str, str] = {}
        self.draw = 0
        self.evaluation_error: str | None = None
        self.extra_limitations: list[str] = []
        self._failures = 0
        self._attempted = 0
        self._cost = self.s.items.column(COST_COL)
        self._pos = {sid: i for i, sid in enumerate(self.s.items.ids)}
        self._old = self.s.items.column(OLD_LOSS_COL)
        self._reveal_start = len(getattr(session._provider, "revealed_ids", ()))

    # -- paying for candidate evaluations ------------------------------------------------------
    def pay(
        self, phase: str, ids: Sequence[str], *, max_attempts: int | None = None
    ) -> list[EvalOutcome]:
        for sid in ids:
            if sid in self.phase_of:
                raise ProtocolViolation(f"{sid} already evaluated in phase {self.phase_of[sid]}")
        outcomes: list[EvalOutcome] = []
        start_calls = self.meter.calls
        rows = cast(list[dict[str, Any]], self.s.items.subset(ids).to_dict(orient="records"))
        for sid, row in zip(ids, rows, strict=True):
            unit_cost = (
                1.0
                if self.cfg.budget.cost_unit == "candidate_calls"
                else float(self._cost[self._pos[sid]])
            )

            def reserve(cost: float = unit_cost) -> bool:
                if max_attempts is not None and self.meter.calls - start_calls >= max_attempts:
                    return False
                if not self.meter.can_afford(phase, 1, cost):
                    return False
                self.meter.charge(phase, 1, cost)
                return True

            o = evaluate_paid(self.s._provider, row, reserve)
            if o.attempts == 0:
                break
            outcomes.append(o)
            self.draw += 1
            self._attempted += 1
            self.phase_of[sid] = phase
            old = float(self._old[self._pos[sid]])
            self.meter.phase(phase).failed += o.attempts - int(o.loss is not None)
            if o.loss is None:
                self._failures += 1
                # Missing confirmation outcomes may depend on loss. Never skip them and
                # certify performance on the surviving, potentially biased subset.
                if phase != "discovery":
                    self.evaluation_error = (
                        "candidate confirmation failed; missing outcomes may be informative"
                    )
            else:
                self.revealed[sid] = float(o.loss)
            self.events.append(
                {
                    "run_id": self.cfg.run_id,
                    "sample_id": sid,
                    "phase": phase,
                    "policy": getattr(self.policy, "name", "?"),
                    "draw_index": self.draw,
                    "selection_probability": None,
                    "new_output": None if not self.s.log_outputs else _jsonable(o.output),
                    "old_loss": old,
                    "new_loss": o.loss,
                    "delta": None if o.loss is None else float(o.loss) - old,
                    "candidate_calls": o.attempts,
                    "cost_actual": unit_cost * o.attempts,
                    "attempts": o.attempts,
                    "error": o.error,
                    "seed": self.cfg.seed,
                }
            )
            if self._attempted >= 20 and self._failures / self._attempted > MAX_FAILURE_RATE:
                self.evaluation_error = (
                    f"candidate evaluation failure rate {self._failures}/{self._attempted} exceeds "
                    f"{MAX_FAILURE_RATE:.0%}; failures may be informative"
                )
            if self.evaluation_error:
                break
        return outcomes

    # -- discovery -----------------------------------------------------------------------------
    def discovery(self, rng: np.random.Generator) -> None:
        d_ids = list(self.part.discovery)
        if not d_ids:
            return
        d_set = set(d_ids)
        public = self.s.items.subset(d_ids)
        batch = self.cfg.policy.batch_size
        disc_revealed: dict[str, float] = {}
        while self.evaluation_error is None:
            remaining = self.meter.remaining_calls("discovery")
            if remaining <= 0 or len(disc_revealed) + self._disc_failed() >= len(d_ids):
                break
            k = min(batch, remaining)
            view = PolicyView.build(
                public,
                disc_revealed,
                remaining,
                history=self.s.history,
                meta={"batch_size": batch, "seed": self.cfg.seed},
            )
            chosen = list(self.policy.select(view, k, rng))
            if not chosen:
                break
            if len(chosen) > k or len(set(chosen)) != len(chosen):
                raise ProtocolViolation("policy returned too many or duplicate IDs")
            bad = [c for c in chosen if c not in d_set or c in self.phase_of]
            if bad:
                raise ProtocolViolation(f"policy selected ineligible IDs: {bad[:3]}")
            outcomes = self.pay("discovery", chosen)
            if not outcomes:
                break
            for o in outcomes:
                if o.loss is not None:
                    disc_revealed[o.sample_id] = float(o.loss)

    def _disc_failed(self) -> int:
        return sum(
            1 for sid, ph in self.phase_of.items() if ph == "discovery" and sid not in self.revealed
        )

    def freeze_slices(self) -> tuple[list[str], dict[str, Any]]:
        plan = self.cfg.plan
        disc = {k: v for k, v in self.revealed.items() if self.phase_of[k] == "discovery"}
        public = self.s.items.subset(list(self.part.discovery)) if self.part.discovery else None
        found: list[str] = []
        exploratory: dict[str, Any] = {"label": "EXPLORATORY - not confirmed"}
        if public is not None and disc:
            view = PolicyView.build(public, disc, 0, history=self.s.history)
            ranker = getattr(self.policy, "suspicious_slices", None)
            found = (
                list(ranker(view, plan.max_slices))
                if callable(ranker)
                else suspicious_slices(view, plan.max_slices)
            )
            old = np.array([self._old[self._pos[i]] for i in disc])
            new = np.array(list(disc.values()))
            exploratory.update(
                {
                    "n_discovery_evaluated": len(disc),
                    "n_down_found": int(np.sum((old == 0) & (new == 1))),
                    "n_up_found": int(np.sum((old == 1) & (new == 0))),
                    "sum_delta_selected": float(np.sum(new - old)),
                    "note": "discovery items are adaptively selected; their mean is NOT an "
                    "estimate of the pool difference",
                }
            )
        slices: list[str] = []
        for s in list(plan.predeclared_slices) + found:
            if s not in slices:
                slices.append(s)
        slices = slices[: max(plan.max_slices, len(plan.predeclared_slices))]
        exploratory["suspicious_slices"] = found
        exploratory["slices_frozen_for_confirmation"] = slices
        return slices, exploratory

    # -- confirmation --------------------------------------------------------------------------
    def _run_test(
        self, test: ConfirmationTest, order: Sequence[str], phase: str, call_cap: int
    ) -> TestEvidence:
        pos = 0
        spent = 0
        old_vals: list[float] = []
        new_vals: list[float] = []
        ev: TestEvidence | None = None
        if not test.schedule:
            return test.evaluate(np.array([]), np.array([]), 0, final=True)
        for li, target in enumerate(test.schedule):
            exhausted = False
            while len(new_vals) < target:
                if pos >= len(order) or spent >= call_cap or self.evaluation_error:
                    exhausted = True
                    break
                if self.meter.remaining_calls(phase) <= 0:
                    exhausted = True
                    break
                sid = order[pos]
                pos += 1
                if not self._affordable(phase, sid):
                    exhausted = True
                    break
                (o,) = self.pay(phase, [sid], max_attempts=call_cap - spent)
                spent += o.attempts
                if o.loss is not None:
                    old_vals.append(float(self._old[self._pos[sid]]))
                    new_vals.append(float(o.loss))
            final = exhausted or li == len(test.schedule) - 1
            if self.evaluation_error or (
                exhausted and len(new_vals) < target and test.method in FIXED_SAMPLE_METHODS
            ):
                # Never turn a cost/retry-dependent stopping time into a fixed-sample look.
                # A previously completed scheduled look remains valid.
                if ev is not None and not self.evaluation_error:
                    return ev.model_copy(
                        update={"reason": "budget_exhausted_before_next_checkpoint"}
                    )
                return TestEvidence(
                    scope=test.scope,
                    estimand=test.estimand,
                    method=test.method,
                    decision=Decision.INCONCLUSIVE,
                    reason="confirmation_evaluation_error"
                    if self.evaluation_error
                    else "budget_exhausted_before_checkpoint",
                    n=len(new_vals),
                    alpha_allocated=test.alpha,
                    looks_used=0,
                    looks_planned=len(test.schedule),
                    population_size=test.population_size,
                )
            ev = test.evaluate(np.array(old_vals), np.array(new_vals), li, final=final)
            if ev.decision != Decision.INCONCLUSIVE or final:
                break
        assert ev is not None
        return ev

    def _affordable(self, phase: str, sid: str) -> bool:
        cost = (
            1.0
            if self.cfg.budget.cost_unit == "candidate_calls"
            else float(self._cost[self._pos[sid]])
        )
        return self.meter.can_afford(phase, 1, cost)

    def global_confirmation(self) -> TestEvidence | None:
        plan = self.cfg.plan
        n_cap = self.cfg.budget.phase_calls()["global_confirmation"]
        order = list(self.part.global_confirm)
        n_planned = min(n_cap, len(order))
        if n_planned == 0:
            self.extra_limitations.append("No global confirmation budget: no global decision.")
            return None
        method = resolve_method(plan, self.s.items.is_binary, plan.regression_margin)
        test = ConfirmationTest(
            scope="global",
            estimand="pool mean paired loss difference Delta (new - old)",
            method=method,
            margin=plan.regression_margin,
            ni_margin=plan.noninferiority_margin,
            alpha=plan.global_alpha(),
            n_planned=n_planned,
            looks=plan.looks,
            population_size=len(self.s.items),
        )
        return self._run_test(test, order, "global_confirmation", n_planned)

    def slice_confirmation(self, slices: Sequence[str]) -> list[TestEvidence]:
        plan = self.cfg.plan
        if not slices:
            return []
        g = len(slices)
        cap_total = self.cfg.budget.phase_calls()["slice_confirmation"]
        per_slice = cap_total // g if g else 0
        slice_col = {sid: self.s.items.value(sid, SLICE_COL) for sid in self.part.slice_confirm}
        alpha_each = plan.slice_alpha() / g  # Bonferroni threshold (strictest Holm step)
        evs: list[TestEvidence] = []
        tests: list[ConfirmationTest] = []
        for name in slices:
            members = [sid for sid in self.part.slice_confirm if slice_col[sid] == name]
            n_planned = min(per_slice, len(members))
            method = resolve_method(plan, self.s.items.is_binary, plan.slice_margin)
            test = ConfirmationTest(
                scope=f"slice:{name}",
                estimand=f"mean paired difference over slice '{name}' members of the "
                "slice-confirmation partition",
                method=method,
                margin=plan.slice_margin,
                ni_margin=None,
                alpha=alpha_each,
                n_planned=max(n_planned, 0),
                looks=plan.looks,
                population_size=max(len(members), 1),
            )
            tests.append(test)
            if len(members) < plan.min_slice_confirm or n_planned < plan.min_slice_confirm:
                evs.append(
                    TestEvidence(
                        scope=test.scope,
                        estimand=test.estimand,
                        method=method,
                        decision=Decision.INCONCLUSIVE,
                        reason=(
                            f"insufficient_fresh_examples ({len(members)} in slice-confirmation "
                            f"pool, {n_planned} affordable; need {plan.min_slice_confirm})"
                        ),
                        n=0,
                        alpha_allocated=alpha_each,
                        looks_planned=len(test.schedule),
                        population_size=len(members),
                    )
                )
                continue
            evs.append(self._run_test(test, members, "slice_confirmation", n_planned))
        # Holm across slices on final p-values (fixed-sample p-values Bonferroni'd over looks).
        pvals = [
            slice_p_value(e, t.k_looks) if e.n > 0 else 1.0 for e, t in zip(evs, tests, strict=True)
        ]
        rejected = holm(pvals, plan.slice_alpha())
        out: list[TestEvidence] = []
        for e, rej, p in zip(evs, rejected, pvals, strict=True):
            upd: dict[str, Any] = {"alpha_allocated": plan.slice_alpha()}
            if rej and e.decision != Decision.CONFIRMED_REGRESSION:
                upd.update(decision=Decision.CONFIRMED_REGRESSION, reason="holm_rejects_H0")
            elif not rej and e.decision == Decision.CONFIRMED_REGRESSION:
                upd.update(decision=Decision.INCONCLUSIVE, reason="not_rejected_after_holm")
            upd["p_value"] = p if e.n > 0 else e.p_value
            out.append(e.model_copy(update=upd))
        return out

    # -- bookkeeping ---------------------------------------------------------------------------
    def selected_ids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for ev in self.events:
            out.setdefault(ev["phase"], []).append(ev["sample_id"])
        return out

    def audit(self) -> None:
        """Post-run integrity checks (budget, uniqueness, oracle reveal log if available)."""
        if self.meter.calls > self.cfg.budget.max_candidate_calls:
            raise ProtocolViolation("budget exceeded")
        if sum(e["candidate_calls"] for e in self.events) != self.meter.calls:
            raise ProtocolViolation("event attempt count does not match paid calls")
        ids = [e["sample_id"] for e in self.events]
        if len(ids) != len(set(ids)):
            raise ProtocolViolation("an item was evaluated twice")
        membership = self.part.membership()
        for e in self.events:
            want = {
                "discovery": "discovery",
                "global_confirmation": "global_confirm",
                "slice_confirmation": "slice_confirm",
            }[e["phase"]]
            if membership[e["sample_id"]] != want:
                raise ProtocolViolation(f"{e['sample_id']} evaluated outside its partition")
        revealed = getattr(self.s._provider, "revealed_ids", None)
        if revealed is not None and sorted(revealed[self._reveal_start :]) != sorted(ids):
            raise ProtocolViolation("oracle reveal log does not match paid evaluations")


def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)
