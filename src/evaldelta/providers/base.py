"""Candidate-evaluation provider interface and metric adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalOutcome:
    """Result of one paid candidate evaluation.

    ``loss is None`` together with ``error`` marks an evaluation failure. Failures are tracked
    separately and are never scored as wrong.
    """

    sample_id: str
    loss: float | None
    output: Any = None
    error: str | None = None
    attempts: int = 1


@runtime_checkable
class CandidateProvider(Protocol):
    """Evaluates the candidate system on items and scores them with a bounded loss."""

    name: str

    def evaluate(self, items: Sequence[Mapping[str, Any]]) -> list[EvalOutcome]: ...


def zero_one_loss(output: Any, item: Mapping[str, Any]) -> float:
    """0/1 loss against ``reference_answer`` (exact string match after strip)."""
    ref = item.get("reference_answer")
    if ref is None:
        raise KeyError("zero_one_loss requires a 'reference_answer' field")
    return 0.0 if str(output).strip() == str(ref).strip() else 1.0


@runtime_checkable
class BudgetedProvider(Protocol):
    """Optional retry-aware interface: reserve each attempt before executing it."""

    def evaluate_budgeted(
        self, item: Mapping[str, Any], before_attempt: Callable[[], bool]
    ) -> EvalOutcome: ...


def evaluate_paid(
    provider: CandidateProvider, item: Mapping[str, Any], before_attempt: Callable[[], bool]
) -> EvalOutcome:
    """Evaluate one item, charging every attempt before execution.

    Custom providers without evaluate_budgeted must perform exactly one attempt per item.
    Zero attempts means that no call could be afforded and no outcome was observed.
    """
    charged = 0

    def reserve() -> bool:
        nonlocal charged
        if not before_attempt():
            return False
        charged += 1
        return True

    if isinstance(provider, BudgetedProvider):
        outcome = provider.evaluate_budgeted(item, reserve)
    else:
        if not reserve():
            return EvalOutcome(str(item["sample_id"]), None, error="budget_exhausted", attempts=0)
        (outcome,) = provider.evaluate([item])
        if outcome.attempts != 1:
            raise ValueError("retrying providers must implement evaluate_budgeted")
    if outcome.attempts != charged or (charged == 0 and outcome.loss is not None):
        raise ValueError("provider attempt count does not match reserved budget")
    if outcome.sample_id != str(item["sample_id"]):
        raise ValueError("provider returned an outcome for the wrong sample_id")
    if outcome.loss is not None and not 0.0 <= outcome.loss <= 1.0:
        raise ValueError("provider loss must be finite and in [0, 1]")
    return outcome
