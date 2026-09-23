"""Candidate-evaluation provider interface and metric adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
