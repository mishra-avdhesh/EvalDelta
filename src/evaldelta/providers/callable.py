"""Provider wrapping an arbitrary Python callable plus a scorer, with retries."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evaldelta.providers.base import EvalOutcome


class CallableProvider:
    """Calls ``fn(item) -> output`` and ``scorer(output, item) -> loss in [0, 1]``.

    Exceptions from ``fn`` are retried up to ``retries`` times with exponential backoff; if all
    attempts fail the outcome is an evaluation failure (``loss=None``), never a wrong answer.
    """

    def __init__(
        self,
        fn: Callable[[Mapping[str, Any]], Any],
        scorer: Callable[[Any, Mapping[str, Any]], float],
        *,
        retries: int = 2,
        backoff_s: float = 0.0,
        name: str = "callable",
    ) -> None:
        self.fn = fn
        self.scorer = scorer
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.retries = retries
        self.backoff_s = backoff_s
        self.name = name

    def evaluate(self, items: Sequence[Mapping[str, Any]]) -> list[EvalOutcome]:
        return [self._one(it) for it in items]

    def evaluate_budgeted(
        self, item: Mapping[str, Any], before_attempt: Callable[[], bool]
    ) -> EvalOutcome:
        return self._one(item, before_attempt)

    def _one(
        self, item: Mapping[str, Any], before_attempt: Callable[[], bool] | None = None
    ) -> EvalOutcome:
        sid = str(item["sample_id"])
        err: str | None = None
        for attempt in range(1, self.retries + 2):
            if before_attempt is not None and not before_attempt():
                return EvalOutcome(sid, None, error="budget_exhausted", attempts=attempt - 1)
            try:
                output = self.fn(item)
            except Exception as exc:  # provider failures are data, not crashes
                err = f"{type(exc).__name__}: {exc}"
                if self.backoff_s:
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
                continue
            loss = float(self.scorer(output, item))
            if not (0.0 <= loss <= 1.0):
                raise ValueError(f"scorer returned loss {loss} outside [0, 1] for {sid}")
            return EvalOutcome(sid, loss, output=output, attempts=attempt)
        return EvalOutcome(sid, None, error=err, attempts=self.retries + 1)
