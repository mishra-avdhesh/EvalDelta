"""Provider that runs a user-declared shell command per item (item JSON on stdin, output on stdout).

The command comes from the user's own config file. It is never taken from uploaded data. The
public demo does not expose this provider.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evaldelta.providers.base import EvalOutcome


class CommandProvider:
    name = "command"

    def __init__(
        self,
        command: str,
        scorer: Callable[[Any, Mapping[str, Any]], float],
        *,
        timeout_s: float = 60.0,
        retries: int = 1,
    ) -> None:
        self.argv = shlex.split(command)
        self.scorer = scorer
        self.timeout_s = timeout_s
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.retries = retries

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
        payload = json.dumps({k: v for k, v in item.items()}, default=str)
        err = None
        for attempt in range(1, self.retries + 2):
            if before_attempt is not None and not before_attempt():
                return EvalOutcome(sid, None, error="budget_exhausted", attempts=attempt - 1)
            try:
                proc = subprocess.run(
                    self.argv,
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=True,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                err = f"{type(exc).__name__}"
                continue
            output = proc.stdout.strip()
            loss = float(self.scorer(output, item))
            if not 0.0 <= loss <= 1.0:
                raise ValueError(f"scorer returned {loss} outside [0, 1]")
            return EvalOutcome(sid, loss, output=output, attempts=attempt)
        return EvalOutcome(sid, None, error=err, attempts=self.retries + 1)
