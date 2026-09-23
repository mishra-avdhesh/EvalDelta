"""Private replay oracle.

Holds the full candidate-outcome column of a benchmark episode behind a closure. The only way
to obtain an outcome is :meth:`ReplayOracle.evaluate`, which the session calls **after**
charging the budget meter. Every reveal is logged, so a run can be audited
(``revealed_ids == paid IDs``).

This protects against accidental leakage in policy code. It does not defend against a policy
that deliberately walks the interpreter's object graph. Policies never receive a reference to
the oracle (see ``tests/security/test_access_control.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evaldelta.providers.base import EvalOutcome


class OracleAccessError(PermissionError):
    pass


def _make_store(hidden: Mapping[str, float]) -> Callable[[str], float]:
    store = {str(k): float(v) for k, v in hidden.items()}

    def lookup(sid: str) -> float:
        if sid not in store:
            raise OracleAccessError(f"unknown sample_id {sid!r}")
        return store[sid]

    return lookup


class ReplayOracle:
    """Replay provider backed by sequestered, precomputed candidate losses."""

    name = "replay"

    __slots__ = ("_lookup", "_revealed", "_fail_ids", "_n")

    def __init__(self, hidden_losses: Mapping[str, float], fail_ids: Sequence[str] = ()) -> None:
        self._lookup = _make_store(hidden_losses)
        self._revealed: list[str] = []
        self._fail_ids = frozenset(fail_ids)
        self._n = len(hidden_losses)

    def __len__(self) -> int:
        return self._n

    def evaluate(self, items: Sequence[Mapping[str, Any]]) -> list[EvalOutcome]:
        out: list[EvalOutcome] = []
        for it in items:
            sid = str(it["sample_id"])
            self._revealed.append(sid)
            if sid in self._fail_ids:
                out.append(EvalOutcome(sid, None, error="simulated_provider_failure"))
            else:
                out.append(EvalOutcome(sid, self._lookup(sid)))
        return out

    @property
    def revealed_ids(self) -> tuple[str, ...]:
        return tuple(self._revealed)

    # Defensive: forbid common accidental bulk-access patterns.
    def __iter__(self) -> Any:
        raise OracleAccessError("the replay oracle cannot be iterated")

    def __getitem__(self, key: Any) -> Any:
        raise OracleAccessError("the replay oracle cannot be indexed; use evaluate()")

    def __reduce__(self) -> Any:
        raise OracleAccessError("the replay oracle cannot be pickled or copied")
