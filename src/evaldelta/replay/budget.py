"""Budget meter: counts only explicitly defined paid candidate evaluations (protocol §8)."""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised *before* a charge that would exceed a ceiling. Nothing is charged."""


@dataclass
class _PhaseLedger:
    cap_calls: int
    calls: int = 0
    cost: float = 0.0
    failed: int = 0


@dataclass
class BudgetMeter:
    """Tracks paid calls and declared cost per phase with hard ceilings."""

    max_calls: int
    max_cost: float | None = None
    phase_caps: dict[str, int] = field(default_factory=dict)
    _phases: dict[str, _PhaseLedger] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if sum(self.phase_caps.values()) > self.max_calls:
            raise ValueError("phase caps exceed the total budget")
        for name, cap in self.phase_caps.items():
            self._phases[name] = _PhaseLedger(cap_calls=cap)

    # -- queries -------------------------------------------------------------------------------
    @property
    def calls(self) -> int:
        return sum(p.calls for p in self._phases.values())

    @property
    def cost(self) -> float:
        return sum(p.cost for p in self._phases.values())

    def phase(self, name: str) -> _PhaseLedger:
        if name not in self._phases:
            self._phases[name] = _PhaseLedger(cap_calls=self.max_calls)
        return self._phases[name]

    def remaining_calls(self, phase: str) -> int:
        p = self.phase(phase)
        return max(0, min(p.cap_calls - p.calls, self.max_calls - self.calls))

    def remaining_cost(self) -> float | None:
        return None if self.max_cost is None else self.max_cost - self.cost

    def can_afford(self, phase: str, n_calls: int, cost: float) -> bool:
        if n_calls > self.remaining_calls(phase):
            return False
        rc = self.remaining_cost()
        return rc is None or cost <= rc + 1e-12

    # -- mutation ------------------------------------------------------------------------------
    def charge(self, phase: str, n_calls: int, cost: float, failed: int = 0) -> None:
        if n_calls < 0 or cost < 0:
            raise ValueError("charges must be non-negative")
        if not self.can_afford(phase, n_calls, cost):
            raise BudgetExceeded(
                f"charging {n_calls} calls / {cost:.4g} cost in phase '{phase}' would exceed budget "
                f"(spent {self.calls}/{self.max_calls} calls)"
            )
        p = self.phase(phase)
        p.calls += n_calls
        p.cost += cost
        p.failed += failed

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            k: {"cap_calls": v.cap_calls, "calls": v.calls, "cost": v.cost, "failed": v.failed}
            for k, v in self._phases.items()
        }
