"""Typed configuration and result schemas (pydantic v2).

Every field here corresponds to a predeclared quantity in ``docs/STATISTICAL_PROTOCOL.md``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"


class Decision(StrEnum):
    """Possible outcomes of a regression test. ``inconclusive`` is never a pass."""

    CONFIRMED_REGRESSION = "confirmed_regression"
    EVIDENCE_OF_NONINFERIORITY = "evidence_of_noninferiority"
    INCONCLUSIVE = "inconclusive"
    EVALUATION_ERROR = "evaluation_error"


EXIT_CODES: dict[Decision, int] = {
    Decision.EVIDENCE_OF_NONINFERIORITY: 0,
    Decision.CONFIRMED_REGRESSION: 2,
    Decision.INCONCLUSIVE: 3,
    Decision.EVALUATION_ERROR: 4,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SplitConfig(_Frozen):
    """Sealed partition of item IDs into discovery / global / slice confirmation pools."""

    discovery: float = Field(0.6, ge=0.0, le=1.0)
    global_confirm: float = Field(0.2, gt=0.0, le=1.0)
    slice_confirm: float = Field(0.2, ge=0.0, le=1.0)
    seed: int = 0

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        total = self.discovery + self.global_confirm + self.slice_confirm
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return self


class Budget(_Frozen):
    """Paid-evaluation budget. Only candidate evaluations are charged (see protocol §8)."""

    max_candidate_calls: int = Field(..., gt=0)
    max_cost: float | None = Field(None, gt=0.0)
    cost_unit: str = "candidate_calls"
    discovery_fraction: float = Field(0.3, ge=0.0, le=1.0)
    global_fraction: float = Field(0.5, ge=0.0, le=1.0)
    slice_fraction: float = Field(0.2, ge=0.0, le=1.0)
    charge_failed_attempts: bool = True
    old_cached: bool = True

    @model_validator(mode="after")
    def _fractions(self) -> Budget:
        total = self.discovery_fraction + self.global_fraction + self.slice_fraction
        if total > 1.0 + 1e-9:
            raise ValueError(f"phase budget fractions sum to {total} > 1")
        if self.max_cost is not None and self.cost_unit == "candidate_calls":
            raise ValueError("max_cost requires a declared cost_unit other than 'candidate_calls'")
        return self

    def phase_calls(self) -> dict[str, int]:
        n = self.max_candidate_calls
        disc = int(n * self.discovery_fraction)
        glob = int(n * self.global_fraction)
        slc = int(n * self.slice_fraction)
        return {"discovery": disc, "global_confirmation": glob, "slice_confirmation": slc}


GlobalMethod = Literal["auto", "mcnemar_exact", "betting_wor", "hoeffding_wor"]


class ConfirmPlan(_Frozen):
    """Predeclared confirmation plan. Frozen before any confirmation call."""

    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    regression_margin: float = Field(0.0, ge=0.0, lt=1.0)
    noninferiority_margin: float | None = Field(0.02, gt=0.0, lt=1.0)
    method: GlobalMethod = "auto"
    looks: int = Field(3, ge=1, le=50)
    alpha_slice: float = Field(0.05, gt=0.0, lt=0.5)
    slice_margin: float = Field(0.0, ge=0.0, lt=1.0)
    max_slices: int = Field(3, ge=0, le=50)
    min_slice_confirm: int = Field(30, ge=1)
    predeclared_slices: tuple[str, ...] = ()
    joint_fwer: bool = False

    @model_validator(mode="after")
    def _margins(self) -> ConfirmPlan:
        if self.method == "mcnemar_exact" and self.regression_margin != 0.0:
            raise ValueError("mcnemar_exact only tests regression_margin == 0; use betting_wor")
        return self

    def global_alpha(self) -> float:
        return self.alpha / 2 if self.joint_fwer else self.alpha

    def slice_alpha(self) -> float:
        return self.alpha / 2 if self.joint_fwer else self.alpha_slice


class PolicyConfig(_Frozen):
    name: str = "uniform"
    params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(25, ge=1)


class RunConfig(_Frozen):
    """Full configuration of one comparison run."""

    run_id: str = "run"
    seed: int = 0
    budget: Budget
    split: SplitConfig = SplitConfig()
    plan: ConfirmPlan = ConfirmPlan()
    policy: PolicyConfig = PolicyConfig()
    protocol_version: str = PROTOCOL_VERSION

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------


class TestEvidence(BaseModel):
    """Evidence from one confirmation test (global or slice)."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    estimand: str
    method: str
    decision: Decision
    reason: str
    n: int
    n_down: int | None = None
    n_up: int | None = None
    effect: float | None = None
    lower: float | None = None
    upper: float | None = None
    interval_method: str | None = None
    interval_level: float | None = None
    p_value: float | None = None
    alpha_allocated: float
    looks_used: int = 0
    looks_planned: int = 0
    population_size: int | None = None


class PhaseSpend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planned_calls: int
    paid_calls: int
    paid_cost: float
    failed_attempts: int = 0


class RunReport(BaseModel):
    """Machine-readable report (``report.json``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REPORT_SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    run_id: str
    episode_id: str | None = None
    old_version_id: str | None = None
    new_version_id: str | None = None
    config: dict[str, Any]
    config_hash: str
    split_hash: str
    pool_size: int
    partition_sizes: dict[str, int]
    decision: Decision
    exit_code: int
    global_result: TestEvidence | None
    slice_results: list[TestEvidence]
    exploratory: dict[str, Any]
    spend: dict[str, PhaseSpend]
    total_paid_calls: int
    total_paid_cost: float
    cost_unit: str
    selected_ids: dict[str, list[str]]
    limitations: list[str]
    wall_time_s: float
    extra: dict[str, Any] = Field(default_factory=dict)
