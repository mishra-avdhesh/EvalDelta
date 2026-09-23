"""EvalDelta: budgeted, statistically valid paired regression testing for ML model versions."""

from evaldelta.bench.episode import Episode
from evaldelta.bench.synthetic import generate_episode
from evaldelta.data.io import ItemTable
from evaldelta.policies.paired_shift import PairedShift, PairedShiftModel
from evaldelta.policies.uniform import StratifiedPolicy, UniformPolicy
from evaldelta.providers.base import EvalOutcome, zero_one_loss
from evaldelta.providers.callable import CallableProvider
from evaldelta.replay.adaptive import AdaptiveGlobalPlan, AdaptiveGlobalSession
from evaldelta.replay.oracle import ReplayOracle
from evaldelta.replay.session import EvalSession, RunResult
from evaldelta.schemas import (
    PROTOCOL_VERSION,
    Budget,
    ConfirmPlan,
    Decision,
    PolicyConfig,
    RunConfig,
    SplitConfig,
)

__version__ = "0.1.0"

__all__ = [
    "PROTOCOL_VERSION",
    "AdaptiveGlobalPlan",
    "AdaptiveGlobalSession",
    "PairedShift",
    "PairedShiftModel",
    "Budget",
    "CallableProvider",
    "ConfirmPlan",
    "Decision",
    "Episode",
    "EvalOutcome",
    "EvalSession",
    "ItemTable",
    "PolicyConfig",
    "ReplayOracle",
    "RunConfig",
    "RunResult",
    "SplitConfig",
    "StratifiedPolicy",
    "UniformPolicy",
    "generate_episode",
    "zero_one_loss",
]
