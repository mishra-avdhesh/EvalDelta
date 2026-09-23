"""Acquisition-policy interface and the read-only view that policies receive.

A :class:`PolicyView` holds copies of permitted information only:

* public metadata / cached old results / allowed features for discovery items,
* candidate losses for discovery items **already paid for**,
* optional historical tables (other version pairs; never the current candidate).

It holds no reference to the replay oracle, the provider or the confirmation pools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from evaldelta.data.io import ID_COL, OLD_LOSS_COL, SLICE_COL


@dataclass(frozen=True)
class PolicyView:
    items: pd.DataFrame  # public columns of the discovery pool (copy)
    revealed: Mapping[str, float]  # candidate losses already purchased in discovery
    remaining_calls: int
    history: pd.DataFrame | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def build(
        items: pd.DataFrame,
        revealed: Mapping[str, float],
        remaining_calls: int,
        history: pd.DataFrame | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> PolicyView:
        return PolicyView(
            items=items.copy(),
            revealed=MappingProxyType(dict(revealed)),
            remaining_calls=remaining_calls,
            history=None if history is None else history.copy(),
            meta=MappingProxyType(dict(meta or {})),
        )

    @property
    def unqueried(self) -> pd.DataFrame:
        mask = ~self.items[ID_COL].isin(list(self.revealed.keys()))
        return self.items.loc[mask]

    def revealed_frame(self) -> pd.DataFrame:
        """Discovery items already purchased, with their candidate loss as ``cand_loss``."""
        df = self.items[self.items[ID_COL].isin(list(self.revealed.keys()))].copy()
        df["cand_loss"] = df[ID_COL].map(self.revealed).astype(float)
        return df


@runtime_checkable
class Policy(Protocol):
    """Discovery acquisition policy (deterministic ranking or randomised)."""

    name: str

    def select(self, view: PolicyView, k: int, rng: np.random.Generator) -> list[str]:
        """Return up to ``k`` *unqueried* discovery IDs to pay for next."""
        ...


def suspicious_slices(view: PolicyView, max_slices: int, min_revealed: int = 5) -> list[str]:
    """Default exploratory slice ranking from purchased discovery outcomes only.

    Ranks slices by the smoothed net negative-flip rate (Beta(1,1)-style shrinkage toward 0)
    among revealed items; returns slices whose smoothed estimate is positive.
    """
    rev = view.revealed_frame()
    if rev.empty or max_slices == 0:
        return []
    rev["d"] = rev["cand_loss"] - rev[OLD_LOSS_COL]
    stats = rev.groupby(SLICE_COL)["d"].agg(["sum", "count"])
    stats = stats[stats["count"] >= min_revealed]
    score = stats["sum"] / (stats["count"] + 2.0)
    score = score[score > 0].sort_values(ascending=False)
    return [str(s) for s in score.index[:max_slices]]
