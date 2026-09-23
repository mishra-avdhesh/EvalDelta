"""Construct acquisition policies by name."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaldelta.data.io import ItemTable
from evaldelta.policies.base import Policy
from evaldelta.policies.uniform import StratifiedPolicy, UniformPolicy
from evaldelta.schemas import PolicyConfig


def make_policy(
    cfg: PolicyConfig, items: ItemTable | None = None, history: pd.DataFrame | None = None
) -> Policy:
    name = cfg.name
    params: dict[str, Any] = dict(cfg.params)
    if name == "uniform":
        return UniformPolicy()
    if name == "stratified":
        return StratifiedPolicy()
    # Optional policies are imported lazily to keep the default import light.
    from evaldelta.policies.static import STATIC_POLICIES

    if name in STATIC_POLICIES:
        return STATIC_POLICIES[name](**params)
    if name == "paired_shift":
        from evaldelta.policies.paired_shift import PairedShift

        ps: Policy = PairedShift(**params)
        return ps
    raise ValueError(f"unknown policy {name!r}")
