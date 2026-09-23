"""YAML run specification for ``evaldelta compare``."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaldelta.data.io import ItemTable, load_hidden_outcomes
from evaldelta.providers.base import CandidateProvider, zero_one_loss
from evaldelta.replay.oracle import ReplayOracle
from evaldelta.schemas import RunConfig

SCORERS = {"zero_one": zero_one_loss}


@dataclass
class RunSpec:
    run: RunConfig
    items: ItemTable
    provider: CandidateProvider
    history: pd.DataFrame | None
    meta: dict[str, Any]


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def _import_object(ref: str, base: Path | None = None) -> Any:
    """Resolve ``module:attr`` or ``path/to/file.py:attr`` (path relative to the config file)."""
    mod, _, attr = ref.rpartition(":")
    if not mod or not attr:
        raise ValueError(f"expected 'module:attribute' or 'file.py:attribute', got {ref!r}")
    if mod.endswith(".py"):
        path = _resolve(base or Path.cwd(), mod)
        spec = importlib.util.spec_from_file_location(f"evaldelta_user_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr)
    return getattr(importlib.import_module(mod), attr)


def build_provider(spec: dict[str, Any], items: ItemTable, base: Path) -> CandidateProvider:
    kind = spec.get("type")
    if kind == "replay":
        hidden = load_hidden_outcomes(_resolve(base, spec["outcomes"]), items)
        return ReplayOracle(hidden)
    scorer_name = spec.get("scorer", "zero_one")
    scorer = SCORERS[scorer_name] if scorer_name in SCORERS else _import_object(scorer_name, base)
    if kind == "python":
        from evaldelta.providers.callable import CallableProvider

        return CallableProvider(
            _import_object(spec["callable"], base), scorer, retries=int(spec.get("retries", 2))
        )
    if kind == "command":
        from evaldelta.providers.command import CommandProvider

        return CommandProvider(
            spec["command"],
            scorer,
            timeout_s=float(spec.get("timeout_s", 60)),
            retries=int(spec.get("retries", 1)),
        )
    if kind == "http":
        from evaldelta.providers.http import HTTPProvider

        http: CandidateProvider = HTTPProvider.from_spec(spec, scorer)
        return http
    raise ValueError(f"unknown provider type {kind!r}")


def load_run_spec(path: str | Path) -> RunSpec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    base = path.parent
    unknown = set(raw) - {"run", "items", "history", "provider", "meta"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    run = RunConfig.model_validate(raw["run"])
    items = ItemTable.from_path(_resolve(base, raw["items"]))
    history = pd.read_parquet(_resolve(base, raw["history"])) if raw.get("history") else None
    provider = build_provider(raw["provider"], items, base)
    return RunSpec(run, items, provider, history, dict(raw.get("meta") or {}))
