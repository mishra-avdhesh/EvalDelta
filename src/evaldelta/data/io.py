"""Loading and validating public item tables.

A *public* item table contains only information a selector may see: stable IDs, slice/task
metadata, cached old-version results, allowed features (``f_*``) and historical signals
(``hist_*``). Columns that look like candidate (new-version) outcomes are rejected outright.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "sample_id"
SLICE_COL = "slice"
TASK_COL = "task"
OLD_LOSS_COL = "old_loss"
COST_COL = "estimated_candidate_cost"

#: Column prefixes that would leak candidate outcomes into a public table.
FORBIDDEN_PREFIXES = ("new_", "candidate_", "d_", "delta")
FEATURE_PREFIX = "f_"
HISTORY_PREFIX = "hist_"


class DataValidationError(ValueError):
    """Raised for malformed or ambiguous datasets."""


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in {".csv", ".tsv"}:
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",", dtype={ID_COL: str})
    if p.suffix in {".jsonl", ".json"}:
        return pd.read_json(p, lines=p.suffix == ".jsonl", dtype={ID_COL: str})
    raise DataValidationError(f"unsupported file type: {p.suffix}")


def _check_ids(df: pd.DataFrame, what: str) -> None:
    if ID_COL not in df.columns:
        raise DataValidationError(f"{what}: missing required column '{ID_COL}'")
    if df[ID_COL].isna().any():
        raise DataValidationError(f"{what}: null {ID_COL}")
    ids = df[ID_COL].astype(str)
    dup = ids[ids.duplicated()]
    if len(dup):
        raise DataValidationError(
            f"{what}: duplicate {ID_COL} values (first: {dup.iloc[0]!r}); IDs must be unique"
        )


def _check_loss(values: pd.Series, name: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise DataValidationError(f"{name}: NaN/inf values are not allowed (tag failures instead)")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise DataValidationError(
            f"{name}: losses must lie in [0, 1]; values are never silently clipped"
        )
    return arr


class ItemTable:
    """Validated, immutable public item table."""

    def __init__(self, df: pd.DataFrame, *, require_old_loss: bool = True) -> None:
        _check_ids(df, "items")
        bad = [c for c in df.columns if c.lower().startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise DataValidationError(
                f"public item table contains candidate-outcome-like columns {bad}; "
                "candidate outcomes must be supplied only to a provider/oracle"
            )
        df = df.copy()
        df[ID_COL] = df[ID_COL].astype(str)
        if SLICE_COL not in df.columns:
            df[SLICE_COL] = "all"
        df[SLICE_COL] = df[SLICE_COL].astype(str)
        if TASK_COL not in df.columns:
            df[TASK_COL] = "default"
        if COST_COL not in df.columns:
            df[COST_COL] = 1.0
        cost = pd.to_numeric(df[COST_COL], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(cost)) or np.any(cost <= 0):
            raise DataValidationError(f"{COST_COL} must be finite and > 0")
        df[COST_COL] = cost
        if require_old_loss:
            if OLD_LOSS_COL not in df.columns:
                raise DataValidationError(f"items: missing '{OLD_LOSS_COL}' (cached old results)")
            df[OLD_LOSS_COL] = _check_loss(df[OLD_LOSS_COL], OLD_LOSS_COL)
        for c in df.columns:
            if c.startswith((FEATURE_PREFIX, HISTORY_PREFIX)):
                col = pd.to_numeric(df[c], errors="coerce")
                if col.isna().any():
                    raise DataValidationError(f"feature column {c} has missing/non-numeric values")
                df[c] = col.astype(float)
        self._df = df.reset_index(drop=True)
        self._pos = {sid: i for i, sid in enumerate(self._df[ID_COL])}

    # -- construction --------------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path, **kw: bool) -> ItemTable:
        return cls(_read_any(path), **kw)

    # -- accessors (all return copies) ---------------------------------------------------------
    @property
    def ids(self) -> list[str]:
        return list(self._df[ID_COL])

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, sid: object) -> bool:
        return sid in self._pos

    @property
    def frame(self) -> pd.DataFrame:
        return self._df.copy()

    def subset(self, ids: Iterable[str]) -> pd.DataFrame:
        idx = [self._pos[i] for i in ids]
        return self._df.iloc[idx].copy()

    def column(self, name: str) -> np.ndarray:
        return self._df[name].to_numpy().copy()

    def value(self, sid: str, name: str) -> object:
        return self._df[name].iloc[self._pos[sid]]

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self._df.columns if c.startswith(FEATURE_PREFIX)]

    @property
    def history_columns(self) -> list[str]:
        return [c for c in self._df.columns if c.startswith(HISTORY_PREFIX)]

    @property
    def is_binary(self) -> bool:
        if OLD_LOSS_COL not in self._df.columns:
            return False
        v = self._df[OLD_LOSS_COL].to_numpy()
        return bool(np.all((v == 0.0) | (v == 1.0)))

    def slice_sizes(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self._df[SLICE_COL].value_counts().items()}


def load_hidden_outcomes(
    path_or_frame: str | Path | pd.DataFrame, items: ItemTable, loss_col: str = "new_loss"
) -> dict[str, float]:
    """Load the private candidate-loss table used only by a replay provider.

    Joins by stable ID and fails on missing, extra or conflicting IDs.
    """
    df = path_or_frame if isinstance(path_or_frame, pd.DataFrame) else _read_any(path_or_frame)
    _check_ids(df, "hidden outcomes")
    if loss_col not in df.columns:
        raise DataValidationError(f"hidden outcomes: missing '{loss_col}'")
    ids = df[ID_COL].astype(str)
    missing = set(items.ids) - set(ids)
    extra = set(ids) - set(items.ids)
    if missing:
        raise DataValidationError(f"hidden outcomes missing {len(missing)} item IDs")
    if extra:
        raise DataValidationError(f"hidden outcomes contain {len(extra)} unknown IDs")
    losses = _check_loss(df[loss_col], loss_col)
    return dict(zip(ids, losses.tolist(), strict=True))


def as_mapping(ids: Iterable[str], values: Iterable[float]) -> Mapping[str, float]:
    return dict(zip(ids, values, strict=True))
