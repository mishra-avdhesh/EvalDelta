"""Benchmark episode container: public items + sequestered candidate outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaldelta.data.io import ID_COL, OLD_LOSS_COL, SLICE_COL, ItemTable
from evaldelta.replay.oracle import ReplayOracle


@dataclass(frozen=True)
class Episode:
    """One old/new version pair on a fixed pool.

    ``_hidden`` holds the candidate losses. Pass it only to :meth:`oracle`, never to a policy.
    The ground-truth summary in ``truth`` is for *evaluation after the run*.
    """

    episode_id: str
    items: pd.DataFrame
    _hidden: Mapping[str, float] = field(repr=False)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def item_table(self) -> ItemTable:
        return ItemTable(self.items)

    def oracle(self, fail_ids: tuple[str, ...] = ()) -> ReplayOracle:
        return ReplayOracle(self._hidden, fail_ids=fail_ids)

    # -- ground truth (evaluation only) --------------------------------------------------------
    def d_vector(self) -> pd.Series:
        new = self.items[ID_COL].map(self._hidden).astype(float)
        return pd.Series(
            new.to_numpy() - self.items[OLD_LOSS_COL].to_numpy(), index=self.items[ID_COL]
        )

    def truth(self) -> dict[str, Any]:
        d = self.d_vector()
        sl = self.items.set_index(ID_COL)[SLICE_COL]
        by_slice = d.groupby(sl).mean().to_dict()
        new = self.items[ID_COL].map(self._hidden).astype(float).to_numpy()
        old = self.items[OLD_LOSS_COL].to_numpy()
        return {
            "delta": float(d.mean()),
            "slice_delta": {str(k): float(v) for k, v in by_slice.items()},
            "n_down": int(np.sum((old == 0) & (new == 1))),
            "n_up": int(np.sum((old == 1) & (new == 0))),
            "old_error": float(old.mean()),
            "new_error": float(new.mean()),
        }

    def hidden_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {ID_COL: list(self._hidden.keys()), "new_loss": list(self._hidden.values())}
        )

    def save(self, out_dir: str | Path) -> Path:
        """Write public items and (separately) the private oracle table."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.items.to_parquet(out / "items.parquet", index=False)
        self.hidden_frame().to_parquet(out / "oracle_private.parquet", index=False)
        pd.Series(dict(self.meta)).to_json(out / "meta.json")
        return out

    def swapped(self) -> Episode:
        """Return the episode with old and new swapped (Delta negates)."""
        items = self.items.copy()
        new = items[ID_COL].map(self._hidden).astype(float)
        hidden = dict(zip(items[ID_COL], items[OLD_LOSS_COL].astype(float), strict=True))
        items[OLD_LOSS_COL] = new.to_numpy()
        meta = dict(self.meta)
        meta["old_version_id"], meta["new_version_id"] = (
            self.meta.get("new_version_id"),
            self.meta.get("old_version_id"),
        )
        return Episode(self.episode_id + "__swapped", items, hidden, meta)
