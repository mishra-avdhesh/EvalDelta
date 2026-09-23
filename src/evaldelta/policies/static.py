"""Static (non-adaptive) discovery baselines B3, B4, B5.

They use only public features and never look at purchased outcomes. Each returns the next
items of a fixed ranking. Deterministic rank selection has **no** selection probabilities, so
none are reported.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from evaldelta.data.io import ID_COL, OLD_LOSS_COL
from evaldelta.policies.base import Policy, PolicyView


class _RankPolicy:
    name = "rank"

    def __init__(self) -> None:
        self._order: list[str] | None = None

    def score(self, items: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def select(self, view: PolicyView, k: int, rng: np.random.Generator) -> list[str]:
        if self._order is None:
            items = view.items.sort_values(ID_COL).reset_index(drop=True)
            s = self.score(items, rng)
            tiebreak = rng.random(len(items))
            order = np.lexsort((tiebreak, -s))
            self._order = items[ID_COL].to_numpy()[order].tolist()
        eligible = set(view.unqueried[ID_COL])
        out: list[str] = []
        for i in self._order:
            if i in eligible:
                out.append(i)
                if len(out) == k:
                    break
        return out


class HistoricalCohortPolicy(_RankPolicy):
    """B3: fixed 'golden regression cohort' of historically unstable items.

    Ranks by ``hist_flip_rate`` (flips across prior version pairs), then historical error.
    """

    name = "historical_cohort"

    def score(self, items: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if "hist_flip_rate" not in items.columns:
            raise ValueError("historical_cohort requires a 'hist_flip_rate' column")
        s = items["hist_flip_rate"].to_numpy(dtype=float)
        if "hist_err_rate" in items.columns:
            s = s + 1e-3 * items["hist_err_rate"].to_numpy(dtype=float)
        return s


class OldUncertaintyPolicy(_RankPolicy):
    """B4: rank by old-model uncertainty (1 - confidence), falling back to old error."""

    name = "old_uncertainty"

    def score(self, items: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if "f_old_conf" in items.columns:
            return 1.0 - items["f_old_conf"].to_numpy(dtype=float)
        return items[OLD_LOSS_COL].to_numpy(dtype=float)


class DiversityPolicy(_RankPolicy):
    """B5: greedy k-center (farthest-point) ordering on standardised allowed features."""

    name = "diversity"

    def __init__(self, max_items: int = 5000) -> None:
        super().__init__()
        self.max_items = max_items

    def score(self, items: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        cols = [c for c in items.columns if c.startswith(("f_", "hist_"))] or [OLD_LOSS_COL]
        x = items[cols].to_numpy(dtype=float)
        x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        n = len(x)
        m = min(n, self.max_items)
        order = np.empty(m, dtype=int)
        order[0] = int(rng.integers(n))
        dist = np.linalg.norm(x - x[order[0]], axis=1)
        for j in range(1, m):
            order[j] = int(np.argmax(dist))
            dist = np.minimum(dist, np.linalg.norm(x - x[order[j]], axis=1))
        score = np.full(n, -1.0)
        score[order] = np.arange(m, 0, -1, dtype=float)
        return score


STATIC_POLICIES: dict[str, Callable[..., Policy]] = {
    "historical_cohort": HistoricalCohortPolicy,
    "old_uncertainty": OldUncertaintyPolicy,
    "diversity": DiversityPolicy,
}


def available() -> list[str]:
    return sorted(STATIC_POLICIES)
