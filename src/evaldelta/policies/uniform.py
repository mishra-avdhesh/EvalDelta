"""B1 uniform-random and B2 stratified-random discovery selectors."""

from __future__ import annotations

import numpy as np

from evaldelta.data.io import ID_COL, SLICE_COL
from evaldelta.policies.base import PolicyView


class UniformPolicy:
    """B1: uniform random selection without replacement among unqueried discovery items."""

    name = "uniform"

    def select(self, view: PolicyView, k: int, rng: np.random.Generator) -> list[str]:
        pool = sorted(view.unqueried[ID_COL].tolist())
        if not pool:
            return []
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]


class StratifiedPolicy:
    """B2: stratified random selection with proportional allocation across slices.

    Allocation targets the slice's share of the *discovery pool*; within a slice items are
    chosen uniformly at random. Largest-remainder rounding keeps the batch size exact.
    """

    name = "stratified"

    def select(self, view: PolicyView, k: int, rng: np.random.Generator) -> list[str]:
        un = view.unqueried
        if un.empty:
            return []
        total = view.items[SLICE_COL].value_counts()
        taken = total - un[SLICE_COL].value_counts().reindex(total.index, fill_value=0)
        n_done = int(taken.sum())
        target = total / total.sum() * (n_done + k)
        need = (target - taken).clip(lower=0)
        avail = un[SLICE_COL].value_counts().reindex(total.index, fill_value=0)
        need = need.clip(upper=avail)
        alloc = need.apply(np.floor).astype(int)
        short = min(k, int(avail.sum())) - int(alloc.sum())
        if short > 0:
            rem = (need - alloc).sort_values(ascending=False)
            for s in rem.index:
                if short == 0:
                    break
                if alloc[s] < avail[s]:
                    alloc[s] += 1
                    short -= 1
            for s in sorted(total.index):  # fill if still short (tiny slices exhausted)
                while short > 0 and alloc[s] < avail[s]:
                    alloc[s] += 1
                    short -= 1
        out: list[str] = []
        for s in sorted(alloc.index):
            n_s = int(alloc[s])
            if n_s == 0:
                continue
            ids = sorted(un.loc[un[SLICE_COL] == s, ID_COL].tolist())
            idx = rng.choice(len(ids), size=n_s, replace=False)
            out.extend(ids[i] for i in idx)
        return out[:k]
