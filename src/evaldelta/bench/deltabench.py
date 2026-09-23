"""DeltaBench: model-version pair episodes built from cached per-item outcome matrices.

A *source* directory contains

* ``matrix.parquet``: one row per pool item. Columns ``sample_id``, ``slice``, ``label``,
  ``estimated_candidate_cost``, public metadata features ``f_*``, and for every version ``v``:
  ``loss__v``, ``conf__v`` (max class probability), ``ptrue__v`` (probability of the true class),
  ``pred__v``.
* ``versions.json``: the ordered release history, with change types and provenance, plus
  source-level license and provenance.

For an episode ``A -> B`` the *public* table contains only A's cached results, metadata features and
historical signals computed from versions released **before B, excluding B**. B's losses go only to
the replay oracle. ``leakage_check`` verifies this.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaldelta.bench.episode import Episode
from evaldelta.data.io import COST_COL, ID_COL, OLD_LOSS_COL, SLICE_COL, TASK_COL

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "data"


@dataclass(frozen=True)
class Source:
    name: str
    matrix: pd.DataFrame
    versions: list[dict[str, Any]]
    meta: dict[str, Any]

    @property
    def version_ids(self) -> list[str]:
        return [v["id"] for v in self.versions]

    def index(self, vid: str) -> int:
        return self.version_ids.index(vid)

    def loss(self, vid: str) -> np.ndarray:
        return self.matrix[f"loss__{vid}"].to_numpy(dtype=float)

    def conf(self, vid: str) -> np.ndarray:
        return self.matrix[f"conf__{vid}"].to_numpy(dtype=float)


@lru_cache(maxsize=16)
def load_source(path: str) -> Source:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = DEFAULT_ROOT / path
    matrix = pd.read_parquet(p / "matrix.parquet")
    spec = json.loads((p / "versions.json").read_text())
    return Source(spec["source"], matrix, spec["versions"], spec.get("meta", {}))


def history_features(src: Source, new_id: str, old_id: str) -> dict[str, np.ndarray]:
    """Item-level signals from versions released before ``new_id`` (never ``new_id`` itself)."""
    j = src.index(new_id)
    prior = [v for v in src.version_ids[:j] if v != new_id]
    if old_id not in prior:
        prior.append(old_id)
    losses = np.stack([src.loss(v) for v in prior]) if prior else np.zeros((0, len(src.matrix)))
    n = len(src.matrix)
    err = losses.mean(0) if len(prior) else np.zeros(n)
    flips = np.abs(np.diff(losses, axis=0)).mean(0) if len(prior) >= 2 else np.zeros(n)
    return {
        "hist_err_rate": err,
        "hist_flip_rate": flips,
        "hist_n_versions": np.full(n, float(len(prior))),
    }


def build_pair_episode(
    src: Source, old_id: str, new_id: str, *, episode_id: str | None = None
) -> Episode:
    if src.index(old_id) >= src.index(new_id):
        raise ValueError("the old version must be released before the new version")
    m = src.matrix
    items = pd.DataFrame(
        {
            ID_COL: m[ID_COL].astype(str),
            TASK_COL: src.meta.get("task", src.name),
            SLICE_COL: m[SLICE_COL].astype(str),
            OLD_LOSS_COL: src.loss(old_id),
            COST_COL: m[COST_COL].to_numpy(dtype=float) if COST_COL in m else 1.0,
            "f_old_conf": src.conf(old_id),
            "f_old_ptrue": m[f"ptrue__{old_id}"].to_numpy(dtype=float),
        }
    )
    for c in m.columns:
        if c.startswith("f_"):
            items[c] = m[c].to_numpy(dtype=float)
    for k, v in history_features(src, new_id, old_id).items():
        items[k] = v
    hidden = dict(zip(items[ID_COL], src.loss(new_id).tolist(), strict=True))
    vmeta = {v["id"]: v for v in src.versions}
    meta = {
        "episode_id": episode_id or f"{src.name}:{old_id}->{new_id}",
        "data_source": src.name,
        "task": src.meta.get("task", src.name),
        "model_family": src.meta.get("model_family", src.name),
        "old_version_id": old_id,
        "new_version_id": new_id,
        "change_type": vmeta[new_id].get("change_type"),
        "change_description": vmeta[new_id].get("description"),
        "is_synthetic": False,
        "regression_pattern": "real_version_change",
        "license": src.meta.get("license"),
        "provenance": src.meta.get("provenance"),
        "target_slice": None,
    }
    return Episode(meta["episode_id"], items, hidden, meta)


def inject(
    base: Episode,
    *,
    pattern: str,
    seed: int,
    severity: float = 0.03,
    slice_name: str | None = None,
    slice_severity: float = 0.2,
    weighting: str = "unstable",
) -> Episode:
    """Controlled perturbation of a real pair's candidate outcomes (marked ``is_synthetic``).

    Patterns: ``null`` (B := A), ``global_degradation``, ``slice_only``, ``compensating``,
    ``rare_severe``, ``improvement``. Extra flips are drawn from items with the relevant old
    outcome. ``weighting="unstable"`` makes them proportional to historical instability plus old
    uncertainty (realistic). ``weighting="uniform"`` draws them at random (adversarial to PairedShift).
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A1E]))
    items = base.items
    ids = items[ID_COL].to_numpy()
    old = items[OLD_LOSS_COL].to_numpy()
    new = np.array([base._hidden[i] for i in ids], dtype=float)
    n = len(ids)
    if weighting == "unstable":
        w = 0.02 + items["hist_flip_rate"].to_numpy() + (1 - items["f_old_conf"].to_numpy())
    else:
        w = np.ones(n)
    in_slice = (
        (items[SLICE_COL] == slice_name).to_numpy() if slice_name else np.zeros(n, dtype=bool)
    )

    def flip(mask: np.ndarray, k: int, to_wrong: bool) -> None:
        # flip items where new currently equals the "good" state
        cand = np.flatnonzero(mask & ((new == 0) if to_wrong else (new == 1)))
        k = min(k, len(cand))
        if k <= 0:
            return
        keys = rng.random(len(cand)) ** (1.0 / (w[cand] + 1e-12))
        pick = cand[np.argsort(-keys)[:k]]
        new[pick] = 1.0 if to_wrong else 0.0

    everywhere = np.ones(n, dtype=bool)
    if pattern == "null":
        new = old.copy()
    elif pattern == "global_degradation":
        flip(everywhere, int(round(severity * n)), True)
    elif pattern == "improvement":
        flip(everywhere, int(round(severity * n)), False)
    elif pattern in {"slice_only", "rare_severe"}:
        if slice_name is None:
            raise ValueError("slice patterns need slice_name")
        flip(in_slice, int(round(slice_severity * in_slice.sum())), True)
    elif pattern == "compensating":
        if slice_name is None:
            raise ValueError("compensating needs slice_name")
        before = new.sum()
        flip(in_slice, int(round(slice_severity * in_slice.sum())), True)
        flip(~in_slice, int(new.sum() - before), False)
    else:
        raise ValueError(f"unknown injection pattern {pattern!r}")
    hidden = dict(zip(ids, new.tolist(), strict=True))
    meta = dict(base.meta)
    meta.update(
        episode_id=f"{base.episode_id}+{pattern}"
        + (f"[{slice_name}]" if slice_name else "")
        + f"#{seed}",
        is_synthetic=True,
        regression_pattern=pattern,
        injection_weighting=weighting,
        target_slice=slice_name,
        base_episode=base.episode_id,
    )
    return Episode(meta["episode_id"], items, hidden, meta)


def episode_from_spec(spec: Mapping[str, Any], trial: int) -> Episode:
    """Runner entry point: {'type': 'deltabench', 'source', 'old', 'new', 'inject'?: {...}}."""
    src = load_source(str(spec["source"]))
    ep = build_pair_episode(src, spec["old"], spec["new"])
    inj = spec.get("inject")
    if inj:
        kw = dict(inj)
        kw.setdefault("seed", trial)
        pick = kw.pop("slice_pick", None)
        if pick is not None:
            kw["slice_name"] = pick_slice(src, pick, int(kw["seed"]))
        ep = inject(ep, **kw)
    return ep


def pick_slice(src: Source, how: str, seed: int) -> str:
    """Deterministically choose an injection slice: 'common' (>= 5% prevalence) or 'rare'."""
    share = src.matrix[SLICE_COL].value_counts(normalize=True)
    if how == "common":
        cands = sorted(str(s) for s in share[share >= 0.05].index)
    elif how == "rare":
        cands = sorted(str(s) for s in share[(share >= 0.005) & (share < 0.05)].index)
        if not cands:
            cands = [str(share.idxmin())]
    else:
        return how  # explicit slice name
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x511CE]))
    return cands[int(rng.integers(len(cands)))]


def leakage_check(ep: Episode) -> None:
    """Assert that no public column reproduces the hidden candidate losses."""
    new = ep.items[ID_COL].map(ep._hidden).to_numpy(dtype=float)
    if np.all(new == ep.items[OLD_LOSS_COL].to_numpy()) and ep.meta.get("regression_pattern") != (
        "null"
    ):
        return  # identical outcomes are legitimate for real near-identical versions
    for c in ep.items.columns:
        if c in {ID_COL, TASK_COL, SLICE_COL, OLD_LOSS_COL}:
            continue
        col = ep.items[c].to_numpy()
        if np.issubdtype(col.dtype, np.number) and np.array_equal(col.astype(float), new):
            raise AssertionError(f"public column {c} equals hidden candidate losses")


# ------------------------------------------------------------------------------------------------
# Registry and episode-level splits
# ------------------------------------------------------------------------------------------------

SPLIT_RULE = {
    # new-version release index -> split, for 12-version ladders
    "train": range(2, 7),
    "validation": range(7, 10),
    "test": range(10, 100),
}
TRANSFER_TEST_SOURCES = ("cifar10",)  # held-out model family: never used for training/tuning
NEAR_NULL_BASES = {  # real near-identical pairs used as backgrounds for injected stress episodes
    "covtype": ("v06", "v07"),
    "adult": ("v04", "v06"),
    "agnews": ("v06", "v08"),
    "bank": ("v06", "v08"),
    "cifar10": ("v04", "v10"),
}


def available_sources(root: Path = DEFAULT_ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "matrix.parquet").exists())


def registry(root: Path = DEFAULT_ROOT, max_gap: int = 3) -> pd.DataFrame:
    """All real pair episodes (i < j, j - i <= max_gap) with their split and true Delta."""
    rows = []
    for name in available_sources(root):
        src = load_source(str(root / name))
        v = src.version_ids
        for j in range(1, len(v)):
            if name in TRANSFER_TEST_SOURCES:
                split = "test_transfer"
            else:
                split = next(s for s, r in SPLIT_RULE.items() if (j + 1) in r)
            for i in range(max(0, j - max_gap), j):
                d = src.loss(v[j]) - src.loss(v[i])
                rows.append(
                    {
                        "source": name,
                        "old": v[i],
                        "new": v[j],
                        "split": split,
                        "change_type": src.versions[j].get("change_type"),
                        "true_delta": float(d.mean()),
                        "down_rate": float((d > 0).mean()),
                        "up_rate": float((d < 0).mean()),
                        "pool_size": len(d),
                    }
                )
    return pd.DataFrame(rows)


def eligible_slices(src: Source, min_frac: float = 0.05) -> list[str]:
    share = src.matrix[SLICE_COL].value_counts(normalize=True)
    return sorted(str(s) for s in share[share >= min_frac].index)
