"""Deterministic synthetic episode generator (CPU, seeded).

Items have a latent old-model logit ``m_i``. Items near the decision boundary (|m_i| small)
are *unstable*: they flip between versions more often. That is the pattern that real
negative-flip studies report (Yan et al. 2021; Xie et al. 2021). Public features are a noisy
old-confidence proxy and flip / error rates across *simulated prior versions* (historical signals).
The candidate's outcomes are built with **exact** discordance counts, so the finite-pool Delta
equals the requested value up to integer rounding.

Scenarios: null_identical, null_noisy, global_regression, slice_regression, improvement,
compensating, rare_severe.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from evaldelta.bench.episode import Episode
from evaldelta.data.io import COST_COL, ID_COL, OLD_LOSS_COL, SLICE_COL, TASK_COL

SCENARIOS = (
    "null_identical",
    "null_noisy",
    "global_regression",
    "slice_regression",
    "improvement",
    "compensating",
    "rare_severe",
)

Pattern = Literal["unstable", "uniform"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


def _weighted_pick(
    rng: np.random.Generator, candidates: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """Pick k distinct indices from ``candidates`` with probability proportional to weights."""
    if k <= 0:
        return np.array([], dtype=int)
    if k > len(candidates):
        raise ValueError(f"cannot pick {k} of {len(candidates)} candidates")
    w = np.asarray(weights, dtype=float) + 1e-12
    # Efraimidis-Spirakis weighted sampling without replacement
    keys = rng.random(len(candidates)) ** (1.0 / w)
    return candidates[np.argsort(-keys)[:k]]


def generate_episode(
    scenario: str = "slice_regression",
    *,
    n_items: int = 5000,
    n_slices: int = 8,
    seed: int = 0,
    flip_rate: float = 0.04,
    severity: float = 0.03,
    slice_severity: float = 0.15,
    target_slice: str | None = None,
    rare_prevalence: float = 0.01,
    pattern: Pattern = "unstable",
    heterogeneous_cost: bool = False,
    n_history_versions: int = 5,
) -> Episode:
    """Generate one synthetic old/new episode.

    ``flip_rate`` is the background symmetric discordance rate (half down, half up).
    ``severity`` is the extra global Delta for global_regression / improvement.
    ``slice_severity`` is the extra Delta inside the target slice.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5157]))
    n = n_items

    # slices with decreasing prevalence (plus an optional rare slice)
    prev = np.array([0.5**0.35 * (0.8**k) for k in range(n_slices)])
    prev = prev / prev.sum()
    slice_names = [f"s{k}" for k in range(n_slices)]
    slices = rng.choice(n_slices, size=n, p=prev)
    slice_arr = np.array(slice_names, dtype=object)[slices]
    if scenario == "rare_severe":
        n_rare = max(1, int(round(rare_prevalence * n)))
        rare_idx = rng.choice(n, size=n_rare, replace=False)
        slice_arr[rare_idx] = "rare"
    slice_offset = rng.normal(0.0, 0.6, size=n_slices)
    z = rng.normal(0.0, 1.0, size=n) + slice_offset[slices]
    logit = 2.2 - 1.6 * z
    old_correct = rng.random(n) < _sigmoid(logit)
    old_loss = (~old_correct).astype(float)
    instability = 4.0 * _sigmoid(logit) * (1 - _sigmoid(logit))  # peaks at boundary
    instability = instability * np.exp(rng.normal(0, 0.3, size=n))

    # public features
    f_old_conf = _sigmoid(np.abs(logit) + rng.normal(0, 0.5, size=n))
    f_difficulty = z + rng.normal(0, 0.5, size=n)
    hist_flips = np.zeros(n)
    hist_err = np.zeros(n)
    for _ in range(n_history_versions):
        flip = rng.random(n) < np.clip(flip_rate * 2.5 * instability, 0, 1)
        prior_correct = np.where(flip, ~old_correct, old_correct)
        hist_flips += flip
        hist_err += ~prior_correct
    hist_flip_rate = hist_flips / max(n_history_versions, 1)
    hist_err_rate = hist_err / max(n_history_versions, 1)

    # --- build exact discordance -------------------------------------------------------------
    idx = np.arange(n)
    oc = idx[old_correct]
    ow = idx[~old_correct]
    w = instability if pattern == "unstable" else np.ones(n)
    k_bg = int(round(flip_rate * n / 2))
    k_bg = min(k_bg, len(ow) // 2, len(oc) // 2)
    new_correct = old_correct.copy()
    target = target_slice or ("rare" if scenario == "rare_severe" else slice_names[2])
    in_target = slice_arr == target

    down: list[int] = []
    up: list[int] = []
    if scenario != "null_identical":
        down.extend(_weighted_pick(rng, oc, w[oc], k_bg))
        up.extend(_weighted_pick(rng, ow, w[ow], k_bg))

    def extra_down(pool_mask: np.ndarray, k: int) -> None:
        cand = np.setdiff1d(idx[pool_mask & old_correct], np.array(down, dtype=int))
        down.extend(_weighted_pick(rng, cand, w[cand], min(k, len(cand))))

    def extra_up(pool_mask: np.ndarray, k: int) -> None:
        cand = np.setdiff1d(idx[pool_mask & ~old_correct], np.array(up, dtype=int))
        up.extend(_weighted_pick(rng, cand, w[cand], min(k, len(cand))))

    everywhere = np.ones(n, dtype=bool)
    n_target = int(in_target.sum())
    if scenario == "global_regression":
        extra_down(everywhere, int(round(severity * n)))
    elif scenario == "improvement":
        extra_up(everywhere, int(round(severity * n)))
    elif scenario in {"slice_regression", "rare_severe"}:
        sev = slice_severity if scenario == "slice_regression" else max(slice_severity, 0.3)
        extra_down(in_target, int(round(sev * n_target)))
    elif scenario == "compensating":
        k = int(round(slice_severity * n_target))
        extra_down(in_target, k)
        extra_up(~in_target, k)

    new_correct[np.array(down, dtype=int)] = False
    new_correct[np.array(up, dtype=int)] = True
    new_loss = (~new_correct).astype(float)

    cost: np.ndarray
    if heterogeneous_cost:
        cost = np.round(np.exp(rng.normal(0.0, 0.8, size=n) + 0.3 * np.abs(z)), 3)
        cost = np.maximum(cost, 0.05)
    else:
        cost = np.ones(n)

    ids = [f"syn{seed}_{i:06d}" for i in range(n)]
    items = pd.DataFrame(
        {
            ID_COL: ids,
            TASK_COL: "synthetic_classification",
            SLICE_COL: slice_arr.astype(str),
            OLD_LOSS_COL: old_loss,
            COST_COL: cost,
            "f_old_conf": f_old_conf,
            "f_difficulty": f_difficulty,
            "hist_flip_rate": hist_flip_rate,
            "hist_err_rate": hist_err_rate,
        }
    )
    hidden = dict(zip(ids, new_loss.tolist(), strict=True))
    meta: dict[str, Any] = {
        "episode_id": f"synthetic-{scenario}-s{seed}",
        "data_source": "synthetic",
        "task": "synthetic_classification",
        "model_family": "synthetic",
        "old_version_id": "synthetic_old",
        "new_version_id": f"synthetic_new_{scenario}",
        "is_synthetic": True,
        "regression_pattern": scenario,
        "flip_pattern": pattern,
        "target_slice": target
        if scenario in {"slice_regression", "rare_severe", "compensating"}
        else None,
        "seed": seed,
        "license": "CC0 (generated)",
        "provenance": "evaldelta.bench.synthetic.generate_episode",
    }
    ep = Episode(meta["episode_id"], items, hidden, meta)
    truth = ep.truth()
    meta["regression_severity"] = truth["delta"]
    return Episode(meta["episode_id"], items, hidden, meta)
