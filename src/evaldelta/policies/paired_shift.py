"""PairedShift: interpretable paired-discordance discovery policy (spec §5).

For each unqueried discovery item it estimates

    p_down(i) = P(new wrong | old correct, allowed features)   (negative flip)
    p_up(i)   = P(new correct | old wrong, allowed features)   (positive flip)

with small L2-regularised logistic models trained on **historical version-pair episodes only**.
Online, a per-episode intercept and per-slice offsets are refit (MAP, Gaussian priors) on
already-purchased discovery outcomes at fixed increments. The acquisition score is the
specification's heuristic:

    score_i = [max(0, p_down - p_up) + lam * sqrt(p_down + p_up) + eta * undercoverage(slice_i)]
              / max(cost_i, cost_floor)

where p_down applies only to old-correct items and p_up only to old-wrong items. Selection takes
the top scores subject to a per-slice cap per batch, plus a random-exploration fraction.

This is a heuristic, not a new algorithm. See docs/NOVELTY_AUDIT.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from evaldelta.data.io import COST_COL, ID_COL, OLD_LOSS_COL, SLICE_COL
from evaldelta.policies.base import PolicyView

FEATURES = (
    "logit_old_conf",
    "old_ptrue",
    "hist_flip_rate",
    "hist_err_rate",
    "hist_available",
    "slice_hist_flip",
    "slice_old_err",
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out: np.ndarray = np.log(p / (1 - p))
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return out


def feature_matrix(items: pd.DataFrame) -> np.ndarray:
    """Source-agnostic features computed from *public* columns only."""
    n = len(items)
    old = items[OLD_LOSS_COL].to_numpy(dtype=float)
    conf = items["f_old_conf"].to_numpy(dtype=float) if "f_old_conf" in items else np.full(n, 0.8)
    ptrue = items["f_old_ptrue"].to_numpy(dtype=float) if "f_old_ptrue" in items else (1.0 - old)
    has_hist = (
        "hist_flip_rate" in items and float(items.get("hist_n_versions", pd.Series([2])).max()) >= 2
    )
    flip = (
        items["hist_flip_rate"].to_numpy(dtype=float) if "hist_flip_rate" in items else np.zeros(n)
    )
    err = items["hist_err_rate"].to_numpy(dtype=float) if "hist_err_rate" in items else old.copy()
    if not has_hist:
        flip = np.zeros(n)
    sl = items[SLICE_COL].astype(str)
    slice_flip = pd.Series(flip).groupby(sl.to_numpy()).transform("mean").to_numpy()
    slice_err = pd.Series(old).groupby(sl.to_numpy()).transform("mean").to_numpy()
    return np.column_stack(
        [
            _logit(conf),
            ptrue,
            flip,
            err,
            np.full(n, 1.0 if has_hist else 0.0),
            slice_flip,
            slice_err,
        ]
    )


@dataclass
class LogisticModel:
    """Standardised L2 logistic regression with explicit coefficients (JSON-serialisable)."""

    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)

    def logits(self, x: np.ndarray) -> np.ndarray:
        if not self.coef:
            return np.full(len(x), self.intercept)
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        out: np.ndarray = z @ np.asarray(self.coef) + self.intercept
        return out

    @staticmethod
    def fit(
        x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, l2: float = 1.0
    ) -> LogisticModel:
        from sklearn.linear_model import LogisticRegression

        mean = x.mean(0)
        scale = x.std(0) + 1e-6
        z = (x - mean) / scale
        if len(np.unique(y)) < 2:
            p = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            return LogisticModel(
                [0.0] * x.shape[1], math.log(p / (1 - p)), mean.tolist(), scale.tolist()
            )
        clf = LogisticRegression(C=1.0 / l2, max_iter=1000)
        clf.fit(z, y, sample_weight=w)
        return LogisticModel(
            clf.coef_[0].tolist(), float(clf.intercept_[0]), mean.tolist(), scale.tolist()
        )


@dataclass
class PairedShiftModel:
    """Historical p_down / p_up models."""

    down: LogisticModel
    up: LogisticModel
    trained_on: list[str] = field(default_factory=list)
    features: tuple[str, ...] = FEATURES

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "down": self.down.__dict__,
                    "up": self.up.__dict__,
                    "trained_on": self.trained_on,
                    "features": list(self.features),
                },
                indent=2,
            )
        )

    @staticmethod
    def load(path: str | Path) -> PairedShiftModel:
        d = json.loads(Path(path).read_text())
        if tuple(d["features"]) != FEATURES:
            raise ValueError("feature set of saved PairedShift model does not match this version")
        return PairedShiftModel(
            LogisticModel(**d["down"]), LogisticModel(**d["up"]), d.get("trained_on", [])
        )

    @staticmethod
    def default() -> PairedShiftModel:
        """Weak prior used when no historical episodes are available.

        Flips are more likely for low-confidence and historically unstable items.
        """
        k = len(FEATURES)
        mean = [0.0] * k
        scale = [1.0] * k
        coef_down = [-0.6, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0]
        coef_up = [0.3, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0]
        return PairedShiftModel(
            LogisticModel(coef_down, -3.2, mean, scale),
            LogisticModel(coef_up, -1.8, mean, scale),
            ["<default prior>"],
        )

    @staticmethod
    def train(
        episodes: list[tuple[str, pd.DataFrame, np.ndarray]], l2: float = 1.0
    ) -> PairedShiftModel:
        """Train from historical episodes: (episode_id, public items, candidate losses).

        Each episode gets equal total weight so large pools do not dominate.
        """
        xd, yd, wd, xu, yu, wu = [], [], [], [], [], []
        for _, items, new_loss in episodes:
            x = feature_matrix(items)
            old = items[OLD_LOSS_COL].to_numpy(dtype=float)
            oc, ow = old == 0, old == 1
            xd.append(x[oc])
            yd.append((new_loss[oc] == 1).astype(int))
            wd.append(np.full(oc.sum(), 1.0 / max(oc.sum(), 1)))
            xu.append(x[ow])
            yu.append((new_loss[ow] == 0).astype(int))
            wu.append(np.full(ow.sum(), 1.0 / max(ow.sum(), 1)))
        down = LogisticModel.fit(np.vstack(xd), np.concatenate(yd), np.concatenate(wd) * 1000, l2)
        up = LogisticModel.fit(np.vstack(xu), np.concatenate(yu), np.concatenate(wu) * 1000, l2)
        return PairedShiftModel(down, up, [e[0] for e in episodes])


def _fit_offsets(
    base: np.ndarray,
    y: np.ndarray,
    slice_idx: np.ndarray,
    n_slices: int,
    prior_a: float,
    prior_b: float,
) -> np.ndarray:
    """MAP intercept + slice offsets for a logistic model with fixed base logits."""
    if len(y) == 0:
        return np.zeros(n_slices + 1)

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = base + theta[0] + theta[1:][slice_idx]
        p = _sigmoid(z)
        nll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        nll += theta[0] ** 2 / (2 * prior_a**2) + np.sum(theta[1:] ** 2) / (2 * prior_b**2)
        r = p - y
        g = np.empty_like(theta)
        g[0] = r.sum() + theta[0] / prior_a**2
        g[1:] = np.bincount(slice_idx, weights=r, minlength=n_slices) + theta[1:] / prior_b**2
        return float(nll), g

    res = minimize(f, np.zeros(n_slices + 1), jac=True, method="L-BFGS-B")
    out: np.ndarray = res.x
    return out


class PairedShift:
    """Discovery policy (OURS-1). Stateless across runs; per-run state is rebuilt from the view."""

    name = "paired_shift"

    def __init__(
        self,
        model_path: str | None = None,
        lam: float = 0.5,
        eta: float = 0.1,
        explore_frac: float = 0.2,
        slice_cap: float = 0.5,
        refit_every: int = 25,
        cost_floor: float = 0.1,
        prior_a: float = 1.0,
        prior_b: float = 0.7,
        min_slice_size: int = 90,
        min_slice_revealed: int = 5,
        slice_rank: str = "model_assisted",
    ) -> None:
        if slice_rank not in {"model_assisted", "residual_z"}:
            raise ValueError("slice_rank must be 'model_assisted' or 'residual_z'")
        self.slice_rank = slice_rank
        self.model = PairedShiftModel.load(model_path) if model_path else PairedShiftModel.default()
        self.lam = lam
        self.eta = eta
        self.explore_frac = explore_frac
        self.slice_cap = slice_cap
        self.refit_every = refit_every
        self.cost_floor = cost_floor
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.min_slice_size = min_slice_size
        self.min_slice_revealed = min_slice_revealed
        self._cache_key: tuple[Any, ...] | None = None
        self._cache: dict[str, Any] = {}

    # -- probability estimates -----------------------------------------------------------------
    def probabilities(self, view: PolicyView) -> pd.DataFrame:
        """p_down / p_up for all discovery items, recalibrated on purchased outcomes."""
        items = view.items
        n_rev = len(view.revealed)
        step = (n_rev // self.refit_every) * self.refit_every  # refit only at fixed increments
        rev_prefix = tuple(list(view.revealed.keys())[:step])
        key = (
            len(items),
            str(items[ID_COL].iloc[0]),
            str(items[ID_COL].iloc[-1]),
            step,
            hash(rev_prefix),
        )
        if self._cache_key == key:
            return self._cache["probs"]  # type: ignore[no-any-return]
        x = feature_matrix(items)
        base_d = self.model.down.logits(x)
        base_u = self.model.up.logits(x)
        slices = items[SLICE_COL].astype(str).to_numpy()
        names, sidx = np.unique(slices, return_inverse=True)
        old = items[OLD_LOSS_COL].to_numpy(dtype=float)
        ids = items[ID_COL].to_numpy()
        # use only the first `step` purchased outcomes (in purchase order) -> fixed increments
        rev_ids = list(view.revealed.keys())[:step]
        pos = {sid: i for i, sid in enumerate(ids)}
        ri = np.array([pos[s] for s in rev_ids], dtype=int)
        new = np.array([view.revealed[s] for s in rev_ids], dtype=float)
        mask_d = old[ri] == 0 if len(ri) else np.zeros(0, dtype=bool)
        th_d = _fit_offsets(
            base_d[ri][mask_d],
            (new[mask_d] == 1).astype(float),
            sidx[ri][mask_d],
            len(names),
            self.prior_a,
            self.prior_b,
        )
        th_u = _fit_offsets(
            base_u[ri][~mask_d],
            (new[~mask_d] == 0).astype(float),
            sidx[ri][~mask_d],
            len(names),
            self.prior_a,
            self.prior_b,
        )
        p_down = np.where(old == 0, _sigmoid(base_d + th_d[0] + th_d[1:][sidx]), 0.0)
        p_up = np.where(old == 1, _sigmoid(base_u + th_u[0] + th_u[1:][sidx]), 0.0)
        probs = pd.DataFrame(
            {ID_COL: ids, SLICE_COL: slices, "p_down": p_down, "p_up": p_up, OLD_LOSS_COL: old}
        )
        probs["slice_offset_down"] = th_d[1:][sidx]
        self._cache_key, self._cache = key, {"probs": probs}
        return probs

    # -- acquisition ---------------------------------------------------------------------------
    def select(self, view: PolicyView, k: int, rng: np.random.Generator) -> list[str]:
        un = view.unqueried
        if un.empty or k <= 0:
            return []
        probs = self.probabilities(view).set_index(ID_COL)
        cand = probs.loc[un[ID_COL].to_numpy()]
        cost = view.items.set_index(ID_COL).loc[cand.index, COST_COL].to_numpy(dtype=float)
        # undercoverage bonus relative to proportional allocation across slices
        share = view.items[SLICE_COL].value_counts(normalize=True)
        rev_sl = (
            view.items.set_index(ID_COL).loc[list(view.revealed.keys()), SLICE_COL]
            if (view.revealed)
            else pd.Series(dtype=str)
        )
        got = rev_sl.value_counts().reindex(share.index, fill_value=0)
        expected = share * max(len(view.revealed), 1)
        under = (1 - got / expected).clip(lower=0).reindex(cand[SLICE_COL]).to_numpy()
        pd_, pu = cand["p_down"].to_numpy(), cand["p_up"].to_numpy()
        score = (
            np.maximum(0, pd_ - pu) + self.lam * np.sqrt(np.maximum(0, pd_ + pu)) + self.eta * under
        ) / np.maximum(cost, self.cost_floor)
        score = score + 1e-9 * rng.random(len(score))  # random tie-break
        ids = cand.index.to_numpy()
        slices = cand[SLICE_COL].to_numpy()

        n_explore = int(round(self.explore_frac * k))
        chosen: list[str] = []
        taken: set[str] = set()
        if n_explore:
            for j in rng.choice(len(ids), size=min(n_explore, len(ids)), replace=False):
                chosen.append(str(ids[j]))
                taken.add(str(ids[j]))
        cap = max(1, math.ceil(self.slice_cap * k))
        per_slice: dict[str, int] = {}
        for j in np.argsort(-score):
            if len(chosen) >= k:
                break
            sid = str(ids[j])
            if sid in taken:
                continue
            s = str(slices[j])
            if per_slice.get(s, 0) >= cap:
                continue
            per_slice[s] = per_slice.get(s, 0) + 1
            chosen.append(sid)
            taken.add(sid)
        if len(chosen) < k:  # cap too tight: fill by score
            for j in np.argsort(-score):
                if len(chosen) >= k:
                    break
                if str(ids[j]) not in taken:
                    chosen.append(str(ids[j]))
                    taken.add(str(ids[j]))
        return chosen[:k]

    # -- slice triage --------------------------------------------------------------------------
    def slice_estimates(self, view: PolicyView) -> pd.DataFrame:
        """Model-assisted exploratory slice regression estimates over the discovery pool."""
        probs = self.probabilities(view)
        g = (probs["p_down"] - probs["p_up"]).to_numpy()
        rev = view.revealed
        ids = probs[ID_COL].to_numpy()
        is_rev = np.array([s in rev for s in ids])
        d = g.copy()
        if is_rev.any():
            d[is_rev] = (
                np.array([rev[s] for s in ids[is_rev]]) - probs[OLD_LOSS_COL].to_numpy()[is_rev]
            )
        df = pd.DataFrame(
            {SLICE_COL: probs[SLICE_COL], "d": d, "rev": is_rev, "down": is_rev & (d > 0)}
        )
        out = df.groupby(SLICE_COL).agg(
            n=("d", "size"), est=("d", "mean"), revealed=("rev", "sum"), downs=("down", "sum")
        )
        # baseline: what the historical model alone would predict for this slice
        x = feature_matrix(view.items)
        old = view.items[OLD_LOSS_COL].to_numpy(dtype=float)
        g0 = np.where(old == 0, _sigmoid(self.model.down.logits(x)), 0) - np.where(
            old == 1, _sigmoid(self.model.up.logits(x)), 0
        )
        out["baseline"] = pd.Series(g0).groupby(view.items[SLICE_COL].astype(str).to_numpy()).mean()
        out["excess"] = out["est"] - out["baseline"]
        return out

    def residual_scores(self, view: PolicyView) -> pd.DataFrame:
        """Observed-vs-expected negative flips per slice among *purchased* items.

        z_g = sum_{i in revealed, slice g} (d_i - g0_i) / sqrt(sum var0_i), where g0 and var0 come
        from the historical model *without* episode recalibration. Conditioning on each purchased
        item's own features removes the bias from PairedShift's preferential sampling of
        unstable items.
        """
        rev = view.revealed_frame()
        if rev.empty:
            return pd.DataFrame(columns=["z", "revealed", "downs", "n"])
        x = feature_matrix(view.items)
        old_all = view.items[OLD_LOSS_COL].to_numpy(dtype=float)
        pd0 = np.where(old_all == 0, _sigmoid(self.model.down.logits(x)), 0.0)
        pu0 = np.where(old_all == 1, _sigmoid(self.model.up.logits(x)), 0.0)
        g0 = pd.Series(pd0 - pu0, index=view.items[ID_COL].to_numpy())
        v0 = pd.Series(pd0 + pu0 - (pd0 - pu0) ** 2, index=view.items[ID_COL].to_numpy())
        rev = rev.assign(
            d=rev["cand_loss"] - rev[OLD_LOSS_COL],
            g0=g0.loc[rev[ID_COL]].to_numpy(),
            v0=v0.loc[rev[ID_COL]].to_numpy(),
        )
        rev["resid"] = rev["d"] - rev["g0"]
        agg = rev.groupby(SLICE_COL).agg(
            resid=("resid", "sum"),
            var=("v0", "sum"),
            revealed=("d", "size"),
            downs=("d", lambda z: (z > 0).sum()),
        )
        agg["z"] = agg["resid"] / np.sqrt(agg["var"] + 1.0)
        agg["n"] = view.items[SLICE_COL].astype(str).value_counts().reindex(agg.index)
        return agg

    def suspicious_slices(self, view: PolicyView, max_slices: int) -> list[str]:
        if max_slices == 0 or not view.revealed:
            return []
        if self.slice_rank == "residual_z":
            agg = self.residual_scores(view)
            agg = agg[
                (agg["n"] >= self.min_slice_size)
                & (agg["revealed"] >= self.min_slice_revealed)
                & (agg["downs"] >= 2)
                & (agg["z"] > 0)
            ]
            return [str(s) for s in agg.sort_values("z", ascending=False).index[:max_slices]]
        est = self.slice_estimates(view)
        est = est[
            (est["n"] >= self.min_slice_size)
            & (est["revealed"] >= self.min_slice_revealed)
            & (est["downs"] >= 2)
            & (est["est"] > 0)
        ]
        est = est.sort_values(["est", "excess"], ascending=False)
        return [str(s) for s in est.index[:max_slices]]
