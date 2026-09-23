"""Build DeltaBench source matrices from free, public datasets (CPU).

    python benchmarks/prepare_real.py --sources covtype adult agnews

Each source trains a *release history* of genuinely different model/pipeline versions on the
training split, then caches per-item outcomes on a fixed evaluation pool. Only derived outcomes
(loss / confidence / prediction per item) are stored. Raw inputs are not redistributed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent / "data"
RAW = ROOT / "raw"
POOL_SIZE = 10_000


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _proba(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(x))
        except AttributeError:
            pass
    d = np.asarray(model.decision_function(x))
    if d.ndim == 1:
        d = np.stack([-d, d], 1)
    return _softmax(d)


def _record(
    matrix: dict[str, Any], vid: str, proba: np.ndarray, y: np.ndarray, classes: np.ndarray
) -> None:
    pred = classes[proba.argmax(1)]
    col = {c: i for i, c in enumerate(classes)}
    ptrue = proba[np.arange(len(y)), [col[v] for v in y]]
    matrix[f"loss__{vid}"] = (pred != y).astype(float)
    matrix[f"conf__{vid}"] = proba.max(1)
    matrix[f"ptrue__{vid}"] = ptrue
    matrix[f"pred__{vid}"] = pred.astype(str)


def _write(
    name: str,
    base: pd.DataFrame,
    cols: dict[str, Any],
    versions: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([base.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    df.to_parquet(out / "matrix.parquet", index=False)
    (out / "versions.json").write_text(
        json.dumps({"source": name, "versions": versions, "meta": meta}, indent=2)
    )
    acc = {v["id"]: round(1 - float(cols[f"loss__{v['id']}"].mean()), 4) for v in versions}
    print(f"[{name}] wrote {len(df)} items x {len(versions)} versions; accuracy: {acc}")


def _ladder(
    name: str,
    versions: list[tuple[str, str, str, Callable[[], tuple[Any, Any]]]],
    y_pool: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cols: dict[str, Any] = {}
    vmeta = []
    for k, (vid, change, desc, fit) in enumerate(versions, start=1):
        t0 = time.time()
        model, x_pool = fit()
        proba = _proba(model, x_pool)
        model_classes = np.asarray(getattr(model, "classes_", classes))
        if len(model_classes) != len(classes) or not np.array_equal(model_classes, classes):
            full = np.zeros((len(proba), len(classes)))
            for j, c in enumerate(model_classes):
                full[:, int(np.flatnonzero(classes == c)[0])] = proba[:, j]
            proba = full
        _record(cols, vid, proba, y_pool, classes)
        vmeta.append(
            {
                "id": vid,
                "release_index": k,
                "change_type": change,
                "description": desc,
                "fit_seconds": round(time.time() - t0, 1),
            }
        )
        print(f"  [{name}] {vid} ({change}) {time.time() - t0:.1f}s")
    return cols, vmeta


# ------------------------------------------------------------------------------------------------
# Covertype (UCI; CC BY 4.0), fetched through scikit-learn
# ------------------------------------------------------------------------------------------------


def prepare_covtype() -> None:
    from sklearn.datasets import fetch_covtype

    d = fetch_covtype(data_home=str(RAW), as_frame=True)
    X = d.data.copy()
    y = d.target.to_numpy()
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(X))
    pool_idx, train_idx = perm[:POOL_SIZE], perm[POOL_SIZE:]
    Xp, yp = X.iloc[pool_idx], y[pool_idx]
    wild = X[[c for c in X.columns if c.startswith("Wilderness_Area")]].to_numpy().argmax(1)
    elev = X["Elevation"].to_numpy()
    ebin = np.digitize(elev, np.quantile(elev[train_idx], [1 / 3, 2 / 3]))
    slice_all = np.array([f"wild{w}_elev{e}" for w, e in zip(wild, ebin, strict=True)])

    def tr(n: int, seed: int = 0, mask: np.ndarray | None = None) -> np.ndarray:
        idx = train_idx if mask is None else train_idx[mask[train_idx]]
        r = np.random.default_rng(seed)
        return r.choice(idx, size=min(n, len(idx)), replace=False)

    def hgb(**kw: Any) -> HistGradientBoostingClassifier:
        base = {"max_iter": 200, "learning_rate": 0.1, "random_state": 0, "early_stopping": False}
        base.update(kw)
        return HistGradientBoostingClassifier(**base)

    def fit(
        model: Any,
        idx: np.ndarray,
        cols: list[str] | None = None,
        ytr: np.ndarray | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> tuple[Any, Any]:
        cols = cols or list(X.columns)
        f = transform or (lambda z: z)
        model.fit(f(X.iloc[idx][cols]), y[idx] if ytr is None else ytr)
        return model, f(Xp[cols])

    noisy = tr(100_000, 1)
    y_noisy = y[noisy].copy()
    flip = np.random.default_rng(5).random(len(noisy)) < 0.10
    y_noisy[flip] = np.random.default_rng(6).choice(np.unique(y), size=flip.sum())
    no_road = [c for c in X.columns if c != "Horizontal_Distance_To_Roadways"]

    def coarse(z: pd.DataFrame) -> pd.DataFrame:
        z = z.copy()
        for c in [c for c in z.columns if "Distance" in c]:
            z[c] = (z[c] / 150).round() * 150
        return z

    versions = [
        (
            "v01",
            "architecture",
            "standardised logistic regression, 50k rows",
            lambda: fit(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)), tr(50_000)
            ),
        ),
        (
            "v02",
            "architecture",
            "random forest (100 trees, depth 18), 50k rows",
            lambda: fit(
                RandomForestClassifier(100, max_depth=18, n_jobs=8, random_state=0), tr(50_000)
            ),
        ),
        ("v03", "architecture", "HistGradientBoosting, 50k rows", lambda: fit(hgb(), tr(50_000))),
        ("v04", "data", "HGB retrained on 100k rows", lambda: fit(hgb(), tr(100_000))),
        (
            "v05",
            "feature",
            "HGB without Horizontal_Distance_To_Roadways (feature removed)",
            lambda: fit(hgb(), tr(100_000), no_road),
        ),
        (
            "v06",
            "data",
            "HGB on 100k rows with 10% label noise (bad labelling batch)",
            lambda: fit(hgb(), noisy, ytr=y_noisy),
        ),
        (
            "v07",
            "seed",
            "HGB 100k rows, different sample and seed (near-null)",
            lambda: fit(hgb(random_state=7), tr(100_000, 7)),
        ),
        (
            "v08",
            "data",
            "HGB trained without wilderness area 3 (ingestion filter bug)",
            lambda: fit(hgb(), tr(100_000, 2, mask=(wild != 3))),
        ),
        (
            "v09",
            "hyperparameter",
            "HGB with aggressive schedule: 400 iterations, lr 0.15, 150k rows",
            lambda: fit(hgb(max_iter=400, learning_rate=0.15), tr(150_000, 3)),
        ),
        (
            "v10",
            "compression",
            "HGB compressed: 15 leaves, 100 iterations, 150k rows",
            lambda: fit(hgb(max_leaf_nodes=15, max_iter=100), tr(150_000, 3)),
        ),
        (
            "v11",
            "pipeline",
            "v09 model with distance features coarsened to 150 m (pipeline change)",
            lambda: fit(hgb(max_iter=400, learning_rate=0.15), tr(150_000, 3), transform=coarse),
        ),
        (
            "v12",
            "hyperparameter",
            "v09 schedule with balanced class weights",
            lambda: fit(
                hgb(max_iter=400, learning_rate=0.15, class_weight="balanced"), tr(150_000, 3)
            ),
        ),
    ]
    cols, vmeta = _ladder("covtype", versions, yp, np.unique(y))
    base = pd.DataFrame(
        {
            "sample_id": [f"covtype_{i}" for i in pool_idx],
            "slice": slice_all[pool_idx],
            "label": yp.astype(str),
            "estimated_candidate_cost": 1.0,
            "f_elevation_z": (elev[pool_idx] - elev.mean()) / elev.std(),
        }
    )
    _write(
        "covtype",
        base,
        cols,
        vmeta,
        {
            "task": "tabular_multiclass",
            "model_family": "sklearn_tabular",
            "license": "CC BY 4.0 (UCI Covertype); derived outcomes only",
            "provenance": "sklearn.datasets.fetch_covtype; pool = 10k random rows (seed 2026)",
            "slices": "wilderness area x elevation tercile",
        },
    )


# ------------------------------------------------------------------------------------------------
# Adult (UCI / OpenML; CC BY 4.0)
# ------------------------------------------------------------------------------------------------


def prepare_adult() -> None:
    from sklearn.datasets import fetch_openml

    d = fetch_openml("adult", version=2, as_frame=True, data_home=str(RAW))
    X = d.data.copy()
    y = (d.target.astype(str).str.strip() == ">50K").astype(int).to_numpy()
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(X))
    pool_idx, train_idx = perm[:POOL_SIZE], perm[POOL_SIZE:]
    cat = [c for c in X.columns if str(X[c].dtype) in {"category", "object"}]
    for c in cat:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cat]
    race = X["race"].str.strip()
    race_g = np.where(race == "White", "white", np.where(race == "Black", "black", "other"))
    slice_all = np.array(
        [f"{s.strip().lower()}_{r}" for s, r in zip(X["sex"], race_g, strict=True)]
    )

    def pre(cols: list[str], scale: bool = True) -> ColumnTransformer:
        c_cat = [c for c in cols if c in cat]
        c_num = [c for c in cols if c in num]
        return ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), c_cat),
                ("num", StandardScaler() if scale else "passthrough", c_num),
            ]
        )

    def tr(n: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).choice(
            train_idx, size=min(n, len(train_idx)), replace=False
        )

    def fit(
        model: Any,
        idx: np.ndarray,
        cols: list[str] | None = None,
        ytr: np.ndarray | None = None,
        dense: bool = False,
    ) -> tuple[Any, Any]:
        cols = cols or list(X.columns)
        steps = [pre(cols)]
        if dense:
            from sklearn.preprocessing import FunctionTransformer

            steps.append(
                FunctionTransformer(
                    lambda z: z.toarray() if hasattr(z, "toarray") else z, accept_sparse=True
                )
            )
        pipe = make_pipeline(*steps, model)
        pipe.fit(X.iloc[idx][cols], y[idx] if ytr is None else ytr)
        return pipe, X.iloc[pool_idx][cols]

    def hgb(**kw: Any) -> HistGradientBoostingClassifier:
        base = {"max_iter": 200, "random_state": 0, "early_stopping": False}
        base.update(kw)
        return HistGradientBoostingClassifier(**base)

    noisy = tr(30_000, 4)
    y_noisy = y[noisy].copy()
    flip = np.random.default_rng(9).random(len(noisy)) < 0.1
    y_noisy[flip] = 1 - y_noisy[flip]
    no_protected = [c for c in X.columns if c not in {"sex", "race"}]
    no_rel = [c for c in X.columns if c not in {"relationship", "marital-status"}]
    versions = [
        (
            "v01",
            "architecture",
            "logistic regression, 15k rows",
            lambda: fit(LogisticRegression(max_iter=500), tr(15_000)),
        ),
        (
            "v02",
            "architecture",
            "random forest (200 trees), 15k rows",
            lambda: fit(
                RandomForestClassifier(200, min_samples_leaf=3, n_jobs=8, random_state=0),
                tr(15_000),
            ),
        ),
        (
            "v03",
            "architecture",
            "HistGradientBoosting, 15k rows",
            lambda: fit(hgb(), tr(15_000), dense=True),
        ),
        ("v04", "data", "HGB, 30k rows", lambda: fit(hgb(), tr(30_000), dense=True)),
        (
            "v05",
            "feature",
            "HGB without sex and race (fairness-motivated feature removal)",
            lambda: fit(hgb(), tr(30_000), no_protected, dense=True),
        ),
        (
            "v06",
            "seed",
            "HGB, different sample and seed (near-null)",
            lambda: fit(hgb(random_state=3), tr(30_000, 3), dense=True),
        ),
        (
            "v07",
            "data",
            "HGB with 10% flipped labels",
            lambda: fit(hgb(), noisy, ytr=y_noisy, dense=True),
        ),
        (
            "v08",
            "feature",
            "HGB without relationship / marital-status",
            lambda: fit(hgb(), tr(30_000), no_rel, dense=True),
        ),
        (
            "v09",
            "hyperparameter",
            "HGB tuned (lr 0.05, 500 iterations, l2 1.0)",
            lambda: fit(
                hgb(learning_rate=0.05, max_iter=500, l2_regularization=1.0),
                tr(30_000, 5),
                dense=True,
            ),
        ),
        (
            "v10",
            "hyperparameter",
            "v09 schedule with balanced class weights (threshold shift)",
            lambda: fit(
                hgb(
                    learning_rate=0.05, max_iter=500, l2_regularization=1.0, class_weight="balanced"
                ),
                tr(30_000, 5),
                dense=True,
            ),
        ),
        (
            "v11",
            "compression",
            "HGB compressed (8 leaves, 60 iterations)",
            lambda: fit(hgb(max_leaf_nodes=8, max_iter=60), tr(30_000, 5), dense=True),
        ),
        (
            "v12",
            "architecture",
            "logistic regression with interactions-free L1 (C=0.05)",
            lambda: fit(
                LogisticRegression(penalty="l1", C=0.05, solver="liblinear"), tr(30_000, 5)
            ),
        ),
    ]
    cols, vmeta = _ladder("adult", versions, y[pool_idx], np.array([0, 1]))
    age = X["age"].astype(float).to_numpy()
    base = pd.DataFrame(
        {
            "sample_id": [f"adult_{i}" for i in pool_idx],
            "slice": slice_all[pool_idx],
            "label": y[pool_idx].astype(str),
            "estimated_candidate_cost": 1.0,
            "f_age_z": (age[pool_idx] - age.mean()) / age.std(),
        }
    )
    _write(
        "adult",
        base,
        cols,
        vmeta,
        {
            "task": "tabular_binary",
            "model_family": "sklearn_tabular",
            "license": "CC BY 4.0 (UCI Adult via OpenML); derived outcomes only",
            "provenance": "sklearn.datasets.fetch_openml('adult', version=2); pool = 10k rows",
            "slices": "sex x race group (protected attributes used only for slicing)",
        },
    )


# ------------------------------------------------------------------------------------------------
# Bank Marketing (UCI / OpenML; CC BY 4.0) — 4th source family, added to tighten source-level CIs
# ------------------------------------------------------------------------------------------------


def prepare_bank() -> None:
    from sklearn.datasets import fetch_openml

    d = fetch_openml("bank-marketing", version=1, as_frame=True, data_home=str(RAW))
    X = d.data.copy()
    y = (d.target.astype(str).str.strip().isin({"2", "yes"})).astype(int).to_numpy()
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(X))
    pool_idx, train_idx = perm[:POOL_SIZE], perm[POOL_SIZE:]
    cat = [c for c in X.columns if str(X[c].dtype) in {"category", "object"}]
    for c in cat:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cat]
    job_col = next((c for c in X.columns if c.lower() in {"job", "v2"}), cat[0])
    age_col = next((c for c in X.columns if c.lower() in {"age", "v1"}), num[0])
    age_all = pd.to_numeric(X[age_col], errors="coerce").fillna(X[age_col].median()).to_numpy()
    age_bin = np.where(age_all < 35, "young", np.where(age_all < 60, "mid", "senior"))
    slice_all = np.array([f"{j}_{a}" for j, a in zip(X[job_col].astype(str), age_bin, strict=True)])

    def pre(cols: list[str], scale: bool = True) -> ColumnTransformer:
        c_cat = [c for c in cols if c in cat]
        c_num = [c for c in cols if c in num]
        return ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), c_cat),
                ("num", StandardScaler() if scale else "passthrough", c_num),
            ]
        )

    def tr(n: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).choice(
            train_idx, size=min(n, len(train_idx)), replace=False
        )

    def fit(
        model: Any,
        idx: np.ndarray,
        cols: list[str] | None = None,
        ytr: np.ndarray | None = None,
        dense: bool = False,
    ) -> tuple[Any, Any]:
        cols = cols or list(X.columns)
        steps = [pre(cols)]
        if dense:
            from sklearn.preprocessing import FunctionTransformer

            steps.append(
                FunctionTransformer(
                    lambda z: z.toarray() if hasattr(z, "toarray") else z, accept_sparse=True
                )
            )
        pipe = make_pipeline(*steps, model)
        pipe.fit(X.iloc[idx][cols], y[idx] if ytr is None else ytr)
        return pipe, X.iloc[pool_idx][cols]

    def hgb(**kw: Any) -> HistGradientBoostingClassifier:
        base = {"max_iter": 200, "random_state": 0, "early_stopping": False}
        base.update(kw)
        return HistGradientBoostingClassifier(**base)

    duration_col = next((c for c in X.columns if c.lower() in {"duration", "v12"}), None)
    no_duration = [c for c in X.columns if c != duration_col] if duration_col else list(X.columns)
    contact_col = next((c for c in X.columns if c.lower() in {"contact", "v8"}), None)
    no_contact = [c for c in X.columns if c != contact_col] if contact_col else list(X.columns)
    noisy = tr(30_000, 4)
    y_noisy = y[noisy].copy()
    flip = np.random.default_rng(9).random(len(noisy)) < 0.10
    y_noisy[flip] = 1 - y_noisy[flip]

    versions = [
        (
            "v01",
            "architecture",
            "logistic regression, 15k rows",
            lambda: fit(LogisticRegression(max_iter=500), tr(15_000)),
        ),
        (
            "v02",
            "architecture",
            "random forest (200 trees), 15k rows",
            lambda: fit(
                RandomForestClassifier(200, min_samples_leaf=3, n_jobs=8, random_state=0),
                tr(15_000),
            ),
        ),
        (
            "v03",
            "architecture",
            "HistGradientBoosting, 15k rows",
            lambda: fit(hgb(), tr(15_000), dense=True),
        ),
        ("v04", "data", "HGB, 30k rows", lambda: fit(hgb(), tr(30_000), dense=True)),
        (
            "v05",
            "feature",
            "HGB without call-duration feature (leakage-motivated removal)",
            lambda: fit(hgb(), tr(30_000), no_duration, dense=True),
        ),
        (
            "v06",
            "seed",
            "HGB, different sample and seed (near-null)",
            lambda: fit(hgb(random_state=3), tr(30_000, 3), dense=True),
        ),
        (
            "v07",
            "data",
            "HGB with 10% flipped labels",
            lambda: fit(hgb(), noisy, ytr=y_noisy, dense=True),
        ),
        (
            "v08",
            "feature",
            "HGB without contact-method feature",
            lambda: fit(hgb(), tr(30_000), no_contact, dense=True),
        ),
        (
            "v09",
            "hyperparameter",
            "HGB tuned (lr 0.05, 500 iterations, l2 1.0)",
            lambda: fit(
                hgb(learning_rate=0.05, max_iter=500, l2_regularization=1.0),
                tr(30_000, 5),
                dense=True,
            ),
        ),
        (
            "v10",
            "hyperparameter",
            "v09 schedule with balanced class weights (threshold shift)",
            lambda: fit(
                hgb(
                    learning_rate=0.05, max_iter=500, l2_regularization=1.0, class_weight="balanced"
                ),
                tr(30_000, 5),
                dense=True,
            ),
        ),
        (
            "v11",
            "compression",
            "HGB compressed (8 leaves, 60 iterations)",
            lambda: fit(hgb(max_leaf_nodes=8, max_iter=60), tr(30_000, 5), dense=True),
        ),
        (
            "v12",
            "architecture",
            "logistic regression with L1 penalty (C=0.05)",
            lambda: fit(
                LogisticRegression(penalty="l1", C=0.05, solver="liblinear"), tr(30_000, 5)
            ),
        ),
    ]
    cols, vmeta = _ladder("bank", versions, y[pool_idx], np.array([0, 1]))
    base = pd.DataFrame(
        {
            "sample_id": [f"bank_{i}" for i in pool_idx],
            "slice": slice_all[pool_idx],
            "label": y[pool_idx].astype(str),
            "estimated_candidate_cost": 1.0,
            "f_age_z": (age_all[pool_idx] - age_all.mean()) / age_all.std(),
        }
    )
    _write(
        "bank",
        base,
        cols,
        vmeta,
        {
            "task": "tabular_binary",
            "model_family": "sklearn_tabular",
            "license": "CC BY 4.0 (UCI Bank Marketing via OpenML); derived outcomes only",
            "provenance": "sklearn.datasets.fetch_openml('bank-marketing', version=1); pool = 10k rows",
            "slices": "job x age band",
        },
    )


# ------------------------------------------------------------------------------------------------
# AG News (text classification), Hugging Face parquet mirror
# ------------------------------------------------------------------------------------------------

AG_URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{split}-00000-of-00001.parquet"


def _agnews(split: str) -> pd.DataFrame:
    path = RAW / f"ag_news_{split}.parquet"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AG_URL.format(split=split), path)
    return pd.read_parquet(path)


def prepare_agnews() -> None:
    train = _agnews("train")
    test = _agnews("test")
    xt, yt = train["text"].to_numpy(), train["label"].to_numpy()
    xp, yp = test["text"].to_numpy(), test["label"].to_numpy()
    rng = np.random.default_rng(2026)
    n_tok = np.array([len(t.split()) for t in xp])
    med = np.median([len(t.split()) for t in xt])
    names = np.array(["world", "sports", "business", "scitech"])
    slice_all = np.array(
        [f"{names[c]}_{'long' if n > med else 'short'}" for c, n in zip(yp, n_tok, strict=True)]
    )

    def sub(frac: float, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.choice(len(xt), size=int(frac * len(xt)), replace=False)

    all_idx = np.arange(len(xt))

    def fit(
        vec: Any,
        clf: Any,
        idx: np.ndarray = all_idx,
        ytr: np.ndarray | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[Any, Any]:
        f = transform or (lambda z: z)
        pipe = make_pipeline(vec, clf)
        pipe.fit(f(xt[idx]), yt[idx] if ytr is None else ytr)
        return pipe, f(xp)

    def lr(C: float = 4.0) -> LogisticRegression:
        return LogisticRegression(C=C, max_iter=300)

    def word(**kw: Any) -> TfidfVectorizer:
        base: dict[str, Any] = {
            "ngram_range": (1, 2),
            "max_features": 200_000,
            "sublinear_tf": True,
            "min_df": 2,
        }
        base.update(kw)
        return TfidfVectorizer(**base)

    noisy = sub(1.0, 11)
    y_noisy = yt[noisy].copy()
    flip = rng.random(len(noisy)) < 0.10
    y_noisy[flip] = rng.integers(0, 4, flip.sum())

    def trunc(z: np.ndarray) -> np.ndarray:
        return np.array([" ".join(t.split()[:20]) for t in z])

    versions = [
        (
            "v01",
            "architecture",
            "unigram TF-IDF (20k) + multinomial naive Bayes",
            lambda: fit(TfidfVectorizer(max_features=20_000), MultinomialNB()),
        ),
        (
            "v02",
            "architecture",
            "uni+bigram TF-IDF + logistic regression (C=1)",
            lambda: fit(word(), lr(1.0)),
        ),
        (
            "v03",
            "hyperparameter",
            "uni+bigram TF-IDF + logistic regression (C=4)",
            lambda: fit(word(), lr(4.0)),
        ),
        ("v04", "data", "v03 retrained on a 30% subsample", lambda: fit(word(), lr(), sub(0.3, 1))),
        (
            "v05",
            "architecture",
            "character 3-5-gram TF-IDF + logistic regression",
            lambda: fit(
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=200_000,
                    sublinear_tf=True,
                    min_df=3,
                ),
                lr(),
            ),
        ),
        ("v06", "pipeline", "v03 without lowercasing", lambda: fit(word(lowercase=False), lr())),
        (
            "v07",
            "pipeline",
            "v03 with English stop-word removal",
            lambda: fit(word(stop_words="english"), lr()),
        ),
        (
            "v08",
            "seed",
            "v03 on a different 90% subsample (near-null)",
            lambda: fit(word(), lr(), sub(0.9, 8)),
        ),
        (
            "v09",
            "architecture",
            "hashing vectoriser (2^18) + SGD logistic",
            lambda: fit(
                HashingVectorizer(n_features=2**18, ngram_range=(1, 2), alternate_sign=False),
                SGDClassifier(loss="log_loss", alpha=2e-6, random_state=0),
            ),
        ),
        (
            "v10",
            "compression",
            "v03 with vocabulary compressed to 5k features",
            lambda: fit(word(max_features=5_000), lr()),
        ),
        (
            "v11",
            "data",
            "v03 trained with 10% label noise",
            lambda: fit(word(), lr(), noisy, ytr=y_noisy),
        ),
        (
            "v12",
            "pipeline",
            "v03 with inputs truncated to 20 tokens (truncation bug)",
            lambda: fit(word(), lr(), transform=trunc),
        ),
    ]
    cols, vmeta = _ladder("agnews", versions, yp, np.arange(4))
    base = pd.DataFrame(
        {
            "sample_id": [f"agnews_test_{i}" for i in range(len(xp))],
            "slice": slice_all,
            "label": yp.astype(str),
            "estimated_candidate_cost": np.maximum(n_tok, 1).astype(float) / float(np.mean(n_tok)),
            "f_log_len": np.log1p(n_tok),
        }
    )
    _write(
        "agnews",
        base,
        cols,
        vmeta,
        {
            "task": "text_topic_classification",
            "model_family": "sklearn_text",
            "license": "AG News: academic/non-commercial research use; derived outcomes only",
            "provenance": "huggingface.co/datasets/fancyzhx/ag_news (test split = pool, 7,600 items)",
            "cost_unit_note": "estimated_candidate_cost = token count / mean token count",
            "slices": "class x length (above/below median)",
        },
    )


PREPARERS = {
    "covtype": prepare_covtype,
    "adult": prepare_adult,
    "agnews": prepare_agnews,
    "bank": prepare_bank,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(PREPARERS))
    args = ap.parse_args()
    for s in args.sources:
        t0 = time.time()
        PREPARERS[s]()
        print(f"== {s} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
