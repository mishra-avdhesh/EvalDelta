"""Prepare the predeclared UCI MAGIC external-source transfer case study.

Read docs/EXTERNAL_CASE_STUDY.md before changing the split, ladder, or pair set.
This writes only ignored local row-level data; the public repository contains code and aggregates.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "magic_gamma_telescope.zip"
OUT = ROOT / "data" / "magic"
URL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
SEED = 20260923
POOL_SIZE = 10_000
FEATURES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]


def source_zip() -> bytes:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise ValueError("Downloaded UCI archive is not a ZIP file")
        RAW.write_bytes(data)
    data = RAW.read_bytes()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError("Cached UCI archive is not a ZIP file")
    return data


def prepare() -> None:
    data = source_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = [n for n in archive.namelist() if n.endswith("magic04.data")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one magic04.data file, found {matches}")
        raw = pd.read_csv(archive.open(matches[0]), header=None)
    if raw.shape != (19_020, 11):
        raise ValueError(f"Unexpected UCI table shape: {raw.shape}")
    x = raw.iloc[:, :10].apply(pd.to_numeric, errors="raise")
    x.columns = FEATURES
    if x.isna().any().any():
        raise ValueError("MAGIC feature table has missing values")
    labels = raw.iloc[:, 10].astype(str).str.strip()
    if set(labels) != {"g", "h"}:
        raise ValueError(f"Unexpected MAGIC classes: {set(labels)}")
    y = labels.eq("g").to_numpy(dtype=int)
    perm = np.random.default_rng(SEED).permutation(len(raw))
    pool, train = perm[:POOL_SIZE], perm[POOL_SIZE:]
    x_train, y_train = x.iloc[train], y[train]
    x_pool, y_pool = x.iloc[pool], y[pool]
    median = float(x_train["fLength"].median())
    base = pd.DataFrame(
        {
            "sample_id": [f"magic_{i}" for i in pool],
            "slice": np.where(x_pool["fLength"].to_numpy() <= median, "short", "long"),
            "label": y_pool,
            "estimated_candidate_cost": np.ones(len(pool)),
        }
    )
    # These are public metadata; the candidate's loss/probability stays in the oracle.
    for name in FEATURES:
        col = x_pool[name].to_numpy(dtype=float)
        center = float(x_train[name].median())
        scale = max(float(x_train[name].quantile(0.75) - x_train[name].quantile(0.25)), 1e-6)
        base[f"f_{name}"] = (col - center) / scale

    versions = [
        (
            "v01",
            "architecture",
            "standardised logistic regression C=1",
            make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000)),
            FEATURES,
        ),
        (
            "v02",
            "architecture",
            "HistGradientBoosting, 100 iterations, 31 leaves",
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=31, random_state=0, early_stopping=False
            ),
            FEATURES,
        ),
        (
            "v03",
            "compression",
            "compressed HGB, 50 iterations, 7 leaves",
            HistGradientBoostingClassifier(
                max_iter=50, max_leaf_nodes=7, random_state=0, early_stopping=False
            ),
            FEATURES,
        ),
        (
            "v04",
            "feature",
            "v02 HGB architecture without first feature fLength",
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=31, random_state=0, early_stopping=False
            ),
            FEATURES[1:],
        ),
    ]
    columns: dict[str, np.ndarray] = {}
    version_meta = []
    for index, (vid, change, description, model, used) in enumerate(versions, start=1):
        model.fit(x_train[used], y_train)
        proba = model.predict_proba(x_pool[used])
        pred = proba.argmax(1)
        columns[f"loss__{vid}"] = (pred != y_pool).astype(float)
        columns[f"conf__{vid}"] = proba.max(1)
        columns[f"ptrue__{vid}"] = proba[np.arange(len(y_pool)), y_pool]
        columns[f"pred__{vid}"] = pred.astype(str)
        version_meta.append(
            {"id": vid, "release_index": index, "change_type": change, "description": description}
        )
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.concat([base, pd.DataFrame(columns)], axis=1)
    matrix.to_parquet(OUT / "matrix.parquet", index=False)
    (OUT / "versions.json").write_text(
        json.dumps(
            {
                "source": "magic",
                "versions": version_meta,
                "meta": {
                    "task": "binary_classification",
                    "model_family": "sklearn_tabular_physics",
                    "license": "CC BY 4.0 (UCI MAGIC Gamma Telescope)",
                    "provenance": URL,
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "split_seed": SEED,
                    "evaluation_pool": len(pool),
                    "training_rows": len(train),
                    "first_feature_train_median": median,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"MAGIC: {len(train)} training rows, {len(pool)} evaluation rows")
    for vid, *_ in versions:
        print(f"{vid}: pool accuracy {1 - columns[f'loss__{vid}'].mean():.4f}")
    print(f"source sha256: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    prepare()
