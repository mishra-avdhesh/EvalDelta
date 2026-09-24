"""Old model vs new model, end to end, on real data (CPU only, no downloads, a few seconds).

Trains two versions of a digit classifier on scikit-learn's bundled digits dataset, then asks
EvalDelta whether the new version regressed, using at most 300 paid evaluations of the new model.

    python examples/sklearn_versions/run.py           # new version trained on too little data
    python examples/sklearn_versions/run.py --same    # routine retrain of the same model

The exit code follows the EvalDelta convention: 0 non-inferiority evidence, 2 confirmed
regression, 3 inconclusive, 4 evaluation error. The report is written to --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from evaldelta import (
    Budget,
    CallableProvider,
    ConfirmPlan,
    EvalSession,
    ItemTable,
    RunConfig,
    UniformPolicy,
    zero_one_loss,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--same", action="store_true", help="new version = retrain on a resample")
    ap.add_argument("--budget", type=int, default=300, help="max paid evaluations of the new model")
    ap.add_argument("--out", type=Path, default=Path("runs/sklearn_versions"))
    args = ap.parse_args(argv)

    X, y = load_digits(return_X_y=True)
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=0.4, stratify=y, random_state=0
    )

    # Old (production) version: trained on all training data.
    old = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    # New (candidate) version: either a data-pipeline bug (only 72 training rows) or a routine retrain.
    rng = np.random.default_rng(1)
    rows = rng.choice(len(X_train), size=len(X_train) if args.same else 72, replace=args.same)
    new = LogisticRegression(max_iter=2000).fit(X_train[rows], y_train[rows])

    # Public item table: labels and the *cached old-version* results. Nothing about the new model.
    ids = [f"digit_{i}" for i in range(len(y_pool))]
    old_pred = old.predict(X_pool)
    items = ItemTable(
        pd.DataFrame(
            {
                "sample_id": ids,
                "slice": [f"digit_{d}" for d in y_pool],
                "reference_answer": [str(d) for d in y_pool],
                "old_loss": (old_pred != y_pool).astype(float),
                "f_old_conf": old.predict_proba(X_pool).max(1),
            }
        )
    )

    # Live candidate: each paid evaluation runs the new model on one item.
    row_of = {sid: i for i, sid in enumerate(ids)}

    def candidate(item: dict) -> str:
        return str(new.predict(X_pool[[row_of[item["sample_id"]]]])[0])

    session = EvalSession(
        items,
        CallableProvider(candidate, zero_one_loss),
        episode_meta={
            "episode_id": "sklearn-digits",
            "old_version_id": "logreg-all-train-rows",
            "new_version_id": "logreg-retrain" if args.same else "logreg-72-train-rows",
        },
    )
    config = RunConfig(
        run_id="sklearn-digits-old-vs-new",
        seed=0,
        budget=Budget(max_candidate_calls=args.budget),
        plan=ConfirmPlan(alpha=0.05, noninferiority_margin=0.03),
    )
    result = session.compare(config, UniformPolicy())
    result.save(args.out)

    report = result.report
    g = report.global_result
    print(f"Decision: {report.decision.value} (exit code {report.exit_code})")
    print(f"Paid evaluations of the new model: {report.total_paid_calls} of {args.budget}")
    if g is not None and g.effect is not None:
        print(
            f"Paired loss difference (new - old): {g.effect:+.3f}, "
            f"interval [{g.lower:+.3f}, {g.upper:+.3f}], n = {g.n}, method {g.method}"
        )
    full = float((new.predict(X_pool) != y_pool).mean() - (old_pred != y_pool).mean())
    print(f"For reference only (needs a full evaluation): true difference {full:+.3f}")
    print(f"Report written to {args.out}/report.md")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
