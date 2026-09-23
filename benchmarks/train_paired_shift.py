"""Train PairedShift's historical p_down / p_up models on historical_train episodes ONLY.

    python benchmarks/train_paired_shift.py [--out configs/paired_shift_model.json]

Validation and test episodes (and the whole CIFAR-10 transfer family) are never read here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaldelta.bench.deltabench import build_pair_episode, load_source, registry
from evaldelta.data.io import ID_COL
from evaldelta.policies.paired_shift import FEATURES, PairedShiftModel, feature_matrix

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "configs/paired_shift_model.json"))
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    reg = registry()
    train = reg[reg.split == "train"]
    assert not (
        reg[reg.split != "train"]
        .set_index(["source", "old", "new"])
        .index.isin(train.set_index(["source", "old", "new"]).index)
    ).any()
    episodes = []
    for r in train.itertuples():
        ep = build_pair_episode(load_source(r.source), r.old, r.new)
        new = ep.items[ID_COL].map(ep._hidden).to_numpy(dtype=float)
        episodes.append((ep.episode_id, ep.items, new))
    model = PairedShiftModel.train(episodes, l2=args.l2)
    model.save(args.out)
    print(f"trained on {len(episodes)} historical_train episodes -> {args.out}")
    for name, m in (("p_down", model.down), ("p_up", model.up)):
        print(name, "intercept", round(m.intercept, 3))
        for f, c in zip(FEATURES, m.coef, strict=True):
            print(f"   {f:18s} {c:+.3f}")
    # in-sample sanity check: AUC of p_down on the training episodes
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _, items, new in episodes:
        x = feature_matrix(items)
        old = items["old_loss"].to_numpy()
        oc = old == 0
        y = (new[oc] == 1).astype(int)
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, model.down.logits(x[oc])))
    print(
        f"p_down in-sample AUC across train episodes: mean {np.mean(aucs):.3f}, "
        f"min {np.min(aucs):.3f}"
    )


if __name__ == "__main__":
    main()
