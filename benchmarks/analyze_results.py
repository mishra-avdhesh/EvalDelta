"""Reproducible descriptive audit of frozen DeltaBench results.

No method selection is performed here. Repeated seeds on one episode are Monte Carlo
replicates, not independent data. Primary bootstrap comparison resamples source families.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "frozen"
FIG = ROOT / "docs" / "figures"


def load_run(name: str) -> pd.DataFrame | None:
    root = DATA / name
    path = root / "trials.parquet"
    manifest = root / "manifest.json"
    if not path.exists():
        return None
    if manifest.exists() and json.loads(manifest.read_text()).get("status") != "complete":
        raise ValueError(f"{name}: manifest is not complete")
    x = pd.read_parquet(path)
    if (x.paid_calls > x.budget).any() or (x.decision == "evaluation_error").any():
        raise ValueError(f"{name}: budget violation or evaluation error")
    if "tag_episode_spec" in x:
        spec = x.tag_episode_spec.map(json.loads)
        x["source"] = spec.map(lambda z: z["source"])
        x["candidate"] = spec.map(lambda z: z["new"])
        x["old"] = spec.map(lambda z: z["old"])
        x["origin"] = spec.map(lambda z: z.get("inject", {}).get("pattern", "real"))
        x["weighting"] = spec.map(lambda z: z.get("inject", {}).get("weighting", "none"))
    if "tag_split" not in x:
        x["tag_split"] = x.source.map(
            lambda source: "test_transfer" if source == "cifar10" else "test"
        )
    x["detected_global"] = x.global_decision.eq("confirmed_regression")
    x["hit_slice"] = x.target_confirmed.fillna(False).astype(bool)
    return x


def cp_interval(events: int, count: int) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    lo = 0.0 if events == 0 else float(beta.ppf(0.025, events, count - events + 1))
    hi = 1.0 if events == count else float(beta.ppf(0.975, events + 1, count - events))
    return lo, hi


def summarize_global(x: pd.DataFrame) -> pd.DataFrame:
    real = x[(x.tag_kind == "global") & (x.origin == "real") & (x.true_delta > 0)]
    rows = []
    for (split, budget, method), g in real.groupby(["tag_split", "tag_budget", "tag_method"]):
        rows.append(
            {
                "split": split,
                "budget": budget,
                "method": method,
                "sources": g.source.nunique(),
                "pairs": g.tag_episode_spec.nunique(),
                "trials": len(g),
                "detections": int(g.detected_global.sum()),
                "detection_rate": g.detected_global.mean(),
                "mean_paid": g.paid_calls.mean(),
                "mean_paid_when_detected": g.loc[g.detected_global, "paid_calls"].mean(),
                "interval_coverage": g.covered.mean(),
                "mean_wall_seconds": g.wall_time_s.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "budget", "method"])


def summarize_slice(x: pd.DataFrame) -> pd.DataFrame:
    active = x[(x.tag_kind == "slice") & x.target_slice.notna() & (x.target_slice_delta > 0)]
    rows = []
    for (split, budget, method), g in active.groupby(["tag_split", "tag_budget", "tag_method"]):
        rows.append(
            {
                "split": split,
                "budget": budget,
                "method": method,
                "sources": g.source.nunique(),
                "episode_specs": g.tag_episode_spec.nunique(),
                "trials": len(g),
                "confirmed_targets": int(g.hit_slice.sum()),
                "target_detection_rate": g.hit_slice.mean(),
                "mean_paid": g.paid_calls.mean(),
                "false_slice_confirmations": int(g.false_slice_confirmations.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "budget", "method"])


def summarize_null(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, budget, method), g in x.groupby(["tag_split", "tag_budget", "tag_method"]):
        events = int(g.false_regression.sum())
        lo, hi = cp_interval(events, len(g))
        rows.append(
            {
                "split": split,
                "budget": budget,
                "method": method,
                "sources": g.source.nunique(),
                "pairs": g.tag_episode_spec.nunique(),
                "trials": len(g),
                "false_alarms": events,
                "rate": events / len(g),
                "mc_binomial_lo": lo,
                "mc_binomial_hi": hi,
                "coverage": g.covered.mean(),
                "mean_paid": g.paid_calls.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "budget", "method"])


def source_bootstrap(
    x: pd.DataFrame, method: str, baseline: str, budget: int, split: str = "test", reps: int = 5000
) -> dict[str, object]:
    """Difference in detection probability; resample whole source families, never seeds."""
    g = x[
        (x.tag_kind == "global")
        & (x.origin == "real")
        & (x.true_delta > 0)
        & (x.tag_budget == budget)
        & (x.tag_split == split)
        & x.tag_method.isin([method, baseline])
    ]
    paired = g.pivot_table(
        index=["source", "candidate", "old", "trial"],
        columns="tag_method",
        values="detected_global",
        aggfunc="first",
    )
    paired = paired.dropna(subset=[method, baseline])
    if paired.empty:
        raise ValueError(f"no matched trials for {method} vs {baseline}")
    p = paired.reset_index()
    p["diff"] = p[method].astype(float) - p[baseline].astype(float)
    source = p.groupby("source").agg(
        diff_sum=("diff", "sum"), n=("diff", "size"), diff_rate=("diff", "mean")
    )
    ci: list[float] | None = None
    if len(source) >= 2:
        rng = np.random.default_rng(20260923)
        idx = rng.integers(0, len(source), size=(reps, len(source)))
        a = source.diff_sum.to_numpy()[idx].sum(1)
        b = source.n.to_numpy()[idx].sum(1)
        ci = np.quantile(a / b, [0.025, 0.975]).tolist()
    return {
        "budget": budget,
        "method": method,
        "baseline": baseline,
        "split": split,
        "sources": len(source),
        "pairs": p[["source", "candidate", "old"]].drop_duplicates().shape[0],
        "trials": len(p),
        "difference": float(p["diff"].mean()),
        "source_bootstrap_95": ci,
        "by_source": {name: float(row.diff_rate) for name, row in source.iterrows()},
    }


def plot_rates(table: pd.DataFrame, value: str, split: str, title: str, path: Path) -> None:
    sub = table[table.split == split]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, g in sub.groupby("method"):
        ax.plot(g.budget, g[value], marker="o", label=method)
    ax.set(
        xlabel="Maximum paid candidate calls",
        ylabel=value.replace("_", " "),
        title=title,
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-partial", action="store_true", help="Generate CPU-only interim analysis"
    )
    ap.add_argument(
        "--suffix",
        default="v1",
        help="Run-directory version suffix, e.g. 'v2' reads results/frozen/test_cpu_v2 etc.",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Suffix for output files under results/frozen/analysis/ (default: overwrite v1 files)",
    )
    args = ap.parse_args()
    sfx = args.suffix
    names = [
        f"test_cpu_{sfx}",
        f"nullreal_cpu_{sfx}",
        f"ablate_cpu_{sfx}",
        f"test_transfer_{sfx}",
        f"nullreal_transfer_{sfx}",
    ]
    test_names = (f"test_cpu_{sfx}", f"test_transfer_{sfx}")
    null_names = (f"nullreal_cpu_{sfx}", f"nullreal_transfer_{sfx}")
    found = {name: load_run(name) for name in names}
    missing = [name for name, frame in found.items() if frame is None]
    if missing and not args.allow_partial:
        ap.error(f"incomplete experiment set: {missing}")
    test = pd.concat(
        [x for name in test_names if (x := found[name]) is not None],
        ignore_index=True,
    )
    null = pd.concat(
        [x for name in null_names if (x := found[name]) is not None],
        ignore_index=True,
    )
    global_table = summarize_global(test)
    slice_table = summarize_slice(test)
    null_table = summarize_null(null)
    out = DATA / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    osfx = f"_{args.out_suffix}" if args.out_suffix else ""
    global_table.to_csv(out / f"global_real{osfx}.csv", index=False)
    slice_table.to_csv(out / f"slice_injected{osfx}.csv", index=False)
    null_table.to_csv(out / f"null_boundary{osfx}.csv", index=False)
    comparisons = []
    for split in test.tag_split.unique():
        for budget in (200, 500):
            for method in ("OURS2_paired_shift_ipw", "B6_active_testing_ipw", "IPW_uniform"):
                comparisons.append(
                    source_bootstrap(test, method, "B1_fixed_mcnemar", budget, split)
                )
    (out / f"comparisons{osfx}.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    for split in test.tag_split.unique():
        plot_rates(
            global_table,
            "detection_rate",
            split,
            f"Real regressive pairs — {split}",
            FIG / f"F1_global_{split}{osfx}.png",
        )
        plot_rates(
            slice_table,
            "target_detection_rate",
            split,
            f"Injected target slice detection — {split}",
            FIG / f"F4_slice_{split}{osfx}.png",
        )
    plot_rates(
        null_table,
        "rate",
        "test",
        "Boundary-null false alarms — CPU families",
        FIG / f"F3_null_cpu{osfx}.png",
    )
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": names,
        "missing": missing,
        "trial_counts": {k: len(v) for k, v in found.items() if v is not None},
        "bootstrap_unit": "source family",
        "bootstrap_reps": 5000,
        "seed": 20260923,
        "warning": "Repeated trial seeds are Monte Carlo replicates, not independent datasets.",
    }
    (out / f"meta{osfx}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    for result in comparisons:
        if result["method"] == "OURS2_paired_shift_ipw":
            print(result)


if __name__ == "__main__":
    main()
