"""M2 null calibration: session-level false-alarm, coverage and power simulations.

Usage:  python benchmarks/calibrate.py [--trials 400] [--out results/frozen/calibration]
Writes trials.parquet + summary.csv and regenerates docs/CALIBRATION.md.
Every number in that document comes from this script. Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaldelta.experiments.runner import expand, run_trials
from evaldelta.statistics.paired_tests import mcnemar_exact_one_sided

ROOT = Path(__file__).resolve().parents[1]


def cp_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def run_spec(budget: int, method: str = "auto", looks: int = 3, **plan: object) -> dict:
    return {
        "budget": {"max_candidate_calls": budget},
        "plan": {"method": method, "looks": looks, **plan},
        "policy": {"name": "uniform"},
    }


def build(trials: int) -> list:
    specs = []
    n10k = {"type": "synthetic", "n_items": 10000}
    # A. global false alarm, delta = 0 (Delta_pool = 0 exactly or < 0)
    for scen, extra in [
        ("null_noisy", {}),
        ("null_noisy", {"flip_rate": 0.15}),
        ("improvement", {"severity": 0.01}),
        ("null_identical", {}),
    ]:
        for method in ["auto", "betting_wor", "hoeffding_wor"]:
            for budget in [300, 1000]:
                name = f"A_null|{scen}{'_f15' if extra else ''}|{method}|B{budget}"
                specs += expand(
                    name,
                    {**n10k, "scenario": scen, **extra},
                    run_spec(budget, method),
                    trials,
                    kind="null_global",
                )
    # A2. looks sensitivity for the fixed-sample default
    for looks in [1, 5, 10]:
        specs += expand(
            f"A_looks|null_noisy|auto|L{looks}",
            {**n10k, "scenario": "null_noisy"},
            run_spec(1000, "auto", looks),
            trials,
            kind="null_global",
        )
    # B. non-zero margin at the boundary: Delta_pool = 0.02 = delta
    specs += expand(
        "B_margin|delta=0.02|betting_wor",
        {**n10k, "scenario": "global_regression", "severity": 0.02},
        run_spec(1000, "betting_wor", regression_margin=0.02),
        trials,
        kind="null_margin",
    )
    # C. non-inferiority false acceptance: Delta_pool = 0.02 = NI margin
    specs += expand(
        "C_ni|Delta=0.02|ni=0.02",
        {**n10k, "scenario": "global_regression", "severity": 0.02},
        run_spec(1000, "auto", noninferiority_margin=0.02),
        trials,
        kind="null_ni",
    )
    # D. slice family FWER with discovery-selected slices (all slice Deltas exactly 0)
    specs += expand(
        "D_slices|null_sliced|max3",
        {**n10k, "scenario": "null_sliced", "flip_rate": 0.1},
        {
            **run_spec(1000, "auto"),
            "budget": {
                "max_candidate_calls": 1000,
                "discovery_fraction": 0.3,
                "global_fraction": 0.3,
                "slice_fraction": 0.4,
            },
        },
        trials,
        kind="null_slice",
    )
    # E. power (alternatives) for reference
    for sev in [0.01, 0.03, 0.05]:
        for method in ["auto", "betting_wor", "hoeffding_wor"]:
            specs += expand(
                f"E_power|Delta={sev}|{method}|B1000",
                {**n10k, "scenario": "global_regression", "severity": sev},
                run_spec(1000, method),
                max(100, trials // 2),
                kind="power",
            )
    return specs


def naive_peeking(trials: int, seed: int = 0) -> dict[str, float]:
    """Stream-level illustration: uncorrected McNemar at every 10th sample (40 looks)."""
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(trials):
        d = rng.choice([1.0, -1.0, 0.0], size=400, p=[0.05, 0.05, 0.9])
        for n in range(10, 401, 10):
            x = d[:n]
            if mcnemar_exact_one_sided(int((x == 1).sum()), int((x == -1).sum())) <= 0.05:
                rej += 1
                break
    lo, hi = cp_interval(rej, trials)
    return {"rate": rej / trials, "lo": lo, "hi": hi, "trials": trials}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("config", sort=True):
        kind = g["tag_kind"].iloc[0]
        n = len(g)
        if kind in {"null_global", "null_margin"}:
            k = int(g["false_regression"].sum())
            metric = "false confirmed-regression rate"
        elif kind == "null_ni":
            k = int(g["false_noninferiority"].sum())
            metric = "false non-inferiority rate"
        elif kind == "null_slice":
            k = int((g["false_slice_confirmations"] > 0).sum())
            metric = "slice-family FWER"
        else:
            k = int((g["global_decision"] == "confirmed_regression").sum())
            metric = "power (confirmed)"
        lo, hi = cp_interval(k, n)
        rows.append(
            {
                "config": name,
                "kind": kind,
                "metric": metric,
                "trials": n,
                "events": k,
                "rate": k / n,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "coverage": float(g["covered"].mean()) if "covered" in g else np.nan,
                "mean_paid_calls": float(g["paid_calls"].mean()),
                "mean_global_n": float(g["global_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(summ: pd.DataFrame, naive: dict, trials: int, secs: float, path: Path) -> None:
    def table(sub: pd.DataFrame) -> list[str]:
        out = [
            "| configuration | trials | events | rate | 95% CI (Clopper-Pearson) | interval coverage | mean paid calls |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
        for _, r in sub.iterrows():
            cov = "" if np.isnan(r.coverage) else f"{r.coverage:.3f}"
            out.append(
                f"| `{r.config}` | {r.trials} | {r.events} | {r.rate:.3f} | "
                f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {cov} | {r.mean_paid_calls:.0f} |"
            )
        return out

    lines = [
        "# EvalDelta: null calibration report (M2)",
        "",
        "Generated by `python benchmarks/calibrate.py` "
        f"({trials} trials per null configuration; {secs:.0f} s wall time). "
        "Do not edit by hand.",
        "",
        "Each trial draws a **fresh synthetic episode** (10,000 items), a fresh sealed partition and a "
        "fresh policy seed, then runs a full `EvalSession` (uniform discovery, predeclared global "
        "confirmation with 3 looks unless stated otherwise, α = 0.05). Rates are compared with the "
        "nominal α using exact binomial 95% intervals. Zero events in a few hundred trials is "
        "**not** proof of control; the upper CI limit is the relevant number.",
        "",
        "Configuration names: `A_null|<scenario>|<method>|B<budget>`. The `auto` method is the exact "
        "McNemar test with Bonferroni over looks, `betting_wor` is the finite-population betting CS "
        "(anytime-valid), and `hoeffding_wor` is the conservative reference. Interval coverage is the "
        "fraction of runs whose final reported interval for Δ contains the true pool Δ. The nominal "
        "value is ≥ 0.90 (two one-sided 95% bounds).",
        "",
        "## A. Global false confirmed regressions (H0: Δ ≤ 0 true)",
        "",
        *table(summ[summ.kind == "null_global"]),
        "",
        "## B. Non-zero regression margin at the boundary (Δ = δ = 0.02)",
        "",
        *table(summ[summ.kind == "null_margin"]),
        "",
        "## C. False evidence of non-inferiority (Δ = δ_NI = 0.02)",
        "",
        *table(summ[summ.kind == "null_ni"]),
        "",
        "## D. Slice family: discovery-selected slices, all slice Δ exactly 0",
        "",
        *table(summ[summ.kind == "null_slice"]),
        "",
        "## E. Power under global regressions (reference only; uniform discovery, B = 1000)",
        "",
        *table(summ[summ.kind == "power"]),
        "",
        "## Negative control: naive peeking",
        "",
        f"Uncorrected one-sided McNemar at α = 0.05 checked after every 10 samples (40 looks, "
        f"i.i.d. null stream): false-alarm rate **{naive['rate']:.3f}** "
        f"(95% CI [{naive['lo']:.3f}, {naive['hi']:.3f}], {naive['trials']} trials). This is why "
        "EvalDelta only uses predeclared looks with Bonferroni, or anytime-valid confidence "
        "sequences.",
        "",
        "## Verdict",
        "",
    ]
    nulls = summ[summ.kind.isin(["null_global", "null_margin", "null_ni", "null_slice"])]
    bad = nulls[nulls.ci95_lo > 0.05]
    if bad.empty:
        lines.append(
            "No null configuration has a false-alarm rate significantly above α = 0.05: every "
            "Clopper-Pearson lower limit is ≤ 0.05. The M2 gate is met."
        )
    else:
        lines.append("**GATE FAILED**: these configurations exceed α significantly:")
        lines += [f"* `{c}`" for c in bad.config]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "results/frozen/calibration"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    df = run_trials(build(args.trials), args.workers)
    naive = naive_peeking(args.trials)
    secs = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "trials.parquet", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    write_markdown(summ, naive, args.trials, secs, ROOT / "docs/CALIBRATION.md")
    print(summ.to_string(index=False))
    print("naive peeking", naive, f"{secs:.0f}s")


if __name__ == "__main__":
    main()
