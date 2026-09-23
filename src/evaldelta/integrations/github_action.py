"""Helpers for the GitHub Action: PR summary rendering and exit-code policy."""

from __future__ import annotations

import os
from pathlib import Path

from evaldelta.schemas import Decision, RunReport

_ICON = {
    Decision.CONFIRMED_REGRESSION: ":x:",
    Decision.EVIDENCE_OF_NONINFERIORITY: ":white_check_mark:",
    Decision.INCONCLUSIVE: ":warning:",
    Decision.EVALUATION_ERROR: ":x:",
}


def pr_summary(report: RunReport) -> str:
    g = report.global_result
    lines = [
        f"### EvalDelta {_ICON[report.decision]} `{report.decision.value}`",
        "",
        f"Paid candidate evaluations: **{report.total_paid_calls}** / "
        f"{report.config['budget']['max_candidate_calls']} ({report.cost_unit})",
    ]
    if g is not None and g.effect is not None:
        lines.append(
            f"Global paired difference Δ (new − old loss): {g.effect:+.4f}, "
            f"interval [{g.lower:+.4f}, {g.upper:+.4f}], n = {g.n}, method `{g.method}`"
        )
    for e in report.slice_results:
        lines.append(f"* slice `{e.scope.split(':', 1)[1]}`: {e.decision.value} ({e.reason})")
    if report.decision == Decision.INCONCLUSIVE:
        lines.append("")
        lines.append("_Inconclusive is not a pass: escalate to a larger or full evaluation._")
    lines.append("")
    lines.append(
        "<sub>Exploratory findings are not confirmed. See report.html for limitations.</sub>"
    )
    return "\n".join(lines)


def write_step_summary(report: RunReport) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a") as fh:
            fh.write(pr_summary(report) + "\n")


def exit_code(report: RunReport, mode: str) -> int:
    """``strict``: block on regression, inconclusive or error. ``report-only``: block on error only.

    ``allow-inconclusive``: block on regression or error only.
    """
    if mode == "report-only":
        return 4 if report.decision == Decision.EVALUATION_ERROR else 0
    if mode == "allow-inconclusive" and report.decision == Decision.INCONCLUSIVE:
        return 0
    if mode not in {"strict", "allow-inconclusive"}:
        raise ValueError(f"unknown mode {mode!r}")
    return report.exit_code
