"""Human-readable Markdown rendering of a :class:`RunReport`."""

from __future__ import annotations

from evaldelta.schemas import Decision, RunReport, TestEvidence

_HEADLINE = {
    Decision.CONFIRMED_REGRESSION: "CONFIRMED REGRESSION: block release",
    Decision.EVIDENCE_OF_NONINFERIORITY: "EVIDENCE OF NON-INFERIORITY at the declared margin",
    Decision.INCONCLUSIVE: "INCONCLUSIVE: escalate evaluation (this is not a pass)",
    Decision.EVALUATION_ERROR: "EVALUATION ERROR: block release",
}


def _fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _row(e: TestEvidence) -> str:
    interval = (
        f"[{_fmt(e.lower)}, {_fmt(e.upper)}]"
        if e.lower is not None and e.upper is not None
        else "n/a"
    )
    disc = f"{e.n_down}/{e.n_up}" if e.n_down is not None else "n/a"
    p = "n/a" if e.p_value is None else f"{e.p_value:.3g}"
    return (
        f"| {e.scope} | {e.decision.value} | {e.n} | {disc} | {_fmt(e.effect)} | {interval} | "
        f"{e.method} | {p} | {e.alpha_allocated:.3g} | {e.looks_used}/{e.looks_planned} | {e.reason} |"
    )


def render_markdown(r: RunReport) -> str:
    lines = [
        f"# EvalDelta report: `{r.run_id}`",
        "",
        f"**Decision:** `{r.decision.value}`. {_HEADLINE[r.decision]} (exit code {r.exit_code})",
        "",
        f"* Episode: `{r.episode_id}` | old `{r.old_version_id}` -> new `{r.new_version_id}`",
        f"* Pool size: {r.pool_size}; partitions: {r.partition_sizes}; split hash `{r.split_hash[:12]}`",
        f"* Paid candidate evaluations: **{r.total_paid_calls}** "
        f"(cost {r.total_paid_cost:.4g} {r.cost_unit}) of "
        f"{r.config['budget']['max_candidate_calls']} budgeted",
        f"* Protocol v{r.protocol_version}; config hash `{r.config_hash[:12]}`",
        "",
        "Sign convention: Delta = mean(loss_new - loss_old). Positive means the candidate is worse. "
        "`down/up` = old-correct->new-wrong / old-wrong->new-correct.",
        "",
        "## Confirmed statistical evidence",
        "",
        "| scope | decision | n | down/up | effect | interval | method | p | alpha | looks | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if r.global_result is not None:
        lines.append(_row(r.global_result))
    for e in r.slice_results:
        lines.append(_row(e))
    lines += [
        "",
        "## Exploratory findings (not confirmed)",
        "",
        "```",
        *[f"{k}: {v}" for k, v in r.exploratory.items()],
        "```",
        "",
        "## Budget",
        "",
        "| phase | planned calls | paid calls | paid cost | failed |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v.planned_calls} | {v.paid_calls} | {v.paid_cost:.4g} | {v.failed_attempts} |"
            for k, v in r.spend.items()
        ],
        "",
        "## Limitations",
        "",
        *[f"* {x}" for x in r.limitations],
        "",
    ]
    return "\n".join(lines)
