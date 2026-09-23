"""Self-contained static HTML report (no external assets, safe to upload as a CI artifact)."""

from __future__ import annotations

import html

from evaldelta.schemas import Decision, RunReport, TestEvidence

_COLOR = {
    Decision.CONFIRMED_REGRESSION: "#b42318",
    Decision.EVIDENCE_OF_NONINFERIORITY: "#067647",
    Decision.INCONCLUSIVE: "#b54708",
    Decision.EVALUATION_ERROR: "#b42318",
}


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def _interval_svg(e: TestEvidence, margin: float, ni: float | None) -> str:
    """Tiny inline SVG showing the interval against the margins on a symmetric axis."""
    if e.lower is None or e.upper is None:
        return ""
    span = max(0.05, abs(e.lower), abs(e.upper), abs(margin), abs(ni or 0.0)) * 1.15
    w, h = 260, 26

    def x(v: float) -> float:
        return (v + span) / (2 * span) * w

    parts = [
        f'<svg width="{w}" height="{h}" role="img" aria-label="interval">',
        f'<line x1="0" y1="13" x2="{w}" y2="13" stroke="#d0d5dd"/>',
        f'<line x1="{x(0):.1f}" y1="3" x2="{x(0):.1f}" y2="23" stroke="#98a2b3"/>',
        f'<line x1="{x(margin):.1f}" y1="3" x2="{x(margin):.1f}" y2="23" stroke="#b42318" '
        'stroke-dasharray="3,2"/>',
    ]
    if ni is not None:
        parts.append(
            f'<line x1="{x(ni):.1f}" y1="3" x2="{x(ni):.1f}" y2="23" stroke="#067647" '
            'stroke-dasharray="3,2"/>'
        )
    parts.append(
        f'<line x1="{x(e.lower):.1f}" y1="13" x2="{x(e.upper):.1f}" y2="13" stroke="#101828" '
        'stroke-width="4"/>'
    )
    if e.effect is not None:
        parts.append(f'<circle cx="{x(e.effect):.1f}" cy="13" r="4" fill="#101828"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(r: RunReport) -> str:
    esc = html.escape
    plan = r.config.get("plan", {})
    margin = float(plan.get("regression_margin", 0.0))
    ni = plan.get("noninferiority_margin")
    rows = []
    for e in ([r.global_result] if r.global_result else []) + list(r.slice_results):
        rows.append(
            "<tr>"
            f"<td>{esc(e.scope)}</td>"
            f"<td style='color:{_COLOR[e.decision]};font-weight:600'>{esc(e.decision.value)}</td>"
            f"<td>{e.n}</td><td>{e.n_down if e.n_down is not None else 'n/a'}/"
            f"{e.n_up if e.n_up is not None else 'n/a'}</td>"
            f"<td>{_num(e.effect)}</td><td>[{_num(e.lower)}, {_num(e.upper)}]</td>"
            f"<td>{_interval_svg(e, margin if e.scope == 'global' else float(plan.get('slice_margin', 0.0)), ni if e.scope == 'global' else None)}</td>"
            f"<td>{esc(e.method)}</td><td>{'n/a' if e.p_value is None else f'{e.p_value:.3g}'}</td>"
            f"<td>{esc(e.reason)}</td></tr>"
        )
    spend = "".join(
        f"<tr><td>{esc(k)}</td><td>{v.planned_calls}</td><td>{v.paid_calls}</td>"
        f"<td>{v.paid_cost:.4g}</td><td>{v.failed_attempts}</td></tr>"
        for k, v in r.spend.items()
    )
    expl = "".join(
        f"<tr><td>{esc(str(k))}</td><td>{esc(str(v))}</td></tr>" for k, v in r.exploratory.items()
    )
    lim = "".join(f"<li>{esc(x)}</li>" for x in r.limitations)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvalDelta report {esc(r.run_id)}</title>
<style>
body{{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:#101828;background:#fcfcfd;
margin:0;padding:24px max(16px,4vw)}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.badge{{display:inline-block;padding:6px 12px;border-radius:6px;color:#fff;font-weight:600;
background:{_COLOR[r.decision]}}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;min-width:640px}}
td,th{{border-bottom:1px solid #eaecf0;padding:6px 10px;text-align:left;vertical-align:middle}}
th{{background:#f2f4f7;font-weight:600}} .muted{{color:#475467}}
code{{background:#f2f4f7;padding:1px 4px;border-radius:4px}}
</style></head><body>
<h1>EvalDelta report <code>{esc(r.run_id)}</code></h1>
<p class="muted">Episode {esc(str(r.episode_id))}: {esc(str(r.old_version_id))} &rarr;
{esc(str(r.new_version_id))} | protocol v{esc(r.protocol_version)} | config
<code>{r.config_hash[:12]}</code> | split <code>{r.split_hash[:12]}</code></p>
<p><span class="badge">{esc(r.decision.value)}</span> &nbsp; exit code {r.exit_code}</p>
<p>Paid candidate evaluations: <b>{r.total_paid_calls}</b> of
{r.config["budget"]["max_candidate_calls"]} ({r.total_paid_cost:.4g} {esc(r.cost_unit)}).
Delta = mean(loss<sub>new</sub> &minus; loss<sub>old</sub>); positive means the candidate is worse.
Red dashed line = regression margin; green dashed line = non-inferiority margin.</p>
<h2>Confirmed statistical evidence</h2><div class="wrap"><table>
<tr><th>scope</th><th>decision</th><th>n</th><th>down/up</th><th>effect</th><th>interval</th>
<th></th><th>method</th><th>p</th><th>reason</th></tr>{"".join(rows)}</table></div>
<h2>Exploratory findings (not confirmed)</h2><div class="wrap"><table>{expl}</table></div>
<h2>Budget</h2><div class="wrap"><table><tr><th>phase</th><th>planned</th><th>paid calls</th>
<th>paid cost</th><th>failed</th></tr>{spend}</table></div>
<h2>Limitations</h2><ul>{lim}</ul>
</body></html>"""
