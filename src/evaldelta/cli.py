"""EvalDelta command-line interface (Typer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from evaldelta.bench.synthetic import SCENARIOS, generate_episode
from evaldelta.policies.registry import make_policy
from evaldelta.replay.session import EvalSession, RunResult
from evaldelta.schemas import EXIT_CODES, Budget, ConfirmPlan, Decision, PolicyConfig, RunConfig

app = typer.Typer(
    add_completion=False, help="Budgeted, statistically valid paired regression tests."
)
console = Console(stderr=True)

_SCENARIO_ALIASES = {s.replace("_", "-"): s for s in SCENARIOS}


def _print_summary(result: RunResult) -> None:
    r = result.report
    t = Table(title=f"EvalDelta: {r.run_id}", show_lines=False)
    for col in ("scope", "decision", "n", "down/up", "effect", "interval", "method"):
        t.add_column(col)
    for e in ([r.global_result] if r.global_result else []) + list(r.slice_results):
        t.add_row(
            e.scope,
            e.decision.value,
            str(e.n),
            f"{e.n_down}/{e.n_up}" if e.n_down is not None else "n/a",
            "n/a" if e.effect is None else f"{e.effect:+.4f}",
            "n/a" if e.lower is None else f"[{e.lower:+.4f}, {e.upper:+.4f}]",
            e.method,
        )
    console.print(t)
    console.print(
        f"Decision: [bold]{r.decision.value}[/bold] (exit {r.exit_code}); paid candidate "
        f"evaluations: {r.total_paid_calls}/{r.config['budget']['max_candidate_calls']} "
        f"({r.total_paid_cost:.4g} {r.cost_unit})"
    )


def _exit(result: RunResult, report_only: bool) -> None:
    if report_only and result.decision != Decision.EVALUATION_ERROR:
        raise typer.Exit(0)
    raise typer.Exit(result.report.exit_code)


@app.command()
def demo(
    scenario: Annotated[str, typer.Option(help=f"one of {sorted(_SCENARIO_ALIASES)}")] = (
        "slice-regression"
    ),
    budget: Annotated[int, typer.Option(help="max paid candidate evaluations")] = 100,
    seed: int = 42,
    policy: str = "uniform",
    n_items: int = 5000,
    output: Annotated[Path, typer.Option(help="output directory")] = Path("runs/demo"),
    report_only: Annotated[
        bool, typer.Option(help="always exit 0 unless evaluation error")
    ] = False,
) -> None:
    """Offline replay on a bundled synthetic episode (CPU only, deterministic)."""
    key = _SCENARIO_ALIASES.get(scenario, scenario)
    if key not in SCENARIOS:
        raise typer.BadParameter(f"unknown scenario {scenario}")
    ep = generate_episode(key, n_items=n_items, seed=seed)
    cfg = RunConfig(
        run_id=f"demo-{key}-b{budget}-s{seed}",
        seed=seed,
        budget=Budget(max_candidate_calls=budget),
        policy=PolicyConfig(name=policy),
    )
    session = EvalSession(ep.item_table(), ep.oracle(), episode_meta=ep.meta)
    result = session.compare(cfg, make_policy(cfg.policy, ep.item_table()))
    out = result.save(output)
    (out / "truth_offline_only.json").write_text(json.dumps(ep.truth(), indent=2))
    _print_summary(result)
    console.print(f"Saved report.json, events.jsonl, report.md to {out}")
    typer.echo(
        json.dumps(
            {
                "decision": result.report.decision.value,
                "exit_code": result.report.exit_code,
                "paid_calls": result.report.total_paid_calls,
                "output": str(out),
            }
        )
    )
    _exit(result, report_only)


@app.command()
def compare(
    config: Annotated[Path, typer.Option(help="YAML run config")],
    output: Annotated[Path, typer.Option(help="output directory")] = Path("runs/compare"),
    report_only: bool = False,
) -> None:
    """Compare a cached old system with a candidate as declared in a YAML config.

    The config names an items table (public) and a provider: either ``replay`` (a private
    outcome table, for offline replay) or ``command`` (a shell command evaluated per item).
    """
    from evaldelta.config import load_run_spec

    try:
        spec = load_run_spec(config)
    except Exception as exc:  # configuration errors -> exit 4
        console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(EXIT_CODES[Decision.EVALUATION_ERROR]) from exc
    session = EvalSession(spec.items, spec.provider, history=spec.history, episode_meta=spec.meta)
    result = session.compare(spec.run, make_policy(spec.run.policy, spec.items, spec.history))
    out = result.save(output)
    from evaldelta.reporting.html_report import render_html

    (out / "report.html").write_text(render_html(result.report))
    _print_summary(result)
    typer.echo(json.dumps({"decision": result.decision.value, "output": str(out)}))
    _exit(result, report_only)


@app.command()
def report(
    run_dir: Path,
    format: Annotated[str, typer.Option("--format", help="md | html | json")] = "html",
) -> None:
    """Re-render a saved run as Markdown, HTML or JSON."""
    result = RunResult.load(run_dir)
    if format == "html":
        from evaldelta.reporting.html_report import render_html

        path = run_dir / "report.html"
        path.write_text(render_html(result.report))
    elif format == "md":
        from evaldelta.reporting.markdown_report import render_markdown

        path = run_dir / "report.md"
        path.write_text(render_markdown(result.report))
    elif format == "json":
        path = run_dir / "report.json"
    else:
        raise typer.BadParameter("format must be md, html or json")
    typer.echo(str(path))


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(help="benchmark YAML (see configs/)")],
    output: Path = Path("results/benchmark"),
) -> None:
    """Run a replay benchmark sweep declared in YAML."""
    from evaldelta.experiments.runner import run_sweep_config

    cfg = yaml.safe_load(config.read_text())
    summary = run_sweep_config(cfg, output)
    typer.echo(json.dumps(summary, indent=2, default=str))


def _entry() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app()

__all__ = ["app", "Budget", "ConfirmPlan"]
