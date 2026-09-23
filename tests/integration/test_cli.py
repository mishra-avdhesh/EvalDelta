from __future__ import annotations

import json

from typer.testing import CliRunner

from evaldelta.cli import app

runner = CliRunner()


def _demo(tmp_path, name, *extra):
    out = tmp_path / name
    res = runner.invoke(
        app,
        [
            "demo",
            "--scenario",
            "slice-regression",
            "--budget",
            "100",
            "--seed",
            "42",
            "--n-items",
            "3000",
            "--output",
            str(out),
            *extra,
        ],
    )
    return res, out


def test_demo_cli_roundtrip_and_determinism(tmp_path):
    r1, o1 = _demo(tmp_path, "a")
    r2, o2 = _demo(tmp_path, "b")
    assert r1.exit_code in (0, 2, 3), r1.output
    assert r1.exit_code == r2.exit_code
    for f in ("report.json", "events.jsonl", "report.md"):
        assert (o1 / f).exists()
    j1 = json.loads((o1 / "report.json").read_text())
    j2 = json.loads((o2 / "report.json").read_text())
    assert j1["selected_ids"] == j2["selected_ids"]
    assert j1["total_paid_calls"] <= 100
    last = json.loads(r1.stdout.strip().splitlines()[-1])
    assert last["paid_calls"] == j1["total_paid_calls"]


def test_report_only_exits_zero(tmp_path):
    res, _ = _demo(tmp_path, "c", "--report-only")
    assert res.exit_code == 0


def test_report_render(tmp_path):
    _, out = _demo(tmp_path, "d")
    for fmt in ("html", "md", "json"):
        res = runner.invoke(app, ["report", str(out), "--format", fmt])
        assert res.exit_code == 0, res.output
    html = (out / "report.html").read_text()
    assert "<title>" in html and "Limitations" in html
