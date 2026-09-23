from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest
from typer.testing import CliRunner

from evaldelta import Budget, Decision, EvalSession, RunConfig, UniformPolicy, generate_episode
from evaldelta.cli import app
from evaldelta.integrations.github_action import exit_code, pr_summary
from evaldelta.providers.callable import CallableProvider
from evaldelta.providers.command import CommandProvider
from evaldelta.providers.http import HTTPProvider


def _items(n=400):
    ep = generate_episode("global_regression", n_items=n, seed=61, severity=0.05)
    items = ep.items.copy()
    items["reference_answer"] = "yes"
    return ep, items


class _Handler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        _Handler.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _Handler.calls % 7 == 0:  # transient failure -> retried
            self.send_response(503)
            self.end_headers()
            return
        out = {"output": "yes" if int(body["sample_id"].split("_")[-1]) % 5 else "no"}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}/predict"
    srv.shutdown()


def test_http_provider_with_retries(server):
    from evaldelta.providers.base import zero_one_loss

    p = HTTPProvider(server, zero_one_loss, backoff_s=0.0, allowed_hosts=["127.0.0.1"])
    ep, items = _items()
    rows = items.head(30).to_dict(orient="records")
    out = p.evaluate(rows)
    assert all(o.loss is not None for o in out)  # 503s were retried
    assert any(o.attempts > 1 for o in out)
    with pytest.raises(ValueError):
        HTTPProvider(server, zero_one_loss, allowed_hosts=["example.org"])
    with pytest.raises(ValueError):
        HTTPProvider("file:///etc/passwd", zero_one_loss)


def test_callable_provider_failures_are_not_wrong_answers():
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        raise RuntimeError("boom")

    p = CallableProvider(flaky, lambda o, i: 0.0, retries=2)
    (o,) = p.evaluate([{"sample_id": "a"}])
    assert o.loss is None and o.error.startswith("RuntimeError") and o.attempts == 3
    assert calls["n"] == 3


def test_command_provider(tmp_path):
    script = tmp_path / "cand.py"
    script.write_text("import sys, json\nitem = json.load(sys.stdin)\nprint('yes')\n")
    p = CommandProvider(f"{sys.executable} {script}", lambda o, i: 0.0 if o == "yes" else 1.0)
    (o,) = p.evaluate([{"sample_id": "a", "x": 1}])
    assert o.loss == 0.0 and o.output == "yes"


def test_session_with_live_callable_matches_replay():
    ep, items = _items()
    truth = ep._hidden

    def cand(item):
        return "wrong" if truth[item["sample_id"]] == 1.0 else "right"

    live = CallableProvider(cand, lambda o, i: 1.0 if o == "wrong" else 0.0)
    cfg = RunConfig(run_id="l", seed=3, budget=Budget(max_candidate_calls=150))
    a = EvalSession(ep.item_table(), live).compare(cfg, UniformPolicy())
    b = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, UniformPolicy())
    assert a.report.selected_ids == b.report.selected_ids
    assert a.report.global_result == b.report.global_result


def _write_cfg(tmp_path, provider, run=None):
    ep, items = _items()
    items.to_parquet(tmp_path / "items.parquet")
    pd.DataFrame({"sample_id": list(ep._hidden), "new_loss": list(ep._hidden.values())}).to_parquet(
        tmp_path / "out.parquet"
    )
    cfg = {
        "run": run or {"run_id": "c", "budget": {"max_candidate_calls": 200}},
        "items": "items.parquet",
        "provider": provider,
    }
    path = tmp_path / "cfg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(cfg))
    return path


def test_cli_compare_modes(tmp_path):
    runner = CliRunner()
    path = _write_cfg(tmp_path, {"type": "replay", "outcomes": "out.parquet"})
    strict = runner.invoke(
        app, ["compare", "--config", str(path), "--output", str(tmp_path / "r1")]
    )
    assert strict.exit_code in (0, 2, 3)
    ro = runner.invoke(
        app,
        [
            "compare",
            "--config",
            str(path),
            "--output",
            str(tmp_path / "r2"),
            "--mode",
            "report-only",
        ],
    )
    assert ro.exit_code == 0
    for f in ("report.json", "report.html", "report.md", "events.jsonl", "summary.md"):
        assert (tmp_path / "r1" / f).exists()


def test_cli_config_errors_exit_4(tmp_path):
    runner = CliRunner()
    bad = _write_cfg(tmp_path, {"type": "nope"})
    assert runner.invoke(app, ["compare", "--config", str(bad)]).exit_code == 4
    bad2 = _write_cfg(
        tmp_path,
        {"type": "replay", "outcomes": "out.parquet"},
        run={"run_id": "x", "budget": {"max_candidate_calls": -5}},
    )
    assert runner.invoke(app, ["compare", "--config", str(bad2)]).exit_code == 4


def test_action_exit_policy():
    ep, _ = _items()
    cfg = RunConfig(run_id="a", budget=Budget(max_candidate_calls=30))
    r = EvalSession(ep.item_table(), ep.oracle()).compare(cfg, UniformPolicy()).report
    assert r.decision == Decision.INCONCLUSIVE
    assert exit_code(r, "strict") == 3
    assert exit_code(r, "allow-inconclusive") == 0
    assert exit_code(r, "report-only") == 0
    assert "not a pass" in pr_summary(r)
    with pytest.raises(ValueError):
        exit_code(r, "yolo")
