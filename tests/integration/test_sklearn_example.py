"""The README's real end-to-end example must keep working and keep telling the truth."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "sklearn_versions" / "run.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("sklearn_versions_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weaker_new_version_is_a_confirmed_regression(example, tmp_path, capsys):
    code = example.main(["--out", str(tmp_path / "weak")])
    assert code == 2
    report = json.loads((tmp_path / "weak" / "report.json").read_text())
    assert report["decision"] == "confirmed_regression"
    assert report["total_paid_calls"] <= 300
    assert "true difference" in capsys.readouterr().out


def test_routine_retrain_is_never_reported_as_a_regression_or_a_pass(example, tmp_path):
    code = example.main(["--same", "--out", str(tmp_path / "same")])
    report = json.loads((tmp_path / "same" / "report.json").read_text())
    assert code == report["exit_code"] == 3  # inconclusive is not approval
    assert report["decision"] == "inconclusive"
