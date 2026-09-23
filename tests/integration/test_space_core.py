from __future__ import annotations

import pandas as pd
import pytest

from evaldelta.demo import run_replay


def test_space_replay_reports_two_budgeted_decisions():
    table, summary, details = run_replay(budget=100, seed=7)
    assert set(table.selector) == {"uniform", "paired_shift"}
    assert (table.paid_calls <= 100).all()
    assert details["replay_only"] is True
    assert "Offline replay" in summary
    assert all(row["split_hash"] for row in details["aggregate_reports"].values())


def test_space_rejects_invalid_or_executable_upload(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("print('no')\n")
    with pytest.raises(ValueError, match="CSV or Parquet"):
        run_replay(str(code))
    invalid = tmp_path / "out.csv"
    pd.DataFrame(
        {"sample_id": [f"i{k}" for k in range(100)], "old_loss": [0] * 100, "new_loss": [2] * 100}
    ).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="binary"):
        run_replay(str(invalid))
