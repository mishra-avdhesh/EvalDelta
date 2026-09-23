"""Toy 'candidate model' used by the free CI example (no network, no API keys)."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).parent


@lru_cache(maxsize=1)
def _answers() -> dict[str, str]:
    df = pd.read_parquet(HERE / "candidate_behaviour.parquet")
    return dict(zip(df["sample_id"], df["answer"], strict=True))


def predict(item: dict[str, Any]) -> str:
    return _answers()[str(item["sample_id"])]
