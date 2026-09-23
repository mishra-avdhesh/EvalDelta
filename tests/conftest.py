from __future__ import annotations

import pytest

from evaldelta.bench.synthetic import generate_episode


@pytest.fixture(scope="session")
def slice_episode():
    return generate_episode("slice_regression", n_items=3000, seed=7)


@pytest.fixture(scope="session")
def null_episode():
    return generate_episode("null_identical", n_items=2000, seed=3)
