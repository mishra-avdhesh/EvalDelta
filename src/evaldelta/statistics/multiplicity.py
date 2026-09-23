"""Multiplicity corrections used by the confirmation plans."""

from __future__ import annotations

from collections.abc import Sequence


def bonferroni_alpha(alpha: float, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return alpha / k


def holm(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Holm step-down procedure; returns rejection flags aligned with ``p_values``."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adj[idx] = running
    return adj
