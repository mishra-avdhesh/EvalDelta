"""Deterministic sealed partition into discovery / global-confirmation / slice-confirmation pools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from evaldelta.schemas import SplitConfig

DISCOVERY = "discovery"
GLOBAL_CONFIRM = "global_confirm"
SLICE_CONFIRM = "slice_confirm"


@dataclass(frozen=True)
class Partition:
    """Sealed partition. Confirmation tuples are in their pre-randomised sampling order."""

    discovery: tuple[str, ...]
    global_confirm: tuple[str, ...]
    slice_confirm: tuple[str, ...]
    seed: int
    split_hash: str

    def sizes(self) -> dict[str, int]:
        return {
            DISCOVERY: len(self.discovery),
            GLOBAL_CONFIRM: len(self.global_confirm),
            SLICE_CONFIRM: len(self.slice_confirm),
        }

    def membership(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, ids in (
            (DISCOVERY, self.discovery),
            (GLOBAL_CONFIRM, self.global_confirm),
            (SLICE_CONFIRM, self.slice_confirm),
        ):
            for i in ids:
                out[i] = name
        return out


def make_partition(ids: Iterable[str], config: SplitConfig) -> Partition:
    """Partition IDs using only the *sorted* ID set and the seed.

    Re-ordering the input rows therefore cannot change the assignment (tested property).
    """
    id_list = [str(i) for i in ids]
    sorted_ids = sorted(set(id_list))
    if len(id_list) != len(sorted_ids):
        raise ValueError("duplicate IDs passed to make_partition")
    n = len(sorted_ids)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x5EA1]))
    perm = rng.permutation(n)
    order = [sorted_ids[j] for j in perm]
    n_d = int(round(n * config.discovery))
    n_g = int(round(n * config.global_confirm))
    if config.slice_confirm == 0.0:
        n_g = n - n_d
    d = tuple(order[:n_d])
    g = tuple(order[n_d : n_d + n_g])
    s = tuple(order[n_d + n_g :])
    payload = json.dumps(
        {
            "seed": config.seed,
            "fractions": [config.discovery, config.global_confirm, config.slice_confirm],
            "discovery": d,
            "global_confirm": g,
            "slice_confirm": s,
        }
    ).encode()
    return Partition(d, g, s, config.seed, hashlib.sha256(payload).hexdigest())
