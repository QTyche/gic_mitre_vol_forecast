"""Deterministic seed utilities shared by experiment backends."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy, and return an explicit NumPy generator.

    Future ML and quantum adapters should receive a seed derived with
    :func:`derive_seed` instead of relying only on process-global state.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def derive_seed(seed: int, namespace: str) -> int:
    """Derive a stable 32-bit seed for a named future backend or component."""

    if not namespace:
        raise ValueError("namespace must be non-empty")
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")
