"""Deterministic paired resampling for time series and stratified classes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _bootstrap_controls(
    sample_count: int,
    repetitions: int,
    seed: int,
) -> None:
    for name, value in (
        ("sample_count", sample_count),
        ("repetitions", repetitions),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if sample_count < 2 or repetitions <= 0 or seed < 0:
        raise ValueError("bootstrap controls require n >= 2, repetitions > 0, and seed >= 0")


def circular_block_bootstrap_indices(
    sample_count: int,
    repetitions: int,
    block_length: int,
    seed: int,
) -> NDArray[np.int64]:
    """Draw circular blocks and truncate each replicate to the original length."""

    _bootstrap_controls(sample_count, repetitions, seed)
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or block_length <= 0
        or block_length > sample_count
    ):
        raise ValueError("block_length must be an integer in [1, sample_count]")
    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(sample_count / block_length))
    starts = generator.integers(
        0,
        sample_count,
        size=(repetitions, block_count),
        dtype=np.int64,
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % sample_count
    return np.asarray(indices.reshape(repetitions, -1)[:, :sample_count], dtype=np.int64)


def stratified_bootstrap_indices(
    labels: NDArray[np.integer[Any]],
    repetitions: int,
    seed: int,
) -> NDArray[np.int64]:
    """Resample independently within every observed class, preserving class counts."""

    values = np.asarray(labels, dtype=int).reshape(-1)
    _bootstrap_controls(len(values), repetitions, seed)
    classes = np.unique(values)
    if len(classes) < 2:
        raise ValueError("stratified bootstrap requires at least two classes")
    generator = np.random.default_rng(seed)
    sampled: list[NDArray[np.int64]] = []
    for label in classes:
        positions = np.flatnonzero(values == label).astype(np.int64)
        choices = generator.integers(
            0,
            len(positions),
            size=(repetitions, len(positions)),
            dtype=np.int64,
        )
        sampled.append(positions[choices])
    return np.asarray(np.concatenate(sampled, axis=1), dtype=np.int64)


def indices_to_counts(
    indices: NDArray[np.integer[Any]],
    sample_count: int,
    *,
    chunk_size: int = 512,
) -> NDArray[np.int32]:
    """Convert paired resampling indices into per-observation multiplicities."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 2 or sample_count < 2 or np.any(values < 0) or np.any(values >= sample_count):
        raise ValueError("bootstrap indices are invalid for the requested sample count")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = np.zeros((values.shape[0], sample_count), dtype=np.int32)
    for start in range(0, len(values), chunk_size):
        end = min(start + chunk_size, len(values))
        selected = values[start:end]
        rows = np.broadcast_to(
            np.arange(end - start, dtype=np.int64)[:, None],
            selected.shape,
        )
        np.add.at(output[start:end], (rows, selected), 1)
    if not np.all(output.sum(axis=1) == values.shape[1]):
        raise RuntimeError("bootstrap count conversion lost paired observations")
    return output


def bootstrap_interval(
    draws: NDArray[np.floating[Any]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize finite bootstrap draws and count invalid/non-finite replicates."""

    values = np.asarray(draws, dtype=float).reshape(-1)
    if not 0.0 < confidence_level < 1.0 or not len(values):
        raise ValueError("bootstrap interval controls are invalid")
    valid_mask = np.isfinite(values)
    valid = values[valid_mask]
    if not len(valid):
        raise ValueError("bootstrap produced no valid draws")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
    probability_below = float(np.mean(valid < 0.0))
    probability_above = float(np.mean(valid > 0.0))
    probability_equal = float(np.mean(valid == 0.0))
    two_sided_p = float(
        min(
            1.0,
            2.0
            * min(
                probability_below + probability_equal / 2.0,
                probability_above + probability_equal / 2.0,
            ),
        )
    )
    return {
        "confidence_level": confidence_level,
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "probability_below_zero": probability_below,
        "probability_above_zero": probability_above,
        "probability_equal_zero": probability_equal,
        "two_sided_p_value": two_sided_p,
        "valid_bootstrap_count": int(valid_mask.sum()),
        "invalid_bootstrap_count": int((~valid_mask).sum()),
    }
