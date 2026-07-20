"""Causal feature construction using current and trailing observations only."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "spy_log_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "spy_rv_5d",
    "spy_rv_10d",
    "spy_rv_20d",
    "spy_parkinson_vol_5d",
    "spy_volume_zscore_20d",
    "spy_high_low_range",
    "vix_log_level",
    "vix_change_1d",
    "vix_change_5d",
    "qqq_log_return_1d",
    "qqq_rv_5d",
    "spy_qqq_return_spread",
    "day_of_week",
)


def minimum_required_rows(history: int = 20, horizon: int = 5) -> int:
    """Return the minimum rows needed for history plus one complete target."""

    return history + horizon + 1


def ensure_sufficient_rows(frame: pd.DataFrame, history: int = 20, horizon: int = 5) -> None:
    """Fail clearly when rolling features and a forward target cannot coexist."""

    required = minimum_required_rows(history, horizon)
    if len(frame) < required:
        raise ValueError(
            f"insufficient history: received {len(frame)} rows; at least {required} are required"
        )


def _annualized_realized_variance(
    returns: pd.Series[float], window: int, annualization: float
) -> pd.Series[float]:
    return (annualization / window) * returns.pow(2).rolling(
        window=window,
        min_periods=window,
    ).sum()


def build_features(
    canonical: pd.DataFrame,
    requested_features: tuple[str, ...] = FEATURE_NAMES,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Add the requested causal features without centered windows or backfilling."""

    unknown = sorted(set(requested_features) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unsupported feature names: {unknown}")
    ensure_sufficient_rows(canonical)

    frame = canonical.copy()
    spy_log_return = np.log(frame["spy_close"] / frame["spy_close"].shift(1))
    qqq_log_return = np.log(frame["qqq_close"] / frame["qqq_close"].shift(1))
    high_low_log_range = np.log(frame["spy_high"] / frame["spy_low"])

    calculated: dict[str, pd.Series[Any]] = {
        "spy_log_return_1d": spy_log_return,
        "spy_return_5d": np.log(frame["spy_close"] / frame["spy_close"].shift(5)),
        "spy_return_20d": np.log(frame["spy_close"] / frame["spy_close"].shift(20)),
        "spy_rv_5d": _annualized_realized_variance(spy_log_return, 5, annualization),
        "spy_rv_10d": _annualized_realized_variance(spy_log_return, 10, annualization),
        "spy_rv_20d": _annualized_realized_variance(spy_log_return, 20, annualization),
        "spy_parkinson_vol_5d": np.sqrt(
            (annualization / (4.0 * math.log(2.0) * 5.0))
            * high_low_log_range.pow(2).rolling(window=5, min_periods=5).sum()
        ),
        "spy_volume_zscore_20d": (
            frame["spy_volume"] - frame["spy_volume"].rolling(20, min_periods=20).mean()
        )
        / frame["spy_volume"].rolling(20, min_periods=20).std(ddof=0),
        "spy_high_low_range": high_low_log_range,
        "vix_log_level": np.log(frame["vix_close"]),
        "vix_change_1d": frame["vix_close"].diff(1),
        "vix_change_5d": frame["vix_close"].diff(5),
        "qqq_log_return_1d": qqq_log_return,
        "qqq_rv_5d": _annualized_realized_variance(qqq_log_return, 5, annualization),
        "spy_qqq_return_spread": spy_log_return - qqq_log_return,
        "day_of_week": frame["date"].dt.dayofweek.astype(float),
    }
    for name in requested_features:
        frame[name] = calculated[name]
    return frame
