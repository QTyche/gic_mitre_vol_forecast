"""Versioned five-trading-day volatility and regime target definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_NAMES = (
    "target_rv_5d",
    "target_regime_5d",
    "current_regime",
    "target_transition",
    "target_upward_transition",
    "target_downward_transition",
)


@dataclass(frozen=True)
class RegimeThresholds:
    """Training-fitted boundaries for low, medium, and high realized variance."""

    low_medium: float
    medium_high: float
    quantiles: tuple[float, float]
    training_rows: int


def add_continuous_targets(
    frame: pd.DataFrame,
    horizon: int = 5,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Add forward RV using exactly returns t+1 through t+horizon."""

    if horizon != 5:
        raise ValueError("target definition v1 requires a five-trading-day horizon")
    result = frame.copy()
    returns = np.log(result["spy_close"] / result["spy_close"].shift(1))
    squared_forward_returns = [returns.shift(-offset).pow(2) for offset in range(1, horizon + 1)]
    result["target_rv_5d"] = (annualization / horizon) * sum(squared_forward_returns)
    result["target_window_end"] = result["date"].shift(-horizon)
    return result


def fit_regime_thresholds(
    split_frame: pd.DataFrame,
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> RegimeThresholds:
    """Fit target-RV quantiles from rows explicitly assigned to training."""

    if "split" not in split_frame or "target_rv_5d" not in split_frame:
        raise ValueError("threshold fitting requires split and target_rv_5d columns")
    training_values = split_frame.loc[split_frame["split"].eq("train"), "target_rv_5d"].dropna()
    if training_values.empty:
        raise ValueError("cannot fit regime thresholds without training targets")
    values = training_values.quantile(list(quantiles), interpolation="linear")
    low_medium = float(values.iloc[0])
    medium_high = float(values.iloc[1])
    if not np.isfinite(low_medium) or not np.isfinite(medium_high):
        raise ValueError("training regime thresholds are not finite")
    if low_medium >= medium_high:
        raise ValueError("training regime thresholds are not distinct")
    return RegimeThresholds(
        low_medium=low_medium,
        medium_high=medium_high,
        quantiles=quantiles,
        training_rows=int(training_values.size),
    )


def _regime(values: pd.Series[float], thresholds: RegimeThresholds) -> pd.Series[int]:
    labels = np.select(
        [values <= thresholds.low_medium, values <= thresholds.medium_high],
        [0, 1],
        default=2,
    )
    return pd.Series(labels, index=values.index, dtype="int64")


def add_regime_and_transition_targets(
    frame: pd.DataFrame,
    thresholds: RegimeThresholds,
) -> pd.DataFrame:
    """Apply frozen training thresholds and construct exact transition labels."""

    if frame[["target_rv_5d", "spy_rv_5d"]].isna().any().any():
        raise ValueError("regime labels require complete current and forward realized variance")
    result = frame.copy()
    result["target_regime_5d"] = _regime(result["target_rv_5d"], thresholds)
    result["current_regime"] = _regime(result["spy_rv_5d"], thresholds)
    result["target_transition"] = (result["target_regime_5d"] != result["current_regime"]).astype(
        int
    )
    result["target_upward_transition"] = (
        result["target_regime_5d"] > result["current_regime"]
    ).astype(int)
    result["target_downward_transition"] = (
        result["target_regime_5d"] < result["current_regime"]
    ).astype(int)
    return result
