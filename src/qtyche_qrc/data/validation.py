"""Canonical-table validation, missing-data accounting, and output audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtyche_qrc.data.config import CANONICAL_COLUMNS
from qtyche_qrc.data.targets import TARGET_NAMES


class DataValidationError(ValueError):
    """Raised when market inputs or processed outputs violate the data contract."""


def _date_strings(values: pd.Series[Any]) -> list[str]:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d").tolist()


def audit_raw_market_frames(
    raw_frames: dict[str, pd.DataFrame],
    *,
    large_move_threshold: float,
    fatal_conditions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Inspect raw observations without repairing, sorting, or filling values."""

    parsed: dict[str, pd.DataFrame] = {}
    instruments: dict[str, Any] = {}
    duplicate_total = 0
    non_increasing_count = 0
    missing_total = 0
    for name, source in raw_frames.items():
        frame = source.copy()
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        parsed[name] = frame
        duplicate_count = int(frame["date"].duplicated().sum())
        non_increasing = not frame["date"].is_monotonic_increasing
        missing = {key: int(value) for key, value in frame.isna().sum().items() if value}
        weekend_dates = _date_strings(frame.loc[frame["date"].dt.dayofweek.ge(5), "date"])
        duplicate_total += duplicate_count
        non_increasing_count += int(non_increasing)
        missing_total += sum(missing.values())
        instruments[name] = {
            "rows": len(frame),
            "actual_date_range": {
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            },
            "duplicate_dates": duplicate_count,
            "non_increasing_dates": non_increasing,
            "missing_values": missing,
            "weekend_count": len(weekend_dates),
            "weekend_dates_sample": weekend_dates[:20],
        }

    spy = parsed["spy"]
    qqq = parsed["qqq"]
    vix = parsed["vix"]
    spy_dates = set(spy["date"])
    qqq_dates = set(qqq["date"])
    vix_dates = set(vix["date"])
    date_alignment = {
        "spy_missing_qqq": sorted(value.date().isoformat() for value in spy_dates - qqq_dates),
        "spy_missing_vix": sorted(value.date().isoformat() for value in spy_dates - vix_dates),
        "vix_not_spy": sorted(value.date().isoformat() for value in vix_dates - spy_dates),
        "qqq_not_spy": sorted(value.date().isoformat() for value in qqq_dates - spy_dates),
    }

    price_columns = {
        "spy": ["open", "high", "low", "close", "adjusted_close"],
        "qqq": ["close"],
        "vix": ["close"],
    }
    zero_prices: dict[str, int] = {}
    negative_prices: dict[str, int] = {}
    for name, columns in price_columns.items():
        zero_prices[name] = int(parsed[name][columns].eq(0).sum().sum())
        negative_prices[name] = int(parsed[name][columns].lt(0).sum().sum())
    non_positive_volume = {
        "spy": int(spy["volume"].le(0).sum()),
        "qqq": int(qqq["volume"].le(0).sum()),
    }
    ohlc_masks = {
        "high_below_low": spy["high"].lt(spy["low"]),
        "high_below_open": spy["high"].lt(spy["open"]),
        "high_below_close": spy["high"].lt(spy["close"]),
        "low_above_open": spy["low"].gt(spy["open"]),
        "low_above_close": spy["low"].gt(spy["close"]),
    }
    ohlc = {
        name: {
            "count": int(mask.sum()),
            "dates_sample": _date_strings(spy.loc[mask, "date"])[:20],
        }
        for name, mask in ohlc_masks.items()
    }
    ohlc_violation_count = sum(int(mask.sum()) for mask in ohlc_masks.values())
    moves = spy["close"].pct_change().abs()
    large_move_mask = moves.gt(large_move_threshold)
    large_moves = [
        {"date": date.date().isoformat(), "absolute_return": float(value)}
        for date, value in zip(spy.loc[large_move_mask, "date"], moves.loc[large_move_mask])
    ]
    adjusted_difference = (spy["adjusted_close"] / spy["close"] - 1.0).abs()
    adjusted = {
        "discrepant_rows": int(adjusted_difference.gt(1e-10).sum()),
        "maximum_absolute_relative_difference": float(adjusted_difference.max()),
        "median_absolute_relative_difference": float(adjusted_difference.median()),
    }
    condition_counts = {
        "duplicate_dates": duplicate_total,
        "non_increasing_dates": non_increasing_count,
        "missing_required_values": missing_total,
        "non_positive_prices": sum(zero_prices.values()) + sum(negative_prices.values()),
        "non_positive_volume": sum(non_positive_volume.values()),
        "ohlc_violations": ohlc_violation_count,
    }
    fatal_hits = {name: condition_counts.get(name, 0) for name in fatal_conditions}
    fatal_hits = {name: count for name, count in fatal_hits.items() if count}
    report = {
        "instruments": instruments,
        "date_alignment": {
            name: {"count": len(dates), "dates_sample": dates[:20]}
            for name, dates in date_alignment.items()
        },
        "zero_prices": zero_prices,
        "negative_prices": negative_prices,
        "non_positive_volume": non_positive_volume,
        "ohlc_consistency": ohlc,
        "large_move_threshold": large_move_threshold,
        "large_one_day_moves": {"count": len(large_moves), "observations": large_moves},
        "adjusted_close_vs_close": adjusted,
        "condition_counts": condition_counts,
        "fatal_conditions": list(fatal_conditions),
        "fatal_hits": fatal_hits,
    }
    if fatal_hits:
        raise DataValidationError(f"fatal raw-market audit conditions: {fatal_hits}")
    return report


def _validate_raw_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} contains an invalid date") from exc
    if result["date"].duplicated().any():
        raise DataValidationError(f"{name} contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise DataValidationError(f"{name} dates must be strictly increasing")
    return result


def align_on_spy_calendar(
    raw_frames: dict[str, pd.DataFrame],
    missing_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-align secondary instruments to SPY and drop gaps only with a report."""

    spy = _validate_raw_frame(raw_frames["spy"], "SPY").rename(
        columns={
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "adjusted_close": "spy_adjusted_close",
            "volume": "spy_volume",
        }
    )
    vix = _validate_raw_frame(raw_frames["vix"], "VIX").rename(columns={"close": "vix_close"})
    qqq = _validate_raw_frame(raw_frames["qqq"], "QQQ").rename(
        columns={"close": "qqq_close", "volume": "qqq_volume"}
    )
    if spy.isna().any().any():
        missing = spy.isna().sum()
        raise DataValidationError(
            f"SPY snapshot has missing required values: {missing[missing.gt(0)].to_dict()}"
        )

    canonical = spy.merge(vix, on="date", how="left", validate="one_to_one").merge(
        qqq,
        on="date",
        how="left",
        validate="one_to_one",
    )
    vix_missing = canonical["vix_close"].isna()
    qqq_missing = canonical[["qqq_close", "qqq_volume"]].isna().any(axis=1)
    missing_secondary = vix_missing | qqq_missing
    missing_dates = canonical.loc[missing_secondary, "date"].dt.strftime("%Y-%m-%d").tolist()
    if missing_dates and missing_policy == "error":
        raise DataValidationError(
            f"secondary instruments are missing on {len(missing_dates)} SPY dates; "
            f"first dates: {missing_dates[:5]}"
        )
    canonical = canonical.loc[~missing_secondary, list(CANONICAL_COLUMNS)].reset_index(drop=True)
    report: dict[str, Any] = {
        "spy_calendar_rows": len(spy),
        "vix_source_rows": len(vix),
        "qqq_source_rows": len(qqq),
        "rows_missing_vix": int(vix_missing.sum()),
        "rows_missing_qqq": int(qqq_missing.sum()),
        "rows_removed_missing_secondary": int(missing_secondary.sum()),
        "removed_dates_sample": missing_dates[:20],
        "missing_data_policy": missing_policy,
        "rows_after_alignment": len(canonical),
    }
    validate_canonical_table(canonical)
    return canonical, report


def validate_canonical_table(frame: pd.DataFrame) -> None:
    """Validate types, ordering, finiteness, and elementary market invariants."""

    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(f"canonical table omits columns: {missing_columns}")
    if frame.empty:
        raise DataValidationError("canonical table is empty")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise DataValidationError("canonical dates must be strictly increasing")
    numeric_columns = list(CANONICAL_COLUMNS[1:])
    if frame[numeric_columns].isna().any().any():
        missing = frame[numeric_columns].isna().sum()
        raise DataValidationError(
            f"canonical table has unexpected missing values: {missing[missing.gt(0)].to_dict()}"
        )
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataValidationError("canonical table contains non-finite numeric values")
    price_columns = [name for name in numeric_columns if "volume" not in name]
    if frame[price_columns].le(0).any().any():
        raise DataValidationError("canonical market prices must be strictly positive")
    if frame[["spy_volume", "qqq_volume"]].lt(0).any().any():
        raise DataValidationError("market volume must be non-negative")
    if frame["spy_high"].lt(frame[["spy_open", "spy_close"]].max(axis=1)).any():
        raise DataValidationError("SPY high is below open or close")
    if frame["spy_high"].lt(frame["spy_low"]).any():
        raise DataValidationError("SPY high is below low")
    if frame["spy_low"].gt(frame[["spy_open", "spy_close"]].min(axis=1)).any():
        raise DataValidationError("SPY low is above open or close")


def audit_processed_data(processed_dir: Path) -> dict[str, Any]:
    """Audit persisted split files against their saved manifest contract."""

    manifest_path = processed_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise DataValidationError(f"missing data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = manifest["feature_names"]
    summaries: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        path = processed_dir / f"{split_name}.csv"
        if not path.is_file():
            raise DataValidationError(f"missing processed split: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        expected = ["date", "split", *feature_names, *TARGET_NAMES]
        if list(frame.columns) != expected:
            raise DataValidationError(f"{split_name} columns differ from the data manifest")
        if frame.empty:
            raise DataValidationError(f"{split_name} split is empty")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise DataValidationError(f"{split_name} dates are not strictly increasing")
        if frame.isna().any().any():
            raise DataValidationError(f"{split_name} contains unexpected missing values")
        for label in ("target_regime_5d", "current_regime"):
            if not set(frame[label].unique()).issubset({0, 1, 2}):
                raise DataValidationError(f"{split_name}.{label} contains invalid regimes")
        transition = frame["target_regime_5d"].ne(frame["current_regime"]).astype(int)
        upward = frame["target_regime_5d"].gt(frame["current_regime"]).astype(int)
        downward = frame["target_regime_5d"].lt(frame["current_regime"]).astype(int)
        if not transition.equals(frame["target_transition"]):
            raise DataValidationError(f"{split_name} transition labels are inconsistent")
        if not upward.equals(frame["target_upward_transition"]):
            raise DataValidationError(f"{split_name} upward-transition labels are inconsistent")
        if not downward.equals(frame["target_downward_transition"]):
            raise DataValidationError(f"{split_name} downward-transition labels are inconsistent")
        summaries[split_name] = {
            "rows": len(frame),
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
            "missing_values": int(frame.isna().sum().sum()),
        }
    return {"status": "passed", "splits": summaries}
