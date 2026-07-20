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
