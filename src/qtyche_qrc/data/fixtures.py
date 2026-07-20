"""Deterministic, non-financial market-shaped fixtures for offline tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from qtyche_qrc.data.config import DataPreparationConfig
from qtyche_qrc.data.download import RAW_COLUMNS


def _fixture_frames(config: DataPreparationConfig) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(config.source_dates.start, config.source_dates.end)
    index = np.arange(len(dates), dtype=float)
    spy_returns = (
        0.00022
        + 0.0045 * np.sin(index / 11.0)
        + 0.0025 * np.cos(index / 37.0)
        + 0.0015 * np.sin(index / 3.0)
    )
    qqq_returns = 1.08 * spy_returns + 0.0012 * np.cos(index / 7.0)
    spy_close = 100.0 * np.exp(np.cumsum(spy_returns))
    qqq_close = 80.0 * np.exp(np.cumsum(qqq_returns))
    previous_spy = np.concatenate(([spy_close[0]], spy_close[:-1]))
    spy_open = previous_spy * np.exp(0.0007 * np.sin(index / 5.0))
    intraday_width = 0.004 + 0.002 * (1.0 + np.sin(index / 17.0))
    spy_high = np.maximum(spy_open, spy_close) * (1.0 + intraday_width)
    spy_low = np.minimum(spy_open, spy_close) * (1.0 - intraday_width)
    spy_volume = (75_000_000 + 8_000_000 * np.sin(index / 9.0) + (index % 23) * 110_000).astype(
        "int64"
    )
    qqq_volume = (45_000_000 + 5_000_000 * np.cos(index / 13.0) + (index % 17) * 90_000).astype(
        "int64"
    )
    vix_close = 14.0 + 5.5 * np.abs(np.sin(index / 29.0)) + 1.5 * np.cos(index / 8.0)

    return {
        "spy": pd.DataFrame(
            {
                "date": dates,
                "open": spy_open,
                "high": spy_high,
                "low": spy_low,
                "close": spy_close,
                "adjusted_close": spy_close * (1.0 + 0.00001 * index),
                "volume": spy_volume,
            }
        ),
        "vix": pd.DataFrame({"date": dates, "close": vix_close}),
        "qqq": pd.DataFrame({"date": dates, "close": qqq_close, "volume": qqq_volume}),
    }


def _csv_bytes(frame: pd.DataFrame, columns: tuple[str, ...]) -> bytes:
    buffer = io.StringIO()
    frame.loc[:, list(columns)].to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def create_or_verify_fixture_snapshots(config: DataPreparationConfig) -> dict[str, Path]:
    """Create fixtures once, or verify existing fixture bytes are identical."""

    if config.mode != "cached_csv":
        raise ValueError("fixture generation is permitted only with cached_csv mode")
    frames = _fixture_frames(config)
    for name, frame in frames.items():
        path = config.raw_paths[name]
        expected = _csv_bytes(frame, RAW_COLUMNS[name])
        if path.is_file():
            if path.read_bytes() != expected:
                raise ValueError(
                    f"existing fixture snapshot differs from deterministic content: {path}"
                )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    return dict(config.raw_paths)


def fixture_summary(config: DataPreparationConfig) -> str:
    """Return a concise statement that prevents fixtures being mistaken for market data."""

    rows = len(pd.bdate_range(config.source_dates.start, config.source_dates.end))
    return f"deterministic synthetic fixtures: {rows} rows per instrument (not financial data)"
