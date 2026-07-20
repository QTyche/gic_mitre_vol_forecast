"""Immutable raw CSV snapshots for cached and public-download input modes."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from qtyche_qrc.data.config import DataPreparationConfig

RAW_COLUMNS = {
    "spy": ("date", "open", "high", "low", "close", "adjusted_close", "volume"),
    "vix": ("date", "close"),
    "qqq": ("date", "close", "volume"),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yahoo_chart(symbol: str, start: str, end_exclusive: str) -> MappingPayload:
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    query = urllib.parse.urlencode(
        {
            "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "period2": int(pd.Timestamp(end_exclusive, tz="UTC").timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "qtyche-qrc/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected download response for {symbol}")
    return cast(MappingPayload, payload)


MappingPayload = dict[str, Any]


def _download_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end).date() + timedelta(days=1)).isoformat()
    payload = _yahoo_chart(symbol, start, end_exclusive)
    chart = cast(dict[str, Any], payload.get("chart", {}))
    if chart.get("error") is not None:
        raise ValueError(f"download failed for {symbol}: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        raise ValueError(f"download returned no rows for {symbol}")
    result = cast(dict[str, Any], results[0])
    timestamps = result.get("timestamp")
    indicators = cast(dict[str, Any], result.get("indicators", {}))
    quotes = indicators.get("quote")
    adjusted = indicators.get("adjclose")
    if not isinstance(timestamps, list) or not isinstance(quotes, list) or not quotes:
        raise ValueError(f"download response lacks price arrays for {symbol}")
    quote = cast(dict[str, Any], quotes[0])
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    if isinstance(adjusted, list) and adjusted:
        frame["adjusted_close"] = cast(dict[str, Any], adjusted[0]).get("adjclose")
    return frame


def _write_immutable_snapshot(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    if path.exists():
        return
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"download for {path.name} omitted required columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.loc[:, list(columns)].to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
        lineterminator="\n",
    )
    temporary.replace(path)


def ensure_raw_snapshots(config: DataPreparationConfig) -> None:
    """Require cached files or download only missing immutable snapshots."""

    missing_paths = [path for path in config.raw_paths.values() if not path.is_file()]
    if config.mode == "cached_csv":
        if missing_paths:
            joined = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"cached_csv mode is missing raw snapshots: {joined}")
        return

    start = config.source_dates.start.isoformat()
    end = config.source_dates.end.isoformat()
    for name in ("spy", "vix", "qqq"):
        path = config.raw_paths[name]
        if path.is_file():
            continue
        downloaded = _download_frame(config.symbols[name], start, end)
        _write_immutable_snapshot(downloaded, RAW_COLUMNS[name], path)


def load_raw_frames(config: DataPreparationConfig) -> dict[str, pd.DataFrame]:
    """Load raw snapshots without filling, sorting, or rewriting observations."""

    ensure_raw_snapshots(config)
    frames: dict[str, pd.DataFrame] = {}
    for name, path in config.raw_paths.items():
        frame = pd.read_csv(path)
        required = RAW_COLUMNS[name]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} omits required raw columns: {missing}")
        frames[name] = frame.loc[:, list(required)].copy()
    return frames
