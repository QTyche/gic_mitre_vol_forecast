"""Immutable raw CSV snapshots for cached and public-download input modes."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from qtyche_qrc.data.config import DataPreparationConfig

RAW_COLUMNS = {
    "spy": ("date", "open", "high", "low", "close", "adjusted_close", "volume"),
    "vix": ("date", "close"),
    "qqq": ("date", "close", "volume"),
}


class SnapshotIntegrityError(ValueError):
    """Raised when an immutable public snapshot is incomplete or changed."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yahoo_url(symbol: str, start: str, end_exclusive: str) -> str:
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
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{query}"


def _yahoo_chart(symbol: str, start: str, end_exclusive: str) -> MappingPayload:
    url = _yahoo_url(symbol, start, end_exclusive)
    request = urllib.request.Request(url, headers={"User-Agent": "qtyche-qrc/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected download response for {symbol}")
    return cast(MappingPayload, payload)


MappingPayload = dict[str, Any]


def _frame_from_payload(
    payload: MappingPayload, symbol: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert one Yahoo chart response to a raw frame plus provider metadata."""

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
    adjusted_record = adjusted[0] if isinstance(adjusted, list) and adjusted else None
    adjusted_available = isinstance(adjusted_record, dict)
    if isinstance(adjusted_record, dict):
        frame["adjusted_close"] = adjusted_record.get("adjclose")
    metadata = cast(dict[str, Any], result.get("meta", {}))
    return frame, {
        "provider_timezone": metadata.get("exchangeTimezoneName") or metadata.get("timezone"),
        "exchange_timezone_offset_seconds": metadata.get("gmtoffset"),
        "currency": metadata.get("currency"),
        "exchange": metadata.get("exchangeName"),
        "data_granularity": metadata.get("dataGranularity"),
        "adjusted_close_available": adjusted_available,
        "adjustment_source": "Yahoo chart adjclose indicator" if adjusted_available else None,
    }


def _download_frame_with_metadata(
    symbol: str, start: str, end: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    end_exclusive = (pd.Timestamp(end).date() + timedelta(days=1)).isoformat()
    payload = _yahoo_chart(symbol, start, end_exclusive)
    frame, metadata = _frame_from_payload(payload, symbol)
    metadata["source_url"] = _yahoo_url(symbol, start, end_exclusive)
    return frame, metadata


def _download_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Compatibility wrapper used by the original unversioned fixture tests."""

    frame, _ = _download_frame_with_metadata(symbol, start, end)
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


def _write_snapshot_file(
    frame: pd.DataFrame, columns: tuple[str, ...], path: Path, *, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise SnapshotIntegrityError(f"snapshot file already exists and is immutable: {path}")
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


def verify_public_snapshot(config: DataPreparationConfig) -> dict[str, Any]:
    """Verify a public snapshot manifest, identity, files, and byte checksums."""

    manifest_path = config.snapshot_manifest_path
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"public snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != config.snapshot_id:
        raise SnapshotIntegrityError("snapshot manifest ID disagrees with configuration")
    if manifest.get("data_source_type") != "public_market" or manifest.get("is_synthetic"):
        raise SnapshotIntegrityError("snapshot manifest source flags are invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SnapshotIntegrityError("snapshot manifest has no file records")
    for name, path in config.raw_paths.items():
        record = files.get(name)
        if not isinstance(record, dict):
            raise SnapshotIntegrityError(f"snapshot manifest omits {name}")
        if Path(str(record.get("file"))).name != path.name:
            raise SnapshotIntegrityError(f"snapshot filename mismatch for {name}")
        if not path.is_file():
            raise FileNotFoundError(f"snapshot file is missing: {path}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise SnapshotIntegrityError(
                f"snapshot checksum mismatch for {name}: {actual} != {record.get('sha256')}"
            )
    return cast(dict[str, Any], manifest)


def download_public_snapshot(
    config: DataPreparationConfig, *, force: bool = False
) -> dict[str, Any]:
    """Download one immutable, versioned public-market snapshot and its manifest."""

    if config.data_source_type != "public_market" or config.snapshot_manifest_path is None:
        raise ValueError("download-public-data requires a public_market data configuration")
    manifest_path = config.snapshot_manifest_path
    existing = manifest_path.exists() or any(path.exists() for path in config.raw_paths.values())
    if existing and not force:
        return verify_public_snapshot(config)

    start = config.source_dates.start.isoformat()
    end = config.source_dates.end.isoformat()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    file_records: dict[str, Any] = {}
    for name in ("spy", "qqq", "vix"):
        symbol = config.symbols[name]
        frame, provider_metadata = _download_frame_with_metadata(symbol, start, end)
        path = config.raw_paths[name]
        _write_snapshot_file(frame, RAW_COLUMNS[name], path, overwrite=force)
        saved = pd.read_csv(path, parse_dates=["date"])
        file_records[name] = {
            "file": path.name,
            "symbol": symbol,
            "rows": len(saved),
            "requested_date_range": {"start": start, "end": end},
            "actual_date_range": {
                "start": saved["date"].min().date().isoformat(),
                "end": saved["date"].max().date().isoformat(),
            },
            "sha256": sha256_file(path),
            **provider_metadata,
        }
    manifest = {
        "schema_version": 1,
        "snapshot_id": config.snapshot_id,
        "provider": config.provider,
        "retrieval_timestamp": retrieved_at,
        "requested_date_range": {"start": start, "end": end},
        "data_source_type": "public_market",
        "is_synthetic": False,
        "files": file_records,
        "redistribution_notice": "Provider terms must be reviewed before redistributing raw files.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return verify_public_snapshot(config)


def ensure_raw_snapshots(config: DataPreparationConfig) -> None:
    """Require cached files or download only missing immutable snapshots."""

    if config.data_source_type == "public_market":
        if config.mode == "cached_csv":
            verify_public_snapshot(config)
        else:
            download_public_snapshot(config)
        return

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
