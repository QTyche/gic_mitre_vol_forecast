import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from qtyche_qrc.config import ConfigError
from qtyche_qrc.data import download as download_module
from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.description import describe_public_data
from qtyche_qrc.data.download import (
    SnapshotIntegrityError,
    download_public_snapshot,
    load_raw_frames,
    sha256_file,
    verify_public_snapshot,
)
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.data.validation import (
    DataValidationError,
    align_on_spy_calendar,
    audit_raw_market_frames,
)
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.run import run_baseline_experiment
from tests.data_helpers import (
    canonical_frame,
    write_test_model_config,
    write_test_public_data_config,
)


def _raw_frames(rows: int = 270) -> dict[str, pd.DataFrame]:
    canonical = canonical_frame(rows)
    return {
        "spy": canonical[
            [
                "date",
                "spy_open",
                "spy_high",
                "spy_low",
                "spy_close",
                "spy_adjusted_close",
                "spy_volume",
            ]
        ].rename(
            columns={
                "spy_open": "open",
                "spy_high": "high",
                "spy_low": "low",
                "spy_close": "close",
                "spy_adjusted_close": "adjusted_close",
                "spy_volume": "volume",
            }
        ),
        "vix": canonical[["date", "vix_close"]].rename(columns={"vix_close": "close"}),
        "qqq": canonical[["date", "qqq_close", "qqq_volume"]].rename(
            columns={"qqq_close": "close", "qqq_volume": "volume"}
        ),
    }


def _write_public_snapshot(config: Any) -> dict[str, pd.DataFrame]:
    frames = _raw_frames()
    for name, frame in frames.items():
        path = config.raw_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    manifest = {
        "schema_version": 1,
        "snapshot_id": config.snapshot_id,
        "provider": "yahoo_chart",
        "retrieval_timestamp": "2026-01-01T00:00:00+00:00",
        "requested_date_range": {
            "start": config.source_dates.start.isoformat(),
            "end": config.source_dates.end.isoformat(),
        },
        "data_source_type": "public_market",
        "is_synthetic": False,
        "files": {
            name: {
                "file": path.name,
                "symbol": config.symbols[name],
                "rows": len(frames[name]),
                "sha256": sha256_file(path),
            }
            for name, path in config.raw_paths.items()
        },
    }
    config.snapshot_manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frames


def test_public_configuration_is_non_synthetic_and_isolated(tmp_path: Path) -> None:
    config = write_test_public_data_config(tmp_path)

    assert config.data_source_type == "public_market"
    assert config.is_synthetic is False
    assert config.processed_path == tmp_path / "data/processed/public_market"
    assert config.processed_path != tmp_path / "data/processed"


def test_public_configuration_cannot_overwrite_fixture_processed_dir(tmp_path: Path) -> None:
    config = write_test_public_data_config(tmp_path)
    raw = yaml.safe_load(config.source.read_text(encoding="utf-8"))
    raw["data"]["processed_path"] = "data/processed"
    config.source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="must not overwrite"):
        load_data_config(config.source)


def test_public_snapshot_is_immutable_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_test_public_data_config(tmp_path, mode="download")
    _write_public_snapshot(config)
    original = {name: path.read_bytes() for name, path in config.raw_paths.items()}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verified immutable snapshots must not be redownloaded")

    monkeypatch.setattr(download_module, "_download_frame_with_metadata", forbidden)
    download_public_snapshot(config)

    assert original == {name: path.read_bytes() for name, path in config.raw_paths.items()}


def test_public_snapshot_force_redownloads_every_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_test_public_data_config(tmp_path, mode="download")
    frames = _write_public_snapshot(config)
    by_symbol = {config.symbols[name]: frame for name, frame in frames.items()}
    calls: list[str] = []

    def fake_download(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(symbol)
        return by_symbol[symbol], {
            "source_url": f"https://example.invalid/{symbol}",
            "provider_timezone": "America/New_York",
            "adjusted_close_available": symbol != "^VIX",
        }

    monkeypatch.setattr(download_module, "_download_frame_with_metadata", fake_download)
    manifest = download_public_snapshot(config, force=True)

    assert set(calls) == {"SPY", "QQQ", "^VIX"}
    assert manifest["snapshot_id"] == config.snapshot_id


def test_snapshot_checksum_change_fails_clearly(tmp_path: Path) -> None:
    config = write_test_public_data_config(tmp_path)
    _write_public_snapshot(config)
    config.raw_paths["spy"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="checksum mismatch"):
        verify_public_snapshot(config)


def test_cached_public_snapshot_uses_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_test_public_data_config(tmp_path)
    _write_public_snapshot(config)

    def forbidden_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached mode must not use the network")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    frames = load_raw_frames(config)

    assert set(frames) == {"spy", "qqq", "vix"}


def test_duplicate_dates_are_fatal_in_raw_audit() -> None:
    frames = _raw_frames(40)
    frames["spy"] = pd.concat([frames["spy"], frames["spy"].iloc[[-1]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate_dates"):
        audit_raw_market_frames(
            frames, large_move_threshold=0.2, fatal_conditions=("duplicate_dates",)
        )


def test_ohlc_inconsistency_is_reported_and_can_be_fatal() -> None:
    frames = _raw_frames(40)
    low_value = frames["spy"]["low"].to_numpy(dtype=float)[5]
    frames["spy"].loc[5, "high"] = low_value - 1.0
    report = audit_raw_market_frames(frames, large_move_threshold=0.2)

    assert report["ohlc_consistency"]["high_below_low"]["count"] == 1
    with pytest.raises(DataValidationError, match="ohlc_violations"):
        audit_raw_market_frames(
            frames, large_move_threshold=0.2, fatal_conditions=("ohlc_violations",)
        )


def test_missing_secondary_date_is_reported_without_forward_fill() -> None:
    frames = _raw_frames(40)
    missing_date = frames["vix"].loc[8, "date"]
    frames["vix"] = frames["vix"].drop(index=8).reset_index(drop=True)

    report = audit_raw_market_frames(frames, large_move_threshold=0.2)
    aligned, alignment = align_on_spy_calendar(frames, "drop_secondary_and_report")

    assert report["date_alignment"]["spy_missing_vix"]["count"] == 1
    assert alignment["rows_removed_missing_secondary"] == 1
    assert missing_date not in set(aligned["date"])


def test_missing_vix_placeholder_outside_spy_calendar_is_reported_and_excluded() -> None:
    frames = _raw_frames(40)
    placeholder = pd.DataFrame({"date": [pd.Timestamp("2020-03-14")], "close": [float("nan")]})
    frames["vix"] = (
        pd.concat([frames["vix"], placeholder], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    report = audit_raw_market_frames(frames, large_move_threshold=0.2)
    aligned, alignment = align_on_spy_calendar(frames, "drop_secondary_and_report")

    assert report["instruments"]["vix"]["missing_values"] == {"close": 1}
    assert len(aligned) == len(frames["spy"])
    assert alignment["rows_missing_vix"] == 0


def test_public_model_configs_reference_public_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs/models/public_market").glob("*.yaml")):
        config = load_model_config(path)
        assert config.processed_dir == root / "data/processed/public_market"


def test_public_manifest_and_experiment_include_snapshot_provenance(tmp_path: Path) -> None:
    config = write_test_public_data_config(tmp_path)
    _write_public_snapshot(config)
    prepare_data(config)
    model_path = write_test_model_config(tmp_path)
    model_raw = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_raw["data"] = {
        "processed_dir": "data/processed/public_market",
        "manifest": "data/processed/public_market/data_manifest.json",
    }
    model_path.write_text(yaml.safe_dump(model_raw, sort_keys=False), encoding="utf-8")

    experiment_dir = run_baseline_experiment(model_path)
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["data_snapshot_id"] == config.snapshot_id
    assert manifest["data_manifest_checksum"]
    assert manifest["is_synthetic"] is False


def test_public_description_writes_tables_and_seven_figures(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    config = write_test_public_data_config(tmp_path)
    _write_public_snapshot(config)
    prepare_data(config)

    describe_public_data(config.processed_path, tmp_path / "results/data_audit")

    assert (tmp_path / "results/data_audit/public_market_summary.json").is_file()
    assert (tmp_path / "results/data_audit/public_market_summary.csv").is_file()
    assert len(list((tmp_path / "results/data_audit/figures").glob("*.png"))) == 7
