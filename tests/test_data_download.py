from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from qtyche_qrc.data import download as download_module
from tests.data_helpers import canonical_frame, write_test_data_config


def test_download_mode_creates_immutable_snapshots_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(write_test_data_config(tmp_path), mode="download")
    canonical = canonical_frame(rows=30)
    frames = {
        "SPY": canonical[
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
        "^VIX": canonical[["date", "vix_close"]].rename(columns={"vix_close": "close"}),
        "QQQ": canonical[["date", "qqq_close", "qqq_volume"]].rename(
            columns={"qqq_close": "close", "qqq_volume": "volume"}
        ),
    }

    def fake_download(symbol: str, start: str, end: str) -> pd.DataFrame:
        assert start == config.source_dates.start.isoformat()
        assert end == config.source_dates.end.isoformat()
        return frames[symbol]

    monkeypatch.setattr(download_module, "_download_frame", fake_download)
    download_module.ensure_raw_snapshots(config)
    original_bytes = {name: path.read_bytes() for name, path in config.raw_paths.items()}

    def forbidden_redownload(symbol: str, start: str, end: str) -> pd.DataFrame:
        raise AssertionError(f"existing snapshot for {symbol} must not be overwritten")

    monkeypatch.setattr(download_module, "_download_frame", forbidden_redownload)
    download_module.ensure_raw_snapshots(config)

    assert original_bytes == {name: path.read_bytes() for name, path in config.raw_paths.items()}
