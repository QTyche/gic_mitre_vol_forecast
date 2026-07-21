import pytest

from qtyche_qrc.data.download import _download_frame_with_metadata


@pytest.mark.network
def test_yahoo_chart_public_download_adapter() -> None:
    frame, metadata = _download_frame_with_metadata("SPY", "2025-01-02", "2025-01-10")

    assert not frame.empty
    assert {"date", "open", "high", "low", "close", "adjusted_close", "volume"}.issubset(
        frame.columns
    )
    assert metadata["source_url"].startswith("https://query1.finance.yahoo.com/")
