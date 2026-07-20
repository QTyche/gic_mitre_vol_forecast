from typing import cast

import pandas as pd
import pytest

from qtyche_qrc.data.validation import DataValidationError, align_on_spy_calendar
from tests.data_helpers import canonical_frame


def _raw_frames() -> dict[str, pd.DataFrame]:
    canonical = canonical_frame(rows=30)
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


def test_missing_secondary_observation_is_removed_and_documented() -> None:
    frames = _raw_frames()
    missing_date = cast(pd.Timestamp, frames["vix"].loc[5, "date"])
    frames["vix"] = frames["vix"].drop(index=5).reset_index(drop=True)

    aligned, report = align_on_spy_calendar(frames, "drop_secondary_and_report")

    assert len(aligned) == 29
    assert report["rows_removed_missing_secondary"] == 1
    assert missing_date.strftime("%Y-%m-%d") in report["removed_dates_sample"]


def test_missing_secondary_observation_can_fail_clearly() -> None:
    frames = _raw_frames()
    frames["qqq"] = frames["qqq"].drop(index=3).reset_index(drop=True)

    with pytest.raises(DataValidationError, match="secondary instruments are missing"):
        align_on_spy_calendar(frames, "error")
