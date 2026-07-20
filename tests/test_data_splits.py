import pandas as pd

from qtyche_qrc.data.config import SplitBoundary
from qtyche_qrc.data.splits import (
    assign_chronological_splits,
    validate_forward_window_containment,
)


def test_split_purging_prevents_forward_window_crossing() -> None:
    dates = pd.bdate_range("2020-01-01", periods=18)
    frame = pd.DataFrame({"date": dates, "target_window_end": dates.to_series().shift(-5).array})
    boundary = SplitBoundary("train", dates[0].date(), dates[11].date())

    split, report = assign_chronological_splits(frame, (boundary,))

    assert report["purged_forward_window_rows"]["train"] == 5
    assert split["target_window_end"].le(pd.Timestamp(boundary.end)).all()
    validate_forward_window_containment(split, (boundary,))


def test_split_dates_are_strictly_chronological() -> None:
    dates = pd.bdate_range("2020-01-01", periods=20)
    frame = pd.DataFrame({"date": dates, "target_window_end": dates.to_series().shift(-5).array})
    boundary = SplitBoundary("train", dates[0].date(), dates[15].date())

    split, _ = assign_chronological_splits(frame, (boundary,))

    assert split["date"].is_monotonic_increasing
    assert not split["date"].duplicated().any()
