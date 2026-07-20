from typing import Any, cast

import pandas as pd
import pytest

from qtyche_qrc.data.features import FEATURE_NAMES, build_features, ensure_sufficient_rows
from tests.data_helpers import canonical_frame


def test_changing_future_observations_does_not_change_features_at_t() -> None:
    frame = canonical_frame()
    observation_index = 35
    feature_names = list(FEATURE_NAMES)
    original = cast(
        "pd.Series[Any]", build_features(frame).loc[observation_index].loc[feature_names]
    )
    changed = frame.copy()
    future = changed.index > observation_index
    for column in (
        "spy_open",
        "spy_high",
        "spy_low",
        "spy_close",
        "spy_volume",
        "vix_close",
        "qqq_close",
        "qqq_volume",
    ):
        changed.loc[future, column] *= 10.0
    after_change = cast(
        "pd.Series[Any]", build_features(changed).loc[observation_index].loc[feature_names]
    )

    pd.testing.assert_series_equal(original, after_change)


def test_insufficient_history_fails_clearly() -> None:
    with pytest.raises(ValueError, match="insufficient history"):
        ensure_sufficient_rows(canonical_frame(rows=20))
