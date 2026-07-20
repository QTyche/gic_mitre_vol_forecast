from typing import cast

import numpy as np
import pandas as pd

from qtyche_qrc.data.targets import (
    RegimeThresholds,
    add_continuous_targets,
    add_regime_and_transition_targets,
    fit_regime_thresholds,
)


def test_forward_target_uses_exactly_t_plus_one_through_t_plus_five() -> None:
    returns = np.array([0.0, 0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08])
    close = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2020-01-01", periods=len(close)), "spy_close": close}
    )

    targeted = add_continuous_targets(frame)
    observation_index = 1
    expected = (252.0 / 5.0) * np.square(returns[2:7]).sum()

    actual = cast(float, targeted.loc[observation_index, "target_rv_5d"])
    assert np.isclose(actual, expected)
    assert targeted.loc[observation_index, "target_window_end"] == frame.loc[6, "date"]


def test_regime_thresholds_ignore_validation_and_test_targets() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train"] * 6 + ["validation"] * 2 + ["test"] * 2,
            "target_rv_5d": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 100.0, 200.0, 300.0, 400.0],
        }
    )
    original = fit_regime_thresholds(frame)
    frame.loc[frame["split"].ne("train"), "target_rv_5d"] *= 1_000_000

    assert fit_regime_thresholds(frame) == original


def test_regime_and_transition_labels_are_consistent() -> None:
    frame = pd.DataFrame(
        {
            "target_rv_5d": [0.5, 2.0, 5.0, 0.5],
            "spy_rv_5d": [0.5, 0.5, 5.0, 5.0],
        }
    )
    thresholds = RegimeThresholds(1.0, 3.0, (0.33, 0.66), 10)

    labeled = add_regime_and_transition_targets(frame, thresholds)

    assert set(labeled["target_regime_5d"]).issubset({0, 1, 2})
    assert set(labeled["current_regime"]).issubset({0, 1, 2})
    assert labeled["target_transition"].tolist() == [0, 1, 0, 1]
    assert labeled["target_upward_transition"].tolist() == [0, 1, 0, 0]
    assert labeled["target_downward_transition"].tolist() == [0, 0, 0, 1]
