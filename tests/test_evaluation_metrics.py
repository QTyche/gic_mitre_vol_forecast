import numpy as np
import pytest

from qtyche_qrc.evaluation.metrics import qlike, regression_metrics


def test_qlike_matches_hand_calculation() -> None:
    truth = np.array([2.0, 4.0])
    prediction = np.array([1.0, 2.0])
    expected = np.mean(np.log(prediction) + truth / prediction)

    assert qlike(truth, prediction) == pytest.approx(expected)


def test_invalid_variance_predictions_are_floored_and_reported() -> None:
    evaluated = regression_metrics(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, np.nan, -2.0]), epsilon=1e-6
    )

    assert evaluated.metrics["non_finite_prediction_count"] == 1
    assert evaluated.metrics["floored_prediction_count"] == 3
    assert evaluated.metrics["prediction_floor"] == 1e-6
    np.testing.assert_array_equal(evaluated.predictions, np.full(3, 1e-6))


def test_non_positive_true_variance_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        regression_metrics(np.array([0.0]), np.array([1.0]))
