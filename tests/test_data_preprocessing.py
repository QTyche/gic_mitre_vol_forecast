import pandas as pd
import pytest

from qtyche_qrc.data.preprocessing import TrainStandardizer


def test_normalization_parameters_are_fit_on_training_only() -> None:
    training = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0]})
    validation = pd.DataFrame({"a": [10.0], "b": [20.0]})
    scaler = TrainStandardizer.fit(training)
    validation.loc[:, :] = 1_000_000.0

    assert TrainStandardizer.fit(training) == scaler
    assert scaler.means == (2.0, 14.0 / 3.0)


def test_zero_variance_and_column_mismatch_are_explicit() -> None:
    training = pd.DataFrame({"variable": [1.0, 2.0], "constant": [5.0, 5.0]})
    scaler = TrainStandardizer.fit(training)

    assert scaler.zero_variance_features == ("constant",)
    assert scaler.transform(training)["constant"].eq(0.0).all()
    with pytest.raises(ValueError, match="feature columns differ"):
        scaler.transform(training[["constant", "variable"]])
