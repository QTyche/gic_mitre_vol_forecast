from pathlib import Path

import numpy as np
import pytest

from qtyche_qrc.evaluation.metrics import transition_probabilities
from qtyche_qrc.models.baselines.logistic import MultinomialLogisticClassifier
from qtyche_qrc.models.baselines.persistence import (
    CurrentRegimePersistenceClassifier,
    MajorityClassClassifier,
    RealizedVariancePersistenceRegressor,
)


def test_majority_classifier_uses_training_labels_and_empirical_frequencies() -> None:
    X_train = np.zeros((6, 2))
    y_train = np.array([0, 0, 0, 1, 1, 2])
    model = MajorityClassClassifier(("a", "b"), seed=3)
    model.fit(X_train, y_train)
    validation = np.full((4, 2), 999.0)

    assert model.predict(validation).tolist() == [0, 0, 0, 0]
    np.testing.assert_allclose(
        model.predict_proba(validation)[0], np.array([0.5, 1.0 / 3.0, 1.0 / 6.0])
    )


def test_persistence_classifier_predicts_explicit_current_regime() -> None:
    current = np.array([[0.0], [1.0], [2.0]])
    model = CurrentRegimePersistenceClassifier()
    model.fit(current, np.array([2, 2, 0]))

    np.testing.assert_array_equal(model.predict(current), np.array([0, 1, 2]))
    probabilities = model.predict_proba(current)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_transition_probability_uses_each_current_regime_column() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])

    actual = transition_probabilities(probabilities, np.array([0, 1, 2]))

    np.testing.assert_allclose(actual, np.array([0.3, 0.4, 0.5]))


def test_rv_persistence_uses_unscaled_physical_variance() -> None:
    current_rv = np.array([[0.01], [0.03], [0.02]])
    model = RealizedVariancePersistenceRegressor()
    model.fit(current_rv, np.array([0.02, 0.04, 0.01]))

    np.testing.assert_array_equal(model.predict(current_rv), current_rv.reshape(-1))
    assert model.get_params()["annualization"] == "252/5 trailing squared log returns"


def test_predict_before_fit_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        MajorityClassClassifier(("a",)).predict(np.ones((1, 1)))


def test_logistic_serialization_preserves_probabilities(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(60, 3))
    targets = np.repeat(np.array([0, 1, 2]), 20)
    model = MultinomialLogisticClassifier(("a", "b", "c"), seed=4)
    model.fit(features, targets)
    expected = model.predict_proba(features[:5])
    model.save(tmp_path)

    loaded = MultinomialLogisticClassifier.load(tmp_path)

    np.testing.assert_allclose(loaded.predict_proba(features[:5]), expected)
