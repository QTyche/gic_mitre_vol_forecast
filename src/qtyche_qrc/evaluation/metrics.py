"""Classification, transition, and realized-variance evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

CLASS_NAMES = ("low", "medium", "high")


def validate_class_probabilities(probabilities: NDArray[np.float64]) -> None:
    """Require finite three-class probability rows summing to one."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("classification probabilities must have shape (n, 3)")
    if not np.isfinite(probabilities).all():
        raise ValueError("classification probabilities must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("classification probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("classification probability rows must sum to one")


def classification_metrics(
    true_regime: NDArray[np.int_],
    probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    """Compute the complete three-class metric contract."""

    validate_class_probabilities(probabilities)
    predicted = np.argmax(probabilities, axis=1).astype(int)
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        true_regime,
        predicted,
        labels=[0, 1, 2],
        zero_division=0,
    )
    one_hot = np.eye(3, dtype=float)[true_regime.astype(int)]
    return {
        "accuracy": float(accuracy_score(true_regime, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true_regime, predicted)),
        "macro_f1": float(f1_score(true_regime, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_regime, predicted, average="weighted", zero_division=0)),
        "per_class_precision": dict(zip(CLASS_NAMES, map(float, precision))),
        "per_class_recall": dict(zip(CLASS_NAMES, map(float, recall))),
        "per_class_f1": dict(zip(CLASS_NAMES, map(float, per_class_f1))),
        "log_loss": float(log_loss(true_regime, probabilities, labels=[0, 1, 2])),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(true_regime, predicted, labels=[0, 1, 2]).tolist(),
    }


def transition_probabilities(
    regime_probabilities: NDArray[np.float64],
    current_regime: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Derive P(transition) = 1 - P(regime equals current regime)."""

    validate_class_probabilities(regime_probabilities)
    current = np.asarray(current_regime, dtype=int).reshape(-1)
    if len(current) != len(regime_probabilities) or not set(np.unique(current)).issubset({0, 1, 2}):
        raise ValueError("current regimes must align and be in {0, 1, 2}")
    return np.asarray(1.0 - regime_probabilities[np.arange(len(current)), current], dtype=float)


def transition_metrics(
    true_transition: NDArray[np.int_],
    regime_probabilities: NDArray[np.float64],
    current_regime: NDArray[np.int_],
    threshold: float = 0.5,
) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.int_]]:
    """Evaluate binary transitions derived from regime probabilities."""

    if not 0 <= threshold <= 1:
        raise ValueError("transition threshold must lie in [0, 1]")
    probabilities = transition_probabilities(regime_probabilities, current_regime)
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "transition_rate": float(np.mean(true_transition)),
        "transition_accuracy": float(accuracy_score(true_transition, predicted)),
        "transition_balanced_accuracy": float(balanced_accuracy_score(true_transition, predicted)),
        "transition_f1": float(f1_score(true_transition, predicted, zero_division=0)),
        "transition_roc_auc": float(roc_auc_score(true_transition, probabilities)),
        "transition_pr_auc": float(average_precision_score(true_transition, probabilities)),
        "transition_brier_score": float(np.mean((probabilities - true_transition) ** 2)),
        "transition_probability_threshold": threshold,
    }
    return metrics, probabilities, predicted


def transition_subgroup_metrics(
    true_regime: NDArray[np.int_],
    current_regime: NDArray[np.int_],
    regime_probabilities: NDArray[np.float64],
    threshold: float = 0.5,
    unstable_below: int = 30,
) -> dict[str, Any]:
    """Report transition detection by current regime and movement direction."""

    truth = np.asarray(true_regime, dtype=int)
    current = np.asarray(current_regime, dtype=int)
    probabilities = transition_probabilities(regime_probabilities, current)

    def evaluate(mask: NDArray[np.bool_], binary_truth: NDArray[np.int_]) -> dict[str, Any]:
        values = binary_truth[mask]
        scores = probabilities[mask]
        predicted = (scores >= threshold).astype(int)
        count = int(mask.sum())
        positives = int(values.sum())
        two_classes = len(np.unique(values)) == 2
        return {
            "count": count,
            "positive_count": positives,
            "positive_rate": float(values.mean()) if count else None,
            "accuracy": float(accuracy_score(values, predicted)) if count else None,
            "balanced_accuracy": float(balanced_accuracy_score(values, predicted))
            if two_classes
            else None,
            "f1": float(f1_score(values, predicted, zero_division=0)) if count else None,
            "roc_auc": float(roc_auc_score(values, scores)) if two_classes else None,
            "pr_auc": float(average_precision_score(values, scores)) if two_classes else None,
            "unstable": count < unstable_below or not two_classes,
        }

    transition_truth = (truth != current).astype(int)
    upward_truth = (truth > current).astype(int)
    downward_truth = (truth < current).astype(int)
    return {
        "low_origin": evaluate(current == 0, transition_truth),
        "medium_origin": evaluate(current == 1, transition_truth),
        "high_origin": evaluate(current == 2, transition_truth),
        "upward": evaluate(current < 2, upward_truth),
        "downward": evaluate(current > 0, downward_truth),
    }


def qlike(true_variance: NDArray[np.float64], predicted_variance: NDArray[np.float64]) -> float:
    """Return mean(log(y_hat) + y / y_hat) for strictly positive variances."""

    true_values = np.asarray(true_variance, dtype=float)
    predicted_values = np.asarray(predicted_variance, dtype=float)
    if not np.isfinite(true_values).all() or np.any(true_values <= 0):
        raise ValueError("true target variance must be finite and strictly positive")
    if not np.isfinite(predicted_values).all() or np.any(predicted_values <= 0):
        raise ValueError("QLIKE predictions must be finite and strictly positive")
    return float(np.mean(np.log(predicted_values) + true_values / predicted_values))


@dataclass(frozen=True)
class RegressionEvaluation:
    """Metrics plus explicit positive-floor transformations and diagnostics."""

    metrics: dict[str, Any]
    predictions: NDArray[np.float64]
    floored: NDArray[np.bool_]


def regression_metrics(
    true_variance: NDArray[np.float64],
    raw_predictions: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> RegressionEvaluation:
    """Floor invalid predictions at evaluation only and report every adjustment."""

    if epsilon <= 0:
        raise ValueError("variance floor epsilon must be positive")
    true_values = np.asarray(true_variance, dtype=float).reshape(-1)
    predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)
    if len(true_values) != len(predictions):
        raise ValueError("regression targets and predictions must align")
    if not np.isfinite(true_values).all() or np.any(true_values <= 0):
        raise ValueError("true target variance must be finite and strictly positive")
    non_finite = ~np.isfinite(predictions)
    floored = non_finite | (predictions < epsilon)
    adjusted = predictions.copy()
    adjusted[floored] = epsilon
    metrics = {
        "rmse": float(mean_squared_error(true_values, adjusted) ** 0.5),
        "mae": float(mean_absolute_error(true_values, adjusted)),
        "qlike": qlike(true_values, adjusted),
        "r_squared": float(r2_score(true_values, adjusted)),
        "prediction_mean": float(adjusted.mean()),
        "prediction_median": float(np.median(adjusted)),
        "prediction_minimum": float(adjusted.min()),
        "prediction_maximum": float(adjusted.max()),
        "non_finite_prediction_count": int(non_finite.sum()),
        "floored_prediction_count": int(floored.sum()),
        "prediction_floor": epsilon,
        "floor_policy": "non-finite or prediction < epsilon replaced with epsilon at evaluation",
    }
    return RegressionEvaluation(metrics, adjusted, floored)


# Extension points intentionally reserved for later Mincer-Zarnowitz and
# Diebold-Mariano statistical analysis.
