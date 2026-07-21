"""Validation-only deterministic hyperparameter search."""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from qtyche_qrc.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    transition_metrics,
)
from qtyche_qrc.models.baselines.esn import (
    ESNClassifier,
    ESNConfig,
    ESNRegressor,
    split_reservoir_states,
)
from qtyche_qrc.models.baselines.logistic import MultinomialLogisticClassifier
from qtyche_qrc.models.dataset import SelectionDataset


@dataclass(frozen=True)
class CandidateResult:
    """One attempted configuration and its validation-only result."""

    trial: int
    configuration: dict[str, Any]
    selection_metric: str
    validation_score: float | None
    status: str
    error: str | None = None
    validation_transition_pr_auc: float | None = None
    training_seconds: float | None = None
    prediction_seconds: float | None = None
    reservoir_state_dimension: int | None = None
    measured_spectral_radius: float | None = None
    numerical_warnings: str | None = None
    negative_prediction_count: int | None = None
    floored_prediction_count: int | None = None


def deterministic_candidates(
    base: dict[str, Any],
    space: dict[str, list[Any]],
    maximum_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample unique Cartesian configurations without running the full grid."""

    if not space:
        return [dict(base)]
    keys = sorted(space)
    combinations = list(itertools.product(*(space[key] for key in keys)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combinations))[: min(maximum_trials, len(combinations))]
    candidates: list[dict[str, Any]] = []
    for index in order:
        candidate = dict(base)
        candidate.update(dict(zip(keys, combinations[int(index)])))
        candidates.append(candidate)
    return candidates


def select_candidate(
    results: list[CandidateResult], metric: str, *, minimize: bool
) -> dict[str, Any]:
    """Select only from successful validation results using the configured metric."""

    if any(result.selection_metric != metric for result in results):
        raise ValueError("candidate results contain an unexpected selection metric")
    successful = [result for result in results if result.status == "success"]
    if not successful:
        raise ValueError("all hyperparameter trials failed")
    key = lambda result: float(result.validation_score)  # noqa: E731
    selected = min(successful, key=key) if minimize else max(successful, key=key)
    return selected.configuration


def search_logistic(
    data: SelectionDataset,
    base: dict[str, Any],
    space: dict[str, list[Any]],
    maximum_trials: int,
    seed: int,
) -> tuple[dict[str, Any], list[CandidateResult]]:
    """Select logistic regularization from validation macro F1 only."""

    results: list[CandidateResult] = []
    for trial, params in enumerate(
        deterministic_candidates(base, space, maximum_trials, seed), start=1
    ):
        try:
            model = MultinomialLogisticClassifier(
                data.feature_names,
                regularization_c=float(params.get("regularization_c", 1.0)),
                class_weight=params.get("class_weight"),
                max_iterations=int(params.get("max_iterations", 500)),
                seed=seed,
            )
            model.fit(data.train.X, data.train.y_regime)
            score = float(
                classification_metrics(
                    data.validation.y_regime,
                    model.predict_proba(data.validation.X),
                )["macro_f1"]
            )
            results.append(CandidateResult(trial, params, "macro_f1", score, "success"))
        except Exception as exc:  # trial failures are persisted, then selection continues
            results.append(CandidateResult(trial, params, "macro_f1", None, "failure", str(exc)))
    return select_candidate(results, "macro_f1", minimize=False), results


def _esn_config(params: dict[str, Any], seed: int) -> ESNConfig:
    values = dict(params)
    values["seed"] = seed
    return ESNConfig(**values)


def search_esn_classifier(
    data: SelectionDataset,
    base: dict[str, Any],
    space: dict[str, list[Any]],
    maximum_trials: int,
    seed: int,
) -> tuple[dict[str, Any], list[CandidateResult]]:
    """Search ESN validation macro F1 without constructing test representations."""

    results: list[CandidateResult] = []
    for trial, params in enumerate(
        deterministic_candidates(base, space, maximum_trials, seed), start=1
    ):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = _esn_config(params, seed)
                model = ESNClassifier(data.feature_names, config)
                started = time.perf_counter()
                train_states, validation_states, _ = split_reservoir_states(
                    model.reservoir,
                    data.train.X,
                    data.validation.X,
                    None,
                    config.state_policy,
                )
                model.fit_readout(train_states, data.train.y_regime)
                training_seconds = time.perf_counter() - started
                started = time.perf_counter()
                probabilities = model.predict_proba_from_states(validation_states)
                prediction_seconds = time.perf_counter() - started
                score = float(
                    classification_metrics(data.validation.y_regime, probabilities)["macro_f1"]
                )
                transition_values, _, _ = transition_metrics(
                    data.validation.y_transition,
                    probabilities,
                    data.validation.current_regime,
                )
            results.append(
                CandidateResult(
                    trial,
                    params,
                    "macro_f1",
                    score,
                    "success",
                    validation_transition_pr_auc=float(transition_values["transition_pr_auc"]),
                    training_seconds=training_seconds,
                    prediction_seconds=prediction_seconds,
                    reservoir_state_dimension=config.reservoir_size,
                    measured_spectral_radius=model.reservoir.measured_spectral_radius,
                    numerical_warnings=" | ".join(str(item.message) for item in caught) or None,
                )
            )
        except Exception as exc:
            results.append(CandidateResult(trial, params, "macro_f1", None, "failure", str(exc)))
    return select_candidate(results, "macro_f1", minimize=False), results


def search_esn_regressor(
    data: SelectionDataset,
    base: dict[str, Any],
    space: dict[str, list[Any]],
    maximum_trials: int,
    seed: int,
    variance_floor: float,
) -> tuple[dict[str, Any], list[CandidateResult]]:
    """Search ESN validation QLIKE without constructing test representations."""

    results: list[CandidateResult] = []
    for trial, params in enumerate(
        deterministic_candidates(base, space, maximum_trials, seed), start=1
    ):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = _esn_config(params, seed)
                model = ESNRegressor(data.feature_names, config)
                started = time.perf_counter()
                train_states, validation_states, _ = split_reservoir_states(
                    model.reservoir,
                    data.train.X,
                    data.validation.X,
                    None,
                    config.state_policy,
                )
                model.fit_readout(train_states, data.train.y_rv)
                training_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predictions = model.predict_from_states(validation_states)
                prediction_seconds = time.perf_counter() - started
                evaluated = regression_metrics(
                    data.validation.y_rv,
                    predictions,
                    variance_floor,
                )
                score = float(evaluated.metrics["qlike"])
            results.append(
                CandidateResult(
                    trial,
                    params,
                    "qlike",
                    score,
                    "success",
                    training_seconds=training_seconds,
                    prediction_seconds=prediction_seconds,
                    reservoir_state_dimension=config.reservoir_size,
                    measured_spectral_radius=model.reservoir.measured_spectral_radius,
                    numerical_warnings=" | ".join(str(item.message) for item in caught) or None,
                    negative_prediction_count=int(np.sum(predictions < 0)),
                    floored_prediction_count=int(evaluated.floored.sum()),
                )
            )
        except Exception as exc:
            results.append(CandidateResult(trial, params, "qlike", None, "failure", str(exc)))
    return select_candidate(results, "qlike", minimize=True), results


def candidate_rows(results: list[CandidateResult]) -> list[dict[str, Any]]:
    """Flatten candidate records for durable CSV output."""

    rows: list[dict[str, Any]] = []
    for result in results:
        row = asdict(result)
        configuration = row.pop("configuration")
        row["configuration"] = json_configuration(configuration)
        rows.append(row)
    return rows


def json_configuration(configuration: dict[str, Any]) -> str:
    """Stable candidate configuration encoding."""

    import json

    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))
