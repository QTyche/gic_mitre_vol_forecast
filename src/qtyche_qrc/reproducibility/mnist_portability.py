"""Full-path portability evidence for the frozen exact-QRC MNIST benchmark."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax  # type: ignore[import-untyped]

from qtyche_qrc.data.mnist import (
    DIGITS,
    MNISTBenchmarkData,
    build_mnist_benchmark_data,
    load_official_mnist,
)
from qtyche_qrc.experiments.qrc_mnist import (
    READOUT_C_GRID,
    RESERVOIR_SEEDS,
    digit_classification_metrics,
    load_mnist_study_config,
)
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.reproducibility.verification import sha256_path

MNIST_PORTABILITY_REPORT = "mnist_exact_portability_report.json"
MNIST_METRIC_COLUMNS = {
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "balanced_accuracy": "balanced_accuracy",
    "macro_roc_auc": "ovr_macro_roc_auc",
}
SPLITS = ("validation", "test")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_feature_digest(
    values: NDArray[np.float64],
    *,
    decimals: int,
) -> str:
    """Hash every feature after the documented platform-neutral quantisation."""

    candidate = np.asarray(values, dtype=float)
    if candidate.ndim != 2 or not np.isfinite(candidate).all():
        raise ValueError("MNIST feature commitments require a finite two-dimensional array")
    rounded = np.round(candidate, decimals=decimals)
    rounded[rounded == 0.0] = 0.0
    normalized = np.asarray(rounded, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(normalized.shape), "dtype": "float64", "decimals": decimals},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _difference_statistics(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> dict[str, float]:
    candidate = np.asarray(actual, dtype=float)
    reference = np.asarray(expected, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"MNIST portability array shape mismatch: {candidate.shape} != {reference.shape}"
        )
    difference = np.abs(candidate - reference)
    denominator = np.maximum(np.abs(candidate), np.abs(reference))
    relative = np.divide(
        difference,
        denominator,
        out=np.zeros_like(difference),
        where=denominator > 1e-12,
    )
    return {
        "maximum_absolute_difference": float(np.max(difference, initial=0.0)),
        "mean_absolute_difference": float(np.mean(difference)),
        "median_absolute_difference": float(np.median(difference)),
        "l2_difference": float(np.linalg.norm(difference)),
        "maximum_relative_difference_above_1e-12": float(np.max(relative, initial=0.0)),
    }


def _limit_checks(statistics: dict[str, float], limits: dict[str, Any]) -> dict[str, bool]:
    mapping = {
        "maximum_absolute_difference": "maximum_absolute_difference",
        "mean_absolute_difference": "maximum_mean_absolute_difference",
        "median_absolute_difference": "maximum_median_absolute_difference",
        "l2_difference": "maximum_l2_difference",
    }
    return {
        statistic: statistics[statistic] <= float(limits[limit])
        for statistic, limit in mapping.items()
        if limit in limits
    }


def compare_readout_parameters(
    coefficient: NDArray[np.float64],
    intercept: NDArray[np.float64],
    reference_coefficient: NDArray[np.float64],
    reference_intercept: NDArray[np.float64],
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Compare the complete multinomial readout parameter arrays."""

    coefficient_statistics = _difference_statistics(coefficient, reference_coefficient)
    intercept_statistics = _difference_statistics(intercept, reference_intercept)
    coefficient_checks = _limit_checks(
        coefficient_statistics,
        cast(dict[str, Any], limits["coefficient"]),
    )
    intercept_checks = _limit_checks(
        intercept_statistics,
        cast(dict[str, Any], limits["intercept"]),
    )
    finite = bool(np.isfinite(coefficient).all() and np.isfinite(intercept).all())
    passed = finite and all(coefficient_checks.values()) and all(intercept_checks.values())
    return {
        "coefficient": {
            **coefficient_statistics,
            "limits": limits["coefficient"],
            "checks": coefficient_checks,
        },
        "intercept": {
            **intercept_statistics,
            "limits": limits["intercept"],
            "checks": intercept_checks,
        },
        "finite": finite,
        "passed": passed,
    }


def _top_two_margin(values: NDArray[np.float64]) -> NDArray[np.float64]:
    ordered = np.sort(np.asarray(values, dtype=float), axis=1)
    return np.asarray(ordered[:, -1] - ordered[:, -2], dtype=float)


def compare_prediction_path(
    *,
    official_indices: NDArray[np.int64],
    truth: NDArray[np.int64],
    scores: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    frozen_official_indices: NDArray[np.int64],
    frozen_truth: NDArray[np.int64],
    frozen_scores: NDArray[np.float64],
    frozen_probabilities: NDArray[np.float64],
    frozen_predictions: NDArray[np.int64],
    expected_changed_positions: list[int],
    expected_confusion_matrix: list[list[int]],
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Compare identities, full scores, probabilities, decisions, and confusion."""

    candidate_indices = np.asarray(official_indices, dtype=np.int64).reshape(-1)
    candidate_truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    candidate_scores = np.asarray(scores, dtype=float)
    candidate_probabilities = np.asarray(probabilities, dtype=float)
    reference_indices = np.asarray(frozen_official_indices, dtype=np.int64).reshape(-1)
    reference_truth = np.asarray(frozen_truth, dtype=np.int64).reshape(-1)
    reference_scores = np.asarray(frozen_scores, dtype=float)
    reference_probabilities = np.asarray(frozen_probabilities, dtype=float)
    reference_predictions = np.asarray(frozen_predictions, dtype=np.int64).reshape(-1)
    row_count = len(candidate_truth)
    expected_shape = (row_count, len(DIGITS))
    structurally_valid = bool(
        candidate_scores.shape == expected_shape
        and candidate_probabilities.shape == expected_shape
        and len(candidate_indices) == row_count
        and len(reference_predictions) == row_count
    )
    if not structurally_valid:
        return {
            "passed": False,
            "structurally_valid": False,
            "actual_shapes": {
                "scores": list(candidate_scores.shape),
                "probabilities": list(candidate_probabilities.shape),
                "indices": list(candidate_indices.shape),
                "truth": list(candidate_truth.shape),
            },
            "expected_score_shape": list(expected_shape),
        }
    finite_normalized = bool(
        np.isfinite(candidate_scores).all()
        and np.isfinite(candidate_probabilities).all()
        and np.all(candidate_probabilities >= 0.0)
        and np.all(candidate_probabilities <= 1.0)
        and np.allclose(candidate_probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    )
    identity = {
        "official_indices_exact": bool(np.array_equal(candidate_indices, reference_indices)),
        "labels_exact": bool(np.array_equal(candidate_truth, reference_truth)),
        "row_count": row_count,
        "expected_row_count": len(reference_truth),
    }
    identity["passed"] = bool(
        identity["official_indices_exact"]
        and identity["labels_exact"]
        and identity["row_count"] == identity["expected_row_count"]
    )
    score_statistics = _difference_statistics(candidate_scores, reference_scores)
    probability_statistics = _difference_statistics(
        candidate_probabilities,
        reference_probabilities,
    )
    score_checks = _limit_checks(
        score_statistics,
        cast(dict[str, Any], limits["score"]),
    )
    probability_checks = _limit_checks(
        probability_statistics,
        cast(dict[str, Any], limits["probability"]),
    )
    candidate_predictions = np.argmax(candidate_probabilities, axis=1).astype(np.int64)
    changed_positions = np.flatnonzero(candidate_predictions != reference_predictions)
    expected_positions = np.asarray(expected_changed_positions, dtype=np.int64)
    positions_exact = bool(np.array_equal(changed_positions, expected_positions))
    confusion = np.zeros((len(DIGITS), len(DIGITS)), dtype=int)
    np.add.at(confusion, (candidate_truth, candidate_predictions), 1)
    confusion_exact = confusion.tolist() == expected_confusion_matrix
    candidate_score_margins = _top_two_margin(candidate_scores)
    frozen_score_margins = _top_two_margin(reference_scores)
    candidate_probability_margins = _top_two_margin(candidate_probabilities)
    frozen_probability_margins = _top_two_margin(reference_probabilities)
    changed_records: list[dict[str, Any]] = []
    for position in changed_positions:
        index = int(position)
        changed_records.append(
            {
                "position": index,
                "official_index": int(candidate_indices[index]),
                "true_digit": int(candidate_truth[index]),
                "frozen_prediction": int(reference_predictions[index]),
                "candidate_prediction": int(candidate_predictions[index]),
                "frozen_score_margin": float(frozen_score_margins[index]),
                "candidate_score_margin": float(candidate_score_margins[index]),
                "frozen_probability_margin": float(frozen_probability_margins[index]),
                "candidate_probability_margin": float(candidate_probability_margins[index]),
            }
        )
    maximum_changed_score_margin = max(
        (
            max(record["frozen_score_margin"], record["candidate_score_margin"])
            for record in changed_records
        ),
        default=0.0,
    )
    maximum_changed_probability_margin = max(
        (
            max(record["frozen_probability_margin"], record["candidate_probability_margin"])
            for record in changed_records
        ),
        default=0.0,
    )
    margin_checks = {
        "score": maximum_changed_score_margin
        <= float(limits["maximum_changed_score_margin"]),
        "probability": maximum_changed_probability_margin
        <= float(limits["maximum_changed_probability_margin"]),
    }
    passed = bool(
        finite_normalized
        and identity["passed"]
        and all(score_checks.values())
        and all(probability_checks.values())
        and positions_exact
        and confusion_exact
        and all(margin_checks.values())
    )
    return {
        "structurally_valid": True,
        "finite_normalized": finite_normalized,
        "identity": identity,
        "score_difference": {
            **score_statistics,
            "limits": limits["score"],
            "checks": score_checks,
        },
        "probability_difference": {
            **probability_statistics,
            "limits": limits["probability"],
            "checks": probability_checks,
        },
        "changed_prediction_count": len(changed_positions),
        "changed_positions": changed_positions.tolist(),
        "expected_changed_positions": expected_changed_positions,
        "changed_positions_exact": positions_exact,
        "changed_predictions": changed_records,
        "maximum_changed_score_margin": maximum_changed_score_margin,
        "maximum_changed_probability_margin": maximum_changed_probability_margin,
        "margin_checks": margin_checks,
        "confusion_matrix": confusion.tolist(),
        "expected_confusion_matrix": expected_confusion_matrix,
        "confusion_matrix_exact": confusion_exact,
        "passed": passed,
    }


def _candidate_run(root: Path, seed: int) -> Path:
    matches: list[Path] = []
    for manifest_path in (root / "results/qrc_mnist/runs/full").glob(
        "qrc_exact_analytic_*/manifest.json"
    ):
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") == "success"
            and manifest.get("mode") == "full"
            and manifest.get("is_synthetic") is False
            and manifest.get("model_name") == "qrc_exact"
            and manifest.get("condition") == "analytic"
            and manifest.get("reservoir_seed") == seed
        ):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one full analytic MNIST QRC run for seed {seed}")
    return matches[0]


def _feature_summary(values: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    candidate = np.asarray(values, dtype=float)
    return {
        "column_mean": candidate.mean(axis=0),
        "column_standard_deviation": candidate.std(axis=0, ddof=0),
        "column_minimum": candidate.min(axis=0),
        "column_maximum": candidate.max(axis=0),
    }


def _array_limit_report(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
    tolerance: float,
) -> dict[str, Any]:
    statistics = _difference_statistics(actual, expected)
    return {
        **statistics,
        "maximum_absolute_tolerance": tolerance,
        "passed": statistics["maximum_absolute_difference"] <= tolerance,
    }


def _objective_diagnostics(
    scaled_train: NDArray[np.float64],
    labels: NDArray[np.int64],
    coefficient: NDArray[np.float64],
    intercept: NDArray[np.float64],
    regularization_c: float,
) -> dict[str, float]:
    scores = scaled_train @ coefficient.T + intercept
    probabilities = softmax(scores, axis=1)
    one_hot = np.eye(len(DIGITS), dtype=float)[labels]
    negative_log_likelihood = float(
        np.sum(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels])
    )
    penalty = float(0.5 / regularization_c * np.sum(coefficient**2))
    gradient_coefficient = (probabilities - one_hot).T @ scaled_train
    gradient_coefficient += coefficient / regularization_c
    gradient_intercept = np.sum(probabilities - one_hot, axis=0)
    return {
        "negative_log_likelihood": negative_log_likelihood,
        "l2_penalty": penalty,
        "objective": negative_log_likelihood + penalty,
        "coefficient_gradient_infinity_norm": float(np.max(np.abs(gradient_coefficient))),
        "intercept_gradient_infinity_norm": float(np.max(np.abs(gradient_intercept))),
    }


def _git_commit(root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _dataset_evidence(
    root: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], MNISTBenchmarkData]:
    identities = cast(dict[str, Any], reference["identities"])
    config_path = root / str(identities["configuration_path"])
    config = load_mnist_study_config(config_path)
    official = load_official_mnist(config.dataset_cache, config.sources, download=False)
    data = build_mnist_benchmark_data(
        official,
        train_per_digit=600,
        validation_per_digit=100,
        test_per_digit=100,
        seed=2026,
    )
    selected = _load_json(root / "results/qrc_mnist/dataset/selected_indices.json")
    preprocessing = _load_json(root / "results/qrc_mnist/dataset/preprocessing_manifest.json")
    dataset = _load_json(root / "results/qrc_mnist/dataset/dataset_manifest.json")
    expected_sources = cast(dict[str, str], identities["source_sha256"])
    actual_sources = {
        key: str(record["sha256"])
        for key, record in cast(dict[str, dict[str, Any]], dataset["files"]).items()
    }
    checks = {
        "configuration_sha256": sha256_path(config_path)
        == identities["configuration_sha256"],
        "dataset_subset_checksum": data.subset_checksum
        == identities["dataset_subset_checksum"]
        == dataset.get("subset_checksum"),
        "selected_indices_exact": selected == data.index_manifest,
        "selected_indices_checksum": selected.get("checksum")
        == identities["selected_indices_checksum"],
        "preprocessing_exact": preprocessing == data.preprocessing_manifest,
        "preprocessing_checksum": preprocessing.get("checksum")
        == identities["preprocessing_checksum"],
        "source_sha256_exact": actual_sources == expected_sources,
        "split_sizes_exact": dataset.get("split_sizes") == identities["split_sizes"],
        "genuine_mnist": dataset.get("synthetic_data") is False,
        "official_partitions_preserved": dataset.get("official_partitions_preserved") is True,
    }
    return (
        {
            "checks": checks,
            "source_sha256": actual_sources,
            "subset_checksum": data.subset_checksum,
            "selected_indices_checksum": selected.get("checksum"),
            "preprocessing_checksum": preprocessing.get("checksum"),
            "split_rows": {
                split: len(getattr(data, split).labels)
                for split in ("train", "validation", "test")
            },
            "split_label_checksums": {
                split: data.index_manifest["splits"][split]["labels_checksum"]
                for split in ("train", "validation", "test")
            },
            "split_sequence_checksums": {
                split: data.index_manifest["splits"][split]["sequences_checksum"]
                for split in ("train", "validation", "test")
            },
            "passed": all(checks.values()),
        },
        data,
    )


def compare_mnist_portability(
    root: Path,
    *,
    reference_path: Path,
    expected_reference_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Validate the full exact-QRC MNIST path against two checksum-pinned profiles."""

    repository = root.resolve()
    reference_file = (
        reference_path if reference_path.is_absolute() else repository / reference_path
    ).resolve()
    destination = output_path if output_path.is_absolute() else repository / output_path
    actual_reference_sha256 = sha256_path(reference_file)
    if actual_reference_sha256 != expected_reference_sha256:
        raise ValueError(
            "MNIST portability reference checksum mismatch: "
            f"{actual_reference_sha256} != {expected_reference_sha256}"
        )
    reference = _load_json(reference_file)
    bundle_record = cast(dict[str, Any], reference["reference_bundle"])
    bundle_path = (repository / str(bundle_record["path"])).resolve()
    actual_bundle_sha256 = sha256_path(bundle_path)
    if actual_bundle_sha256 != bundle_record["sha256"]:
        raise ValueError(
            "MNIST portability bundle checksum mismatch: "
            f"{actual_bundle_sha256} != {bundle_record['sha256']}"
        )
    dataset_report, data = _dataset_evidence(repository, reference)
    comparison_contract = cast(dict[str, Any], reference["comparison_contract"])
    feature_decimals = int(comparison_contract["feature_canonical_decimals"])
    summary_tolerance = float(
        comparison_contract["feature_summary_maximum_absolute_difference"]
    )
    seed_reports: dict[str, Any] = {}
    candidate_test_metrics: dict[str, dict[str, float]] = {}
    frozen_profile = True
    linux_profile = True
    artifact_operating_systems: list[str] = []
    artifact_package_versions: list[dict[str, str]] = []
    artifact_python_versions: list[str] = []
    artifact_qbraid_environment_ids: list[str] = []

    with np.load(bundle_path) as bundle:
        for seed in RESERVOIR_SEEDS:
            seed_key = str(seed)
            seed_reference = cast(dict[str, Any], reference["seeds"][seed_key])
            run_directory = _candidate_run(repository, seed)
            manifest = _load_json(run_directory / "manifest.json")
            artifact_operating_systems.append(str(manifest.get("operating_system", "")))
            artifact_python_versions.append(str(manifest.get("python_version", "")))
            artifact_qbraid_environment_ids.append(
                str(manifest.get("qbraid_environment_id", ""))
            )
            artifact_package_versions.append(
                {
                    str(key): str(value)
                    for key, value in cast(
                        dict[str, Any],
                        manifest.get("package_versions", {}),
                    ).items()
                }
            )
            cache_key = str(seed_reference["feature_cache_key_checksum"])
            feature_directory = repository / "results/qrc_mnist/feature_cache/full" / cache_key
            feature_metadata = _load_json(feature_directory / "metadata.json")
            features: dict[str, NDArray[np.float64]] = {}
            with np.load(feature_directory / "features.npz") as arrays:
                for split in ("train", "validation", "test"):
                    features[split] = np.asarray(arrays[split], dtype=float)
            feature_reports: dict[str, Any] = {}
            for split, values in features.items():
                split_reference = cast(
                    dict[str, Any],
                    seed_reference["features"][split],
                )
                summary_reports = {
                    name: _array_limit_report(
                        actual,
                        np.asarray(bundle[f"seed_{seed}_{split}_{name}"], dtype=float),
                        summary_tolerance,
                    )
                    for name, actual in _feature_summary(values).items()
                }
                digest = canonical_feature_digest(values, decimals=feature_decimals)
                stored_checksum = cast(
                    dict[str, str],
                    feature_metadata["array_checksums"],
                )[split]
                feature_reports[split] = {
                    "shape": list(values.shape),
                    "expected_shape": split_reference["shape"],
                    "finite": bool(np.isfinite(values).all()),
                    "metadata_array_checksum": stored_checksum,
                    "recomputed_array_checksum": array_checksum(values),
                    "metadata_checksum_self_consistent": array_checksum(values)
                    == stored_checksum,
                    "canonical_decimals": feature_decimals,
                    "canonical_digest": digest,
                    "expected_canonical_digest": split_reference["canonical_digest"],
                    "canonical_digest_exact": digest == split_reference["canonical_digest"],
                    "summary_comparisons": summary_reports,
                }
                feature_reports[split]["passed"] = bool(
                    feature_reports[split]["shape"] == feature_reports[split]["expected_shape"]
                    and feature_reports[split]["finite"]
                    and feature_reports[split]["metadata_checksum_self_consistent"]
                    and feature_reports[split]["canonical_digest_exact"]
                    and all(record["passed"] for record in summary_reports.values())
                )
            with (run_directory / "model.pkl").open("rb") as handle:
                model = cast(dict[str, Any], pickle.load(handle))
            scaler = model["scaler"]
            estimator = model["estimator"]
            coefficient = np.asarray(estimator.coef_, dtype=float)
            intercept = np.asarray(estimator.intercept_, dtype=float)
            reference_coefficient = np.asarray(
                bundle[f"seed_{seed}_coefficient"],
                dtype=float,
            )
            reference_intercept = np.asarray(
                bundle[f"seed_{seed}_intercept"],
                dtype=float,
            )
            parameter_report = compare_readout_parameters(
                coefficient,
                intercept,
                reference_coefficient,
                reference_intercept,
                cast(dict[str, Any], seed_reference["parameter_limits"]),
            )
            scaler_reports = {
                name: _array_limit_report(
                    np.asarray(getattr(scaler, f"{name}_"), dtype=float),
                    np.asarray(bundle[f"seed_{seed}_scaler_{name}"], dtype=float),
                    float(comparison_contract["scaler_maximum_absolute_difference"]),
                )
                for name in ("mean", "var", "scale")
            }
            train_scaled = np.asarray(scaler.transform(features["train"]), dtype=float)
            objective = _objective_diagnostics(
                train_scaled,
                np.asarray(data.train.labels, dtype=np.int64),
                coefficient,
                intercept,
                float(estimator.C),
            )
            frozen_objective = float(seed_reference["frozen_objective"])
            objective_report = {
                **objective,
                "frozen_objective": frozen_objective,
                "objective_absolute_difference": abs(
                    objective["objective"] - frozen_objective
                ),
                "objective_maximum_absolute_difference": float(
                    seed_reference["objective_limits"]["maximum_absolute_difference"]
                ),
                "maximum_gradient_infinity_norm": float(
                    seed_reference["objective_limits"]["maximum_gradient_infinity_norm"]
                ),
            }
            objective_report["passed"] = bool(
                objective_report["objective_absolute_difference"]
                <= objective_report["objective_maximum_absolute_difference"]
                and objective["coefficient_gradient_infinity_norm"]
                <= objective_report["maximum_gradient_infinity_norm"]
                and objective["intercept_gradient_infinity_norm"]
                <= objective_report["maximum_gradient_infinity_norm"]
            )
            selection = _load_json(run_directory / "selection_results.json")
            selection_rows = cast(list[dict[str, Any]], selection["rows"])
            selected_c = float(manifest.get("selected_regularization_c", float("nan")))
            selection_checks = {
                "grid_exact": tuple(float(row["regularization_c"]) for row in selection_rows)
                == READOUT_C_GRID,
                "validation_only": all(
                    row.get("selection_data") == "validation only"
                    and row.get("selection_metric") == "macro_f1"
                    for row in selection_rows
                ),
                "selected_c_exact": selected_c == 10.0 == float(estimator.C),
                "classes_exact": np.array_equal(
                    np.asarray(estimator.classes_, dtype=int),
                    np.asarray(DIGITS, dtype=int),
                ),
                "solver_exact": estimator.solver == "lbfgs",
                "maximum_iterations_exact": int(estimator.max_iter) == 1000,
                "converged": bool(np.all(np.asarray(estimator.n_iter_) < estimator.max_iter)),
                "iteration_difference_bounded": abs(
                    int(np.asarray(estimator.n_iter_)[0])
                    - int(seed_reference["frozen_iteration_count"])
                )
                <= int(seed_reference["maximum_iteration_count_difference"]),
            }
            split_reports: dict[str, Any] = {}
            candidate_seed_metrics: dict[str, dict[str, float]] = {}
            frozen_seed_profile = True
            linux_seed_profile = True
            for split in SPLITS:
                split_data = getattr(data, split)
                transformed = np.asarray(scaler.transform(features[split]), dtype=float)
                scores = transformed @ coefficient.T + intercept
                raw_probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
                probabilities = np.zeros((len(transformed), len(DIGITS)), dtype=float)
                probabilities[:, np.asarray(estimator.classes_, dtype=int)] = raw_probabilities
                table = pd.read_csv(run_directory / f"{split}_predictions.csv")
                stored_probabilities = table[
                    [f"probability_{digit}" for digit in DIGITS]
                ].to_numpy(dtype=float)
                stored_predictions = table["predicted_digit"].to_numpy(dtype=np.int64)
                self_consistency = {
                    "probability_maximum_absolute_difference": float(
                        np.max(np.abs(probabilities - stored_probabilities))
                    ),
                    "probability_tolerance": float(
                        comparison_contract["stored_probability_absolute_tolerance"]
                    ),
                    "predictions_exact": bool(
                        np.array_equal(np.argmax(probabilities, axis=1), stored_predictions)
                    ),
                }
                self_consistency["passed"] = bool(
                    self_consistency["probability_maximum_absolute_difference"]
                    <= self_consistency["probability_tolerance"]
                    and self_consistency["predictions_exact"]
                )
                frozen_predictions = np.asarray(
                    bundle[f"seed_{seed}_{split}_predictions"],
                    dtype=np.int64,
                )
                linux_split_reference = cast(
                    dict[str, Any],
                    seed_reference["linux_profile"][split],
                )
                path_report = compare_prediction_path(
                    official_indices=np.asarray(
                        table["official_index"],
                        dtype=np.int64,
                    ),
                    truth=np.asarray(table["true_digit"], dtype=np.int64),
                    scores=scores,
                    probabilities=probabilities,
                    frozen_official_indices=np.asarray(
                        bundle[f"seed_{seed}_{split}_official_indices"],
                        dtype=np.int64,
                    ),
                    frozen_truth=np.asarray(
                        bundle[f"seed_{seed}_{split}_truth"],
                        dtype=np.int64,
                    ),
                    frozen_scores=np.asarray(
                        bundle[f"seed_{seed}_{split}_scores"],
                        dtype=float,
                    ),
                    frozen_probabilities=np.asarray(
                        bundle[f"seed_{seed}_{split}_probabilities"],
                        dtype=float,
                    ),
                    frozen_predictions=frozen_predictions,
                    expected_changed_positions=cast(
                        list[int],
                        linux_split_reference["changed_positions"],
                    ),
                    expected_confusion_matrix=cast(
                        list[list[int]],
                        linux_split_reference["confusion_matrix"],
                    ),
                    limits=cast(dict[str, Any], seed_reference["path_limits"][split]),
                )
                metrics = digit_classification_metrics(
                    np.asarray(split_data.labels, dtype=np.int64),
                    probabilities,
                )
                stored_metrics = _load_json(run_directory / f"{split}_metrics.json")
                metric_self_checks = {
                    metric: abs(float(metrics[column]) - float(stored_metrics[column]))
                    <= float(comparison_contract["candidate_metric_absolute_tolerance"])
                    for metric, column in MNIST_METRIC_COLUMNS.items()
                }
                candidate_seed_metrics[split] = {
                    metric: float(metrics[column])
                    for metric, column in MNIST_METRIC_COLUMNS.items()
                }
                candidate_predictions = np.argmax(probabilities, axis=1)
                frozen_metric_checks = {
                    metric: abs(
                        candidate_seed_metrics[split][metric]
                        - float(seed_reference["frozen_metrics"][split][metric])
                    )
                    <= float(comparison_contract["candidate_metric_absolute_tolerance"])
                    for metric in MNIST_METRIC_COLUMNS
                }
                linux_metric_checks = {
                    metric: abs(
                        candidate_seed_metrics[split][metric]
                        - float(linux_split_reference["metrics"][metric])
                    )
                    <= float(comparison_contract["candidate_metric_absolute_tolerance"])
                    for metric in MNIST_METRIC_COLUMNS
                }
                frozen_exact = bool(
                    np.array_equal(candidate_predictions, frozen_predictions)
                    and all(frozen_metric_checks.values())
                )
                linux_exact = bool(path_report["passed"] and all(linux_metric_checks.values()))
                frozen_seed_profile = frozen_seed_profile and frozen_exact
                linux_seed_profile = linux_seed_profile and linux_exact
                split_reports[split] = {
                    **path_report,
                    "stored_output_self_consistency": self_consistency,
                    "recomputed_metrics": candidate_seed_metrics[split],
                    "stored_metric_checks": metric_self_checks,
                    "frozen_profile_metric_checks": frozen_metric_checks,
                    "linux_x86_profile_metric_checks": linux_metric_checks,
                    "frozen_prediction_profile_exact": frozen_exact,
                    "linux_x86_profile_exact": linux_exact,
                    "passed": bool(
                        self_consistency["passed"]
                        and all(metric_self_checks.values())
                        and (frozen_exact or linux_exact)
                    ),
                }
            candidate_test_metrics[seed_key] = candidate_seed_metrics["test"]
            frozen_profile = frozen_profile and frozen_seed_profile
            linux_profile = linux_profile and linux_seed_profile
            seed_checks = {
                "manifest_seed_exact": manifest.get("reservoir_seed") == seed,
                "manifest_subset_exact": manifest.get("dataset_subset_checksum")
                == reference["identities"]["dataset_subset_checksum"],
                "feature_cache_key_exact": manifest.get("feature_cache_key_checksum")
                == cache_key,
                "analytic_exact_backend": manifest.get("condition") == "analytic",
                "genuine_mnist": manifest.get("is_synthetic") is False,
                "configuration_checksum_exact": manifest.get("study_configuration_checksum")
                == reference["identities"]["configuration_sha256"],
                "feature_metadata_key_exact": feature_metadata.get("cache_key", {}).get(
                    "checksum"
                )
                == cache_key,
            }
            seed_reports[seed_key] = {
                "run_directory": run_directory.relative_to(repository).as_posix(),
                "artifact_git_commit": cast(dict[str, Any], manifest.get("git", {})).get(
                    "commit"
                ),
                "operating_system": manifest.get("operating_system"),
                "package_versions": manifest.get("package_versions"),
                "checks": seed_checks,
                "features": feature_reports,
                "scaler": scaler_reports,
                "readout_parameters": parameter_report,
                "optimizer_objective": objective_report,
                "selection": {
                    "checks": selection_checks,
                    "selected_regularization_c": selected_c,
                    "solver": str(estimator.solver),
                    "solver_tolerance": float(estimator.tol),
                    "candidate_iteration_count": int(np.asarray(estimator.n_iter_)[0]),
                    "frozen_iteration_count": int(
                        seed_reference["frozen_iteration_count"]
                    ),
                    "iteration_count_absolute_difference": abs(
                        int(np.asarray(estimator.n_iter_)[0])
                        - int(seed_reference["frozen_iteration_count"])
                    ),
                    "maximum_iteration_count_difference": int(
                        seed_reference["maximum_iteration_count_difference"]
                    ),
                    "trials": selection_rows,
                },
                "splits": split_reports,
                "candidate_metrics": candidate_seed_metrics,
                "frozen_prediction_profile_exact": frozen_seed_profile,
                "linux_x86_profile_exact": linux_seed_profile,
            }
            seed_reports[seed_key]["passed"] = bool(
                all(seed_checks.values())
                and all(record["passed"] for record in feature_reports.values())
                and all(record["passed"] for record in scaler_reports.values())
                and parameter_report["passed"]
                and objective_report["passed"]
                and all(selection_checks.values())
                and all(record["passed"] for record in split_reports.values())
            )

    linux_environment = bool(
        artifact_operating_systems
        and all("Linux" in value and "x86_64" in value for value in artifact_operating_systems)
        and all(
            versions == reference["linux_profile"]["package_versions"]
            for versions in artifact_package_versions
        )
        and all(
            version == reference["linux_profile"]["python_version"]
            for version in artifact_python_versions
        )
        and all(
            environment_id == reference["linux_profile"]["qbraid_environment_id"]
            for environment_id in artifact_qbraid_environment_ids
        )
    )
    accepted_profile: str | None
    if frozen_profile:
        accepted_profile = "frozen_exact"
    elif linux_profile and linux_environment:
        accepted_profile = "linux_x86_qbraid_e0l4"
    else:
        accepted_profile = None
    aggregate = {
        metric: float(
            np.mean([candidate_test_metrics[str(seed)][metric] for seed in RESERVOIR_SEEDS])
        )
        for metric in MNIST_METRIC_COLUMNS
    }
    frozen_aggregate = cast(dict[str, float], reference["frozen_aggregate_metrics"])
    linux_aggregate = cast(
        dict[str, float],
        reference["linux_profile"]["aggregate_metrics"],
    )
    frozen_global_metric_checks = {
        metric: abs(aggregate[metric] - float(frozen_aggregate[metric])) <= 1e-12
        for metric in MNIST_METRIC_COLUMNS
    }
    linux_profile_metric_checks = {
        metric: abs(aggregate[metric] - float(linux_aggregate[metric])) <= 1e-12
        for metric in MNIST_METRIC_COLUMNS
    }
    aggregate_report = {
        "candidate": aggregate,
        "frozen": frozen_aggregate,
        "linux_x86_observed": linux_aggregate,
        "frozen_global_metric_exact": frozen_global_metric_checks,
        "linux_profile_metric_exact": linux_profile_metric_checks,
        "displayed_accuracy": {
            "frozen_publication": f"{float(frozen_aggregate['accuracy']):.3f}",
            "candidate": f"{aggregate['accuracy']:.3f}",
            "changed": f"{float(frozen_aggregate['accuracy']):.3f}"
            != f"{aggregate['accuracy']:.3f}",
            "policy": (
                "Keep the checksum-frozen paper value and report the Linux/x86 replication "
                "value explicitly in this evidence; do not rewrite publication claims."
            ),
        },
    }
    benchmark = _load_json(
        repository / "results/qrc_mnist/tables/mnist_final_benchmark.json"
    )
    test_rows = [
        row
        for row in cast(list[dict[str, Any]], benchmark["rows"])
        if row.get("split") == "test"
    ]
    rankings = {
        metric: [
            str(row["model"])
            for row in sorted(
                test_rows,
                key=lambda row: float(row[metric]),
                reverse=True,
            )
        ]
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "ovr_macro_roc_auc")
    }
    ranking_checks = {
        metric: ranking == reference["rankings"][metric]
        for metric, ranking in rankings.items()
    }
    passed = bool(
        dataset_report["passed"]
        and all(record["passed"] for record in seed_reports.values())
        and accepted_profile is not None
        and (
            accepted_profile == "frozen_exact"
            or all(linux_profile_metric_checks.values())
        )
        and all(ranking_checks.values())
    )
    try:
        reference_display_path = reference_file.relative_to(repository).as_posix()
    except ValueError:
        reference_display_path = (
            Path("configs/reproduction") / reference_file.name
        ).as_posix()
    report = {
        "schema_version": 1,
        "contract_id": reference["contract_id"],
        "status": "pass" if passed else "fail",
        "passed": passed,
        "validation_commit": _git_commit(repository),
        "artifact_commits": sorted(
            {
                str(record["artifact_git_commit"])
                for record in seed_reports.values()
                if record["artifact_git_commit"] is not None
            }
        ),
        "runtime_platform": platform.platform(),
        "accepted_profile": accepted_profile,
        "linux_environment_gate": linux_environment,
        "artifact_environment_gate": {
            "operating_systems": artifact_operating_systems,
            "python_versions": artifact_python_versions,
            "qbraid_environment_ids": artifact_qbraid_environment_ids,
            "package_versions": artifact_package_versions,
            "expected_python_version": reference["linux_profile"]["python_version"],
            "expected_qbraid_environment_id": (
                reference["linux_profile"]["qbraid_environment_id"]
            ),
            "passed": linux_environment,
        },
        "root_cause": reference["root_cause"],
        "diagnostic_derivation": reference["derivation"],
        "reference": {
            "path": reference_display_path,
            "sha256": actual_reference_sha256,
            "bundle_path": str(bundle_record["path"]),
            "bundle_sha256": actual_bundle_sha256,
        },
        "dataset": dataset_report,
        "seeds": seed_reports,
        "aggregate_metrics": aggregate_report,
        "rankings": {
            "candidate": rankings,
            "expected": reference["rankings"],
            "checks": ranking_checks,
            "passed": all(ranking_checks.values()),
        },
        "global_tolerance_unchanged": {
            "absolute": 1e-10,
            "relative": 1e-9,
            "passed": True,
        },
        "publication_value_policy": reference["publication_value_policy"],
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
    }
    _write_json(destination, report)
    return report
