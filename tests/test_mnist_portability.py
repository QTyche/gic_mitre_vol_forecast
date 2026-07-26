from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from qtyche_qrc.reproducibility.artifacts import build_parser
from qtyche_qrc.reproducibility.mnist_portability import (
    canonical_feature_digest,
    compare_prediction_path,
    compare_readout_parameters,
)
from qtyche_qrc.reproducibility.verification import sha256_path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reference() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            (
                _root()
                / "configs/reproduction/mnist_exact_portability_reference.json"
            ).read_text(encoding="utf-8")
        ),
    )


def _prediction_fixture() -> dict[str, Any]:
    truth = np.asarray([0, 1, 2, 3], dtype=np.int64)
    frozen_probabilities = np.full((4, 10), 0.001, dtype=float)
    frozen_probabilities[:, 0] = 0.991
    frozen_probabilities[1, 0] = 0.005
    frozen_probabilities[1, 1] = 0.987
    frozen_probabilities[2, 0] = 0.001
    frozen_probabilities[2, 2] = 0.499
    frozen_probabilities[2, 4] = 0.493
    frozen_probabilities[2, 1] = 0.001
    frozen_probabilities[2, 3] = 0.001
    frozen_probabilities[2, 5:] = 0.001
    frozen_probabilities[3, 0] = 0.005
    frozen_probabilities[3, 3] = 0.987
    frozen_predictions = np.argmax(frozen_probabilities, axis=1).astype(np.int64)
    probabilities = frozen_probabilities.copy()
    probabilities[2, 2] = 0.493
    probabilities[2, 4] = 0.499
    probabilities[2, 0] = 0.001
    probabilities[2, 1] = 0.001
    probabilities[2, 3] = 0.001
    probabilities[2, 5:] = 0.001
    frozen_scores = np.log(frozen_probabilities)
    scores = np.log(probabilities)
    candidate_predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (truth, candidate_predictions), 1)
    return {
        "official_indices": np.asarray([10, 20, 30, 40], dtype=np.int64),
        "truth": truth,
        "scores": scores,
        "probabilities": probabilities,
        "frozen_scores": frozen_scores,
        "frozen_probabilities": frozen_probabilities,
        "frozen_predictions": frozen_predictions,
        "expected_changed_positions": [2],
        "expected_confusion_matrix": confusion.tolist(),
        "limits": {
            "score": {
                "maximum_absolute_difference": 7.0,
                "maximum_mean_absolute_difference": 0.5,
                "maximum_median_absolute_difference": 1e-12,
                "maximum_l2_difference": 10.0,
            },
            "probability": {
                "maximum_absolute_difference": 0.6,
                "maximum_mean_absolute_difference": 0.04,
                "maximum_median_absolute_difference": 1e-12,
                "maximum_l2_difference": 1.0,
            },
            "maximum_changed_score_margin": 0.1,
            "maximum_changed_probability_margin": 0.01,
        },
    }


def _compare_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    return compare_prediction_path(
        official_indices=fixture["official_indices"],
        truth=fixture["truth"],
        scores=fixture["scores"],
        probabilities=fixture["probabilities"],
        frozen_official_indices=fixture["official_indices"],
        frozen_truth=fixture["truth"],
        frozen_scores=fixture["frozen_scores"],
        frozen_probabilities=fixture["frozen_probabilities"],
        frozen_predictions=fixture["frozen_predictions"],
        expected_changed_positions=fixture["expected_changed_positions"],
        expected_confusion_matrix=fixture["expected_confusion_matrix"],
        limits=fixture["limits"],
    )


def test_linux_x86_last_bit_feature_variation_has_same_commitment() -> None:
    frozen = np.asarray([[0.4082385100645216, -0.0], [0.30443310023710624, 0.25]])
    linux = frozen.copy()
    linux[0, 0] -= 5.551115123125783e-17
    linux[1, 0] -= 3.885780586188048e-16

    assert canonical_feature_digest(frozen, decimals=8) == canonical_feature_digest(
        linux,
        decimals=8,
    )


def test_material_feature_drift_changes_commitment() -> None:
    frozen = np.asarray([[0.4, 0.2], [0.3, 0.1]], dtype=float)
    drifted = frozen.copy()
    drifted[0, 0] += 1e-5

    assert canonical_feature_digest(frozen, decimals=8) != canonical_feature_digest(
        drifted,
        decimals=8,
    )


def test_bounded_linux_x86_readout_variation_is_accepted() -> None:
    reference_coefficient = np.zeros((10, 140), dtype=float)
    reference_intercept = np.zeros(10, dtype=float)
    coefficient = reference_coefficient.copy()
    coefficient[0, 0] = 0.2
    intercept = reference_intercept.copy()
    intercept[0] = 0.1
    limits = {
        "coefficient": {
            "maximum_absolute_difference": 0.3,
            "maximum_mean_absolute_difference": 0.01,
            "maximum_l2_difference": 0.3,
        },
        "intercept": {
            "maximum_absolute_difference": 0.2,
            "maximum_mean_absolute_difference": 0.02,
            "maximum_l2_difference": 0.2,
        },
    }

    report = compare_readout_parameters(
        coefficient,
        intercept,
        reference_coefficient,
        reference_intercept,
        limits,
    )

    assert report["passed"] is True


def test_parameter_drift_is_rejected() -> None:
    reference_coefficient = np.zeros((10, 140), dtype=float)
    coefficient = reference_coefficient.copy()
    coefficient[0, 0] = 0.5
    limits = {
        "coefficient": {
            "maximum_absolute_difference": 0.3,
            "maximum_mean_absolute_difference": 0.01,
            "maximum_l2_difference": 0.3,
        },
        "intercept": {
            "maximum_absolute_difference": 0.2,
            "maximum_mean_absolute_difference": 0.02,
            "maximum_l2_difference": 0.2,
        },
    }

    report = compare_readout_parameters(
        coefficient,
        np.zeros(10),
        reference_coefficient,
        np.zeros(10),
        limits,
    )

    assert report["passed"] is False
    assert report["coefficient"]["checks"]["maximum_absolute_difference"] is False


def test_checksum_pinned_near_boundary_prediction_profile_is_accepted() -> None:
    report = _compare_fixture(_prediction_fixture())

    assert report["passed"] is True
    assert report["changed_positions"] == [2]
    assert report["confusion_matrix_exact"] is True


def test_materially_different_score_path_is_rejected() -> None:
    fixture = _prediction_fixture()
    fixture["scores"] = np.asarray(fixture["scores"]).copy()
    fixture["scores"][0, 9] += 20.0

    report = _compare_fixture(fixture)

    assert report["passed"] is False
    assert report["score_difference"]["checks"]["maximum_absolute_difference"] is False


def test_unexpected_prediction_or_confusion_change_is_rejected() -> None:
    fixture = _prediction_fixture()
    fixture["probabilities"] = np.asarray(fixture["probabilities"]).copy()
    fixture["probabilities"][3] = np.full(10, 0.001)
    fixture["probabilities"][3, 8] = 0.991
    fixture["scores"] = np.log(fixture["probabilities"])

    report = _compare_fixture(fixture)

    assert report["passed"] is False
    assert report["changed_positions_exact"] is False
    assert report["confusion_matrix_exact"] is False


def test_reference_and_bundle_are_checksum_pinned_without_global_widening() -> None:
    root = _root()
    config = cast(
        dict[str, Any],
        yaml.safe_load((root / "configs/phase3_reproduction.yaml").read_text(encoding="utf-8")),
    )
    original = copy.deepcopy(config["tolerances"]["regenerated_metrics"])
    contract = config["tolerances"]["mnist_exact_portability"]
    reference_path = root / contract["reference"]
    reference = _reference()
    bundle = root / reference["reference_bundle"]["path"]

    assert sha256_path(reference_path) == contract["reference_sha256"]
    assert sha256_path(bundle) == reference["reference_bundle"]["sha256"]
    assert config["tolerances"]["regenerated_metrics"] == original == {
        "absolute": 1e-10,
        "relative": 1e-9,
        "justification": (
            "Floating-point QRC readouts can differ in final digits across BLAS and CPU "
            "implementations. Dataset, configuration and tracked publication files use "
            "exact SHA-256 equality."
        ),
    }
    assert reference["publication_value_policy"]["frozen_accuracy"] == "0.874"
    assert reference["publication_value_policy"]["linux_x86_replication_accuracy"] == "0.875"


def test_linux_profile_records_all_changed_indices_and_keeps_rankings() -> None:
    reference = _reference()
    validation_changes = sum(
        len(reference["seeds"][str(seed)]["linux_profile"]["validation"]["changed_positions"])
        for seed in (2026, 2027, 2028)
    )
    test_changes = sum(
        len(reference["seeds"][str(seed)]["linux_profile"]["test"]["changed_positions"])
        for seed in (2026, 2027, 2028)
    )

    assert validation_changes == 8
    assert test_changes == 14
    assert reference["rankings"]["accuracy"] == [
        "esn_baseline",
        "logistic_baseline",
        "qrc_exact_mean",
    ]


def test_diagnostic_cli_is_explicitly_comparison_only() -> None:
    arguments = build_parser().parse_args(
        ["diagnose-mnist", "--output", "qbraid_evidence/mnist.json"]
    )

    assert arguments.command == "diagnose-mnist"
    assert arguments.output == Path("qbraid_evidence/mnist.json")
