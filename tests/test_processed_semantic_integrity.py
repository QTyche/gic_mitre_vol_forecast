from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qtyche_qrc.data.download import sha256_file
from qtyche_qrc.data.semantic_integrity import (
    CSV_FILES,
    JSON_FILES,
    ProcessedSemanticIntegrityError,
    build_processed_semantic_reference,
    canonicalization_contract,
    require_processed_semantic_integrity,
    verify_processed_semantic_integrity,
)
from tests.data_helpers import write_test_public_data_config


def _write_raw_snapshot(config: Any) -> None:
    for name, path in config.raw_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"date,value\n2020-01-01,{name}\n", encoding="utf-8")
    assert config.snapshot_manifest_path is not None
    manifest = {
        "schema_version": 1,
        "snapshot_id": config.snapshot_id,
        "provider": "yahoo_chart",
        "data_source_type": "public_market",
        "is_synthetic": False,
        "files": {
            name: {
                "file": path.name,
                "symbol": config.symbols[name],
                "sha256": sha256_file(path),
            }
            for name, path in config.raw_paths.items()
        },
    }
    config.snapshot_manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frame(split: str, date: str, offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "split": [split],
            "feature_a": [0.123456789012 + offset],
            "target_rv_5d": [0.019876543219 + offset],
            "target_regime_5d": [0],
            "current_regime": [1],
            "target_transition": [1],
            "target_upward_transition": [0],
            "target_downward_transition": [1],
        }
    )


def _refresh_processed_manifest(processed_dir: Path, snapshot_id: str) -> None:
    manifest = {
        "schema_version": 1,
        "source_snapshot_id": snapshot_id,
        "data_source_type": "public_market",
        "is_synthetic": False,
        "processed_checksums": {
            name: sha256_file(processed_dir / name) for name in (*CSV_FILES, *JSON_FILES)
        },
    }
    (processed_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_processed(processed_dir: Path, snapshot_id: str) -> None:
    processed_dir.mkdir(parents=True)
    frames = {
        "train.csv": _frame("train", "2020-01-02", 0.0),
        "validation.csv": _frame("validation", "2020-07-02", 0.01),
        "test.csv": _frame("test", "2020-10-02", 0.02),
    }
    frames["features_unscaled.csv"] = pd.concat(
        [frames["train.csv"], frames["validation.csv"], frames["test.csv"]],
        ignore_index=True,
    )
    for name, frame in frames.items():
        frame.to_csv(
            processed_dir / name,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
    preprocessing = {
        "schema_version": 1,
        "fit_split": "train",
        "feature_names": ["feature_a"],
        "means": {"feature_a": 0.123456789012},
        "standard_deviations": {"feature_a": 0.0123456789012},
        "fitted_rows": 1,
        "zero_variance_features": [],
        "zero_variance_policy": "scale_by_one_after_centering",
        "standard_deviation_ddof": 0,
    }
    thresholds = {
        "schema_version": 1,
        "fit_split": "train",
        "fit_column": "target_rv_5d",
        "low_medium": 0.01,
        "medium_high": 0.03,
        "quantiles": [0.33, 0.66],
        "training_rows": 1,
    }
    for name, value in (
        ("preprocessing.json", preprocessing),
        ("regime_thresholds.json", thresholds),
    ):
        (processed_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _refresh_processed_manifest(processed_dir, snapshot_id)


def _prepared_contract(tmp_path: Path) -> tuple[Any, Path, Path, str]:
    config = write_test_public_data_config(tmp_path)
    assert config.snapshot_id is not None
    _write_raw_snapshot(config)
    _write_processed(config.processed_path, config.snapshot_id)
    historical = {
        name: sha256_file(config.processed_path / name) for name in (*CSV_FILES, *JSON_FILES)
    }
    reference = build_processed_semantic_reference(
        config.processed_path,
        data_config_sha256=sha256_file(config.source),
        source_snapshot_id=config.snapshot_id,
        historical_file_sha256=historical,
    )
    reference_path = tmp_path / "semantic_reference.json"
    reference_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config, config.processed_path, reference_path, sha256_file(reference_path)


def _verify(
    config: Any,
    processed_dir: Path,
    reference_path: Path,
    reference_sha256: str,
) -> dict[str, Any]:
    return verify_processed_semantic_integrity(
        processed_dir,
        data_config_path=config.source,
        reference_path=reference_path,
        expected_reference_sha256=reference_sha256,
    )


def test_cross_platform_float_and_formatting_differences_pass_semantically(
    tmp_path: Path,
) -> None:
    config, processed_dir, reference_path, reference_sha256 = _prepared_contract(tmp_path)

    for name in CSV_FILES:
        frame = pd.read_csv(processed_dir / name)
        for column in ("feature_a", "target_rv_5d"):
            frame[column] = frame[column].map(lambda value: float(value) + 1e-13)
        frame.to_csv(
            processed_dir / name,
            index=False,
            float_format="%.17g",
            lineterminator="\r\n",
        )
    preprocessing_path = processed_dir / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["means"]["feature_a"] += 1e-13
    preprocessing_path.write_text(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    thresholds_path = processed_dir / "regime_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds_path.write_text(
        json.dumps(thresholds, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _refresh_processed_manifest(processed_dir, config.snapshot_id)

    report = _verify(config, processed_dir, reference_path, reference_sha256)

    assert report["status"] == "pass"
    assert report["processed_semantic_match"] is True
    assert report["all_processed_files_byte_exact_to_historical"] is False
    assert all(row["semantic_match"] is True for row in report["processed_files"].values())
    assert report["canonicalization"]["significant_decimal_digits"] == 10
    assert report["canonicalization"]["maximum_relative_quantization_width"] == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    (
        ("numeric", "numeric_column_sha256"),
        ("label", "discrete_column_sha256"),
        ("date", "date_"),
        ("split", "split_"),
        ("missing", "missing_"),
        ("columns", "columns"),
    ),
)
def test_material_csv_differences_fail(
    tmp_path: Path,
    mutation: str,
    expected_field: str,
) -> None:
    config, processed_dir, reference_path, reference_sha256 = _prepared_contract(tmp_path)
    path = processed_dir / "train.csv"
    frame = pd.read_csv(path)
    if mutation == "numeric":
        values = frame["feature_a"].to_numpy(dtype=float)
        values[0] += 1e-5
        frame["feature_a"] = values
    elif mutation == "label":
        frame.loc[0, "target_regime_5d"] = 2
    elif mutation == "date":
        frame.loc[0, "date"] = "2020-01-03"
    elif mutation == "split":
        frame.loc[0, "split"] = "test"
    elif mutation == "missing":
        frame.loc[0, "feature_a"] = np.nan
    elif mutation == "columns":
        frame = frame.loc[:, list(reversed(frame.columns))]
    frame.to_csv(path, index=False, lineterminator="\n")
    _refresh_processed_manifest(processed_dir, config.snapshot_id)

    report = _verify(config, processed_dir, reference_path, reference_sha256)

    assert report["status"] == "fail"
    assert report["processed_files"]["train.csv"]["passed"] is False
    assert any(
        expected_field in field
        for field in report["processed_files"]["train.csv"]["mismatch_fields"]
    )
    with pytest.raises(ProcessedSemanticIntegrityError, match="semantic integrity"):
        require_processed_semantic_integrity(
            processed_dir,
            data_config_path=config.source,
            reference_path=reference_path,
            expected_reference_sha256=reference_sha256,
        )


def test_threshold_difference_fails(tmp_path: Path) -> None:
    config, processed_dir, reference_path, reference_sha256 = _prepared_contract(tmp_path)
    path = processed_dir / "regime_thresholds.json"
    thresholds = json.loads(path.read_text(encoding="utf-8"))
    thresholds["low_medium"] += 1e-5
    path.write_text(json.dumps(thresholds), encoding="utf-8")
    _refresh_processed_manifest(processed_dir, config.snapshot_id)

    report = _verify(config, processed_dir, reference_path, reference_sha256)

    assert report["status"] == "fail"
    assert report["processed_files"]["regime_thresholds.json"]["passed"] is False


def test_raw_snapshot_byte_tampering_remains_fatal(tmp_path: Path) -> None:
    config, processed_dir, reference_path, reference_sha256 = _prepared_contract(tmp_path)
    config.raw_paths["spy"].write_text("tampered\n", encoding="utf-8")

    report = _verify(config, processed_dir, reference_path, reference_sha256)

    assert report["status"] == "fail"
    assert report["raw_snapshot"]["passed"] is False
    assert report["processed_semantic_match"] is True


def test_qbraid_failure_and_root_cause_are_frozen_in_reference() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = json.loads(
        (root / "configs/reproduction/processed_data_semantic_reference.json").read_text(
            encoding="utf-8"
        )
    )

    assert reference["canonicalization"] == canonicalization_contract()
    assert reference["root_cause"]["classification"] == (
        "cross_platform_floating_point_last_digit_variation"
    )
    assert reference["observed_qbraid_failure_sha256"]["features_unscaled.csv"] == (
        "a1859fde2005709d07df74d24efac905bf07bd673569f064b1384ec8ca2a23d6"
    )
