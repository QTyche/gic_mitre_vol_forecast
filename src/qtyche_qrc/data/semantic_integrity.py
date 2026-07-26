"""Cross-platform semantic integrity checks for generated financial data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import sha256_file, verify_public_snapshot

SCHEMA_VERSION = 1
CANONICALIZATION_ID = "qtyche_processed_financial_semantic_v1"
SIGNIFICANT_DECIMAL_DIGITS = 10
MAXIMUM_RELATIVE_QUANTIZATION_WIDTH = 1e-9
CSV_FILES = (
    "features_unscaled.csv",
    "train.csv",
    "validation.csv",
    "test.csv",
)
JSON_FILES = ("preprocessing.json", "regime_thresholds.json")
DISCRETE_COLUMNS = (
    "target_regime_5d",
    "current_regime",
    "target_transition",
    "target_upward_transition",
    "target_downward_transition",
)


class ProcessedSemanticIntegrityError(ValueError):
    """Raised when generated data differ materially from the frozen reference."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ProcessedSemanticIntegrityError("processed numeric values must be finite")
    if value == 0.0:
        return "0"
    return format(value, f".{SIGNIFICANT_DECIMAL_DIGITS}g")


def canonicalization_contract() -> dict[str, Any]:
    """Return the frozen numeric and structural equality rule."""

    return {
        "id": CANONICALIZATION_ID,
        "numeric_rule": (
            "Parse finite decimal values, normalize signed zero, and format with "
            f"{SIGNIFICANT_DECIMAL_DIGITS} significant decimal digits before SHA-256."
        ),
        "significant_decimal_digits": SIGNIFICANT_DECIMAL_DIGITS,
        "maximum_relative_quantization_width": MAXIMUM_RELATIVE_QUANTIZATION_WIDTH,
        "absolute_floor": 0.0,
        "non_finite_values": "rejected",
        "dates": "parsed and normalized to exact YYYY-MM-DD sequences",
        "columns": "names and order exact",
        "row_order": "exact",
        "split_membership": "date/split pairs exact",
        "labels": "integral values exact",
        "missing_values": "positions exact",
        "json_structure": "keys, list order, scalar types, and non-float values exact",
        "line_endings_and_whitespace": "ignored after parsing",
    }


def _canonical_date(value: str, *, file_name: str, row: int) -> str:
    if not value:
        return "<NA>"
    try:
        parsed = pd.Timestamp(value)
    except ValueError as exc:
        raise ProcessedSemanticIntegrityError(
            f"{file_name} has an invalid date at row {row}: {value!r}"
        ) from exc
    return parsed.date().isoformat()


def _canonical_discrete(value: str, *, file_name: str, column: str, row: int) -> str:
    if not value:
        return "<NA>"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ProcessedSemanticIntegrityError(
            f"{file_name} has a nonnumeric label in {column} at row {row}"
        ) from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ProcessedSemanticIntegrityError(
            f"{file_name} has a non-integral label in {column} at row {row}: {value!r}"
        )
    return str(int(parsed))


def _canonical_numeric(value: str, *, file_name: str, column: str, row: int) -> str:
    if not value:
        return "<NA>"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ProcessedSemanticIntegrityError(
            f"{file_name} has a nonnumeric value in {column} at row {row}"
        ) from exc
    return _canonical_float(parsed)


def csv_semantic_signature(path: Path) -> dict[str, Any]:
    """Build an exact-structure and quantized-numeric signature for one CSV."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = [str(column) for column in frame.columns]
    if "date" not in frame or "split" not in frame:
        raise ProcessedSemanticIntegrityError(f"{path.name} must contain date and split columns")
    dates = [
        _canonical_date(str(value), file_name=path.name, row=index)
        for index, value in enumerate(frame["date"], start=2)
    ]
    splits = [str(value) if str(value) else "<NA>" for value in frame["split"]]
    missing_mask = [
        ["1" if str(value) == "" else "0" for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    discrete_columns = [column for column in DISCRETE_COLUMNS if column in frame]
    numeric_columns = [
        column for column in columns if column not in {"date", "split", *discrete_columns}
    ]
    discrete_digests: dict[str, str] = {}
    for column in discrete_columns:
        values = [
            _canonical_discrete(
                str(value),
                file_name=path.name,
                column=column,
                row=index,
            )
            for index, value in enumerate(frame[column], start=2)
        ]
        discrete_digests[column] = _digest(values)
    numeric_digests: dict[str, str] = {}
    for column in numeric_columns:
        values = [
            _canonical_numeric(
                str(value),
                file_name=path.name,
                column=column,
                row=index,
            )
            for index, value in enumerate(frame[column], start=2)
        ]
        numeric_digests[column] = _digest(values)
    split_counts = {
        split: int(sum(value == split for value in splits)) for split in sorted(set(splits))
    }
    signature: dict[str, Any] = {
        "rows": len(frame),
        "columns": columns,
        "date_sha256": _digest(dates),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "dates_unique": len(set(dates)) == len(dates),
        "dates_increasing": dates == sorted(dates),
        "split_sha256": _digest(splits),
        "split_membership_sha256": _digest(list(zip(dates, splits))),
        "split_counts": split_counts,
        "missing_position_sha256": _digest(missing_mask),
        "missing_value_count": sum(row.count("1") for row in missing_mask),
        "discrete_column_sha256": discrete_digests,
        "numeric_column_sha256": numeric_digests,
    }
    signature["semantic_sha256"] = _digest(signature)
    return signature


def _flatten_json(
    value: Any,
    path: str,
    structure: dict[str, Any],
    numeric: dict[str, str],
) -> None:
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value)
        structure[path] = {"type": "object", "keys": keys}
        for key in keys:
            _flatten_json(value[key], f"{path}/{key}", structure, numeric)
        return
    if isinstance(value, list):
        structure[path] = {"type": "array", "length": len(value)}
        for index, item in enumerate(value):
            _flatten_json(item, f"{path}/{index}", structure, numeric)
        return
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        structure[path] = {"type": type(value).__name__, "value": value}
        return
    if isinstance(value, float):
        numeric[path] = _canonical_float(value)
        return
    raise ProcessedSemanticIntegrityError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def json_semantic_signature(path: Path) -> dict[str, Any]:
    """Build an exact-structure and quantized-numeric signature for one JSON file."""

    value = json.loads(path.read_text(encoding="utf-8"))
    structure: dict[str, Any] = {}
    numeric: dict[str, str] = {}
    _flatten_json(value, "$", structure, numeric)
    numeric_path_sha256 = {name: _digest(token) for name, token in sorted(numeric.items())}
    signature: dict[str, Any] = {
        "structure_sha256": _digest(structure),
        "numeric_path_sha256": numeric_path_sha256,
        "numeric_value_count": len(numeric),
    }
    signature["semantic_sha256"] = _digest(signature)
    return signature


def _semantic_signatures(processed_dir: Path) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for name in CSV_FILES:
        path = processed_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"processed data file is missing: {path}")
        signatures[name] = csv_semantic_signature(path)
    for name in JSON_FILES:
        path = processed_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"processed data file is missing: {path}")
        signatures[name] = json_semantic_signature(path)
    return signatures


def build_processed_semantic_reference(
    processed_dir: Path,
    *,
    data_config_sha256: str,
    source_snapshot_id: str,
    historical_file_sha256: dict[str, str],
    observed_qbraid_failure_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the compact frozen semantic commitment from trusted processed files."""

    signatures = _semantic_signatures(processed_dir)
    if set(historical_file_sha256) != {*CSV_FILES, *JSON_FILES}:
        raise ValueError("historical checksums must cover the six processed files")
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_id": "yahoo_chart_20100101_20251231_processed_semantic_v1",
        "source_snapshot_id": source_snapshot_id,
        "data_config_sha256": data_config_sha256,
        "canonicalization": canonicalization_contract(),
        "root_cause": {
            "classification": "cross_platform_floating_point_last_digit_variation",
            "mechanism": (
                "Platform libm implementations of logarithms and CPU-specific rolling/"
                "reduction paths can differ in final binary digits. The processed CSV "
                "writer exposes up to 12 significant digits and preprocessing JSON "
                "stores full binary64 round trips, so scientifically immaterial final-"
                "digit differences change byte SHA-256 values."
            ),
            "evidence_pattern": (
                "The qBraid failure changed every float-bearing CSV and preprocessing "
                "JSON while the regime-threshold document remained byte-identical."
            ),
        },
        "historical_file_sha256": dict(sorted(historical_file_sha256.items())),
        "observed_qbraid_failure_sha256": dict(
            sorted((observed_qbraid_failure_sha256 or {}).items())
        ),
        "files": signatures,
    }


def _differences(actual: Any, expected: Any, path: str = "$") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        result: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            child = f"{path}.{key}"
            if key not in actual or key not in expected:
                result.append(child)
            else:
                result.extend(_differences(actual[key], expected[key], child))
        return result
    return [] if actual == expected else [path]


def _raw_snapshot_report(data_config_path: Path, reference: dict[str, Any]) -> dict[str, Any]:
    try:
        data_config = load_data_config(data_config_path)
        config_checksum = sha256_file(data_config_path)
        snapshot = verify_public_snapshot(data_config)
        files = {
            name: {
                "path": str(data_config.raw_paths[name].relative_to(data_config.project_root)),
                "manifest_sha256": str(record["sha256"]),
                "actual_sha256": sha256_file(data_config.raw_paths[name]),
                "passed": sha256_file(data_config.raw_paths[name]) == record["sha256"],
            }
            for name, record in sorted(snapshot["files"].items())
        }
        passed = (
            config_checksum == reference.get("data_config_sha256")
            and snapshot.get("snapshot_id") == reference.get("source_snapshot_id")
            and bool(files)
            and all(record["passed"] for record in files.values())
        )
        return {
            "passed": passed,
            "snapshot_id": snapshot.get("snapshot_id"),
            "data_config_expected_sha256": reference.get("data_config_sha256"),
            "data_config_actual_sha256": config_checksum,
            "download_manifest_file_checks": files,
            "verification_rule": (
                "Exact SHA-256 equality between each downloaded raw CSV and its "
                "immutable per-download snapshot-manifest record."
            ),
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "verification_rule": (
                "Exact SHA-256 equality between each downloaded raw CSV and its "
                "immutable per-download snapshot-manifest record."
            ),
        }


def _processed_manifest_report(
    processed_dir: Path,
    reference: dict[str, Any],
) -> dict[str, Any]:
    path = processed_dir / "data_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        saved_checksums = manifest.get("processed_checksums")
        if not isinstance(saved_checksums, dict):
            raise ProcessedSemanticIntegrityError("processed data manifest has no checksum mapping")
        checks = {
            name: {
                "manifest_sha256": saved_checksums.get(name),
                "actual_sha256": sha256_file(processed_dir / name),
                "passed": saved_checksums.get(name) == sha256_file(processed_dir / name),
            }
            for name in (*CSV_FILES, *JSON_FILES)
        }
        passed = (
            manifest.get("source_snapshot_id") == reference.get("source_snapshot_id")
            and manifest.get("data_source_type") == "public_market"
            and manifest.get("is_synthetic") is False
            and all(record["passed"] for record in checks.values())
        )
        return {
            "passed": passed,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "source_snapshot_id": manifest.get("source_snapshot_id"),
            "data_source_type": manifest.get("data_source_type"),
            "is_synthetic": manifest.get("is_synthetic"),
            "generated_file_checksum_checks": checks,
        }
    except Exception as exc:
        return {
            "passed": False,
            "path": path.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def verify_processed_semantic_integrity(
    processed_dir: Path,
    *,
    data_config_path: Path,
    reference_path: Path,
    expected_reference_sha256: str,
) -> dict[str, Any]:
    """Verify raw bytes exactly and generated data by strict semantic commitments."""

    reference_actual_sha256 = sha256_file(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_valid = (
        reference.get("schema_version") == SCHEMA_VERSION
        and reference.get("canonicalization") == canonicalization_contract()
        and reference_actual_sha256 == expected_reference_sha256
    )
    raw_report = _raw_snapshot_report(data_config_path, reference)
    manifest_report = _processed_manifest_report(processed_dir, reference)
    file_reports: dict[str, Any] = {}
    try:
        actual_signatures = _semantic_signatures(processed_dir)
        for name in (*CSV_FILES, *JSON_FILES):
            expected_signature = reference.get("files", {}).get(name)
            actual_signature = actual_signatures[name]
            mismatch_fields = _differences(actual_signature, expected_signature)
            historical_sha256 = reference.get("historical_file_sha256", {}).get(name)
            actual_sha256 = sha256_file(processed_dir / name)
            file_reports[name] = {
                "passed": not mismatch_fields,
                "semantic_match": not mismatch_fields,
                "byte_exact_to_historical": actual_sha256 == historical_sha256,
                "historical_sha256": historical_sha256,
                "actual_sha256": actual_sha256,
                "expected_semantic_sha256": (
                    expected_signature.get("semantic_sha256")
                    if isinstance(expected_signature, dict)
                    else None
                ),
                "actual_semantic_sha256": actual_signature["semantic_sha256"],
                "mismatch_fields": mismatch_fields,
            }
    except Exception as exc:
        file_reports["error"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    processed_passed = bool(file_reports) and all(
        isinstance(record, dict) and record.get("passed") is True
        for record in file_reports.values()
    )
    passed = (
        reference_valid and raw_report["passed"] and manifest_report["passed"] and processed_passed
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "reference": {
            "path": reference_path.as_posix(),
            "expected_sha256": expected_reference_sha256,
            "actual_sha256": reference_actual_sha256,
            "passed": reference_valid,
        },
        "canonicalization": canonicalization_contract(),
        "root_cause": reference.get("root_cause"),
        "observed_qbraid_failure_sha256": reference.get("observed_qbraid_failure_sha256"),
        "raw_snapshot": raw_report,
        "processed_manifest": manifest_report,
        "processed_files": file_reports,
        "processed_semantic_match": processed_passed,
        "all_processed_files_byte_exact_to_historical": (
            processed_passed
            and all(
                record.get("byte_exact_to_historical") is True
                for record in file_reports.values()
                if isinstance(record, dict)
            )
        ),
        "integrity_interpretation": (
            "Byte SHA-256 remains recorded for every generated file. A byte mismatch "
            "is accepted only when the tracked canonical semantic commitments prove "
            "exact structure, dates, splits, labels, missingness, and numeric equality "
            "under the declared tight precision rule."
        ),
    }


def require_processed_semantic_integrity(
    processed_dir: Path,
    *,
    data_config_path: Path,
    reference_path: Path,
    expected_reference_sha256: str,
) -> dict[str, Any]:
    """Return a passing report or raise with the material mismatch fields."""

    report = verify_processed_semantic_integrity(
        processed_dir,
        data_config_path=data_config_path,
        reference_path=reference_path,
        expected_reference_sha256=expected_reference_sha256,
    )
    if report["status"] != "pass":
        failed = {
            name: record
            for name, record in report["processed_files"].items()
            if not isinstance(record, dict) or record.get("passed") is not True
        }
        raise ProcessedSemanticIntegrityError(
            "processed-data semantic integrity failure: "
            + json.dumps(
                {
                    "reference_passed": report["reference"]["passed"],
                    "raw_snapshot_passed": report["raw_snapshot"]["passed"],
                    "processed_manifest_passed": report["processed_manifest"]["passed"],
                    "processed_failures": failed,
                },
                sort_keys=True,
            )
        )
    return report


def _write_report(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_processed_semantic_integrity(
            args.processed_dir.resolve(),
            data_config_path=args.data_config.resolve(),
            reference_path=args.reference.resolve(),
            expected_reference_sha256=args.reference_sha256,
        )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "canonicalization": canonicalization_contract(),
        }
    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
