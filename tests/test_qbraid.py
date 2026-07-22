import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from qtyche_qrc.config import load_config
from qtyche_qrc.data.download import SnapshotIntegrityError
from qtyche_qrc.experiments.manifest import create_manifest
from qtyche_qrc.qbraid import (
    QbraidVerificationError,
    scan_prohibited_paths,
    verify_public_pilot_inputs,
    verify_qbraid_environment,
)
from tests.data_helpers import write_test_public_data_config


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_qbraid_environment_configuration_parses_with_pinned_ranges() -> None:
    root = _repository_root()
    environment = yaml.safe_load((root / "environment-qbraid.yaml").read_text(encoding="utf-8"))
    requirements = (root / "requirements-qbraid.txt").read_text(encoding="utf-8")

    assert environment["name"] == "qtyche-qrc-phase3"
    assert "python>=3.11,<3.13" in environment["dependencies"]
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "pytest",
        "ruff",
        "mypy",
    ):
        assert f"{package}>=" in requirements
        assert ",<" in next(line for line in requirements.splitlines() if line.startswith(package))


def test_prohibited_path_scan_reports_host_specific_text(tmp_path: Path) -> None:
    marker = "/" + "Users/" + "example/results/output.json"
    (tmp_path / "bad.yaml").write_text(f"output: {marker}\n", encoding="utf-8")

    findings = scan_prohibited_paths(tmp_path)

    assert findings == [
        {
            "path": "bad.yaml",
            "line": 1,
            "markers": ["/" + "Users/", "absolute results path"],
        }
    ]


def test_verify_qbraid_writes_success_report(tmp_path: Path) -> None:
    report_path = tmp_path / "qbraid_environment_report.json"

    report = verify_qbraid_environment(
        _repository_root(),
        report_path=report_path,
        minimum_python=(3, 9),
    )

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert persisted["checks"]["qrc_backend"] == {
        "backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "n_qubits": 3,
        "passed": True,
    }
    assert persisted["checks"]["prohibited_path_scan"]["findings"] == []
    assert persisted["git"]["commit"]


def test_verify_qbraid_persists_missing_dependency_failure(tmp_path: Path) -> None:
    report_path = tmp_path / "missing_dependency_report.json"

    with pytest.raises(QbraidVerificationError, match="dependencies"):
        verify_qbraid_environment(
            _repository_root(),
            report_path=report_path,
            required_imports=(),
            required_distributions=("qtyche-intentionally-missing",),
            minimum_python=(3, 9),
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["checks"]["dependencies"]["missing"] == ["qtyche-intentionally-missing"]


def test_experiment_manifest_records_qbraid_runtime_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "experiment:\n"
        "  name: qbraid_metadata\n"
        "  seed: 2026\n"
        "  output_dir: results\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QTYCHE_EXECUTION_PLATFORM", "qbraid_lab")
    monkeypatch.setenv("QTYCHE_QBRAID_ENVIRONMENT_NAME", "qtyche-test")
    monkeypatch.setenv("QTYCHE_QBRAID_ENVIRONMENT_ID", "env-test")
    monkeypatch.setenv("QTYCHE_QBRAID_LAB_IMAGE", "python-3.11")

    manifest_path = create_manifest(load_config(config_path), repository=tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["execution_platform"] == "qbraid_lab"
    assert manifest["qbraid_environment_name"] == "qtyche-test"
    assert manifest["qbraid_environment_id"] == "env-test"
    assert manifest["qbraid_lab_image"] == "python-3.11"
    assert manifest["python_version"]
    assert manifest["operating_system"]
    assert manifest["package_versions"]["numpy"] != "not-installed"


def _write_minimal_snapshot(root: Path) -> tuple[Path, dict[str, Path]]:
    config = write_test_public_data_config(root)
    for path in config.raw_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("date,close\n2020-01-01,1\n", encoding="utf-8")
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
                "rows": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in config.raw_paths.items()
        },
    }
    config.snapshot_manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config.snapshot_manifest_path, dict(config.raw_paths)


def test_public_script_guard_rejects_missing_snapshot(tmp_path: Path) -> None:
    write_test_public_data_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="snapshot manifest is missing"):
        verify_public_pilot_inputs(tmp_path)


def test_public_script_guard_rejects_snapshot_checksum_mismatch(tmp_path: Path) -> None:
    _, raw_paths = _write_minimal_snapshot(tmp_path)
    raw_paths["spy"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="checksum mismatch"):
        verify_public_pilot_inputs(tmp_path)


def test_stage_runner_returns_nonzero_for_invalid_seed() -> None:
    root = _repository_root()
    process = subprocess.run(
        [sys.executable, "scripts/reproduce_phase3.py", "--stage", "public-pilot", "--seed", "1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "--seed must be one of" in process.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="qBraid smoke requires Python 3.11+")
def test_stage_runner_qbraid_smoke_is_offline_exact_and_deterministic() -> None:
    root = _repository_root()
    summaries = (
        root / "results/qbraid/test_qbraid_smoke_first.json",
        root / "results/qbraid/test_qbraid_smoke_second.json",
    )
    environment = dict(os.environ)
    environment["QTYCHE_EXECUTION_PLATFORM"] = "qbraid_lab"
    records: list[dict[str, Any]] = []
    for summary_path in summaries:
        process = subprocess.run(
            [
                sys.executable,
                "scripts/reproduce_phase3.py",
                "--stage",
                "smoke",
                "--summary",
                str(summary_path.relative_to(root)),
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        records.append(json.loads(summary_path.read_text(encoding="utf-8")))

    first, second = records
    assert first["status"] == second["status"] == "success"
    assert first["execution_platform"] == second["execution_platform"] == "qbraid_lab"
    assert first["qrc_backend"] == "numpy_density_matrix_exact"
    assert first["exact_noiseless"] is True
    assert first["smoke_data_is_synthetic"] is True
    assert "NOT A FINANCIAL PERFORMANCE RESULT" in first["synthetic_data_warning"]
    assert first["synthetic_smoke_outputs_are_financial_evidence"] is False
    assert first["deterministic_output_checksums"] == second["deterministic_output_checksums"]
    assert len(first["commands"]) == 14
    assert first["verified_output_checksums"]

    qrc_manifests = []
    for relative in first["outputs"]:
        manifest_path = root / str(relative) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_type") in {"qrc_classifier", "qrc_regressor"}:
            qrc_manifests.append(manifest)
    assert len(qrc_manifests) == 2
    for manifest in qrc_manifests:
        assert manifest["execution_platform"] == "qbraid_lab"
        assert manifest["backend"] == "numpy_density_matrix_exact"
        assert manifest["exact_noiseless"] is True
        assert manifest["qrc_configuration"]["n_qubits"] == 3
