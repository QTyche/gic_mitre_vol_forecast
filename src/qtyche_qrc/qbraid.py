"""qBraid Lab environment and repository portability verification."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import verify_public_snapshot
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.models.dataset import load_model_dataset
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir
from qtyche_qrc.runtime import runtime_metadata

REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "pytest",
    "ruff",
    "mypy",
)
REQUIRED_IMPORTS = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "pytest",
    "qtyche_qrc",
)
EXPECTED_CLI_COMMANDS = (
    "validate-config",
    "create-fixture-data",
    "prepare-data",
    "audit-data",
    "train-baseline",
    "search-baseline",
    "generate-qrc-features",
    "train-qrc",
    "characterize-qrc",
    "compare-qrc-seeds",
    "verify-qbraid",
)
REQUIRED_REPOSITORY_PATHS = (
    "pyproject.toml",
    "requirements-qbraid.txt",
    "environment-qbraid.yaml",
    "configs/data.yaml",
    "configs/data_public_market.yaml",
    "configs/models/qrc_classifier_smoke.yaml",
    "configs/models/qrc_regressor_smoke.yaml",
    "configs/models/qrc_capacity_qbraid_smoke.yaml",
    "scripts/setup_qbraid.sh",
    "scripts/verify_qbraid_environment.sh",
    "scripts/reproduce_qbraid_smoke.sh",
    "scripts/reproduce_qbraid_public_pilot.sh",
    "scripts/reproduce_phase3.py",
    "docs/qbraid_reproduction.md",
)


class QbraidVerificationError(RuntimeError):
    """Raised after a failing verification report has been persisted."""


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_public_pilot_inputs(repository_root: Path) -> dict[str, Any]:
    """Verify the immutable raw snapshot and processed checksums without downloading."""

    root = repository_root.resolve()
    data_config = load_data_config(root / "configs/data_public_market.yaml")
    snapshot = verify_public_snapshot(data_config)
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(root / "data/processed/public_market")
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise ValueError("public pilot requires non-synthetic public-market processed data")
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_manifest": "data/raw/public_market/"
        + f"{snapshot['snapshot_id']}/snapshot_manifest.json",
        "snapshot_manifest_sha256": sha256_file(data_config.snapshot_manifest_path),
        "raw_file_checksums": {
            name: str(record["sha256"]) for name, record in sorted(snapshot["files"].items())
        },
        "processed_manifest": "data/processed/public_market/data_manifest.json",
        "processed_checksums": dataset.processed_checksums,
    }


def _repository_files(repository_root: Path) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return [repository_root / value for value in output.splitlines() if value]
    except (FileNotFoundError, subprocess.CalledProcessError):
        ignored = {".git", ".venv", ".uv-cache", ".pytest_cache", ".mypy_cache"}
        return [
            path
            for path in repository_root.rglob("*")
            if path.is_file() and not any(part in ignored for part in path.parts)
        ]


def scan_prohibited_paths(repository_root: Path) -> list[dict[str, Any]]:
    """Scan repository text for host-specific paths and absolute result destinations."""

    markers = (
        "/" + "Users/",
        "Google" + "Drive",
        "Library/" + "CloudStorage",
        "file" + "://",
    )
    absolute_results = re.compile(r"(?:^|[\s='\"])/(?:[^\s'\"]*/)*results(?:/|\b)")
    findings: list[dict[str, Any]] = []
    for path in _repository_files(repository_root):
        try:
            if path.stat().st_size > 2_000_000:
                continue
            contents = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(repository_root).as_posix()
        for line_number, line in enumerate(contents.splitlines(), start=1):
            matched = [marker for marker in markers if marker in line]
            if absolute_results.search(line):
                matched.append("absolute results path")
            if matched:
                findings.append(
                    {"path": relative, "line": line_number, "markers": sorted(set(matched))}
                )
    return findings


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _check_imports(required_imports: tuple[str, ...]) -> dict[str, Any]:
    failures: dict[str, str] = {}
    for module_name in required_imports:
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            failures[module_name] = str(exc)
    return {"passed": not failures, "failures": failures}


def _check_distributions(required_distributions: tuple[str, ...]) -> dict[str, Any]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution in required_distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    return {"passed": not missing, "versions": versions, "missing": missing}


def _relative_config_paths(repository_root: Path) -> dict[str, Any]:
    absolute_values: list[dict[str, str]] = []
    for path in sorted((repository_root / "configs").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        relative_file = path.relative_to(repository_root).as_posix()

        def visit(value: Any, key_path: str, config_file: str = relative_file) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{key_path}.{key}" if key_path else str(key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{key_path}[{index}]")
            elif isinstance(value, str) and Path(value).is_absolute():
                absolute_values.append(
                    {
                        "file": config_file,
                        "key": key_path,
                        "value": value,
                    }
                )

        visit(raw, "")
    return {"passed": not absolute_values, "absolute_values": absolute_values}


def _available_cli_commands() -> set[str]:
    from qtyche_qrc.cli import build_parser

    help_text = build_parser().format_help()
    return {command for command in EXPECTED_CLI_COMMANDS if command in help_text}


def _check_qrc_backend() -> dict[str, Any]:
    config = QRCConfig(
        n_qubits=3,
        graph="ring",
        virtual_nodes=1,
        j_strength=1.0,
        h_strength=1.0,
        tau=1.0,
        input_scaling=0.5,
        state_policy="carry_inputs",
        reservoir_seed=2026,
        backend="numpy_density_matrix_exact",
    )
    reservoir = QuantumReservoir(1, config)
    metadata = reservoir.resource_metadata()
    return {
        "passed": metadata["backend"] == "numpy_density_matrix_exact",
        "backend": metadata["backend"],
        "exact_noiseless": True,
        "n_qubits": config.n_qubits,
    }


def verify_qbraid_environment(
    repository_root: Path,
    *,
    report_path: Path | None = None,
    required_imports: tuple[str, ...] = REQUIRED_IMPORTS,
    required_distributions: tuple[str, ...] = REQUIRED_DISTRIBUTIONS,
    minimum_python: tuple[int, int] = (3, 11),
) -> dict[str, Any]:
    """Run all qBraid readiness checks, write a report, and fail after persistence."""

    root = repository_root.resolve()
    output = report_path or root / "results/qbraid/qbraid_environment_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    required_paths = {
        relative: (root / relative).is_file() for relative in REQUIRED_REPOSITORY_PATHS
    }
    python_check = {
        "passed": sys.version_info >= minimum_python,
        "minimum": ".".join(str(value) for value in minimum_python),
        "actual": ".".join(str(value) for value in sys.version_info[:3]),
    }
    imports_check = _check_imports(required_imports)
    dependencies_check = _check_distributions(required_distributions)
    config_paths_check = _relative_config_paths(root)
    prohibited_findings = scan_prohibited_paths(root)
    fixture_manifest = root / "data/processed/data_manifest.json"
    public_manifest = root / "data/processed/public_market/data_manifest.json"
    data_check = {
        "passed": (root / "configs/data.yaml").is_file(),
        "fixture_manifest_available": fixture_manifest.is_file(),
        "public_manifest_available": public_manifest.is_file(),
        "bootstrap_command": "python -m qtyche_qrc.cli prepare-data --config configs/data.yaml",
    }
    fixture_configs: dict[str, str] = {}
    fixture_config_passed = True
    for relative in (
        "configs/models/qrc_classifier_smoke.yaml",
        "configs/models/qrc_regressor_smoke.yaml",
    ):
        try:
            model = load_model_config(root / relative)
            fixture_configs[relative] = model.model_type
        except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            fixture_config_passed = False
            fixture_configs[relative] = f"invalid: {exc}"
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix="write-check-", delete=True):
        writable = True
    commands = _available_cli_commands()
    missing_commands = sorted(set(EXPECTED_CLI_COMMANDS) - commands)

    checks: dict[str, dict[str, Any]] = {
        "python": python_check,
        "imports": imports_check,
        "dependencies": dependencies_check,
        "required_repository_paths": {
            "passed": all(required_paths.values()),
            "paths": required_paths,
        },
        "repository_relative_config_paths": config_paths_check,
        "prohibited_path_scan": {
            "passed": not prohibited_findings,
            "findings": prohibited_findings,
        },
        "data_manifests": data_check,
        "fixture_smoke_configs": {
            "passed": fixture_config_passed,
            "configs": fixture_configs,
        },
        "writable_output_directory": {
            "passed": writable,
            "path": "results/qbraid",
        },
        "cli_commands": {
            "passed": not missing_commands,
            "expected": list(EXPECTED_CLI_COMMANDS),
            "missing": missing_commands,
        },
        "qrc_backend": _check_qrc_backend(),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        **runtime_metadata(),
        "repository_root": ".",
        "git": _git_metadata(root),
        "checks": checks,
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failed = ", ".join(name for name, check in checks.items() if not check["passed"])
        raise QbraidVerificationError(f"qBraid verification failed: {failed}; report: {output}")
    return report
