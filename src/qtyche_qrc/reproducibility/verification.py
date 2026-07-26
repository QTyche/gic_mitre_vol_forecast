"""Verify the tracked, frozen Phase 3 contract without executing models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SUBMISSION_COMMIT = "cd9fa988f854009e408af1774a97ed663b0e8b86"
CLONE_URL = "https://github.com/QTyche/gic_mitre_vol_forecast.git"
EXPECTED_CHECKSUMS = {
    "configs/data_public_market.yaml": (
        "07281e4a7c2195f4d7f764cacd83721c36ee254491e9858c09bbcad742e88289"
    ),
    "configs/final_financial_qrc.yaml": (
        "2eb0abdacc8799842b6b609783717da8f2a64bc2d7fc25595709fe4457f74587"
    ),
    "configs/qrc_mnist_benchmark.yaml": (
        "b2afa29a00ae5a1510a22baa2335ba8a5b8e07a3e66f33ca7facc7c9ad484dcd"
    ),
    "data/public_market_snapshot.json": (
        "c3079a9fbd15d1bb23c89d8e8e9059b47cd44cf6a4bf9be75ebf00325bbd7481"
    ),
    "paper_assets/final_results_manifest.json": (
        "a6cc26b63c6931e70e07b4c513fded101786c4dc52c0bc115bdc5294cfbe32d8"
    ),
    "paper_assets/publication_assets_manifest.json": (
        "545f6075746af61882b6ad338b2d088039b86ce9e29210e3978fc1f704e546d8"
    ),
}
EXPECTED_ARCHITECTURE_SHA256 = "10a431b7d047f5e0b18b657815492560a717aca6588067402bf47cb64983190f"
EXPECTED_PUBLICATION_TREE_DIGEST = (
    "70fbb66ed94b74520732c94819dc30e623bd89d5f4d1d3f70df4bba05f174b85"
)
REQUIRED_DISTRIBUTIONS = (
    "matplotlib",
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
)
EXPECTED_SCIENTIFIC_VERSIONS = {
    "matplotlib": "3.11.1",
    "numpy": "2.4.6",
    "pandas": "2.3.3",
    "PyYAML": "6.0.3",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.1",
}
PROHIBITED_CLAIM_IDS = (1, 2, 11, 12, 14)


class ReproductionVerificationError(RuntimeError):
    """Raised after a failed frozen-contract report has been written."""


def sha256_path(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_repository_root(start: Path) -> Path:
    """Find the nearest Git repository containing this project."""

    resolved = start.resolve()
    candidates = (resolved, *resolved.parents) if resolved.is_dir() else resolved.parents
    for candidate in candidates:
        if (
            (candidate / ".git").exists()
            and (candidate / "pyproject.toml").is_file()
            and (candidate / "src/qtyche_qrc").is_dir()
        ):
            return candidate
    raise ReproductionVerificationError(
        f"could not locate the gic_mitre_vol_forecast repository from {start}"
    )


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def git_report(root: Path) -> dict[str, Any]:
    """Capture commit, branch, remote, clean state, and submission ancestry."""

    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    status = _git(root, "status", "--porcelain=v1").stdout
    remote_process = _git(root, "remote", "get-url", "origin", check=False)
    remote = remote_process.stdout.strip() if remote_process.returncode == 0 else None
    branch_process = _git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    branch = branch_process.stdout.strip() if branch_process.returncode == 0 else None
    ancestor = (
        _git(
            root,
            "merge-base",
            "--is-ancestor",
            SUBMISSION_COMMIT,
            commit,
            check=False,
        ).returncode
        == 0
    )
    return {
        "branch": branch,
        "clean": not bool(status.strip()),
        "commit": commit,
        "compatible_submission_descendant": ancestor,
        "origin": remote,
        "submission_commit": SUBMISSION_COMMIT,
    }


def compare_numeric(
    actual: float,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare one regenerated floating-point value with declared tolerances."""

    absolute_difference = abs(float(actual) - float(expected))
    permitted = absolute_tolerance + relative_tolerance * abs(float(expected))
    return {
        "actual": float(actual),
        "expected": float(expected),
        "absolute_difference": absolute_difference,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "permitted_difference": permitted,
        "passed": absolute_difference <= permitted,
    }


def publication_tree_digest(root: Path) -> str:
    """Hash relative names and content hashes for manifest-selected paper assets."""

    manifest_path = root / "paper_assets/publication_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    paths = sorted(str(record["path"]) for record in manifest["selected_assets"])
    for relative in paths:
        path = root / relative
        checksum = sha256_path(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_expected_files(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative, expected in sorted(EXPECTED_CHECKSUMS.items()):
        path = root / relative
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )
    return {"passed": all(row["passed"] for row in rows), "files": rows}


def _verify_publication_assets(root: Path) -> dict[str, Any]:
    manifest_path = root / "paper_assets/publication_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in manifest["selected_assets"]:
        relative = str(record["path"])
        path = root / relative
        actual = sha256_path(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        rows.append(
            {
                "path": relative,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "expected_bytes": record["bytes"],
                "actual_bytes": actual_bytes,
                "passed": actual == record["sha256"] and actual_bytes == record["bytes"],
            }
        )
    expected_paths = {str(record["path"]) for record in manifest["selected_assets"]}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "paper_assets").rglob("*")
        if path.is_file() and path != manifest_path
    }
    tree_digest = publication_tree_digest(root)
    return {
        "asset_count": len(rows),
        "declared_asset_count": manifest["selected_asset_count"],
        "assets": rows,
        "exact_path_set": expected_paths == actual_paths,
        "publication_tree_digest": tree_digest,
        "expected_publication_tree_digest": EXPECTED_PUBLICATION_TREE_DIGEST,
        "passed": (
            all(row["passed"] for row in rows)
            and len(rows) == manifest["selected_asset_count"]
            and expected_paths == actual_paths
            and tree_digest == EXPECTED_PUBLICATION_TREE_DIGEST
        ),
    }


def _fact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["fact_id"]): dict(record) for record in manifest["facts"]}


def _verify_frozen_facts(root: Path) -> dict[str, Any]:
    results = json.loads(
        (root / "paper_assets/final_results_manifest.json").read_text(encoding="utf-8")
    )
    facts = _fact_map(results)
    source_pairs = {(str(record["path"]), str(record["sha256"])) for record in results["sources"]}
    fact_sources_registered = all(
        (str(record["source_artifact_path"]), str(record["source_artifact_sha256"])) in source_pairs
        for record in results["facts"]
    )
    architecture_facts = {
        "backend": facts["financial.architecture.backend"]["exact_value"],
        "feature_dimension": facts["financial.architecture.feature_dimension"]["exact_value"],
        "manifest_sha256": facts["financial.architecture.manifest_sha256"]["exact_value"],
        "n_qubits": facts["financial.architecture.n_qubits"]["exact_value"],
        "state_policy": facts["financial.architecture.state_policy"]["exact_value"],
        "virtual_nodes": facts["financial.architecture.virtual_nodes"]["exact_value"],
    }
    prohibited = {int(record["claim_id"]): str(record["status"]) for record in results["claims"]}
    prohibited_preserved = all(
        prohibited.get(claim_id) == "prohibited" for claim_id in PROHIBITED_CLAIM_IDS
    )
    mnist_subset = facts["mnist.data.subset_checksum"]["exact_value"]
    return {
        "architecture": architecture_facts,
        "architecture_checksum_passed": (
            architecture_facts["manifest_sha256"] == EXPECTED_ARCHITECTURE_SHA256
        ),
        "fact_count": len(results["facts"]),
        "source_count": len(results["sources"]),
        "fact_sources_registered": fact_sources_registered,
        "mnist_subset_checksum": mnist_subset,
        "prohibited_claim_statuses": prohibited,
        "prohibited_claims_preserved": prohibited_preserved,
        "passed": (
            architecture_facts
            == {
                "backend": "numpy_density_matrix_exact",
                "feature_dimension": 6,
                "manifest_sha256": EXPECTED_ARCHITECTURE_SHA256,
                "n_qubits": 2,
                "state_policy": "reset_each_input",
                "virtual_nodes": 2,
            }
            and fact_sources_registered
            and prohibited_preserved
            and mnist_subset == "06c1ee9c6db87efc13ebaacc7f4406297d061d10d573731434c3f957e7c0574e"
        ),
    }


def _verify_data_declarations(root: Path) -> dict[str, Any]:
    public = json.loads((root / "data/public_market_snapshot.json").read_text(encoding="utf-8"))
    mnist = yaml.safe_load((root / "configs/qrc_mnist_benchmark.yaml").read_text(encoding="utf-8"))
    public_checksums = {
        name: str(record["sha256"]) for name, record in sorted(public["files"].items())
    }
    mnist_checksums = {
        name: str(record["sha256"]) for name, record in sorted(mnist["dataset"]["files"].items())
    }
    public_passed = (
        public.get("snapshot_id") == "yahoo_chart_20100101_20251231_v1"
        and public.get("data_source_type") == "public_market"
        and public.get("is_synthetic") is False
        and all(len(value) == 64 for value in public_checksums.values())
    )
    return {
        "public_market": {
            "snapshot_id": public.get("snapshot_id"),
            "snapshot_manifest_sha256": public.get("snapshot_manifest_sha256"),
            "file_checksums": public_checksums,
            "synthetic": public.get("is_synthetic"),
            "passed": public_passed,
        },
        "mnist": {
            "provider": mnist["dataset"]["provider"],
            "file_checksums": mnist_checksums,
            "synthetic_fallback_permitted": False,
            "passed": len(mnist_checksums) == 4
            and all(len(value) == 64 for value in mnist_checksums.values()),
        },
        "passed": public_passed
        and len(mnist_checksums) == 4
        and all(len(value) == 64 for value in mnist_checksums.values()),
    }


def _environment_report() -> dict[str, Any]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    qbraid_name = os.environ.get("QTYCHE_QBRAID_ENVIRONMENT_NAME") or os.environ.get(
        "QBRAID_ENVIRONMENT_NAME"
    )
    qbraid_id = os.environ.get("QTYCHE_QBRAID_ENVIRONMENT_ID") or os.environ.get(
        "QBRAID_ENVIRONMENT_ID"
    )
    execution_platform = os.environ.get("QTYCHE_EXECUTION_PLATFORM", "local")
    version_mismatches = {
        distribution: {
            "expected": expected,
            "actual": versions.get(distribution),
        }
        for distribution, expected in EXPECTED_SCIENTIFIC_VERSIONS.items()
        if versions.get(distribution) != expected
    }
    return {
        "execution_platform": execution_platform,
        "operating_system": platform.platform(),
        "package_versions": versions,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "python_312_preferred": sys.version_info[:2] == (3, 12),
        "qbraid_environment_id": qbraid_id,
        "qbraid_environment_name": qbraid_name,
        "qbraid_identity_complete": bool(qbraid_id and qbraid_name),
        "qbraid_lab": execution_platform == "qbraid_lab",
        "required_distributions_missing": missing,
        "scientific_version_mismatches": version_mismatches,
        "passed": sys.version_info >= (3, 11) and not missing and not version_mismatches,
    }


def verify_frozen_repository(
    root: Path,
    *,
    report_path: Path,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Verify tracked identities and persist a complete machine-readable report."""

    repository_root = find_repository_root(root)
    checks = {
        "tracked_checksums": _verify_expected_files(repository_root),
        "publication_assets": _verify_publication_assets(repository_root),
        "frozen_facts": _verify_frozen_facts(repository_root),
        "data_declarations": _verify_data_declarations(repository_root),
        "environment": _environment_report(),
    }
    git = git_report(repository_root)
    git_passed = git["compatible_submission_descendant"] and (git["clean"] or not require_clean)
    checks["git"] = {
        **git,
        "require_clean": require_clean,
        "passed": git_passed,
    }
    status = "pass" if all(bool(record["passed"]) for record in checks.values()) else "fail"
    report = {
        "schema_version": 1,
        "status": status,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "clone_url": CLONE_URL,
        "checks": checks,
        "exact_checksum_policy": (
            "Tracked deterministic files and selected publication assets require exact SHA-256 "
            "equality."
        ),
        "numeric_tolerance_policy": {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-9,
            "justification": (
                "Regenerated floating-point metrics may vary at final digits across BLAS, "
                "SciPy, and CPU implementations; dataset, config, and display files remain exact."
            ),
        },
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
    }
    destination = report_path if report_path.is_absolute() else repository_root / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if status != "pass":
        failed = [name for name, record in checks.items() if not record["passed"]]
        raise ReproductionVerificationError(
            "frozen Phase 3 verification failed: " + ", ".join(failed)
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_frozen_repository(args.root, report_path=args.output)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
