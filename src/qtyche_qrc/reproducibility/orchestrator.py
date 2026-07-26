"""Checked subprocess orchestration and resumable evidence capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

from qtyche_qrc.reproducibility.verification import (
    CLONE_URL,
    ReproductionVerificationError,
    find_repository_root,
    git_report,
    sha256_path,
)

TIERS = ("verify", "headline", "full")
ARTIFACT_REUSE_EXECUTION_MODE = "artifact_reuse_finalization"
ARTIFACT_REUSE_SOURCE_REPORT = "execution_report.pre_artifact_reuse.json"
ARTIFACT_REUSE_VALIDATION_REPORT = "artifact_reuse_validation_report.json"
ARTIFACT_REUSE_ALLOWED_PATHS = frozenset(
    {
        ".agents/skills/qbraid-phase3-reproduction/SKILL.md",
        ".agents/skills/qbraid-phase3-reproduction/references/contracts.md",
        "README.md",
        "configs/phase3_reproduction.yaml",
        "configs/reproduction/mnist_exact_portability_reference.json",
        "configs/reproduction/mnist_exact_portability_reference.npz",
        "docs/qbraid_reproduction.md",
        "scripts/package_qbraid_evidence.py",
        "scripts/reproduce_phase3.py",
        "src/qtyche_qrc/reproducibility/artifacts.py",
        "src/qtyche_qrc/reproducibility/garch_portability.py",
        "src/qtyche_qrc/reproducibility/mnist_portability.py",
        "src/qtyche_qrc/reproducibility/orchestrator.py",
        "src/qtyche_qrc/reproducibility/verification.py",
        "tests/test_mnist_portability.py",
        "tests/test_phase3_reproduction.py",
    }
)


class ReproductionTaskError(RuntimeError):
    """Raised when a checked reproduction task fails."""


@dataclass(frozen=True)
class Task:
    """One repository-relative subprocess and its watched artifacts."""

    task_id: str
    command: tuple[str, ...]
    watch: tuple[Path, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_reproduction_config(path: Path) -> dict[str, Any]:
    """Load and structurally validate the Stage 3A orchestration contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Phase 3 reproduction config schema_version must be 1")
    if not isinstance(raw.get("tiers"), dict) or not isinstance(raw.get("tasks"), dict):
        raise ValueError("Phase 3 reproduction config requires tiers and tasks mappings")
    for tier in TIERS:
        selected = raw["tiers"].get(tier)
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"tier {tier} must contain task IDs")
        unknown = set(selected) - set(raw["tasks"])
        if unknown:
            raise ValueError(f"tier {tier} contains unknown tasks: {sorted(unknown)}")
    scientific = raw.get("scientific_contract")
    if not isinstance(scientific, dict):
        raise ValueError("scientific_contract must be a mapping")
    if (
        scientific.get("synthetic_fallback_permitted") is not False
        or scientific.get("physical_qpu_execution") is not False
        or scientific.get("quantum_advantage_claim") is not False
    ):
        raise ValueError("Stage 3A safeguards were weakened")
    return raw


def construct_tasks(
    config: dict[str, Any],
    *,
    tier: str,
    root: Path,
    evidence_dir: Path,
    python_executable: str,
) -> list[Task]:
    """Expand placeholders into shell-free subprocess argument vectors."""

    if tier not in TIERS:
        raise ValueError(f"unknown reproduction tier: {tier}")
    tasks: list[Task] = []
    replacements = {
        "{python}": python_executable,
        "{evidence}": evidence_dir.relative_to(root).as_posix(),
    }

    def expand(value: object) -> str:
        result = str(value)
        for marker, replacement in replacements.items():
            result = result.replace(marker, replacement)
        return result

    for task_id in config["tiers"][tier]:
        raw = config["tasks"][task_id]
        command = tuple(expand(value) for value in raw["command"])
        if not command or any(Path(value).is_absolute() for value in command[1:]):
            raise ValueError(f"task {task_id} contains an absolute or empty command")
        watch = tuple(root / expand(value) for value in raw.get("watch", []))
        tasks.append(Task(str(task_id), command, watch))
    return tasks


def _fingerprint(task: Task, config_sha256: str, commit: str) -> str:
    payload = json.dumps(
        {
            "command": task.command,
            "config_sha256": config_sha256,
            "git_commit": commit,
            "task_id": task.task_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _files_under(paths: tuple[Path, ...]) -> set[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
    return files


def _artifact_records(paths: tuple[Path, ...], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(_files_under(paths)):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return records


def _resume_valid(record: dict[str, Any], *, fingerprint: str, root: Path) -> bool:
    if record.get("status") != "success" or record.get("fingerprint") != fingerprint:
        return False
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or sha256_path(path) != artifact["sha256"]
        ):
            return False
    return True


def validate_recorded_artifacts(
    records: list[dict[str, Any]],
    *,
    root: Path,
) -> dict[str, Any]:
    """Rehash every recorded artifact without trusting an obsolete task fingerprint."""

    repository = root.resolve()
    final_identities: dict[str, tuple[int, str, str]] = {}
    superseded: list[dict[str, Any]] = []
    record_count = 0
    for record in records:
        task_id = str(record.get("task_id", ""))
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            raise ReproductionVerificationError(f"task {task_id} has no artifact inventory")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ReproductionVerificationError("invalid recorded artifact entry")
            relative = Path(str(artifact.get("path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ReproductionVerificationError(f"unsafe recorded artifact path: {relative}")
            path = (repository / relative).resolve()
            try:
                path.relative_to(repository)
            except ValueError as exc:
                raise ReproductionVerificationError(
                    f"recorded artifact escapes repository: {relative}"
                ) from exc
            expected = (int(artifact["bytes"]), str(artifact["sha256"]), task_id)
            key = relative.as_posix()
            prior = final_identities.get(key)
            if prior is not None and prior[:2] != expected[:2]:
                superseded.append(
                    {
                        "path": key,
                        "prior_task_id": prior[2],
                        "prior_bytes": prior[0],
                        "prior_sha256": prior[1],
                        "superseding_task_id": task_id,
                        "final_bytes": expected[0],
                        "final_sha256": expected[1],
                    }
                )
            final_identities[key] = expected
            record_count += 1
    total_bytes = 0
    for relative_text, expected in sorted(final_identities.items()):
        relative = Path(relative_text)
        path = (repository / relative).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != expected[0]
            or sha256_path(path) != expected[1]
        ):
            raise ReproductionVerificationError(
                f"final recorded artifact changed or is missing: {relative}"
            )
        total_bytes += expected[0]
    return {
        "recorded_artifact_entries": record_count,
        "unique_artifacts": len(final_identities),
        "unique_artifact_bytes": total_bytes,
        "superseded_artifact_identity_count": len(superseded),
        "superseded_artifact_identities": superseded,
        "all_recorded_artifacts_exact": True,
    }


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _git_changed_paths(root: Path, source_commit: str, current_commit: str) -> list[str]:
    process = subprocess.run(
        ["git", "diff", "--name-only", f"{source_commit}..{current_commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in process.stdout.splitlines() if line)


def _git_file_sha256(root: Path, commit: str, relative: str) -> str:
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(process.stdout).hexdigest()


def validate_artifact_reuse_changed_paths(changed_paths: list[str]) -> list[str]:
    """Reject artifact reuse when any non-validation implementation changed."""

    normalized = sorted(set(changed_paths))
    prohibited = sorted(set(normalized) - ARTIFACT_REUSE_ALLOWED_PATHS)
    if prohibited:
        raise ReproductionVerificationError(
            "artifact reuse crosses non-validation changes: " + ", ".join(prohibited)
        )
    return normalized


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReproductionVerificationError(f"expected JSON object: {path}")
    return dict(value)


def _repository_relative_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ReproductionVerificationError(
            f"expected a safe repository-relative path, got {relative}"
        )
    repository = root.resolve()
    candidate = (repository / relative).resolve()
    try:
        candidate.relative_to(repository)
    except ValueError as exc:
        raise ReproductionVerificationError(
            f"recorded path escapes repository: {relative}"
        ) from exc
    return candidate


def _artifact_source_commit(config: dict[str, Any], root: Path) -> str:
    contract = config.get("tolerances", {}).get("mnist_exact_portability", {})
    reference = root / str(contract.get("reference", ""))
    expected = str(contract.get("reference_sha256", ""))
    if not reference.is_file() or sha256_path(reference) != expected:
        raise ReproductionVerificationError(
            "MNIST portability reference is missing or not checksum-exact"
        )
    value = _load_json_object(reference)
    return str(value.get("linux_profile", {}).get("artifact_commit", ""))


def _validate_source_execution(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    config_source: Path,
    root: Path,
    current_git: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        source.get("schema_version") != 1
        or source.get("status") != "failure"
        or source.get("tier") != "full"
    ):
        raise ReproductionVerificationError(
            "artifact reuse requires the original failed schema-v1 full execution report"
        )
    error = source.get("error")
    if not isinstance(error, dict) or "task full_comparison failed" not in str(
        error.get("message", "")
    ):
        raise ReproductionVerificationError(
            "source execution did not fail solely at the final comparison task"
        )
    source_git = source.get("git")
    source_configuration = source.get("configuration")
    if not isinstance(source_git, dict) or not isinstance(source_configuration, dict):
        raise ReproductionVerificationError("source execution provenance is incomplete")
    source_commit = str(source_git.get("commit", ""))
    current_commit = str(current_git["commit"])
    required_source_commit = _artifact_source_commit(config, root)
    if source_commit != required_source_commit:
        raise ReproductionVerificationError(
            f"source execution commit {source_commit} is not {required_source_commit}"
        )
    if not _git_is_ancestor(root, source_commit, current_commit):
        raise ReproductionVerificationError(
            "source execution commit is not an ancestor of the validation commit"
        )
    source_config_path = str(source_configuration.get("path", ""))
    if source_config_path != config_source.relative_to(root).as_posix() or _git_file_sha256(
        root, source_commit, source_config_path
    ) != source_configuration.get("sha256"):
        raise ReproductionVerificationError(
            "source execution configuration does not match its recorded Git commit"
        )
    changed_paths = validate_artifact_reuse_changed_paths(
        _git_changed_paths(root, source_commit, current_commit)
    )
    records = source.get("tasks")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ReproductionVerificationError("source task records are invalid")
    source_records = [dict(record) for record in records]
    expected_ids = [str(task_id) for task_id in config["tiers"]["full"]]
    actual_ids = [str(record.get("task_id")) for record in source_records]
    if actual_ids != expected_ids[:-1] or expected_ids[-1] != "full_comparison":
        raise ReproductionVerificationError(
            "source execution must contain every full task before full_comparison in order"
        )
    if any(
        record.get("status") != "success" or int(record.get("exit_status", -1)) != 0
        for record in source_records
    ):
        raise ReproductionVerificationError(
            "a source scientific task was not recorded as successful"
        )
    artifacts = validate_recorded_artifacts(source_records, root=root)
    return source_records, {
        "source_commit": source_commit,
        "validation_commit": current_commit,
        "changed_paths": changed_paths,
        "all_changes_validation_only": True,
        **artifacts,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _environment_report() -> dict[str, Any]:
    return {
        "execution_platform": os.environ.get("QTYCHE_EXECUTION_PLATFORM", "local"),
        "operating_system": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "qbraid_environment_id": os.environ.get("QTYCHE_QBRAID_ENVIRONMENT_ID")
        or os.environ.get("QBRAID_ENVIRONMENT_ID"),
        "qbraid_environment_name": os.environ.get("QTYCHE_QBRAID_ENVIRONMENT_NAME")
        or os.environ.get("QBRAID_ENVIRONMENT_NAME"),
        "qbraid_lab_image": os.environ.get("QTYCHE_QBRAID_LAB_IMAGE"),
    }


def _run_task(
    task: Task,
    *,
    root: Path,
    transcript: TextIO,
    fingerprint: str,
) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    command_text = " ".join(task.command)
    heading = f"[{started_at}] START {task.task_id}: {command_text}"
    print(heading, flush=True)
    transcript.write(heading + "\n")
    transcript.flush()
    process = subprocess.Popen(
        list(task.command),
        cwd=root,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "QTYCHE_EXECUTION_PLATFORM": os.environ.get("QTYCHE_EXECUTION_PLATFORM", "local"),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        transcript.write(line)
        transcript.flush()
        output_lines.append(line)
    exit_code = process.wait()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    duration = time.perf_counter() - started
    peak_value = max(usage_before.ru_maxrss, usage_after.ru_maxrss)
    record = {
        "task_id": task.task_id,
        "command": list(task.command),
        "fingerprint": fingerprint,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "duration_seconds": duration,
        "exit_status": exit_code,
        "peak_child_rss": peak_value,
        "peak_child_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
        "output_tail": "".join(output_lines)[-12000:],
        "artifacts": _artifact_records(task.watch, root),
        "status": "success" if exit_code == 0 else "failure",
    }
    ending = f"[{record['completed_at_utc']}] END {task.task_id}: exit={exit_code}"
    print(ending, flush=True)
    transcript.write(ending + "\n")
    transcript.flush()
    if exit_code != 0:
        raise ReproductionTaskError(
            f"task {task.task_id} failed with exit status {exit_code}: {command_text}"
        )
    return record


def _material_inputs_exist(root: Path) -> list[str]:
    candidates = (
        root / "data/raw/public_market/yahoo_chart_20100101_20251231_v1",
        root / "data/raw/mnist",
        root / "data/processed/public_market",
        root / "results/public_market",
        root / "results/final_financial_qrc",
        root / "results/garch_baseline",
        root / "results/qrc_mnist",
    )
    return [path.relative_to(root).as_posix() for path in candidates if path.exists()]


def run_reproduction(
    config_path: Path,
    *,
    tier: str,
    evidence_dir: Path | None = None,
    resume: bool = True,
) -> Path:
    """Execute one tier, recording checked commands and resumable evidence."""

    root = find_repository_root(config_path)
    config_source = config_path if config_path.is_absolute() else root / config_path
    config = load_reproduction_config(config_source)
    git = git_report(root)
    if not git["clean"]:
        raise ReproductionVerificationError(
            "Stage 3A requires a clean Git checkout; commit, stash, or remove changes first"
        )
    if not git["compatible_submission_descendant"]:
        raise ReproductionVerificationError(
            "current commit is not the required submission commit or a compatible descendant"
        )
    configured = root / str(config["study"]["evidence_root"])
    destination = evidence_dir or configured
    destination = destination if destination.is_absolute() else root / destination
    destination = destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence directory must remain inside the repository") from exc
    state_path = destination / "execution_report.json"
    existing = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if tier != "verify" and existing is None:
        material = _material_inputs_exist(root)
        if material:
            raise ReproductionVerificationError(
                "headline/full reproduction requires a fresh clone; pre-existing generated "
                f"inputs were found: {', '.join(material)}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    tasks = construct_tasks(
        config,
        tier=tier,
        root=root,
        evidence_dir=destination,
        python_executable=sys.executable,
    )
    config_checksum = sha256_path(config_source)
    previous_by_id = {
        str(record["task_id"]): record
        for record in (existing or {}).get("tasks", [])
        if isinstance(record, dict) and "task_id" in record
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "tier": tier,
        "clone_url": CLONE_URL,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "runtime_seconds": None,
        "git": git,
        "environment": _environment_report(),
        "configuration": {
            "path": config_source.relative_to(root).as_posix(),
            "sha256": config_checksum,
        },
        "runtime_planning": config["runtime_planning"],
        "resume_enabled": resume,
        "tasks": [],
        "resumed_task_ids": [],
        "recomputed_task_ids": [],
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
        "synthetic_fallback_permitted": False,
    }
    started = time.perf_counter()
    transcript_path = destination / "terminal_transcript.log"
    command_log_path = destination / "command_log.json"
    _write_json(destination / "environment_report.json", report["environment"])
    _write_json(destination / "git_report.json", git)
    with transcript_path.open("a", encoding="utf-8") as transcript:
        try:
            for task in tasks:
                fingerprint = _fingerprint(task, config_checksum, str(git["commit"]))
                prior = previous_by_id.get(task.task_id)
                if (
                    resume
                    and prior is not None
                    and _resume_valid(prior, fingerprint=fingerprint, root=root)
                ):
                    resumed_record = dict(prior)
                    resumed_record["resumed_at_utc"] = _utc_now()
                    resumed_record["execution"] = "resumed_verified"
                    report["tasks"].append(resumed_record)
                    report["resumed_task_ids"].append(task.task_id)
                    print(f"RESUME {task.task_id}: verified recorded artifacts", flush=True)
                    continue
                record = _run_task(
                    task,
                    root=root,
                    transcript=transcript,
                    fingerprint=fingerprint,
                )
                record["execution"] = "recomputed"
                report["tasks"].append(record)
                report["recomputed_task_ids"].append(task.task_id)
                _write_json(state_path, report)
            report["status"] = "success"
            report["error"] = None
        except Exception as exc:
            report["status"] = "failure"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["completed_at_utc"] = _utc_now()
            report["runtime_seconds"] = time.perf_counter() - started
            report["effective_compute_runtime_seconds"] = sum(
                float(record.get("duration_seconds", 0.0)) for record in report["tasks"]
            )
            peak_records = [
                int(record["peak_child_rss"])
                * (1024 if record.get("peak_child_rss_units") == "KiB" else 1)
                for record in report["tasks"]
                if record.get("peak_child_rss") is not None
            ]
            report["maximum_peak_child_rss_bytes"] = max(peak_records) if peak_records else None
            _write_json(state_path, report)
            _write_json(
                command_log_path,
                [
                    {
                        "task_id": record["task_id"],
                        "command": record["command"],
                        "exit_status": record["exit_status"],
                        "status": record["status"],
                        "execution": record["execution"],
                    }
                    for record in report["tasks"]
                ],
            )
    return state_path


def finalize_artifact_reuse_execution(
    config_path: Path,
    *,
    evidence_dir: Path | None = None,
) -> Path:
    """Finalize a full qBraid run after strict revalidation, without model recomputation."""

    root = find_repository_root(config_path)
    config_source = config_path if config_path.is_absolute() else root / config_path
    config = load_reproduction_config(config_source)
    current_git = git_report(root)
    if not current_git["clean"]:
        raise ReproductionVerificationError(
            "artifact-reuse finalization requires a clean Git checkout"
        )
    if not current_git["compatible_submission_descendant"]:
        raise ReproductionVerificationError(
            "validation commit is not a compatible submission descendant"
        )
    configured = root / str(config["study"]["evidence_root"])
    destination = evidence_dir or configured
    destination = destination if destination.is_absolute() else root / destination
    destination = destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence directory must remain inside the repository") from exc
    state_path = destination / "execution_report.json"
    source_path = destination / ARTIFACT_REUSE_SOURCE_REPORT
    if source_path.is_file():
        source = _load_json_object(source_path)
    else:
        if not state_path.is_file():
            raise FileNotFoundError("the failed full execution report is missing")
        source_bytes = state_path.read_bytes()
        source = json.loads(source_bytes)
        if not isinstance(source, dict):
            raise ReproductionVerificationError("failed execution report is invalid")
        if source.get("status") != "failure":
            raise ReproductionVerificationError(
                "refusing to replace an execution report that is not the original failure"
            )
        source_path.write_bytes(source_bytes)
    source_sha256 = sha256_path(source_path)
    source_records, artifact_validation = _validate_source_execution(
        source,
        config=config,
        config_source=config_source,
        root=root,
        current_git=current_git,
    )
    destination.mkdir(parents=True, exist_ok=True)
    finalization_dir = destination / "artifact_reuse_finalization"
    finalization_dir.mkdir(parents=True, exist_ok=True)
    config_checksum = sha256_path(config_source)
    current_full_tasks = construct_tasks(
        config,
        tier="full",
        root=root,
        evidence_dir=destination,
        python_executable=sys.executable,
    )
    comparison_task = current_full_tasks[-1]
    if comparison_task.task_id != "full_comparison":
        raise ReproductionVerificationError("full_comparison is not the final configured task")
    lightweight_tasks = (
        Task(
            "finalization_frozen_verification",
            (
                sys.executable,
                "-m",
                "qtyche_qrc.reproducibility.verification",
                "--root",
                ".",
                "--output",
                (finalization_dir / "fast_verification_report.json").relative_to(root).as_posix(),
            ),
            (finalization_dir / "fast_verification_report.json",),
        ),
        Task(
            "finalization_focused_tests",
            (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_phase3_reproduction.py",
                "tests/test_processed_semantic_integrity.py",
                "tests/test_garch_portability.py",
                "tests/test_mnist_portability.py",
            ),
            (),
        ),
        comparison_task,
    )
    started_at = _utc_now()
    started = time.perf_counter()
    transcript_path = destination / "terminal_transcript.log"
    validation_records: list[dict[str, Any]] = []
    with transcript_path.open("a", encoding="utf-8") as transcript:
        for task in lightweight_tasks:
            record = _run_task(
                task,
                root=root,
                transcript=transcript,
                fingerprint=_fingerprint(task, config_checksum, str(current_git["commit"])),
            )
            record["execution"] = "recomputed_validation_only"
            record["scientific_model_recomputation"] = False
            validation_records.append(record)
    comparison_record = validation_records[-1]
    for name in (
        "full_reproduction_report.json",
        "garch_portability_report.json",
        "mnist_exact_portability_report.json",
        "processed_data_semantic_verification.json",
    ):
        report = _load_json_object(destination / name)
        if report.get("status") != "pass":
            raise ReproductionVerificationError(
                f"artifact-reuse finalization requires passing {name}"
            )
    full_report = _load_json_object(destination / "full_reproduction_report.json")
    if (
        full_report.get("failed_comparison_count") != 0
        or full_report.get("mnist_genuine") is not True
    ):
        raise ReproductionVerificationError(
            "full comparison is not a zero-failure genuine-MNIST reproduction"
        )
    validation_report_path = destination / ARTIFACT_REUSE_VALIDATION_REPORT
    validation_report = {
        "schema_version": 1,
        "status": "pass",
        "execution_mode": ARTIFACT_REUSE_EXECUTION_MODE,
        "validated_at_utc": _utc_now(),
        "git": current_git,
        "configuration": {
            "path": config_source.relative_to(root).as_posix(),
            "sha256": config_checksum,
        },
        "source_execution_report": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": source_sha256,
            "status": source["status"],
            "failed_task": "full_comparison",
        },
        "compatibility": artifact_validation,
        "validation_tasks": validation_records,
        "all_source_artifacts_rehashed": True,
        "scientific_model_recomputation": False,
        "mnist_reservoir_recomputation": False,
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
    }
    _write_json(validation_report_path, validation_report)
    finalized_records: list[dict[str, Any]] = []
    for source_record in source_records:
        record = dict(source_record)
        record["source_execution"] = record.get("execution")
        record["execution"] = "artifact_reused_checksum_verified"
        record["artifact_revalidated_at_utc"] = validation_report["validated_at_utc"]
        record["source_execution_commit"] = artifact_validation["source_commit"]
        record["scientific_model_recomputation"] = False
        finalized_records.append(record)
    finalized_records.append(comparison_record)
    elapsed = time.perf_counter() - started
    peak_records = [
        int(record["peak_child_rss"]) * (1024 if record.get("peak_child_rss_units") == "KiB" else 1)
        for record in (*finalized_records, *validation_records[:-1])
        if record.get("peak_child_rss") is not None
    ]
    report = {
        "schema_version": 2,
        "status": "success",
        "tier": "full",
        "execution_mode": ARTIFACT_REUSE_EXECUTION_MODE,
        "clone_url": CLONE_URL,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "runtime_seconds": elapsed,
        "effective_compute_runtime_seconds": sum(
            float(record.get("duration_seconds", 0.0)) for record in finalized_records
        ),
        "finalization_runtime_seconds": elapsed,
        "maximum_peak_child_rss_bytes": max(peak_records) if peak_records else None,
        "git": current_git,
        "environment": _environment_report(),
        "configuration": {
            "path": config_source.relative_to(root).as_posix(),
            "sha256": config_checksum,
        },
        "scientific_execution": {
            "git": source["git"],
            "environment": source["environment"],
            "configuration": source["configuration"],
            "started_at_utc": source["started_at_utc"],
            "completed_at_utc": source["completed_at_utc"],
            "source_status": source["status"],
            "source_error": source["error"],
            "source_report": source_path.relative_to(root).as_posix(),
            "source_report_sha256": source_sha256,
        },
        "artifact_reuse_validation": {
            "path": validation_report_path.relative_to(root).as_posix(),
            "sha256": sha256_path(validation_report_path),
            "status": "pass",
            "source_artifact_entries": artifact_validation["recorded_artifact_entries"],
            "source_unique_artifacts": artifact_validation["unique_artifacts"],
            "validation_task_ids": [record["task_id"] for record in validation_records[:-1]],
        },
        "runtime_planning": config["runtime_planning"],
        "resume_enabled": True,
        "tasks": finalized_records,
        "resumed_task_ids": [record["task_id"] for record in source_records],
        "recomputed_task_ids": ["full_comparison"],
        "artifact_revalidated_task_ids": [record["task_id"] for record in source_records],
        "error": None,
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
        "synthetic_fallback_permitted": False,
        "scientific_model_recomputation": False,
        "mnist_reservoir_recomputation": False,
    }
    _write_json(state_path, report)
    _write_json(destination / "environment_report.json", report["environment"])
    _write_json(destination / "git_report.json", current_git)
    _write_json(
        destination / "command_log.json",
        [
            {
                "task_id": record["task_id"],
                "command": record["command"],
                "exit_status": record["exit_status"],
                "status": record["status"],
                "execution": record["execution"],
            }
            for record in finalized_records
        ],
    )
    return state_path


def verify_artifact_reuse_execution(
    root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Revalidate a finalized artifact-reuse execution before packaging."""

    execution_path = evidence_dir / "execution_report.json"
    execution = _load_json_object(execution_path)
    if execution.get("execution_mode") != ARTIFACT_REUSE_EXECUTION_MODE:
        return {"applicable": False, "passed": execution.get("status") == "success"}
    if (
        execution.get("schema_version") != 2
        or execution.get("status") != "success"
        or execution.get("tier") != "full"
        or execution.get("scientific_model_recomputation") is not False
        or execution.get("mnist_reservoir_recomputation") is not False
    ):
        raise ReproductionVerificationError(
            "artifact-reuse execution report has invalid status or safeguards"
        )
    current_git = git_report(root)
    if not current_git["clean"] or execution.get("git", {}).get("commit") != current_git["commit"]:
        raise ReproductionVerificationError(
            "artifact-reuse package commit does not match the clean checkout"
        )
    config_path = _repository_relative_path(
        root,
        execution.get("configuration", {}).get("path", ""),
    )
    if not config_path.is_file() or sha256_path(config_path) != execution.get(
        "configuration", {}
    ).get("sha256"):
        raise ReproductionVerificationError("artifact-reuse execution configuration changed")
    source = execution.get("scientific_execution")
    validation = execution.get("artifact_reuse_validation")
    if not isinstance(source, dict) or not isinstance(validation, dict):
        raise ReproductionVerificationError("artifact-reuse provenance is incomplete")
    source_path = _repository_relative_path(root, source.get("source_report", ""))
    validation_path = _repository_relative_path(root, validation.get("path", ""))
    if (
        not source_path.is_file()
        or sha256_path(source_path) != source.get("source_report_sha256")
        or not validation_path.is_file()
        or sha256_path(validation_path) != validation.get("sha256")
    ):
        raise ReproductionVerificationError(
            "artifact-reuse source or validation report checksum mismatch"
        )
    validation_report = _load_json_object(validation_path)
    if (
        validation_report.get("status") != "pass"
        or validation_report.get("scientific_model_recomputation") is not False
        or validation_report.get("mnist_reservoir_recomputation") is not False
        or validation_report.get("git", {}).get("commit") != current_git["commit"]
    ):
        raise ReproductionVerificationError(
            "artifact-reuse validation report is not package-eligible"
        )
    config = load_reproduction_config(config_path)
    source_report = _load_json_object(source_path)
    source_records, source_validation = _validate_source_execution(
        source_report,
        config=config,
        config_source=config_path,
        root=root,
        current_git=current_git,
    )
    if (
        source.get("git") != source_report.get("git")
        or source.get("configuration") != source_report.get("configuration")
        or validation_report.get("source_execution_report", {}).get("sha256")
        != source.get("source_report_sha256")
        or validation_report.get("compatibility") != source_validation
    ):
        raise ReproductionVerificationError(
            "artifact-reuse source provenance does not match its validation chain"
        )
    expected_ids = [str(task_id) for task_id in config["tiers"]["full"]]
    records = execution.get("tasks")
    if (
        not isinstance(records, list)
        or [str(record.get("task_id")) for record in records if isinstance(record, dict)]
        != expected_ids
    ):
        raise ReproductionVerificationError("artifact-reuse execution task sequence is incomplete")
    typed_records = [dict(record) for record in records if isinstance(record, dict)]
    if len(typed_records) != len(records) or any(
        record.get("status") != "success" or int(record.get("exit_status", -1)) != 0
        for record in typed_records
    ):
        raise ReproductionVerificationError(
            "artifact-reuse execution contains an unsuccessful task"
        )
    for index, record in enumerate(typed_records):
        if record.get("scientific_model_recomputation") is not False:
            raise ReproductionVerificationError(
                f"task {record.get('task_id')} lacks the no-recomputation safeguard"
            )
        if index < len(source_records):
            if record.get("execution") != "artifact_reused_checksum_verified" or record.get(
                "source_execution_commit"
            ) != source_report.get("git", {}).get("commit"):
                raise ReproductionVerificationError(
                    f"source task {record.get('task_id')} was not strictly revalidated"
                )
        elif (
            record.get("task_id") != "full_comparison"
            or record.get("execution") != "recomputed_validation_only"
        ):
            raise ReproductionVerificationError(
                "only full_comparison may be recomputed during artifact reuse"
            )
    validation_tasks = validation_report.get("validation_tasks")
    if not isinstance(validation_tasks, list) or [
        record.get("task_id") for record in validation_tasks if isinstance(record, dict)
    ] != [
        "finalization_frozen_verification",
        "finalization_focused_tests",
        "full_comparison",
    ]:
        raise ReproductionVerificationError(
            "artifact-reuse lightweight validation task sequence is incomplete"
        )
    if any(
        not isinstance(record, dict)
        or record.get("status") != "success"
        or int(record.get("exit_status", -1)) != 0
        or record.get("scientific_model_recomputation") is not False
        for record in validation_tasks
    ):
        raise ReproductionVerificationError(
            "artifact-reuse lightweight validation contains a failed task"
        )
    artifacts = validate_recorded_artifacts(typed_records, root=root)
    return {
        "applicable": True,
        "passed": True,
        "execution_report": execution_path.relative_to(root).as_posix(),
        "source_report": source_path.relative_to(root).as_posix(),
        "validation_report": validation_path.relative_to(root).as_posix(),
        **artifacts,
    }
