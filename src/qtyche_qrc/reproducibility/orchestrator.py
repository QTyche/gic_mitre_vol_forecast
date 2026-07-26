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
