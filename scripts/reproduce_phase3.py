#!/usr/bin/env python3
"""One non-notebook entry point for Phase 3 smoke, capacity, and public pilot stages."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from qtyche_qrc.cli import main as cli_main
from qtyche_qrc.qbraid import verify_public_pilot_inputs
from qtyche_qrc.runtime import runtime_metadata

PUBLIC_SEEDS = (2026, 2027, 2028)


class StageFailure(RuntimeError):
    """Raised when a recorded CLI step exits unsuccessfully."""


@dataclass(frozen=True)
class CommandRecord:
    """One in-process invocation of the repository CLI."""

    arguments: list[str]
    exit_code: int
    duration_seconds: float
    output: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _run_cli(arguments: list[str], root: Path) -> CommandRecord:
    stream = io.StringIO()
    started = time.perf_counter()
    exit_code = 0
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            returned = cli_main(arguments)
            exit_code = int(returned)
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:
            exit_code = 1
            print(f"{type(exc).__name__}: {exc}")
    duration = time.perf_counter() - started
    output = stream.getvalue().strip().replace(str(root), ".")
    record = CommandRecord(
        arguments=["python", "-m", "qtyche_qrc.cli", *arguments],
        exit_code=exit_code,
        duration_seconds=duration,
        output=output,
    )
    if exit_code != 0:
        raise StageFailure(
            f"command failed with exit code {exit_code}: {' '.join(record.arguments)}\n{output}"
        )
    return record


def _last_output_path(record: CommandRecord, root: Path) -> Path:
    lines = [line.strip() for line in record.output.splitlines() if line.strip()]
    if not lines:
        raise StageFailure(f"command produced no output path: {' '.join(record.arguments)}")
    value = lines[-1]
    return (root / value).resolve() if not Path(value).is_absolute() else Path(value)


def _record_checksum(path: Path, root: Path, checksums: dict[str, str]) -> None:
    if not path.is_file():
        raise StageFailure(f"expected output is missing: {path}")
    checksum = _sha256(path)
    if checksum != _sha256(path):
        raise StageFailure(f"output checksum was not stable when re-read: {path}")
    checksums[path.relative_to(root).as_posix()] = checksum


def _run_capacity(
    root: Path,
    run_token: str,
    commands: list[CommandRecord],
    outputs: list[str],
    checksums: dict[str, str],
) -> None:
    template_path = root / "configs/models/qrc_capacity_qbraid_smoke.yaml"
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    raw["analysis"]["id"] = f"qrc_capacity_qbraid_smoke_{run_token}"
    effective_path = root / "results/qbraid" / f"capacity_config_{run_token}.yaml"
    effective_path.parent.mkdir(parents=True, exist_ok=True)
    effective_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    record = _run_cli(["characterize-qrc", "--config", str(effective_path)], root)
    commands.append(record)
    output_dir = _last_output_path(record, root)
    outputs.append(output_dir.relative_to(root).as_posix())
    for relative in (
        "linear_memory_by_delay.csv",
        "quadratic_capacity_by_delay.csv",
        "cross_delay_capacity.csv",
        "feature_rank.json",
        "manifest.json",
    ):
        _record_checksum(output_dir / relative, root, checksums)


def _run_smoke(
    root: Path,
    run_token: str,
    commands: list[CommandRecord],
    outputs: list[str],
    checksums: dict[str, str],
) -> None:
    commands.append(_run_cli(["verify-qbraid"], root))
    commands.append(_run_cli(["create-fixture-data", "--config", "configs/data.yaml"], root))
    commands.append(_run_cli(["prepare-data", "--config", "configs/data.yaml"], root))
    commands.append(_run_cli(["audit-data", "--processed-dir", "data/processed"], root))

    baseline_configs = (
        ("train-baseline", "configs/models/majority_classifier.yaml"),
        ("train-baseline", "configs/models/regime_persistence.yaml"),
        ("search-baseline", "configs/models/logistic_regression.yaml"),
        ("train-baseline", "configs/models/rv_persistence.yaml"),
        ("search-baseline", "configs/models/esn_classifier_smoke.yaml"),
        ("search-baseline", "configs/models/esn_regressor_smoke.yaml"),
    )
    for command, config in baseline_configs:
        record = _run_cli([command, "--config", config, "--allow-synthetic-results"], root)
        commands.append(record)
        experiment_dir = _last_output_path(record, root)
        outputs.append(experiment_dir.relative_to(root).as_posix())
        _record_checksum(experiment_dir / "manifest.json", root, checksums)
        _record_checksum(experiment_dir / "test_metrics.json", root, checksums)

    commands.append(
        _run_cli(
            [
                "generate-qrc-features",
                "--config",
                "configs/models/qrc_classifier_smoke.yaml",
                "--reservoir-seed",
                "2026",
                "--allow-synthetic-results",
            ],
            root,
        )
    )
    for config in (
        "configs/models/qrc_classifier_smoke.yaml",
        "configs/models/qrc_regressor_smoke.yaml",
    ):
        record = _run_cli(
            [
                "train-qrc",
                "--config",
                config,
                "--reservoir-seed",
                "2026",
                "--allow-synthetic-results",
            ],
            root,
        )
        commands.append(record)
        experiment_dir = _last_output_path(record, root)
        outputs.append(experiment_dir.relative_to(root).as_posix())
        manifest_path = experiment_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["backend"] != "numpy_density_matrix_exact" or not manifest["exact_noiseless"]:
            raise StageFailure("QRC smoke manifest does not identify the exact noiseless backend")
        _record_checksum(manifest_path, root, checksums)
        _record_checksum(experiment_dir / "test_metrics.json", root, checksums)
        _record_checksum(experiment_dir / "test_predictions.csv", root, checksums)
    _run_capacity(root, run_token, commands, outputs, checksums)


def _run_public_pilot(
    root: Path,
    seeds: tuple[int, ...],
    commands: list[CommandRecord],
    outputs: list[str],
    checksums: dict[str, str],
) -> dict[str, Any]:
    commands.append(_run_cli(["verify-qbraid"], root))
    provenance = verify_public_pilot_inputs(root)
    for seed in seeds:
        commands.append(
            _run_cli(
                [
                    "generate-qrc-features",
                    "--config",
                    "configs/models/qrc_classifier_pilot.yaml",
                    "--reservoir-seed",
                    str(seed),
                ],
                root,
            )
        )
        for config in (
            "configs/models/qrc_classifier_pilot.yaml",
            "configs/models/qrc_regressor_pilot.yaml",
        ):
            record = _run_cli(
                ["train-qrc", "--config", config, "--reservoir-seed", str(seed)], root
            )
            commands.append(record)
            experiment_dir = _last_output_path(record, root)
            outputs.append(experiment_dir.relative_to(root).as_posix())
            manifest_path = experiment_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest["backend"] != "numpy_density_matrix_exact"
                or not manifest["exact_noiseless"]
            ):
                raise StageFailure(
                    "public QRC manifest does not identify the exact noiseless backend"
                )
            _record_checksum(manifest_path, root, checksums)
            _record_checksum(experiment_dir / "test_metrics.json", root, checksums)
    comparison = _run_cli(
        [
            "compare-qrc-seeds",
            "--results-dir",
            "results/qrc_public_pilot",
            "--output-dir",
            "results/tables",
        ],
        root,
    )
    commands.append(comparison)
    for path in (
        root / "results/tables/qrc_pilot_validation_by_seed.csv",
        root / "results/tables/qrc_pilot_test_by_seed.csv",
        root / "results/tables/qrc_pilot_seed_summary.csv",
    ):
        _record_checksum(path, root, checksums)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("smoke", "capacity", "public-pilot", "all")
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=2026)
    seeds.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed not in PUBLIC_SEEDS:
        raise SystemExit(f"--seed must be one of {PUBLIC_SEEDS}")
    if args.all_seeds and args.stage not in {"public-pilot", "all"}:
        raise SystemExit("--all-seeds is only valid for public-pilot or all")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    run_started = datetime.now(timezone.utc)
    run_token = run_started.strftime("%Y%m%dT%H%M%S%fZ")
    commands: list[CommandRecord] = []
    outputs: list[str] = []
    checksums: dict[str, str] = {}
    public_provenance: dict[str, Any] | None = None
    summary_path = args.summary or root / "results/qbraid" / f"phase3_{args.stage}_summary.json"
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if args.stage in {"smoke", "all"}:
            _run_smoke(root, run_token, commands, outputs, checksums)
        elif args.stage == "capacity":
            commands.append(_run_cli(["verify-qbraid"], root))
            _run_capacity(root, run_token, commands, outputs, checksums)
        if args.stage in {"public-pilot", "all"}:
            seeds = PUBLIC_SEEDS if args.all_seeds else (args.seed,)
            public_provenance = _run_public_pilot(root, seeds, commands, outputs, checksums)
        status = "success"
        error = None
    except Exception as exc:
        status = "failure"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "status": status,
        "error": error,
        "started_at_utc": run_started.isoformat(),
        "runtime_seconds": elapsed,
        **runtime_metadata(),
        "git": _git_metadata(root),
        "seeds": list(PUBLIC_SEEDS if args.all_seeds else (args.seed,)),
        "qrc_backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "smoke_data_is_synthetic": args.stage in {"smoke", "all"},
        "synthetic_data_warning": (
            "SYNTHETIC FIXTURE DATA — NOT A FINANCIAL PERFORMANCE RESULT"
            if args.stage in {"smoke", "all"}
            else None
        ),
        "synthetic_smoke_outputs_are_financial_evidence": False,
        "commands": [asdict(record) for record in commands],
        "outputs": outputs,
        "verified_output_checksums": dict(sorted(checksums.items())),
        "deterministic_output_checksums": sorted(
            checksum
            for path, checksum in checksums.items()
            if Path(path).name
            in {
                "test_metrics.json",
                "test_predictions.csv",
                "linear_memory_by_delay.csv",
                "quadratic_capacity_by_delay.csv",
                "cross_delay_capacity.csv",
                "feature_rank.json",
            }
        ),
        "public_data_provenance": public_provenance,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary_path.relative_to(root))
    if status != "success":
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
