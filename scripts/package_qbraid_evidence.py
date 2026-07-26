#!/usr/bin/env python3
"""Package one isolated clean-room evidence directory deterministically."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from qtyche_qrc.reproducibility.verification import find_repository_root, sha256_path

REQUIRED_FILES = (
    "command_log.json",
    "dataset_checksum_report.json",
    "environment_report.json",
    "execution_report.json",
    "git_report.json",
    "terminal_transcript.log",
)


def _dataset_checksum_report(root: Path) -> dict[str, Any]:
    frozen_config = yaml.safe_load(
        (root / "configs/reproduction/final_financial_qrc.yaml").read_text(encoding="utf-8")
    )
    expected_processed = dict(frozen_config["study"]["processed_file_sha256"])
    processed_manifest_path = root / "data/processed/public_market/data_manifest.json"
    processed_manifest = (
        json.loads(processed_manifest_path.read_text(encoding="utf-8"))
        if processed_manifest_path.is_file()
        else None
    )
    processed_rows = []
    for name, expected in sorted(expected_processed.items()):
        path = root / "data/processed/public_market" / name
        actual = sha256_path(path) if path.is_file() else None
        processed_rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )

    frozen_public = json.loads(
        (root / "data/public_market_snapshot.json").read_text(encoding="utf-8")
    )
    current_snapshot_path = (
        root / "data/raw/public_market/yahoo_chart_20100101_20251231_v1/snapshot_manifest.json"
    )
    current_snapshot = (
        json.loads(current_snapshot_path.read_text(encoding="utf-8"))
        if current_snapshot_path.is_file()
        else None
    )
    raw_rows = []
    for name, record in sorted(frozen_public["files"].items()):
        current_record = (
            current_snapshot.get("files", {}).get(name, {})
            if isinstance(current_snapshot, dict)
            else {}
        )
        actual = current_record.get("sha256")
        raw_rows.append(
            {
                "name": name,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "passed": actual == record["sha256"],
            }
        )

    mnist_path = root / "results/qrc_mnist/dataset/dataset_manifest.json"
    mnist = json.loads(mnist_path.read_text(encoding="utf-8")) if mnist_path.is_file() else None
    mnist_passed = mnist is None or (
        mnist.get("dataset") == "MNIST"
        and mnist.get("synthetic_data") is False
        and mnist.get("official_partitions_preserved") is True
        and all(record.get("verified") is True for record in mnist.get("files", {}).values())
    )
    processed_passed = bool(processed_rows) and all(row["passed"] for row in processed_rows)
    return {
        "schema_version": 1,
        "status": "pass" if processed_passed and mnist_passed else "fail",
        "public_market": {
            "snapshot_id": frozen_public["snapshot_id"],
            "is_synthetic": frozen_public["is_synthetic"],
            "historical_snapshot_manifest_sha256": frozen_public["snapshot_manifest_sha256"],
            "current_snapshot_manifest_sha256": (
                sha256_path(current_snapshot_path) if current_snapshot_path.is_file() else None
            ),
            "raw_file_checks": raw_rows,
            "historical_raw_snapshot_byte_exact": bool(raw_rows)
            and all(row["passed"] for row in raw_rows),
            "provider_revision_detected": bool(raw_rows)
            and not all(row["passed"] for row in raw_rows),
            "processed_file_checks": processed_rows,
            "processed_model_inputs_byte_exact": processed_passed,
            "processed_manifest_sha256": (
                sha256_path(processed_manifest_path) if processed_manifest_path.is_file() else None
            ),
            "processed_data_source_type": (
                processed_manifest.get("data_source_type")
                if isinstance(processed_manifest, dict)
                else None
            ),
            "processed_is_synthetic": (
                processed_manifest.get("is_synthetic")
                if isinstance(processed_manifest, dict)
                else None
            ),
            "interpretation": (
                "The exact processed files are the immutable model-input contract. "
                "A provider revision limited to unused raw fields may change the "
                "historical raw-file or manifest hash without changing these files."
            ),
        },
        "mnist": {
            "executed": mnist is not None,
            "dataset": mnist.get("dataset") if isinstance(mnist, dict) else None,
            "mode": mnist.get("mode") if isinstance(mnist, dict) else None,
            "subset_checksum": (mnist.get("subset_checksum") if isinstance(mnist, dict) else None),
            "synthetic_data": (mnist.get("synthetic_data") if isinstance(mnist, dict) else None),
            "official_partitions_preserved": (
                mnist.get("official_partitions_preserved") if isinstance(mnist, dict) else None
            ),
            "passed": mnist_passed,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalized_tar(source: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = Path(source.name) / path.relative_to(source)
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return stream.getvalue()


def package_evidence(root: Path, evidence_dir: Path, archive_path: Path) -> dict[str, Any]:
    """Validate, inventory, and archive one evidence directory."""

    _write_json(
        evidence_dir / "dataset_checksum_report.json",
        _dataset_checksum_report(root),
    )
    missing = [name for name in REQUIRED_FILES if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("evidence directory is incomplete: " + ", ".join(missing))
    execution = json.loads((evidence_dir / "execution_report.json").read_text(encoding="utf-8"))
    if execution.get("status") != "success":
        raise ValueError("cannot package an unsuccessful execution report")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (evidence_dir / "package_freeze.txt").write_text(freeze, encoding="utf-8")
    failures: list[dict[str, Any]] = [
        {
            "task_id": record["task_id"],
            "status": record["status"],
            "output_tail": record.get("output_tail", ""),
        }
        for record in execution["tasks"]
        if record.get("status") != "success"
    ]
    transcript = (evidence_dir / "terminal_transcript.log").read_text(encoding="utf-8")
    prior_failures = [
        {
            "task_id": match.group("task"),
            "status": "failure_in_prior_attempt",
            "exit_status": int(match.group("exit")),
        }
        for match in re.finditer(
            r"\] END (?P<task>[^:]+): exit=(?P<exit>[1-9][0-9]*)",
            transcript,
        )
    ]
    failures.extend(prior_failures)
    _write_json(
        evidence_dir / "failures_and_resolutions.json",
        {
            "schema_version": 1,
            "failures": failures,
            "resolution": (
                "No unresolved failures in the packaged successful run."
                if not failures
                else (
                    "The packaged execution is successful after the recorded prior "
                    "attempt failures were resolved. See the terminal transcript."
                )
            ),
        },
    )
    records = [
        {
            "path": path.relative_to(evidence_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(evidence_dir.rglob("*"))
        if path.is_file() and path.name != "generated_artifact_checksums.json"
    ]
    _write_json(
        evidence_dir / "generated_artifact_checksums.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": records,
        },
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tar_payload = _normalized_tar(evidence_dir)
    with (
        archive_path.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
            compresslevel=9,
        ) as compressed,
    ):
        compressed.write(tar_payload)
    checksum = sha256_path(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return {
        "archive": archive_path.relative_to(root).as_posix(),
        "bytes": archive_path.stat().st_size,
        "sha256": checksum,
        "sidecar": sidecar.relative_to(root).as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("qbraid_evidence/final_clean_room"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("qbraid_evidence/phase3_final_clean_room.tar.gz"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_repository_root(Path.cwd())
    evidence = args.evidence_dir if args.evidence_dir.is_absolute() else root / args.evidence_dir
    archive = args.archive if args.archive.is_absolute() else root / args.archive
    try:
        report = package_evidence(root, evidence.resolve(), archive.resolve())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
