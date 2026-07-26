#!/usr/bin/env python3
"""Audit every locally present frozen Stage 2C source without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qtyche_qrc.reproducibility.verification import find_repository_root, sha256_path


def audit_prior_results(root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "paper_assets/publication_assets_manifest.json").read_text(encoding="utf-8")
    )
    records: list[dict[str, Any]] = []
    for source in manifest["source_artifacts"]:
        path = root / str(source["path"])
        actual = sha256_path(path) if path.is_file() else None
        records.append(
            {
                "path": source["path"],
                "expected_sha256": source["sha256"],
                "actual_sha256": actual,
                "present": path.is_file(),
                "passed": actual == source["sha256"] if path.is_file() else None,
            }
        )
    mismatches = [record for record in records if record["passed"] is False]
    return {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "present_source_count": sum(bool(record["present"]) for record in records),
        "absent_source_count": sum(not bool(record["present"]) for record in records),
        "mismatch_count": len(mismatches),
        "sources": records,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qbraid_evidence/prior_result_checksum_audit.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_repository_root(Path.cwd())
    report = audit_prior_results(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
