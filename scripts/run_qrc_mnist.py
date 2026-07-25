#!/usr/bin/env python3
"""Run or resume the isolated common MNIST QRC benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.qrc_mnist import run_qrc_mnist_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qrc_mnist_benchmark.yaml"),
        help="MNIST benchmark configuration",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing checksum-pinned official MNIST files before running",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download/verify official MNIST and exit without fitting models",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the deterministic 200/50/50 one-seed subset",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="refit requested readouts even when matching runs are complete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        summary = run_qrc_mnist_benchmark(
            config,
            smoke=args.smoke,
            resume=not args.no_resume,
            download=args.download or args.download_only,
            download_only=args.download_only,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "success",
                "summary": summary.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
