#!/usr/bin/env python3
"""Run or resume the controlled exact-noiseless QRC qubit-scaling study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.qrc_scaling import run_qubit_scaling


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qrc_qubit_scaling.yaml"),
        help="scaling study configuration",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        help="qubit counts; defaults to the smoke or full configured grid",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="reservoir seeds; defaults to the smoke or full configured grid",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the small configured grid and enforce at most three sizes and one seed",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun requested tasks even when complete matching runs exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        summary_path = run_qubit_scaling(
            root / args.config if not args.config.is_absolute() else args.config,
            qubits=tuple(args.qubits) if args.qubits else None,
            seeds=tuple(args.seeds) if args.seeds else None,
            smoke=args.smoke,
            resume=not args.no_resume,
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
                "summary": summary_path.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
