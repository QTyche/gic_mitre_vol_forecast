#!/usr/bin/env python3
"""Run or resume the controlled exact-noiseless QRC encoding-density study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.qrc_encoding_density import run_encoding_density


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qrc_encoding_density.yaml"),
        help="encoding-density study configuration",
    )
    parser.add_argument(
        "--virtual-nodes",
        nargs="+",
        type=int,
        help="temporal readout counts; defaults to the smoke or full configured grid",
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
        help="use the configured small grid and enforce at most three densities and one seed",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun requested readouts even when complete matching runs exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        summary_path = run_encoding_density(
            root / args.config if not args.config.is_absolute() else args.config,
            virtual_nodes=tuple(args.virtual_nodes) if args.virtual_nodes else None,
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
