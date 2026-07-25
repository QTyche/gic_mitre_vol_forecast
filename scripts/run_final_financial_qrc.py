#!/usr/bin/env python3
"""Run/resume the frozen final financial QRC and its robustness refresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.final_financial_qrc import run_final_financial_qrc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_financial_qrc.yaml"),
        help="frozen final-QRC configuration",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the deterministic one-reservoir small grid",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="execute requested task runs even when complete matching runs exist",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="run only the exact benchmark and regenerate available reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        summary = run_final_financial_qrc(
            config,
            smoke=args.smoke,
            resume=not args.no_resume,
            refresh_robustness=not args.skip_robustness,
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
