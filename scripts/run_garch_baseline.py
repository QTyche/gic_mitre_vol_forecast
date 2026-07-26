#!/usr/bin/env python3
"""Run or resume the leakage-safe Gaussian GARCH(1,1) public baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.garch_baseline import run_garch_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/garch_baseline.yaml"),
        help="GARCH baseline study configuration",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the deterministic reduced training window and start grid",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="fit a new run even when a complete matching run exists",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="fit and evaluate GARCH without requiring downstream comparison runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = root / args.config if not args.config.is_absolute() else args.config
    try:
        summary_path = run_garch_baseline(
            config,
            smoke=args.smoke,
            resume=not args.no_resume,
            write_comparison=not args.fit_only,
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
