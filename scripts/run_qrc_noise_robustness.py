#!/usr/bin/env python3
"""Run or resume the controlled QRC finite-shot and simulated-noise studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.experiments.qrc_robustness import run_qrc_noise_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qrc_noise_robustness.yaml"),
        help="shot/noise study configuration",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        help="reservoir size; defaults to the selected two-qubit architecture",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the configured small grid instead of the full study",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun requested readout tasks even when matching runs are complete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        summary_path = run_qrc_noise_robustness(
            config_path,
            n_qubits=args.n_qubits,
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
