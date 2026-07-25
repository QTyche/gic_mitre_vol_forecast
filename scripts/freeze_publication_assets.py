#!/usr/bin/env python3
"""Freeze compact Phase 3 paper assets from existing checksum-pinned outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtyche_qrc.publication import freeze_publication_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/publication_assets.yaml"),
        help="checksum-pinned Stage 2C publication configuration",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        manifest = freeze_publication_assets(config)
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
                "manifest": manifest.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
