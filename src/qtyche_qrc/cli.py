"""Command-line entry points for validation and reproducible experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qtyche_qrc.config import ConfigError, load_config
from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots, fixture_summary
from qtyche_qrc.data.pipeline import inspect_processed_targets, prepare_data
from qtyche_qrc.data.validation import DataValidationError, audit_processed_data
from qtyche_qrc.experiments.compare import compare_baselines
from qtyche_qrc.experiments.manifest import create_manifest
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.run import (
    SyntheticResultsError,
    evaluate_experiment,
    inspect_experiment,
    run_baseline_experiment,
)
from qtyche_qrc.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(
        prog="qtyche-qrc",
        description="Team QTyche reproducible benchmark utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate a YAML config")
    validate.add_argument("--config", required=True, help="path to a YAML configuration")

    smoke = subparsers.add_parser("smoke", help="write a manifest-only smoke run")
    smoke.add_argument("--config", required=True, help="path to a YAML configuration")

    fixture = subparsers.add_parser(
        "create-fixture-data",
        help="create or verify deterministic offline raw CSV fixtures",
    )
    fixture.add_argument("--config", required=True, help="path to the data YAML configuration")

    prepare = subparsers.add_parser(
        "prepare-data",
        help="prepare causal features, targets, purged splits, and manifests",
    )
    prepare.add_argument("--config", required=True, help="path to the data YAML configuration")

    audit = subparsers.add_parser("audit-data", help="audit persisted processed data")
    audit.add_argument("--processed-dir", required=True, help="processed data directory")

    inspect = subparsers.add_parser(
        "inspect-targets",
        help="show regime and transition target distributions",
    )
    inspect.add_argument("--processed-dir", required=True, help="processed data directory")

    for command, help_text in (
        ("train-baseline", "train, validate, and test one configured baseline"),
        ("search-baseline", "run validation-only baseline hyperparameter selection"),
    ):
        baseline = subparsers.add_parser(command, help=help_text)
        baseline.add_argument("--config", required=True, help="model YAML configuration")
        baseline.add_argument(
            "--allow-synthetic-results",
            action="store_true",
            help="allow clearly marked fixture-only smoke outputs",
        )

    evaluate = subparsers.add_parser(
        "evaluate-experiment", help="recompute metrics from persisted predictions"
    )
    evaluate.add_argument("--experiment-dir", required=True)

    compare = subparsers.add_parser(
        "compare-baselines", help="write separate validation and test comparison tables"
    )
    compare.add_argument("--results-dir", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument(
        "--latest-per-model",
        action="store_true",
        help="retain only the latest experiment for each task/seed/model tuple",
    )

    inspect_experiment_parser = subparsers.add_parser(
        "inspect-experiment", help="inspect experiment provenance and metrics"
    )
    inspect_experiment_parser.add_argument("--experiment-dir", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"valid configuration: {config.source}")
            return 0
        if args.command == "smoke":
            config = load_config(args.config)
            set_global_seed(config.experiment.seed)
            manifest_path = create_manifest(config)
            print(manifest_path)
            return 0
        if args.command == "create-fixture-data":
            data_config = load_data_config(args.config)
            paths = create_or_verify_fixture_snapshots(data_config)
            print(fixture_summary(data_config))
            for name, path in paths.items():
                print(f"{name}: {path}")
            return 0
        if args.command == "prepare-data":
            data_config = load_data_config(args.config)
            result = prepare_data(data_config)
            construction = result.quality_report["construction"]
            splitting = result.quality_report["splitting"]
            print(f"prepared data: {result.processed_dir}")
            print(
                "construction removals: "
                f"features={construction['rows_removed_missing_features']}, "
                f"targets={construction['rows_removed_missing_targets']}"
            )
            print(
                "purged forward windows: "
                + json.dumps(splitting["purged_forward_window_rows"], sort_keys=True)
            )
            print(f"data manifest: {result.output_paths['data_manifest']}")
            return 0
        if args.command == "audit-data":
            summary = audit_processed_data(Path(args.processed_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-targets":
            summary = inspect_processed_targets(Path(args.processed_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command in {"train-baseline", "search-baseline"}:
            model_config = load_model_config(Path(args.config))
            if args.command == "search-baseline" and not model_config.search_enabled:
                raise ValueError("search-baseline requires search.enabled: true")
            experiment_dir = run_baseline_experiment(
                Path(args.config), allow_synthetic_results=args.allow_synthetic_results
            )
            print(experiment_dir)
            return 0
        if args.command == "evaluate-experiment":
            summary = evaluate_experiment(Path(args.experiment_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "compare-baselines":
            validation_path, test_path = compare_baselines(
                Path(args.results_dir).resolve(),
                Path(args.output_dir).resolve(),
                latest_per_model=args.latest_per_model,
            )
            print(f"validation comparison: {validation_path}")
            print(f"test comparison: {test_path}")
            return 0
        if args.command == "inspect-experiment":
            summary = inspect_experiment(Path(args.experiment_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    except (
        ConfigError,
        DataValidationError,
        FileNotFoundError,
        SyntheticResultsError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
