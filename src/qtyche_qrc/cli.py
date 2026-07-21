"""Command-line entry points for validation and reproducible experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from qtyche_qrc.config import ConfigError, load_config
from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.description import describe_public_data
from qtyche_qrc.data.download import download_public_snapshot
from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots, fixture_summary
from qtyche_qrc.data.pipeline import inspect_processed_targets, prepare_data
from qtyche_qrc.data.validation import DataValidationError, audit_processed_data
from qtyche_qrc.experiments.compare import compare_baselines
from qtyche_qrc.experiments.esn_regression_diagnostics import (
    run_esn_regression_diagnostics,
)
from qtyche_qrc.experiments.manifest import create_manifest
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.public_compare import compare_public_baselines
from qtyche_qrc.experiments.qrc_capacity import characterize_qrc
from qtyche_qrc.experiments.qrc_run import (
    compare_qrc_seeds,
    generate_qrc_features,
    inspect_qrc_experiment,
    run_qrc_experiment,
)
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

    public_download = subparsers.add_parser(
        "download-public-data",
        help="download or verify an immutable versioned public-market snapshot",
    )
    public_download.add_argument("--config", required=True)
    public_download.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing snapshot after redownloading all instruments",
    )

    prepare = subparsers.add_parser(
        "prepare-data",
        help="prepare causal features, targets, purged splits, and manifests",
    )
    prepare.add_argument("--config", required=True, help="path to the data YAML configuration")
    prepare.add_argument(
        "--cached",
        action="store_true",
        help="require the configured public snapshot and perform no network requests",
    )

    audit = subparsers.add_parser("audit-data", help="audit persisted processed data")
    audit.add_argument("--processed-dir", required=True, help="processed data directory")

    inspect = subparsers.add_parser(
        "inspect-targets",
        help="show regime and transition target distributions",
    )
    inspect.add_argument("--processed-dir", required=True, help="processed data directory")

    describe = subparsers.add_parser(
        "describe-data", help="write public-market descriptive tables and figures"
    )
    describe.add_argument("--processed-dir", required=True)
    describe.add_argument("--output-dir")

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

    compare_public = subparsers.add_parser(
        "compare-public-baselines",
        help="write public-market comparison and diagnostic tables",
    )
    compare_public.add_argument("--results-dir", required=True)
    compare_public.add_argument("--output-dir", required=True)

    inspect_experiment_parser = subparsers.add_parser(
        "inspect-experiment", help="inspect experiment provenance and metrics"
    )
    inspect_experiment_parser.add_argument("--experiment-dir", required=True)

    diagnose = subparsers.add_parser(
        "diagnose-esn-regression",
        help="select and audit an ESN variance head using validation data only",
    )
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--output-dir", required=True)

    characterize = subparsers.add_parser(
        "characterize-qrc",
        help="run deterministic synthetic QRC memory and nonlinearity characterization",
    )
    characterize.add_argument("--config", required=True)

    generate_qrc = subparsers.add_parser(
        "generate-qrc-features", help="generate or verify checksum-keyed QRC features"
    )
    generate_qrc.add_argument("--config", required=True)
    generate_qrc.add_argument("--reservoir-seed", type=int)
    generate_qrc.add_argument("--allow-synthetic-results", action="store_true")

    train_qrc = subparsers.add_parser(
        "train-qrc", help="validation-select a QRC ridge head and evaluate its frozen test"
    )
    train_qrc.add_argument("--config", required=True)
    train_qrc.add_argument("--reservoir-seed", type=int)
    train_qrc.add_argument("--allow-synthetic-results", action="store_true")

    inspect_qrc = subparsers.add_parser(
        "inspect-qrc", help="inspect a completed QRC experiment and numerical diagnostics"
    )
    inspect_qrc.add_argument("--experiment-dir", required=True)

    compare_qrc = subparsers.add_parser(
        "compare-qrc-seeds", help="aggregate public QRC pilot metrics across reservoir seeds"
    )
    compare_qrc.add_argument("--results-dir", required=True)
    compare_qrc.add_argument("--output-dir", required=True)

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
        if args.command == "download-public-data":
            data_config = load_data_config(args.config)
            manifest = download_public_snapshot(data_config, force=args.force)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-data":
            data_config = load_data_config(args.config)
            if args.cached:
                if data_config.data_source_type != "public_market":
                    raise ValueError("--cached is only valid for public-market data")
                data_config = replace(data_config, mode="cached_csv")
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
        if args.command == "describe-data":
            output_dir = Path(args.output_dir).resolve() if args.output_dir else None
            summary = describe_public_data(Path(args.processed_dir), output_dir)
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
        if args.command == "compare-public-baselines":
            outputs = compare_public_baselines(
                Path(args.results_dir).resolve(), Path(args.output_dir).resolve()
            )
            print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
            return 0
        if args.command == "inspect-experiment":
            summary = inspect_experiment(Path(args.experiment_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "diagnose-esn-regression":
            summary = run_esn_regression_diagnostics(
                Path(args.config), Path(args.output_dir).resolve()
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "characterize-qrc":
            output = characterize_qrc(Path(args.config))
            print(output)
            return 0
        if args.command == "generate-qrc-features":
            bundle = generate_qrc_features(
                Path(args.config),
                allow_synthetic_results=args.allow_synthetic_results,
                reservoir_seed=args.reservoir_seed,
            )
            print(
                json.dumps(
                    {
                        "cache_directory": str(bundle.cache_dir),
                        "cache_hit": bundle.cache_hit,
                        "split_shapes": bundle.metadata["split_shapes"],
                        "cache_key_checksum": bundle.metadata["cache_key_checksum"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "train-qrc":
            output = run_qrc_experiment(
                Path(args.config),
                allow_synthetic_results=args.allow_synthetic_results,
                reservoir_seed=args.reservoir_seed,
            )
            print(output)
            return 0
        if args.command == "inspect-qrc":
            summary = inspect_qrc_experiment(Path(args.experiment_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "compare-qrc-seeds":
            outputs = compare_qrc_seeds(
                Path(args.results_dir).resolve(), Path(args.output_dir).resolve()
            )
            print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
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
