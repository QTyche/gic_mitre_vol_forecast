"""Build clean-room configs and compare regenerated outputs with frozen facts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from qtyche_qrc.data.semantic_integrity import require_processed_semantic_integrity
from qtyche_qrc.reproducibility.garch_portability import (
    GARCH_PORTABILITY_REPORT,
    GARCH_REGRESSION_METRICS,
    compare_garch_portability,
)
from qtyche_qrc.reproducibility.verification import (
    compare_numeric,
    find_repository_root,
    sha256_path,
)

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-9
SEEDS = (2026, 2027, 2028)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_without_overwrite(path: Path, value: Any) -> None:
    """Create a bootstrap source without changing an existing experiment artifact."""

    if path.is_file():
        if _load_json(path) != value:
            raise FileExistsError(f"refusing to overwrite an existing experiment artifact: {path}")
        return
    _write_json(path, value)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _verify_frozen_processed_inputs(root: Path) -> None:
    reproduction = cast(
        dict[str, Any],
        yaml.safe_load(
            (root / "configs/reproduction/final_financial_qrc.yaml").read_text(encoding="utf-8")
        ),
    )
    study = cast(dict[str, Any], reproduction["study"])
    reference = root / str(study["processed_semantic_reference"])
    require_processed_semantic_integrity(
        root / "data/processed/public_market",
        data_config_path=root / str(study["data_config"]),
        reference_path=reference,
        expected_reference_sha256=str(study["processed_semantic_reference_sha256"]),
    )


def _facts(root: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_json(root / "paper_assets/final_results_manifest.json")
    return {str(record["fact_id"]): dict(record) for record in manifest["facts"]}


def _comparison(
    *,
    fact_id: str,
    actual: float,
    facts: dict[str, dict[str, Any]],
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    tolerance_contract: str = "global_regenerated_metrics",
) -> dict[str, Any]:
    expected = float(facts[fact_id]["exact_value"])
    return {
        "fact_id": fact_id,
        "tolerance_contract": tolerance_contract,
        **compare_numeric(
            actual,
            expected,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
    }


def _find_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _latest_manifest(
    root: Path,
    search_root: Path,
    *,
    model_type: str,
    task: str | None = None,
    seed: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in search_root.rglob("manifest.json"):
        manifest = _load_json(path)
        if manifest.get("status") != "success" or manifest.get("is_synthetic"):
            continue
        if manifest.get("model_type") != model_type:
            continue
        if task is not None and manifest.get("task") != task:
            continue
        manifest_seed = manifest.get("reservoir_seed", manifest.get("seed"))
        if seed is not None and (manifest_seed is None or int(manifest_seed) != seed):
            continue
        matches.append((path.parent, manifest))
    if not matches:
        raise FileNotFoundError(
            f"no completed non-synthetic {model_type}/{task}/{seed} run under "
            f"{_relative(search_root, root)}"
        )
    return sorted(matches, key=lambda item: item[0].as_posix())[-1]


def _mnist_run(
    root: Path,
    *,
    model_name: str,
    condition: str,
    seed: int | None,
) -> Path:
    matches: list[Path] = []
    for manifest_path in (root / "results/qrc_mnist/runs/full").glob("*/manifest.json"):
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") == "success"
            and manifest.get("mode") == "full"
            and manifest.get("is_synthetic") is False
            and manifest.get("model_name") == model_name
            and manifest.get("condition") == condition
            and manifest.get("reservoir_seed") == seed
        ):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one full genuine-MNIST run for {model_name}/{condition}/{seed}")
    return matches[0]


def compare_reproduction(root: Path, *, mode: str, output: Path) -> dict[str, Any]:
    """Compare headline/full regenerated metrics to the frozen facts manifest."""

    destination = output if output.is_absolute() else root / output
    facts = _facts(root)
    comparisons: list[dict[str, Any]] = []
    aggregate = _load_json(
        root / "results/final_financial_qrc/tables/final_qrc_exact_aggregate.json"
    )["rows"]
    classifier = _find_row(aggregate, task="regime_classification", split="test")
    regressor = _find_row(aggregate, task="rv_regression", split="test")
    for metric in ("macro_f1", "balanced_accuracy", "transition_pr_auc"):
        comparisons.append(
            _comparison(
                fact_id=f"financial.test.qrc_mean.{metric}",
                actual=float(classifier[f"{metric}_mean"]),
                facts=facts,
            )
        )
    for metric in ("qlike", "rmse", "mae", "correlation"):
        comparisons.append(
            _comparison(
                fact_id=f"financial.test.qrc_mean.{metric}",
                actual=float(regressor[f"{metric}_mean"]),
                facts=facts,
            )
        )
    garch_dirs = sorted(
        path.parent
        for path in (root / "results/garch_baseline/runs").glob("*/manifest.json")
        if _load_json(path).get("mode") == "full"
    )
    if not garch_dirs:
        raise FileNotFoundError("full GARCH result is missing")
    garch_metrics = _load_json(garch_dirs[-1] / "test_metrics.json")
    reproduction = cast(
        dict[str, Any],
        yaml.safe_load((root / "configs/phase3_reproduction.yaml").read_text(encoding="utf-8")),
    )
    garch_contract = cast(
        dict[str, Any],
        reproduction["tolerances"]["garch_regression_portability"],
    )
    reference_path = root / str(garch_contract["reference"])
    portability_report = compare_garch_portability(
        root,
        experiment_dir=garch_dirs[-1],
        reference_path=reference_path,
        expected_reference_sha256=str(garch_contract["reference_sha256"]),
        output_path=destination.parent / GARCH_PORTABILITY_REPORT,
    )
    special_metric_contract = cast(
        dict[str, Any],
        portability_report["comparison_contract"]["regression_metric_tolerance"],
    )
    for metric in ("macro_f1", "balanced_accuracy", "qlike", "rmse", "mae"):
        use_garch_contract = metric in GARCH_REGRESSION_METRICS and portability_report["passed"]
        comparisons.append(
            _comparison(
                fact_id=f"financial.test.garch_1_1.{metric}",
                actual=float(garch_metrics[metric]),
                facts=facts,
                absolute_tolerance=(
                    float(special_metric_contract["absolute"])
                    if use_garch_contract
                    else ABSOLUTE_TOLERANCE
                ),
                relative_tolerance=(
                    float(special_metric_contract["relative"])
                    if use_garch_contract
                    else RELATIVE_TOLERANCE
                ),
                tolerance_contract=(
                    str(portability_report["contract_id"])
                    if use_garch_contract
                    else "global_regenerated_metrics"
                ),
            )
        )
    mnist_summary = _load_json(root / "results/qrc_mnist/run_summary.json")
    if mnist_summary.get("status") != "success" or mnist_summary.get("mode") not in {
        "smoke",
        "full",
    }:
        raise ValueError("MNIST run summary is not a successful genuine benchmark")
    mnist_dataset = _load_json(root / "results/qrc_mnist/dataset/dataset_manifest.json")
    genuine_mnist = (
        mnist_dataset.get("synthetic_data") is False
        and mnist_dataset.get("official_partitions_preserved") is True
    )
    if not genuine_mnist:
        raise ValueError("MNIST reproduction used synthetic or substituted data")
    if mode == "full":
        mnist_aggregate = _load_json(
            root / "results/qrc_mnist/tables/mnist_qrc_exact_aggregate.json"
        )["rows"]
        mnist_test = _find_row(mnist_aggregate, split="test")
        metric_columns = {
            "accuracy": "accuracy",
            "macro_f1": "macro_f1",
            "balanced_accuracy": "balanced_accuracy",
            "macro_roc_auc": "ovr_macro_roc_auc",
        }
        for metric, column in metric_columns.items():
            comparisons.append(
                _comparison(
                    fact_id=f"mnist.test.exact_qrc_mean.{metric}",
                    actual=float(mnist_test[f"{column}_mean"]),
                    facts=facts,
                )
            )
    failed = [record for record in comparisons if not record["passed"]]
    garch_portability_failed = not portability_report["passed"]
    key_outputs = [
        root / "results/final_financial_qrc/tables/final_qrc_exact_aggregate.json",
        garch_dirs[-1] / "test_metrics.json",
        root / "results/qrc_mnist/run_summary.json",
    ]
    if mode == "full":
        key_outputs.extend(
            (
                root / "results/statistical_validation/tables/"
                "financial_classification_architecture_level.json",
                root / "results/statistical_validation/tables/"
                "financial_regression_architecture_level.json",
                root / "results/statistical_validation/tables/mnist_architecture_level.json",
            )
        )
    checksums = {_relative(path, root): sha256_path(path) for path in key_outputs if path.is_file()}
    report = {
        "schema_version": 1,
        "status": "pass" if not failed and not garch_portability_failed else "fail",
        "mode": mode,
        "data_snapshot_id": "yahoo_chart_20100101_20251231_v1",
        "mnist_genuine": genuine_mnist,
        "mnist_mode": mnist_summary["mode"],
        "comparisons": comparisons,
        "failed_comparison_count": len(failed) + int(garch_portability_failed),
        "tolerances": {
            "global": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
                "unchanged": True,
                "justification": (
                    "The original QRC, GARCH-classification, MNIST, and publication "
                    "comparison contract remains unchanged; tracked files and dataset "
                    "identities remain checksum-exact."
                ),
            },
            "garch_regression_portability": {
                **special_metric_contract,
                "activated": portability_report["passed"],
                "contract_id": portability_report["contract_id"],
                "evidence_report": _relative(
                    destination.parent / GARCH_PORTABILITY_REPORT,
                    root,
                ),
                "scope": "GARCH test QLIKE, RMSE, and MAE only",
            },
        },
        "garch_portability": {
            "status": portability_report["status"],
            "passed": portability_report["passed"],
            "report": _relative(
                destination.parent / GARCH_PORTABILITY_REPORT,
                root,
            ),
        },
        "output_checksums": checksums,
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
    }
    _write_json(destination, report)
    if failed or garch_portability_failed:
        raise ValueError(
            f"{len(failed)} regenerated headline values exceeded tolerance; "
            f"GARCH portability evidence passed={not garch_portability_failed}"
        )
    return report


def _source_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": _relative(path, root), "sha256": sha256_path(path)}


def _update_source(record: dict[str, Any], path: Path, root: Path) -> None:
    record.update(_source_record(path, root))


def _project_setting(config_path: Path, root: Path) -> str:
    return Path(os.path.relpath(root, config_path.parent)).as_posix()


def prepare_statistical_config(root: Path, output: Path) -> Path:
    """Retarget the frozen Stage 2A controls to regenerated prediction files."""

    raw = cast(
        dict[str, Any],
        yaml.safe_load((root / "configs/statistical_validation.yaml").read_text(encoding="utf-8")),
    )
    raw["study"]["project_root"] = _project_setting(output, root)
    architecture = root / "results/final_financial_qrc/final_architecture_manifest.json"
    _update_source(raw["study"]["financial_architecture_manifest"], architecture, root)
    final_root = root / "results/final_financial_qrc/exact/runs"
    for seed in SEEDS:
        for key, model_type, task in (
            ("classifier", "qrc_classifier", "regime_classification"),
            ("regressor", "qrc_regressor", "rv_regression"),
        ):
            directory, _manifest = _latest_manifest(
                root, final_root, model_type=model_type, task=task, seed=seed
            )
            _update_source(
                raw["financial"]["qrc"][seed][key],
                directory / "test_predictions.csv",
                root,
            )
    public_root = root / "results/public_market"
    for source_id, model_type in (
        ("logistic_regression", "logistic_regression"),
        ("esn_classifier", "esn_classifier"),
        ("regime_persistence", "regime_persistence"),
        ("majority_classifier", "majority_classifier"),
    ):
        directory, _ = _latest_manifest(
            root, public_root, model_type=model_type, task="regime_classification"
        )
        _update_source(
            raw["financial"]["classification_baselines"][source_id],
            directory / "test_predictions.csv",
            root,
        )
    for source_id, model_type in (
        ("esn_regressor", "esn_regressor"),
        ("rv_persistence", "rv_persistence"),
    ):
        directory, _ = _latest_manifest(
            root, public_root, model_type=model_type, task="rv_regression"
        )
        _update_source(
            raw["financial"]["regression_baselines"][source_id],
            directory / "test_predictions.csv",
            root,
        )
    garch_directory, _ = _latest_manifest(
        root,
        root / "results/garch_baseline/runs",
        model_type="gaussian_garch_1_1",
        task="rv_regression",
    )
    _update_source(
        raw["financial"]["regression_baselines"]["garch_1_1"],
        garch_directory / "test_predictions.csv",
        root,
    )
    diagnostic = root / "results/diagnostics/esn_regression/test_predictions.csv"
    _update_source(
        raw["financial"]["regression_baselines"]["esn_log_variance_diagnostic"],
        diagnostic,
        root,
    )
    for seed in SEEDS:
        directory = _mnist_run(root, model_name="qrc_exact", condition="analytic", seed=seed)
        _update_source(raw["mnist"]["qrc"][seed], directory / "test_predictions.csv", root)
    for source_id, model_name, condition in (
        ("flattened_logistic", "logistic_baseline", "flattened_28_by_5"),
        ("esn", "esn_baseline", "size_controlled_32"),
    ):
        directory = _mnist_run(
            root,
            model_name=model_name,
            condition=condition,
            seed=None if source_id == "flattened_logistic" else 2026,
        )
        _update_source(
            raw["mnist"]["baselines"][source_id],
            directory / "test_predictions.csv",
            root,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def prepare_diagnostics_config(root: Path, output: Path) -> Path:
    """Retarget Stage 2B controls to the regenerated aligned outputs."""

    _verify_frozen_processed_inputs(root)
    raw = cast(
        dict[str, Any],
        yaml.safe_load((root / "configs/benchmark_diagnostics.yaml").read_text(encoding="utf-8")),
    )
    raw["study"]["project_root"] = _project_setting(output, root)
    for source_id, filename in (
        ("manifest", "data_manifest.json"),
        ("training", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
        ("regime_thresholds", "regime_thresholds.json"),
    ):
        _update_source(
            raw["financial_data"][source_id],
            root / "data/processed/public_market" / filename,
            root,
        )
    _update_source(
        raw["study"]["financial_architecture_manifest"],
        root / "results/final_financial_qrc/final_architecture_manifest.json",
        root,
    )
    _update_source(
        raw["study"]["qrc_exact_table"],
        root / "results/final_financial_qrc/tables/final_qrc_exact_per_run.json",
        root,
    )
    final_root = root / "results/final_financial_qrc/exact/runs"
    public_root = root / "results/public_market"
    financial_directories: dict[str, Path] = {}
    for source_id, model_type, task in (
        ("rv_persistence", "rv_persistence", "rv_regression"),
        ("esn_regressor", "esn_regressor", "rv_regression"),
        ("majority_classifier", "majority_classifier", "regime_classification"),
        ("regime_persistence", "regime_persistence", "regime_classification"),
        ("logistic_regression", "logistic_regression", "regime_classification"),
        ("esn_classifier", "esn_classifier", "regime_classification"),
    ):
        financial_directories[source_id] = _latest_manifest(
            root, public_root, model_type=model_type, task=task
        )[0]
    financial_directories["garch_1_1"] = _latest_manifest(
        root,
        root / "results/garch_baseline/runs",
        model_type="gaussian_garch_1_1",
        task="rv_regression",
    )[0]
    for seed in SEEDS:
        financial_directories[f"qrc_regression_{seed}"] = _latest_manifest(
            root,
            final_root,
            model_type="qrc_regressor",
            task="rv_regression",
            seed=seed,
        )[0]
        financial_directories[f"qrc_classification_{seed}"] = _latest_manifest(
            root,
            final_root,
            model_type="qrc_classifier",
            task="regime_classification",
            seed=seed,
        )[0]
    for source_id in ("rv_persistence", "esn_regressor", "garch_1_1"):
        for split in ("validation", "test"):
            _update_source(
                raw["financial_regression"][source_id][split],
                financial_directories[source_id] / f"{split}_predictions.csv",
                root,
            )
    for seed in SEEDS:
        for split in ("validation", "test"):
            _update_source(
                raw["financial_regression"][f"qrc_{seed}"][split],
                financial_directories[f"qrc_regression_{seed}"] / f"{split}_predictions.csv",
                root,
            )
    for source_id in (
        "majority_classifier",
        "regime_persistence",
        "logistic_regression",
        "esn_classifier",
    ):
        for split in ("validation", "test"):
            _update_source(
                raw["financial_classification"][source_id][split],
                financial_directories[source_id] / f"{split}_predictions.csv",
                root,
            )
    for seed in SEEDS:
        for split in ("validation", "test"):
            _update_source(
                raw["financial_classification"][f"qrc_{seed}"][split],
                financial_directories[f"qrc_classification_{seed}"] / f"{split}_predictions.csv",
                root,
            )
        _update_source(
            raw["qrc_numerical"][seed],
            financial_directories[f"qrc_classification_{seed}"] / "qrc_numerical_diagnostics.json",
            root,
        )
    _update_source(
        raw["mnist"]["selected_indices"],
        root / "results/qrc_mnist/dataset/selected_indices.json",
        root,
    )
    mnist_specs: dict[str, tuple[str, str, int | None]] = {
        "flattened_logistic": ("logistic_baseline", "flattened_28_by_5", None),
        "esn": ("esn_baseline", "size_controlled_32", 2026),
        **{f"qrc_{seed}": ("qrc_exact", "analytic", seed) for seed in SEEDS},
    }
    for source_id, (model_name, condition, mnist_seed) in mnist_specs.items():
        directory = _mnist_run(
            root,
            model_name=model_name,
            condition=condition,
            seed=mnist_seed,
        )
        _update_source(
            raw["mnist"]["models"][source_id],
            directory / "test_predictions.csv",
            root,
        )
    robustness_specs = {
        "analytic": ("qrc_exact", "analytic"),
        "shots_2048": ("qrc_robustness", "shots_2048"),
        "depolarizing_0_01": ("qrc_robustness", "depolarizing_0_01"),
        "measurement_flip_0_02": ("qrc_robustness", "measurement_flip_0_02"),
    }
    for source_id, (model_name, condition) in robustness_specs.items():
        directory = _mnist_run(root, model_name=model_name, condition=condition, seed=2026)
        _update_source(
            raw["mnist"]["robustness"][source_id],
            directory / "test_predictions.csv",
            root,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def prepare_full_configs(root: Path, output_dir: Path) -> list[Path]:
    """Write checksum-pinned Stage 2A/2B configs for current regenerated sources."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        prepare_statistical_config(root, output_dir / "statistical_validation.yaml"),
        prepare_diagnostics_config(root, output_dir / "benchmark_diagnostics.yaml"),
    ]


def _materialize_selection_sources(root: Path) -> None:
    facts = _facts(root)
    scaling_rows: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    for n_qubits in range(2, 7):
        for metric, task in (
            ("macro_f1", "regime_classification"),
            ("transition_pr_auc", "regime_classification"),
            ("qlike", "rv_regression"),
            ("state_generation_seconds", "regime_classification"),
        ):
            scaling_rows.append(
                {
                    "n_qubits": n_qubits,
                    "split": "validation",
                    "task": task,
                    "metric": metric,
                    "mean": facts[f"selection.qubits.{n_qubits}.{metric}"]["exact_value"],
                }
            )
        resources.append(
            {
                "n_qubits": n_qubits,
                "raw_feature_dimension": facts[f"selection.qubits.{n_qubits}.feature_dimension"][
                    "exact_value"
                ],
            }
        )
    virtual_rows: list[dict[str, Any]] = []
    for virtual_nodes in (1, 2, 4, 8):
        virtual_rows.append(
            {
                "virtual_nodes": virtual_nodes,
                "validation_macro_f1_mean": facts[
                    f"selection.virtual_nodes.{virtual_nodes}.macro_f1"
                ]["exact_value"],
                "validation_transition_pr_auc_mean": facts[
                    f"selection.virtual_nodes.{virtual_nodes}.transition_pr_auc"
                ]["exact_value"],
                "validation_qlike_mean": facts[f"selection.virtual_nodes.{virtual_nodes}.qlike"][
                    "exact_value"
                ],
                "condition_number_mean": facts[
                    f"selection.virtual_nodes.{virtual_nodes}.condition_number"
                ]["exact_value"],
                "raw_feature_dimension": facts[
                    f"selection.virtual_nodes.{virtual_nodes}.feature_dimension"
                ]["exact_value"],
            }
        )
    _write_json_without_overwrite(
        root / "results/qrc_qubit_scaling/tables/qrc_qubit_scaling_aggregate.json",
        {"schema_version": 1, "rows": scaling_rows},
    )
    _write_json_without_overwrite(
        root / "results/qrc_qubit_scaling/tables/qrc_qubit_scaling_resources.json",
        {"schema_version": 1, "rows": resources},
    )
    _write_json_without_overwrite(
        root / "results/qrc_encoding_density/tables/"
        "qrc_encoding_density_validation_candidates.json",
        {"schema_version": 1, "rows": virtual_rows},
    )


def prepare_publication_config(
    root: Path,
    *,
    output: Path,
    publication_output: Path,
) -> Path:
    """Retarget Stage 2C to regenerated sources and an isolated evidence tree."""

    _materialize_selection_sources(root)
    raw = cast(
        dict[str, Any],
        yaml.safe_load((root / "configs/publication_assets.yaml").read_text(encoding="utf-8")),
    )
    raw["study"]["project_root"] = _project_setting(output, root)
    raw["study"]["tracked_output_root"] = _relative(publication_output, root)
    raw["study"]["intermediate_output_root"] = _relative(
        publication_output.parent / "publication_intermediate", root
    )
    raw["study"]["frozen_source_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    raw["study"]["generation_command"] = (
        "python scripts/freeze_publication_assets.py --config " + _relative(output, root)
    )
    for record in raw["sources"].values():
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"regenerated publication source is missing: {path}")
        record["sha256"] = sha256_path(path)
    for formats in raw["appendix_assets"].values():
        for record in formats.values():
            path = root / str(record["path"])
            if not path.is_file():
                raise FileNotFoundError(f"regenerated appendix source is missing: {path}")
            record["sha256"] = sha256_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--mode", choices=("headline", "full"), required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    full_parser = subparsers.add_parser("prepare-full")
    full_parser.add_argument("--output-dir", type=Path, required=True)
    publication = subparsers.add_parser("prepare-publication")
    publication.add_argument("--output-dir", type=Path, required=True)
    publication.add_argument("--publication-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_repository_root(Path.cwd())
    try:
        if args.command == "compare":
            compare_reproduction(root, mode=args.mode, output=args.output)
        elif args.command == "prepare-full":
            prepare_full_configs(
                root,
                args.output_dir if args.output_dir.is_absolute() else root / args.output_dir,
            )
        else:
            output_dir = (
                args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
            )
            publication_output = (
                args.publication_output
                if args.publication_output.is_absolute()
                else root / args.publication_output
            )
            prepare_publication_config(
                root,
                output=output_dir / "publication_assets.yaml",
                publication_output=publication_output,
            )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
