"""Compile immutable Phase 3 paper assets from checksum-pinned frozen outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qtyche_qrc.publication.figures import (
    architecture_selection_figure,
    financial_comparison_figure,
    financial_robustness_figure,
    mnist_benchmark_figure,
)
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "phase3_publication_assets_v1"
TABLE_NAMES = (
    "publication_table_1_experiment_design",
    "publication_table_2_financial_benchmark",
    "publication_table_3_mnist_benchmark",
)
FIGURE_NAMES = (
    "publication_figure_1_architecture_selection",
    "publication_figure_2_financial_comparison",
    "publication_figure_3_financial_robustness",
    "publication_figure_4_mnist_benchmark",
)
CLAIM_STATUSES = {
    "supported",
    "supported with qualification",
    "unsupported",
    "prohibited",
}


@dataclass(frozen=True)
class FrozenSource:
    """A repository-relative, checksum-pinned source artifact."""

    source_id: str
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class PublicationConfig:
    """Validated Stage 2C publication compiler configuration."""

    source: Path
    project_root: Path
    tracked_output_root: Path
    intermediate_output_root: Path
    frozen_source_commit: str
    generation_command: str
    publication_assets_frozen: bool
    no_new_model_execution: bool
    no_test_based_selection: bool
    sources: dict[str, FrozenSource]
    appendix_sources: dict[str, dict[str, FrozenSource]]
    dpi: int
    significance_alpha: float
    missing_display: str
    metric_precision: dict[str, int]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, location: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _source(
    source_id: str,
    value: object,
    project_root: Path,
) -> FrozenSource:
    record = _mapping(value, source_id)
    relative_path = str(record.get("path", ""))
    expected = str(record.get("sha256", ""))
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"{source_id}.path must be a non-empty repository-relative path")
    if len(expected) != 64:
        raise ValueError(f"{source_id}.sha256 must be a SHA-256 digest")
    return FrozenSource(
        source_id=source_id,
        path=project_root / relative_path,
        relative_path=relative_path,
        sha256=expected,
    )


def load_publication_config(path: Path) -> PublicationConfig:
    """Load the source contract without reading any scientific values."""

    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("publication asset config schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    if study.get("id") != STUDY_ID:
        raise ValueError(f"study.id must be {STUDY_ID}")
    project_root = (source.parent / str(study.get("project_root", ".."))).resolve()
    tracked_relative = str(study.get("tracked_output_root", ""))
    intermediate_relative = str(study.get("intermediate_output_root", ""))
    if Path(tracked_relative).is_absolute() or Path(intermediate_relative).is_absolute():
        raise ValueError("publication output roots must be repository-relative")
    source_records = {
        str(source_id): _source(
            f"sources.{source_id}",
            value,
            project_root,
        )
        for source_id, value in _mapping(root.get("sources"), "sources").items()
    }
    appendix: dict[str, dict[str, FrozenSource]] = {}
    for asset_id, formats in _mapping(root.get("appendix_assets"), "appendix_assets").items():
        appendix[str(asset_id)] = {
            str(extension): _source(
                f"appendix_assets.{asset_id}.{extension}",
                value,
                project_root,
            )
            for extension, value in _mapping(
                formats,
                f"appendix_assets.{asset_id}",
            ).items()
        }
    formatting = _mapping(root.get("formatting"), "formatting")
    precision = {
        str(metric): int(value)
        for metric, value in _mapping(
            formatting.get("metric_precision"),
            "formatting.metric_precision",
        ).items()
    }
    config = PublicationConfig(
        source=source,
        project_root=project_root,
        tracked_output_root=project_root / tracked_relative,
        intermediate_output_root=project_root / intermediate_relative,
        frozen_source_commit=str(study.get("frozen_source_commit", "")),
        generation_command=str(study.get("generation_command", "")),
        publication_assets_frozen=bool(study.get("publication_assets_frozen")),
        no_new_model_execution=bool(study.get("no_new_model_execution")),
        no_test_based_selection=bool(study.get("no_test_based_selection")),
        sources=source_records,
        appendix_sources=appendix,
        dpi=int(formatting.get("dpi", 300)),
        significance_alpha=float(formatting.get("significance_alpha", 0.05)),
        missing_display=str(formatting.get("missing_display", "—")),
        metric_precision=precision,
    )
    if not all(
        (
            config.publication_assets_frozen,
            config.no_new_model_execution,
            config.no_test_based_selection,
        )
    ):
        raise ValueError("publication freeze safeguards must all be true")
    if len(config.frozen_source_commit) != 40:
        raise ValueError("study.frozen_source_commit must be a full Git commit")
    if config.tracked_output_root == config.intermediate_output_root:
        raise ValueError("tracked and intermediate output roots must be isolated")
    return config


def _all_sources(config: PublicationConfig) -> Iterable[FrozenSource]:
    yield from config.sources.values()
    for formats in config.appendix_sources.values():
        yield from formats.values()


def verify_publication_sources(config: PublicationConfig) -> list[dict[str, Any]]:
    """Verify all frozen input identities and reject fixture financial data."""

    records: list[dict[str, Any]] = []
    seen_paths: dict[Path, str] = {}
    for source in _all_sources(config):
        if not source.path.is_file():
            raise FileNotFoundError(f"frozen publication source is missing: {source.path}")
        actual = sha256_path(source.path)
        if actual != source.sha256:
            raise ValueError(
                f"checksum mismatch for {source.relative_path}: "
                f"expected {source.sha256}, got {actual}"
            )
        previous = seen_paths.get(source.path)
        if previous is not None and previous != source.sha256:
            raise ValueError(f"conflicting checksums configured for {source.relative_path}")
        seen_paths[source.path] = source.sha256
        records.append(
            {
                "source_id": source.source_id,
                "path": source.relative_path,
                "sha256": source.sha256,
                "bytes": source.path.stat().st_size,
            }
        )
    data_manifest = _load_json(config.sources["public_data_manifest"].path)
    if bool(data_manifest.get("is_synthetic")) or data_manifest.get("data_source_type") != (
        "public_market"
    ):
        raise ValueError("publication assets require frozen non-synthetic public-market data")
    return sorted(records, key=lambda row: (str(row["path"]), str(row["source_id"])))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _rows(config: PublicationConfig, source_id: str) -> list[dict[str, Any]]:
    payload = _load_json(config.sources[source_id].path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source_id} must contain an object-valued rows list")
    return [dict(row) for row in rows]


def _find_row(
    rows: Sequence[dict[str, Any]],
    **criteria: object,
) -> dict[str, Any]:
    matches = [
        row for row in rows if all(row.get(key) == expected for key, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row matching {criteria}, found {len(matches)}")
    return matches[0]


class FactRegistry:
    """Collect unique trace records for every number selected for publication."""

    def __init__(self, config: PublicationConfig) -> None:
        self._config = config
        self._facts: dict[str, dict[str, Any]] = {}

    def add(
        self,
        fact_id: str,
        *,
        exact_value: Any,
        display_value: str,
        source_id: str,
        locator: str,
        split: str,
        scope: str,
        significance_adjusted: bool,
        preference: str,
        usages: Sequence[str],
    ) -> str:
        source = self._config.sources[source_id]
        record = {
            "fact_id": fact_id,
            "exact_value": exact_value,
            "display_rounded_value": display_value,
            "source_artifact_path": source.relative_path,
            "source_column_or_key": locator,
            "source_artifact_sha256": source.sha256,
            "split": split,
            "value_scope": scope,
            "significance_adjusted": significance_adjusted,
            "metric_preference": preference,
            "usages": sorted(set(usages)),
        }
        existing = self._facts.get(fact_id)
        if existing is not None and existing != record:
            raise ValueError(f"fact {fact_id} was registered inconsistently")
        self._facts[fact_id] = record
        return fact_id

    def records(self) -> list[dict[str, Any]]:
        return [self._facts[key] for key in sorted(self._facts)]

    def get(self, fact_id: str) -> dict[str, Any]:
        return self._facts[fact_id]


def _display(config: PublicationConfig, metric: str, value: float) -> str:
    precision = config.metric_precision.get(metric, 3)
    return f"{float(value):.{precision}f}"


def _slug(value: str) -> str:
    characters = [character.lower() if character.isalnum() else "_" for character in value]
    return "_".join(part for part in "".join(characters).split("_") if part)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"$\pm$",
        "↑": r"$\uparrow$",
        "↓": r"$\downarrow$",
        "—": r"---",
    }
    return "".join(replacements.get(character, character) for character in value)


def _write_table_bundle(
    *,
    directory: Path,
    name: str,
    title: str,
    columns: Sequence[tuple[str, str]],
    rows: list[dict[str, Any]],
    footnotes: Sequence[str],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": directory / f"{name}.json",
        "csv": directory / f"{name}.csv",
        "tex": directory / f"{name}.tex",
        "md": directory / f"{name}.md",
    }
    payload = {
        "schema_version": 1,
        "title": title,
        "columns": [{"key": key, "label": label} for key, label in columns],
        "rows": rows,
        "footnotes": list(footnotes),
    }
    _write_json(paths["json"], payload)
    fieldnames = [key for key, _ in columns] + ["trace_fact_ids"]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    markdown_lines = [f"# {title}", "", header, divider]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(str(row.get(f"{key}_display", row.get(key, ""))) for key, _ in columns)
            + " |"
        )
    if footnotes:
        markdown_lines.extend(["", *[f"- {note}" for note in footnotes]])
    paths["md"].write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    align = "l" + "r" * (len(columns) - 1)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_latex_escape(title)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_latex_escape(label) for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(
            " & ".join(
                _latex_escape(str(row.get(f"{key}_display", row.get(key, ""))))
                for key, _ in columns
            )
            + r" \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    for note in footnotes:
        latex_lines.append(rf"\parbox{{0.98\linewidth}}{{\footnotesize {_latex_escape(note)}}}")
    latex_lines.append(r"\end{table*}")
    paths["tex"].write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return paths


def _design_fact(
    registry: FactRegistry,
    fact_id: str,
    exact_value: Any,
    display_value: str,
    *,
    source_id: str,
    locator: str,
    split: str = "not_applicable",
    usages: Sequence[str] = ("publication_table_1",),
) -> str:
    return registry.add(
        fact_id,
        exact_value=exact_value,
        display_value=display_value,
        source_id=source_id,
        locator=locator,
        split=split,
        scope="design",
        significance_adjusted=False,
        preference="not_applicable",
        usages=usages,
    )


def _build_design_table(
    config: PublicationConfig,
    registry: FactRegistry,
) -> list[dict[str, Any]]:
    data = _load_json(config.sources["public_data_manifest"].path)
    thresholds = _load_json(config.sources["regime_thresholds"].path)
    architecture = _load_json(config.sources["financial_architecture"].path)
    mnist_dataset = _load_json(config.sources["mnist_dataset"].path)
    mnist_summary = _load_json(config.sources["mnist_summary"].path)

    canonical_range = data["date_ranges"]["canonical_source"]
    split_sizes = data["split_row_counts"]
    arch = architecture["architecture"]
    mnist_arch = mnist_summary["qrc_architecture"]
    mnist_sizes = mnist_dataset["split_sizes"]
    rows: list[dict[str, Any]] = []

    def append(section: str, item: str, value: str, fact_ids: Sequence[str]) -> None:
        rows.append(
            {
                "section": section,
                "item": item,
                "value": value,
                "value_display": value,
                "trace_fact_ids": list(fact_ids),
            }
        )

    date_start = _design_fact(
        registry,
        "financial.data.start_date",
        canonical_range["start"],
        str(canonical_range["start"]),
        source_id="public_data_manifest",
        locator="date_ranges.canonical_source.start",
    )
    date_end = _design_fact(
        registry,
        "financial.data.end_date",
        canonical_range["end"],
        str(canonical_range["end"]),
        source_id="public_data_manifest",
        locator="date_ranges.canonical_source.end",
    )
    snapshot = _design_fact(
        registry,
        "financial.data.snapshot_id",
        data["source_snapshot_id"],
        str(data["source_snapshot_id"]),
        source_id="public_data_manifest",
        locator="source_snapshot_id",
    )
    append(
        "Financial application",
        "Dataset and date range",
        (
            f"SPY, QQQ and VIX public-market snapshot "
            f"({canonical_range['start']} to {canonical_range['end']})"
        ),
        [date_start, date_end, snapshot],
    )
    size_facts = [
        _design_fact(
            registry,
            f"financial.data.{split}_rows",
            int(split_sizes[split]),
            str(int(split_sizes[split])),
            source_id="public_data_manifest",
            locator=f"split_row_counts.{split}",
            split=split,
        )
        for split in ("train", "validation", "test")
    ]
    append(
        "Financial application",
        "Temporal split sizes",
        (
            f"{int(split_sizes['train'])} train / {int(split_sizes['validation'])} "
            f"validation / {int(split_sizes['test'])} test"
        ),
        size_facts,
    )
    horizon = _design_fact(
        registry,
        "financial.target.horizon_trading_days",
        5,
        "5",
        source_id="public_data_manifest",
        locator="target_names[target_rv_5d] and target_definition_version",
    )
    append("Financial application", "Forecast target", "5-trading-day realised variance", [horizon])
    low_threshold = _design_fact(
        registry,
        "financial.regime.low_medium",
        float(thresholds["low_medium"]),
        f"{float(thresholds['low_medium']):.6f}",
        source_id="regime_thresholds",
        locator="low_medium",
        split="train",
    )
    high_threshold = _design_fact(
        registry,
        "financial.regime.medium_high",
        float(thresholds["medium_high"]),
        f"{float(thresholds['medium_high']):.6f}",
        source_id="regime_thresholds",
        locator="medium_high",
        split="train",
    )
    append(
        "Financial application",
        "Frozen regime thresholds",
        (
            f"low/medium={float(thresholds['low_medium']):.6f}; "
            f"medium/high={float(thresholds['medium_high']):.6f}"
        ),
        [low_threshold, high_threshold],
    )
    architecture_facts = [
        _design_fact(
            registry,
            "financial.architecture.n_qubits",
            int(arch["n_qubits"]),
            str(int(arch["n_qubits"])),
            source_id="financial_architecture",
            locator="architecture.n_qubits",
        ),
        _design_fact(
            registry,
            "financial.architecture.virtual_nodes",
            int(arch["virtual_nodes"]),
            str(int(arch["virtual_nodes"])),
            source_id="financial_architecture",
            locator="architecture.virtual_nodes",
        ),
        _design_fact(
            registry,
            "financial.architecture.feature_dimension",
            int(arch["raw_feature_dimension"]),
            str(int(arch["raw_feature_dimension"])),
            source_id="financial_architecture",
            locator="architecture.raw_feature_dimension",
        ),
        _design_fact(
            registry,
            "financial.architecture.state_policy",
            arch["state_policy"],
            str(arch["state_policy"]),
            source_id="financial_architecture",
            locator="architecture.state_policy",
        ),
    ]
    append(
        "Financial application",
        "Frozen QRC architecture",
        (
            f"{arch['n_qubits']} qubits; V={arch['virtual_nodes']}; "
            f"{arch['state_policy']}; feature dimension {arch['raw_feature_dimension']}"
        ),
        architecture_facts,
    )
    seed_facts = [
        _design_fact(
            registry,
            f"financial.architecture.seed_{seed}",
            seed,
            str(seed),
            source_id="financial_architecture",
            locator="readout.selected_ridge_values[].reservoir_seed",
        )
        for seed in (2026, 2027, 2028)
    ]
    backend = _design_fact(
        registry,
        "financial.architecture.backend",
        arch["backend"],
        str(arch["backend"]),
        source_id="financial_architecture",
        locator="architecture.backend",
    )
    append(
        "Financial application",
        "Reservoir seeds and exact backend",
        f"2026, 2027, 2028; {arch['backend']}",
        [*seed_facts, backend],
    )
    shot_facts = [
        _design_fact(
            registry,
            f"financial.robustness.shots_{shots}",
            shots,
            f"{shots:,}",
            source_id="financial_robustness",
            locator=f"rows[study_type=finite_shot,shot_count={shots}].shot_count",
        )
        for shots in (128, 512, 2048, 8192)
    ]
    depolarizing = _design_fact(
        registry,
        "financial.robustness.depolarizing_probability",
        0.01,
        "0.01",
        source_id="financial_robustness",
        locator="rows[study_type=depolarizing_noise,depolarizing_probability=0.01]",
    )
    measurement = _design_fact(
        registry,
        "financial.robustness.measurement_flip_probability",
        0.02,
        "0.02",
        source_id="financial_robustness",
        locator="rows[study_type=measurement_noise,measurement_bit_flip_probability=0.02]",
    )
    append(
        "Financial application",
        "Controlled robustness conditions",
        "analytic; 128/512/2,048/8,192 shots; depolarising 0.01; bit flip 0.02",
        [*shot_facts, depolarizing, measurement],
    )
    architecture_sha = _design_fact(
        registry,
        "financial.architecture.manifest_sha256",
        config.sources["financial_architecture"].sha256,
        config.sources["financial_architecture"].sha256,
        source_id="financial_architecture",
        locator="$artifact_sha256",
    )
    append(
        "Financial application",
        "Final architecture-manifest SHA-256",
        config.sources["financial_architecture"].sha256,
        [architecture_sha],
    )

    mnist_size_facts = [
        _design_fact(
            registry,
            f"mnist.data.{split}_rows",
            int(mnist_sizes[split]),
            str(int(mnist_sizes[split])),
            source_id="mnist_dataset",
            locator=f"split_sizes.{split}",
            split=split,
        )
        for split in ("train", "validation", "test")
    ]
    per_digit = _design_fact(
        registry,
        "mnist.data.test_rows_per_digit",
        100,
        "100",
        source_id="mnist_dataset",
        locator="class_counts.test.*",
        split="test",
    )
    append(
        "MNIST benchmark",
        "Genuine balanced split sizes",
        (
            f"{mnist_sizes['train']} train / {mnist_sizes['validation']} validation / "
            f"{mnist_sizes['test']} test; 100 test images per digit"
        ),
        [*mnist_size_facts, per_digit],
    )
    mnist_arch_facts = [
        _design_fact(
            registry,
            "mnist.architecture.n_qubits",
            int(mnist_arch["n_qubits"]),
            str(int(mnist_arch["n_qubits"])),
            source_id="mnist_summary",
            locator="qrc_architecture.n_qubits",
        ),
        _design_fact(
            registry,
            "mnist.architecture.row_steps",
            28,
            "28",
            source_id="mnist_summary",
            locator="resources.row_steps_per_image",
        ),
        _design_fact(
            registry,
            "mnist.architecture.column_bands",
            5,
            "5",
            source_id="mnist_preprocessing",
            locator="sequence_shape_per_image[1]",
        ),
        _design_fact(
            registry,
            "mnist.architecture.image_feature_dimension",
            int(mnist_arch["image_feature_dimension"]),
            str(int(mnist_arch["image_feature_dimension"])),
            source_id="mnist_summary",
            locator="qrc_architecture.image_feature_dimension",
        ),
    ]
    append(
        "MNIST benchmark",
        "QRC representation",
        (
            f"{mnist_arch['n_qubits']} qubits; 28-row, five-band sequence; "
            f"image feature dimension {mnist_arch['image_feature_dimension']}"
        ),
        mnist_arch_facts,
    )
    mnist_seed_facts = [
        _design_fact(
            registry,
            f"mnist.architecture.seed_{seed}",
            seed,
            str(seed),
            source_id="mnist_qrc_aggregate",
            locator="rows[condition=analytic].reservoir_seeds",
        )
        for seed in (2026, 2027, 2028)
    ]
    mnist_backend = _design_fact(
        registry,
        "mnist.architecture.backend",
        mnist_arch["backend"],
        str(mnist_arch["backend"]),
        source_id="mnist_summary",
        locator="qrc_architecture.backend",
    )
    append(
        "MNIST benchmark",
        "Reservoir seeds and backend",
        f"2026, 2027, 2028; {mnist_arch['backend']}",
        [*mnist_seed_facts, mnist_backend],
    )
    subset = _design_fact(
        registry,
        "mnist.data.subset_checksum",
        mnist_dataset["subset_checksum"],
        str(mnist_dataset["subset_checksum"]),
        source_id="mnist_dataset",
        locator="subset_checksum",
    )
    append(
        "MNIST benchmark",
        "Balanced-subset checksum",
        str(mnist_dataset["subset_checksum"]),
        [subset],
    )
    condition_facts = [
        _design_fact(
            registry,
            f"mnist.robustness.{condition}",
            value,
            display,
            source_id="mnist_robustness",
            locator=f"rows[condition={condition},split=test]",
        )
        for condition, value, display in (
            ("analytic", "analytic", "analytic"),
            ("shots_2048", 2048, "2,048 shots"),
            ("depolarizing_0_01", 0.01, "depolarising 0.01"),
            ("measurement_flip_0_02", 0.02, "measurement flip 0.02"),
        )
    ]
    append(
        "MNIST benchmark",
        "Analytic and robustness conditions",
        "analytic; 2,048 shots; depolarising 0.01; measurement flip 0.02",
        condition_facts,
    )
    return rows


def _metric_fact(
    config: PublicationConfig,
    registry: FactRegistry,
    fact_id: str,
    value: float,
    *,
    metric: str,
    source_id: str,
    locator: str,
    split: str,
    scope: str,
    usages: Sequence[str],
) -> str:
    return registry.add(
        fact_id,
        exact_value=float(value),
        display_value=_display(config, metric, float(value)),
        source_id=source_id,
        locator=locator,
        split=split,
        scope=scope,
        significance_adjusted=False,
        preference="lower" if metric in {"qlike", "rmse", "mae"} else "higher",
        usages=usages,
    )


def _inference_record(
    config: PublicationConfig,
    registry: FactRegistry,
    row: Mapping[str, Any],
    *,
    source_id: str,
    evidence_id: str,
    metric: str,
    comparison: str,
    usages: Sequence[str],
) -> dict[str, Any]:
    difference_key = (
        "observed_metric_difference"
        if "observed_metric_difference" in row
        else "mean_loss_differential"
    )
    difference = float(row[difference_key])
    lower = float(row["confidence_interval_lower"])
    upper = float(row["confidence_interval_upper"])
    adjusted = float(row["adjusted_p_value"])
    locator_prefix = (
        f"rows[baseline={row['baseline']},"
        f"{'metric' if 'metric' in row else 'loss_metric'}={metric}]"
    )
    fact_ids = [
        registry.add(
            f"{evidence_id}.difference",
            exact_value=difference,
            display_value=f"{difference:.4f}",
            source_id=source_id,
            locator=f"{locator_prefix}.{difference_key}",
            split="test",
            scope="architecture_level",
            significance_adjusted=True,
            preference="higher" if metric in {"macro_f1", "transition_pr_auc"} else "lower",
            usages=usages,
        ),
        registry.add(
            f"{evidence_id}.ci_lower",
            exact_value=lower,
            display_value=f"{lower:.4f}",
            source_id=source_id,
            locator=f"{locator_prefix}.confidence_interval_lower",
            split="test",
            scope="architecture_level",
            significance_adjusted=True,
            preference="higher" if metric in {"macro_f1", "transition_pr_auc"} else "lower",
            usages=usages,
        ),
        registry.add(
            f"{evidence_id}.ci_upper",
            exact_value=upper,
            display_value=f"{upper:.4f}",
            source_id=source_id,
            locator=f"{locator_prefix}.confidence_interval_upper",
            split="test",
            scope="architecture_level",
            significance_adjusted=True,
            preference="higher" if metric in {"macro_f1", "transition_pr_auc"} else "lower",
            usages=usages,
        ),
        registry.add(
            f"{evidence_id}.holm_p",
            exact_value=adjusted,
            display_value=f"{adjusted:.3g}",
            source_id=source_id,
            locator=f"{locator_prefix}.adjusted_p_value",
            split="test",
            scope="architecture_level",
            significance_adjusted=True,
            preference="not_applicable",
            usages=usages,
        ),
    ]
    return {
        "comparison": comparison,
        "difference": difference,
        "ci_lower": lower,
        "ci_upper": upper,
        "holm_p": adjusted,
        "holm_significant": adjusted < config.significance_alpha,
        "fact_ids": fact_ids,
    }


def _build_financial_table(
    config: PublicationConfig,
    registry: FactRegistry,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    benchmark = _rows(config, "financial_benchmark")
    qrc_aggregate = _rows(config, "financial_qrc_aggregate")
    variance = _rows(config, "variance_diagnostics")
    classification_inference = _rows(config, "financial_classification_inference")
    regression_inference = _rows(config, "financial_regression_inference")

    qrc_class = _find_row(
        qrc_aggregate,
        split="test",
        task="regime_classification",
    )
    qrc_regression = _find_row(
        qrc_aggregate,
        split="test",
        task="rv_regression",
    )
    correlation_by_model = {
        row["model"]: float(row["correlation"]) for row in variance if row["split"] == "test"
    }
    model_definitions = [
        (
            "Majority classifier",
            "majority_classifier",
            None,
            None,
            "QRC higher on all three classification metrics (Holm-adjusted)",
        ),
        (
            "Regime persistence",
            "regime_persistence",
            None,
            None,
            "QRC higher on transition PR-AUC only (Holm-adjusted)",
        ),
        (
            "Logistic regression",
            "logistic_regression",
            None,
            None,
            "Logistic regression higher on macro-F1 (Holm-adjusted)",
        ),
        (
            "ESN",
            "esn_classifier",
            "esn_regressor",
            "esn_regressor",
            "No Holm-adjusted QRC difference",
        ),
        (
            "RV persistence",
            None,
            "rv_persistence",
            "rv_persistence",
            "No Holm-adjusted QRC difference",
        ),
        (
            "GARCH(1,1)",
            "gaussian_garch_1_1",
            "gaussian_garch_1_1",
            "garch_1_1",
            "No Holm-adjusted regression difference; transition PR-AUC undefined",
        ),
        ("QRC mean", None, None, None, "Reference: mean of three frozen reservoir seeds"),
    ]
    table_rows: list[dict[str, Any]] = []
    metric_names = (
        "macro_f1",
        "balanced_accuracy",
        "transition_pr_auc",
        "qlike",
        "rmse",
        "mae",
        "correlation",
    )
    for display_name, classifier_id, regression_id, correlation_id, annotation in model_definitions:
        values: dict[str, float | None] = {metric: None for metric in metric_names}
        standard_deviations: dict[str, float] = {}
        source_locations: dict[str, tuple[str, str, str]] = {}
        if display_name == "QRC mean":
            for metric in ("macro_f1", "balanced_accuracy", "transition_pr_auc"):
                values[metric] = float(qrc_class[f"{metric}_mean"])
                standard_deviations[metric] = float(qrc_class[f"{metric}_standard_deviation"])
                source_locations[metric] = (
                    "financial_qrc_aggregate",
                    (f"rows[split=test,task=regime_classification].{metric}_mean"),
                    "architecture_level",
                )
            for metric in ("qlike", "rmse", "mae", "correlation"):
                values[metric] = float(qrc_regression[f"{metric}_mean"])
                standard_deviations[metric] = float(qrc_regression[f"{metric}_standard_deviation"])
                source_locations[metric] = (
                    "financial_qrc_aggregate",
                    f"rows[split=test,task=rv_regression].{metric}_mean",
                    "architecture_level",
                )
        else:
            if classifier_id is not None:
                classifier = _find_row(
                    benchmark,
                    split="test",
                    task="regime_classification",
                    model_type=classifier_id,
                )
                for metric in ("macro_f1", "balanced_accuracy", "transition_pr_auc"):
                    if classifier.get(metric) is not None:
                        values[metric] = float(classifier[metric])
                        source_locations[metric] = (
                            "financial_benchmark",
                            (
                                "rows[split=test,task=regime_classification,"
                                f"model_type={classifier_id}].{metric}"
                            ),
                            "baseline",
                        )
            if regression_id is not None:
                regression = _find_row(
                    benchmark,
                    split="test",
                    task="rv_regression",
                    model_type=regression_id,
                )
                for metric in ("qlike", "rmse", "mae"):
                    values[metric] = float(regression[metric])
                    source_locations[metric] = (
                        "financial_benchmark",
                        (
                            "rows[split=test,task=rv_regression,"
                            f"model_type={regression_id}].{metric}"
                        ),
                        "baseline",
                    )
            if correlation_id is not None:
                values["correlation"] = correlation_by_model[correlation_id]
                source_locations["correlation"] = (
                    "variance_diagnostics",
                    f"rows[split=test,model={correlation_id}].correlation",
                    "baseline",
                )
        row: dict[str, Any] = {
            "model": display_name,
            "holm_annotation": annotation,
            "holm_annotation_display": annotation,
        }
        fact_ids: list[str] = []
        for metric in metric_names:
            value = values[metric]
            row[metric] = value
            if value is None:
                row[f"{metric}_display"] = config.missing_display
                row[f"{metric}_sd"] = None
                continue
            source_id, locator, scope = source_locations[metric]
            fact_id = _metric_fact(
                config,
                registry,
                f"financial.test.{_slug(display_name)}.{metric}",
                value,
                metric=metric,
                source_id=source_id,
                locator=locator,
                split="test",
                scope=scope,
                usages=("publication_table_2", "publication_figure_2", "results_factsheet"),
            )
            fact_ids.append(fact_id)
            standard_deviation = standard_deviations.get(metric)
            row[f"{metric}_sd"] = standard_deviation
            if standard_deviation is not None:
                fact_ids.append(
                    registry.add(
                        (f"financial.test.qrc_mean.{metric}.standard_deviation"),
                        exact_value=standard_deviation,
                        display_value=_display(config, metric, standard_deviation),
                        source_id=source_id,
                        locator=locator.replace("_mean", "_standard_deviation"),
                        split="test",
                        scope="architecture_level",
                        significance_adjusted=False,
                        preference="not_applicable",
                        usages=("publication_figure_2",),
                    )
                )
            display = _display(config, metric, value)
            row[f"{metric}_display"] = display
        row["trace_fact_ids"] = fact_ids
        table_rows.append(row)

    classification_evidence = {
        (row["baseline"], row["metric"]): row for row in classification_inference
    }
    regression_evidence = {
        (row["baseline"], row["loss_metric"]): row for row in regression_inference
    }
    inference_for_figure = {
        "macro_f1": [
            _inference_record(
                config,
                registry,
                classification_evidence[(baseline, "macro_f1")],
                source_id="financial_classification_inference",
                evidence_id=f"financial.inference.macro_f1.qrc_vs_{baseline}",
                metric="macro_f1",
                comparison=label,
                usages=("publication_table_2", "publication_figure_2", "claims"),
            )
            for baseline, label in (
                ("logistic_regression", "vs logistic"),
                ("esn_classifier", "vs ESN"),
            )
        ],
        "transition_pr_auc": [
            _inference_record(
                config,
                registry,
                classification_evidence[(baseline, "transition_pr_auc")],
                source_id="financial_classification_inference",
                evidence_id=f"financial.inference.transition_pr_auc.qrc_vs_{baseline}",
                metric="transition_pr_auc",
                comparison=label,
                usages=("publication_table_2", "publication_figure_2", "claims"),
            )
            for baseline, label in (
                ("regime_persistence", "vs persistence"),
                ("esn_classifier", "vs ESN"),
            )
        ],
        "qlike": [
            _inference_record(
                config,
                registry,
                regression_evidence[(baseline, "qlike")],
                source_id="financial_regression_inference",
                evidence_id=f"financial.inference.qlike.qrc_vs_{baseline}",
                metric="qlike",
                comparison=label,
                usages=("publication_table_2", "publication_figure_2", "claims"),
            )
            for baseline, label in (("garch_1_1", "vs GARCH"), ("esn_regressor", "vs ESN"))
        ],
        "rmse": [
            _inference_record(
                config,
                registry,
                regression_evidence[(baseline, "squared_error")],
                source_id="financial_regression_inference",
                evidence_id=f"financial.inference.squared_error.qrc_vs_{baseline}",
                metric="squared_error",
                comparison=label,
                usages=("publication_table_2", "publication_figure_2", "claims"),
            )
            for baseline, label in (("garch_1_1", "vs GARCH"), ("esn_regressor", "vs ESN"))
        ],
    }
    registered: dict[tuple[str, str, str], dict[str, Any]] = {}
    for metric, records in inference_for_figure.items():
        source_metric = "squared_error" if metric == "rmse" else metric
        for record in records:
            comparison_to_baseline = {
                "vs logistic": "logistic_regression",
                "vs ESN": (
                    "esn_regressor"
                    if source_metric in {"qlike", "squared_error"}
                    else "esn_classifier"
                ),
                "vs persistence": "regime_persistence",
                "vs GARCH": "garch_1_1",
            }[record["comparison"]]
            domain = (
                "regression" if source_metric in {"qlike", "squared_error"} else "classification"
            )
            registered[(domain, comparison_to_baseline, source_metric)] = record
    for (baseline, metric), row in classification_evidence.items():
        key = ("classification", baseline, metric)
        if key not in registered:
            registered[key] = _inference_record(
                config,
                registry,
                row,
                source_id="financial_classification_inference",
                evidence_id=f"financial.inference.{metric}.qrc_vs_{baseline}",
                metric=metric,
                comparison=f"vs {row['baseline_display_name']}",
                usages=("publication_table_2", "claims"),
            )
    for (baseline, metric), row in regression_evidence.items():
        key = ("regression", baseline, metric)
        if key not in registered:
            registered[key] = _inference_record(
                config,
                registry,
                row,
                source_id="financial_regression_inference",
                evidence_id=f"financial.inference.{metric}.qrc_vs_{baseline}",
                metric=metric,
                comparison=f"vs {row['baseline_display_name']}",
                usages=("publication_table_2", "claims"),
            )
    table_baselines = {
        "Majority classifier": [("classification", "majority_classifier")],
        "Regime persistence": [("classification", "regime_persistence")],
        "Logistic regression": [("classification", "logistic_regression")],
        "ESN": [
            ("classification", "esn_classifier"),
            ("regression", "esn_regressor"),
        ],
        "RV persistence": [("regression", "rv_persistence")],
        "GARCH(1,1)": [("regression", "garch_1_1")],
        "QRC mean": [],
    }
    for table_row in table_rows:
        for domain, baseline in table_baselines[str(table_row["model"])]:
            table_row["trace_fact_ids"].extend(
                fact_id
                for (record_domain, record_baseline, _), record in registered.items()
                if record_domain == domain and record_baseline == baseline
                for fact_id in record["fact_ids"]
            )
        table_row["trace_fact_ids"] = sorted(set(table_row["trace_fact_ids"]))
    evidence_map = {
        "classification": classification_evidence,
        "regression": regression_evidence,
        "figure": inference_for_figure,
        "registered": registered,
    }
    return table_rows, inference_for_figure, evidence_map


def _mnist_inference_facts(
    config: PublicationConfig,
    registry: FactRegistry,
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        baseline = str(row["baseline"])
        metric = str(row["metric"])
        prefix = f"mnist.inference.{metric}.qrc_vs_{baseline}"
        locator = f"rows[baseline={baseline},metric={metric}]"
        adjusted = float(row["bootstrap_adjusted_p_value"])
        fact_ids = [
            registry.add(
                f"{prefix}.difference",
                exact_value=float(row["observed_metric_difference"]),
                display_value=f"{float(row['observed_metric_difference']):.4f}",
                source_id="mnist_inference",
                locator=f"{locator}.observed_metric_difference",
                split="test",
                scope="architecture_level",
                significance_adjusted=True,
                preference="higher",
                usages=("publication_table_3", "claims"),
            ),
            registry.add(
                f"{prefix}.ci_lower",
                exact_value=float(row["confidence_interval_lower"]),
                display_value=f"{float(row['confidence_interval_lower']):.4f}",
                source_id="mnist_inference",
                locator=f"{locator}.confidence_interval_lower",
                split="test",
                scope="architecture_level",
                significance_adjusted=True,
                preference="higher",
                usages=("publication_table_3", "claims"),
            ),
            registry.add(
                f"{prefix}.ci_upper",
                exact_value=float(row["confidence_interval_upper"]),
                display_value=f"{float(row['confidence_interval_upper']):.4f}",
                source_id="mnist_inference",
                locator=f"{locator}.confidence_interval_upper",
                split="test",
                scope="architecture_level",
                significance_adjusted=True,
                preference="higher",
                usages=("publication_table_3", "claims"),
            ),
            registry.add(
                f"{prefix}.holm_p",
                exact_value=adjusted,
                display_value=f"{adjusted:.3g}",
                source_id="mnist_inference",
                locator=f"{locator}.bootstrap_adjusted_p_value",
                split="test",
                scope="architecture_level",
                significance_adjusted=True,
                preference="not_applicable",
                usages=("publication_table_3", "claims"),
            ),
        ]
        result[(baseline, metric)] = {
            "adjusted_p_value": adjusted,
            "holm_significant": adjusted < config.significance_alpha,
            "fact_ids": fact_ids,
        }
    return result


def _build_mnist_table(
    config: PublicationConfig,
    registry: FactRegistry,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    benchmark = _rows(config, "mnist_benchmark")
    qrc_aggregate = _rows(config, "mnist_qrc_aggregate")
    robustness = _rows(config, "mnist_robustness")
    inference = _mnist_inference_facts(
        config,
        registry,
        _rows(config, "mnist_inference"),
    )
    qrc_test = _find_row(qrc_aggregate, split="test", condition="analytic")
    metric_map = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "balanced_accuracy": "balanced_accuracy",
        "macro_roc_auc": "ovr_macro_roc_auc",
    }
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        section: str,
        model: str,
        source_id: str,
        locator_prefix: str,
        values: Mapping[str, float],
        standard_deviations: Mapping[str, float] | None,
        scope: str,
        annotation: str,
        aggregate: bool = False,
    ) -> None:
        row: dict[str, Any] = {
            "section": section,
            "model_or_condition": model,
            "holm_annotation": annotation,
            "holm_annotation_display": annotation,
        }
        fact_ids: list[str] = []
        for metric, source_metric in metric_map.items():
            value = float(values[metric])
            fact_ids.append(
                _metric_fact(
                    config,
                    registry,
                    f"mnist.test.{_slug(model)}.{metric}",
                    value,
                    metric=metric,
                    source_id=source_id,
                    locator=(
                        f"{locator_prefix}.{source_metric}_mean"
                        if aggregate
                        else f"{locator_prefix}.{source_metric}"
                    ),
                    split="test",
                    scope=scope,
                    usages=("publication_table_3", "publication_figure_4", "results_factsheet"),
                )
            )
            row[metric] = value
            standard_deviation = (
                None if standard_deviations is None else float(standard_deviations[metric])
            )
            row[f"{metric}_sd"] = standard_deviation
            display = _display(config, metric, value)
            if standard_deviation is not None:
                display = f"{display} ± {_display(config, metric, standard_deviation)}"
                fact_ids.append(
                    registry.add(
                        f"mnist.test.{_slug(model)}.{metric}.population_standard_deviation",
                        exact_value=standard_deviation,
                        display_value=_display(config, metric, standard_deviation),
                        source_id=source_id,
                        locator=(f"{locator_prefix}.{source_metric}_population_standard_deviation"),
                        split="test",
                        scope=scope,
                        significance_adjusted=False,
                        preference="not_applicable",
                        usages=(
                            "publication_table_3",
                            "publication_figure_4",
                            "results_factsheet",
                        ),
                    )
                )
            row[f"{metric}_display"] = display
        row["trace_fact_ids"] = fact_ids
        rows.append(row)

    logistic = _find_row(benchmark, split="test", model="logistic_baseline")
    esn = _find_row(benchmark, split="test", model="esn_baseline")
    add_row(
        section="Model comparison",
        model="Flattened logistic",
        source_id="mnist_benchmark",
        locator_prefix="rows[split=test,model=logistic_baseline]",
        values={
            "accuracy": logistic["accuracy"],
            "macro_f1": logistic["macro_f1"],
            "balanced_accuracy": logistic["balanced_accuracy"],
            "macro_roc_auc": logistic["ovr_macro_roc_auc"],
        },
        standard_deviations=None,
        scope="baseline",
        annotation="No Holm-adjusted QRC difference",
    )
    add_row(
        section="Model comparison",
        model="Size-controlled ESN",
        source_id="mnist_benchmark",
        locator_prefix="rows[split=test,model=esn_baseline]",
        values={
            "accuracy": esn["accuracy"],
            "macro_f1": esn["macro_f1"],
            "balanced_accuracy": esn["balanced_accuracy"],
            "macro_roc_auc": esn["ovr_macro_roc_auc"],
        },
        standard_deviations=None,
        scope="baseline",
        annotation="ESN higher than QRC on all metrics (Holm-adjusted)",
    )
    add_row(
        section="Model comparison",
        model="Exact QRC mean",
        source_id="mnist_qrc_aggregate",
        locator_prefix="rows[split=test,condition=analytic]",
        values={
            metric: qrc_test[f"{source_metric}_mean"]
            for metric, source_metric in metric_map.items()
        },
        standard_deviations={
            metric: qrc_test[f"{source_metric}_population_standard_deviation"]
            for metric, source_metric in metric_map.items()
        },
        scope="architecture_level",
        annotation="Mean ± population SD across seeds 2026-2028",
        aggregate=True,
    )
    robustness_names = {
        "analytic": "Analytic (seed 2026)",
        "shots_2048": "2,048 shots (seed 2026)",
        "depolarizing_0_01": "2,048 shots + depolarising 0.01",
        "measurement_flip_0_02": "2,048 shots + measurement flip 0.02",
    }
    for condition, name in robustness_names.items():
        source = _find_row(robustness, split="test", condition=condition)
        add_row(
            section="QRC robustness",
            model=name,
            source_id="mnist_robustness",
            locator_prefix=f"rows[split=test,condition={condition}]",
            values={
                "accuracy": source["accuracy"],
                "macro_f1": source["macro_f1"],
                "balanced_accuracy": source["balanced_accuracy"],
                "macro_roc_auc": source["ovr_macro_roc_auc"],
            },
            standard_deviations=None,
            scope="per_seed",
            annotation="Frozen seed-2026 prediction condition; no significance test",
        )
    inference_facts_by_model = {
        "Flattened logistic": [
            fact_id
            for (baseline, _), record in inference.items()
            if baseline == "flattened_logistic"
            for fact_id in record["fact_ids"]
        ],
        "Size-controlled ESN": [
            fact_id
            for (baseline, _), record in inference.items()
            if baseline == "esn"
            for fact_id in record["fact_ids"]
        ],
        "Exact QRC mean": [
            "mnist.architecture.seed_2026",
            "mnist.architecture.seed_2027",
            "mnist.architecture.seed_2028",
        ],
    }
    for row in rows:
        row["trace_fact_ids"].extend(
            inference_facts_by_model.get(
                str(row["model_or_condition"]),
                ["mnist.architecture.seed_2026"] if row["section"] == "QRC robustness" else [],
            )
        )
        row["trace_fact_ids"] = sorted(set(row["trace_fact_ids"]))
    return rows, inference


def _build_architecture_figure_data(
    config: PublicationConfig,
    registry: FactRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = _rows(config, "qubit_scaling")
    resources = _rows(config, "qubit_resources")
    qubit_rows: list[dict[str, Any]] = []
    for n_qubits in (2, 3, 4, 5, 6):
        values: dict[str, Any] = {"n_qubits": n_qubits}
        fact_ids = [
            registry.add(
                f"selection.qubits.{n_qubits}.n_qubits",
                exact_value=n_qubits,
                display_value=str(n_qubits),
                source_id="qubit_resources",
                locator=f"rows[n_qubits={n_qubits}].n_qubits",
                split="validation",
                scope="architecture_level",
                significance_adjusted=False,
                preference="not_applicable",
                usages=("publication_figure_1",),
            )
        ]
        for metric, task in (
            ("macro_f1", "regime_classification"),
            ("transition_pr_auc", "regime_classification"),
            ("qlike", "rv_regression"),
            ("state_generation_seconds", "regime_classification"),
        ):
            row = _find_row(
                aggregate,
                n_qubits=n_qubits,
                split="validation",
                task=task,
                metric=metric,
            )
            values[metric] = float(row["mean"])
            preference = "lower" if metric in {"qlike", "state_generation_seconds"} else "higher"
            fact_ids.append(
                registry.add(
                    f"selection.qubits.{n_qubits}.{metric}",
                    exact_value=float(row["mean"]),
                    display_value=f"{float(row['mean']):.6g}",
                    source_id="qubit_scaling",
                    locator=(
                        f"rows[n_qubits={n_qubits},split=validation,"
                        f"task={task},metric={metric}].mean"
                    ),
                    split="validation",
                    scope="architecture_level",
                    significance_adjusted=False,
                    preference=preference,
                    usages=("publication_figure_1",),
                )
            )
        resource = _find_row(resources, n_qubits=n_qubits)
        values["feature_dimension"] = int(resource["raw_feature_dimension"])
        fact_ids.append(
            registry.add(
                f"selection.qubits.{n_qubits}.feature_dimension",
                exact_value=int(resource["raw_feature_dimension"]),
                display_value=str(int(resource["raw_feature_dimension"])),
                source_id="qubit_resources",
                locator=f"rows[n_qubits={n_qubits}].raw_feature_dimension",
                split="validation",
                scope="architecture_level",
                significance_adjusted=False,
                preference="lower",
                usages=("publication_figure_1",),
            )
        )
        values["fact_ids"] = fact_ids
        qubit_rows.append(values)

    virtual_rows: list[dict[str, Any]] = []
    for source in _rows(config, "virtual_node_selection"):
        virtual_nodes = int(source["virtual_nodes"])
        values = {
            "virtual_nodes": virtual_nodes,
            "macro_f1": float(source["validation_macro_f1_mean"]),
            "transition_pr_auc": float(source["validation_transition_pr_auc_mean"]),
            "qlike": float(source["validation_qlike_mean"]),
            "condition_number": float(source["condition_number_mean"]),
            "feature_dimension": int(source["raw_feature_dimension"]),
        }
        virtual_fact_ids: list[str] = []
        fields = {
            "virtual_nodes": "virtual_nodes",
            "macro_f1": "validation_macro_f1_mean",
            "transition_pr_auc": "validation_transition_pr_auc_mean",
            "qlike": "validation_qlike_mean",
            "condition_number": "condition_number_mean",
            "feature_dimension": "raw_feature_dimension",
        }
        for metric, field in fields.items():
            value = values[metric]
            virtual_fact_ids.append(
                registry.add(
                    f"selection.virtual_nodes.{virtual_nodes}.{metric}",
                    exact_value=value,
                    display_value=f"{value:.6g}" if isinstance(value, float) else str(value),
                    source_id="virtual_node_selection",
                    locator=f"rows[virtual_nodes={virtual_nodes}].{field}",
                    split="validation",
                    scope="architecture_level",
                    significance_adjusted=False,
                    preference=(
                        "higher"
                        if metric in {"macro_f1", "transition_pr_auc"}
                        else "lower"
                        if metric in {"qlike", "condition_number", "feature_dimension"}
                        else "not_applicable"
                    ),
                    usages=("publication_figure_1",),
                )
            )
        values["fact_ids"] = virtual_fact_ids
        virtual_rows.append(values)
    return qubit_rows, virtual_rows


def _build_financial_robustness_figure_data(
    config: PublicationConfig,
    registry: FactRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _rows(config, "financial_robustness")
    metrics = ("macro_f1", "transition_pr_auc", "qlike")
    task_by_metric = {
        "macro_f1": "regime_classification",
        "transition_pr_auc": "regime_classification",
        "qlike": "rv_regression",
    }
    shot_rows: list[dict[str, Any]] = []
    for order, shots in enumerate((None, 128, 512, 2048, 8192)):
        for metric in metrics:
            criteria: dict[str, Any] = {
                "split": "test",
                "task": task_by_metric[metric],
                "metric": metric,
                "reservoir_seed": None,
            }
            if shots is None:
                criteria.update(
                    study_type="analytic_reference",
                    aggregation_level="reservoir_seeds",
                    shot_count=None,
                )
                label = "analytic"
            else:
                criteria.update(
                    study_type="finite_shot",
                    aggregation_level="all_repetitions",
                    shot_count=shots,
                )
                label = str(shots)
            source = _find_row(rows, **criteria)
            prefix = f"financial.robustness.{label}.{metric}"
            fact_ids = [
                _metric_fact(
                    config,
                    registry,
                    f"{prefix}.mean",
                    float(source["mean"]),
                    metric=metric,
                    source_id="financial_robustness",
                    locator=(
                        f"rows[study_type={criteria['study_type']},"
                        f"aggregation_level={criteria['aggregation_level']},"
                        f"shot_count={shots},split=test,metric={metric}].mean"
                    ),
                    split="test",
                    scope="architecture_level",
                    usages=("publication_figure_3", "results_factsheet"),
                ),
                registry.add(
                    f"{prefix}.standard_deviation",
                    exact_value=float(source["standard_deviation"]),
                    display_value=f"{float(source['standard_deviation']):.4f}",
                    source_id="financial_robustness",
                    locator=(
                        f"rows[study_type={criteria['study_type']},"
                        f"aggregation_level={criteria['aggregation_level']},"
                        f"shot_count={shots},split=test,metric={metric}].standard_deviation"
                    ),
                    split="test",
                    scope="architecture_level",
                    significance_adjusted=False,
                    preference="not_applicable",
                    usages=("publication_figure_3",),
                ),
            ]
            shot_rows.append(
                {
                    "order": order,
                    "shot_count": shots,
                    "metric": metric,
                    "value": float(source["mean"]),
                    "sd": float(source["standard_deviation"]),
                    "fact_ids": fact_ids,
                }
            )

    noise_rows: list[dict[str, Any]] = []
    conditions = (
        (
            0,
            "finite_shot",
            0.0,
            0.0,
            "shots_2048",
        ),
        (
            1,
            "depolarizing_noise",
            0.01,
            0.0,
            "depolarizing_0_01",
        ),
        (
            2,
            "measurement_noise",
            0.0,
            0.02,
            "measurement_flip_0_02",
        ),
    )
    for order, study_type, depolarizing, measurement, condition_id in conditions:
        for metric in metrics:
            source = _find_row(
                rows,
                split="test",
                task=task_by_metric[metric],
                metric=metric,
                reservoir_seed=None,
                study_type=study_type,
                aggregation_level="all_repetitions",
                shot_count=2048,
                depolarizing_probability=depolarizing,
                measurement_bit_flip_probability=measurement,
            )
            prefix = f"financial.robustness.{condition_id}.{metric}"
            fact_ids = [
                _metric_fact(
                    config,
                    registry,
                    f"{prefix}.mean",
                    float(source["mean"]),
                    metric=metric,
                    source_id="financial_robustness",
                    locator=(
                        f"rows[study_type={study_type},aggregation_level=all_repetitions,"
                        f"depolarizing_probability={depolarizing},"
                        f"measurement_bit_flip_probability={measurement},"
                        f"split=test,metric={metric}].mean"
                    ),
                    split="test",
                    scope="architecture_level",
                    usages=("publication_figure_3", "results_factsheet"),
                ),
                registry.add(
                    f"{prefix}.standard_deviation",
                    exact_value=float(source["standard_deviation"]),
                    display_value=f"{float(source['standard_deviation']):.4f}",
                    source_id="financial_robustness",
                    locator=(
                        f"rows[study_type={study_type},aggregation_level=all_repetitions,"
                        f"depolarizing_probability={depolarizing},"
                        f"measurement_bit_flip_probability={measurement},"
                        f"split=test,metric={metric}].standard_deviation"
                    ),
                    split="test",
                    scope="architecture_level",
                    significance_adjusted=False,
                    preference="not_applicable",
                    usages=("publication_figure_3",),
                ),
            ]
            noise_rows.append(
                {
                    "order": order,
                    "condition": condition_id,
                    "metric": metric,
                    "value": float(source["mean"]),
                    "sd": float(source["standard_deviation"]),
                    "fact_ids": fact_ids,
                }
            )
    return shot_rows, noise_rows


def _register_constraint_and_reproducibility_facts(
    config: PublicationConfig,
    registry: FactRegistry,
) -> dict[str, str]:
    architecture = _load_json(config.sources["financial_architecture"].path)
    diagnostic_summary = _load_json(config.sources["benchmark_diagnostics_summary"].path)
    tail_rows = _rows(config, "tail_diagnostics")
    mnist_environment = _load_json(config.sources["mnist_environment"].path)
    mnist_aggregate = _find_row(
        _rows(config, "mnist_qrc_aggregate"),
        split="test",
        condition="analytic",
    )
    fact_ids = {
        "quantum_advantage": registry.add(
            "constraints.quantum_advantage_claim",
            exact_value=bool(architecture["quantum_advantage_claim"]),
            display_value=str(bool(architecture["quantum_advantage_claim"])).lower(),
            source_id="financial_architecture",
            locator="quantum_advantage_claim",
            split="not_applicable",
            scope="constraint",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("claims", "limitations_factsheet"),
        ),
        "physical_qpu": registry.add(
            "constraints.physical_qpu_execution",
            exact_value=bool(architecture["physical_qpu_execution"]),
            display_value=str(bool(architecture["physical_qpu_execution"])).lower(),
            source_id="financial_architecture",
            locator="physical_qpu_execution",
            split="not_applicable",
            scope="constraint",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("claims", "limitations_factsheet"),
        ),
        "lead_identifiable": registry.add(
            "constraints.transition_lead_identifiable",
            exact_value=bool(diagnostic_summary["lead_time_analysis"]["identifiable"]),
            display_value=str(
                bool(diagnostic_summary["lead_time_analysis"]["identifiable"])
            ).lower(),
            source_id="benchmark_diagnostics_summary",
            locator="lead_time_analysis.identifiable",
            split="test",
            scope="diagnostic",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("claims", "limitations_factsheet", "results_factsheet"),
        ),
        "mnist_runtime": registry.add(
            "reproducibility.mnist_exact_total_runtime_seconds_mean",
            exact_value=float(mnist_aggregate["total_runtime_seconds_mean"]),
            display_value=f"{float(mnist_aggregate['total_runtime_seconds_mean']):.1f}",
            source_id="mnist_qrc_aggregate",
            locator="rows[split=test,condition=analytic].total_runtime_seconds_mean",
            split="test",
            scope="architecture_level",
            significance_adjusted=False,
            preference="lower",
            usages=("reproducibility_factsheet",),
        ),
        "diagnostic_runtime": registry.add(
            "reproducibility.stage2b_runtime_seconds",
            exact_value=float(diagnostic_summary["runtime_seconds"]),
            display_value=f"{float(diagnostic_summary['runtime_seconds']):.1f}",
            source_id="benchmark_diagnostics_summary",
            locator="runtime_seconds",
            split="not_applicable",
            scope="runtime",
            significance_adjusted=False,
            preference="lower",
            usages=("reproducibility_factsheet",),
        ),
        "financial_commit": registry.add(
            "reproducibility.financial_architecture_git_commit",
            exact_value=architecture["git"]["commit"],
            display_value=str(architecture["git"]["commit"]),
            source_id="financial_architecture",
            locator="git.commit",
            split="not_applicable",
            scope="provenance",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("reproducibility_factsheet",),
        ),
        "mnist_commit": registry.add(
            "reproducibility.mnist_git_commit",
            exact_value=mnist_environment["git"]["commit"],
            display_value=str(mnist_environment["git"]["commit"]),
            source_id="mnist_environment",
            locator="git.commit",
            split="not_applicable",
            scope="provenance",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("reproducibility_factsheet",),
        ),
        "tail_p90_count": registry.add(
            "limitations.financial_test_training_p90_tail_count",
            exact_value=int(
                _find_row(
                    tail_rows,
                    split="test",
                    model="qrc_2026",
                    tail_threshold_id="training_p90",
                )["sample_count"]
            ),
            display_value="29",
            source_id="tail_diagnostics",
            locator=("rows[split=test,model=qrc_2026,tail_threshold_id=training_p90].sample_count"),
            split="test",
            scope="diagnostic",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("limitations_factsheet",),
        ),
        "tail_p95_count": registry.add(
            "limitations.financial_test_training_p95_tail_count",
            exact_value=int(
                _find_row(
                    tail_rows,
                    split="test",
                    model="qrc_2026",
                    tail_threshold_id="training_p95",
                )["sample_count"]
            ),
            display_value="13",
            source_id="tail_diagnostics",
            locator=("rows[split=test,model=qrc_2026,tail_threshold_id=training_p95].sample_count"),
            split="test",
            scope="diagnostic",
            significance_adjusted=False,
            preference="not_applicable",
            usages=("limitations_factsheet",),
        ),
    }
    return fact_ids


def _build_claims(
    registry: FactRegistry,
    constraint_facts: Mapping[str, str],
) -> list[dict[str, Any]]:
    def claim(
        claim_id: int,
        text: str,
        status: str,
        rationale: str,
        evidence: Sequence[str],
    ) -> dict[str, Any]:
        if status not in CLAIM_STATUSES:
            raise ValueError(f"invalid claim status: {status}")
        for fact_id in evidence:
            registry.get(fact_id)
        return {
            "claim_id": claim_id,
            "claim": text,
            "status": status,
            "rationale": rationale,
            "evidence_fact_ids": list(evidence),
        }

    claims = [
        claim(
            1,
            "QRC achieved quantum advantage.",
            "prohibited",
            (
                "Only classical exact and controlled sampled simulations were run; "
                "the frozen manifest explicitly makes no quantum-advantage claim."
            ),
            [constraint_facts["quantum_advantage"], constraint_facts["physical_qpu"]],
        ),
        claim(
            2,
            "QRC outperformed all classical models.",
            "prohibited",
            (
                "The point estimates contradict this: logistic regression leads financial "
                "macro-F1, GARCH leads QLIKE, ESN leads RMSE and MNIST."
            ),
            [
                "financial.test.logistic_regression.macro_f1",
                "financial.test.qrc_mean.macro_f1",
                "financial.test.garch_1_1.qlike",
                "financial.test.qrc_mean.qlike",
                "financial.test.esn.rmse",
                "financial.test.qrc_mean.rmse",
            ],
        ),
        claim(
            3,
            "QRC was competitive with ESN for financial classification.",
            "supported with qualification",
            (
                "QRC and ESN point estimates are close and none of macro-F1, balanced "
                "accuracy or transition PR-AUC differed after Holm correction. "
                "Competitive does not mean superior or equivalent."
            ),
            [
                "financial.inference.macro_f1.qrc_vs_esn_classifier.holm_p",
                "financial.inference.balanced_accuracy.qrc_vs_esn_classifier.holm_p",
                "financial.inference.transition_pr_auc.qrc_vs_esn_classifier.holm_p",
            ],
        ),
        claim(
            4,
            "QRC significantly outperformed regime persistence for transition PR-AUC.",
            "supported",
            (
                "The architecture-level paired interval is positive and the "
                "Holm-adjusted p-value is below 0.05."
            ),
            [
                "financial.inference.transition_pr_auc.qrc_vs_regime_persistence.difference",
                "financial.inference.transition_pr_auc.qrc_vs_regime_persistence.ci_lower",
                "financial.inference.transition_pr_auc.qrc_vs_regime_persistence.ci_upper",
                "financial.inference.transition_pr_auc.qrc_vs_regime_persistence.holm_p",
            ],
        ),
        claim(
            5,
            "Logistic regression significantly outperformed QRC on financial macro-F1.",
            "supported",
            "The QRC-minus-logistic interval is negative and the Holm-adjusted p-value is 0.0168.",
            [
                "financial.inference.macro_f1.qrc_vs_logistic_regression.difference",
                "financial.inference.macro_f1.qrc_vs_logistic_regression.ci_lower",
                "financial.inference.macro_f1.qrc_vs_logistic_regression.ci_upper",
                "financial.inference.macro_f1.qrc_vs_logistic_regression.holm_p",
            ],
        ),
        claim(
            6,
            "GARCH achieved the best test QLIKE.",
            "supported",
            (
                "GARCH has the lowest frozen test QLIKE point estimate in the "
                "directly comparable table."
            ),
            [
                "financial.test.garch_1_1.qlike",
                "financial.test.esn.qlike",
                "financial.test.qrc_mean.qlike",
                "financial.test.rv_persistence.qlike",
            ],
        ),
        claim(
            7,
            "ESN achieved the best test RMSE.",
            "supported",
            "ESN has the lowest frozen test RMSE point estimate.",
            [
                "financial.test.esn.rmse",
                "financial.test.garch_1_1.rmse",
                "financial.test.qrc_mean.rmse",
                "financial.test.rv_persistence.rmse",
            ],
        ),
        claim(
            8,
            "QRC produced the highest architecture-level variance correlation.",
            "supported",
            (
                "The mean across the three frozen QRC seeds is 0.543, above the "
                "directly comparable ESN, GARCH and persistence correlations."
            ),
            [
                "financial.test.qrc_mean.correlation",
                "financial.test.esn.correlation",
                "financial.test.garch_1_1.correlation",
                "financial.test.rv_persistence.correlation",
            ],
        ),
        claim(
            9,
            (
                "QRC and flattened logistic were statistically indistinguishable "
                "on MNIST after Holm correction."
            ),
            "supported with qualification",
            (
                "No tested MNIST metric was Holm-significant, but failure to reject "
                "is not an equivalence or non-inferiority result."
            ),
            [
                "mnist.inference.accuracy.qrc_vs_flattened_logistic.holm_p",
                "mnist.inference.macro_f1.qrc_vs_flattened_logistic.holm_p",
                "mnist.inference.balanced_accuracy.qrc_vs_flattened_logistic.holm_p",
                "mnist.inference.macro_roc_auc.qrc_vs_flattened_logistic.holm_p",
            ],
        ),
        claim(
            10,
            "ESN significantly outperformed QRC on MNIST.",
            "supported",
            "All four architecture-level MNIST comparisons favour ESN after Holm correction.",
            [
                "mnist.inference.accuracy.qrc_vs_esn.holm_p",
                "mnist.inference.macro_f1.qrc_vs_esn.holm_p",
                "mnist.inference.balanced_accuracy.qrc_vs_esn.holm_p",
                "mnist.inference.macro_roc_auc.qrc_vs_esn.holm_p",
            ],
        ),
        claim(
            11,
            "A reliable transition lead time was demonstrated.",
            "prohibited",
            (
                "Lead time is not identifiable from the frozen aggregate target rows; "
                "no forecast-origin/target/transition-date mapping exists."
            ),
            [constraint_facts["lead_identifiable"]],
        ),
        claim(
            12,
            "The finite-shot simulations represent physical QPU performance.",
            "prohibited",
            "The frozen execution manifest records no physical-QPU execution.",
            [constraint_facts["physical_qpu"]],
        ),
        claim(
            13,
            "Increasing the number of qubits monotonically improved validation performance.",
            "unsupported",
            (
                "Validation macro-F1, transition PR-AUC and QLIKE vary "
                "non-monotonically over two to six qubits."
            ),
            [
                *[
                    f"selection.qubits.{qubits}.{metric}"
                    for qubits in (2, 3, 4, 5, 6)
                    for metric in ("macro_f1", "transition_pr_auc", "qlike")
                ]
            ],
        ),
        claim(
            14,
            "Controlled noise improved QRC performance.",
            "prohibited",
            (
                "No improvement hypothesis was tested; non-monotonic simulation changes "
                "are treated only as stochastic/model variability."
            ),
            [
                "financial.robustness.analytic.macro_f1.mean",
                "financial.robustness.2048.macro_f1.mean",
                "financial.robustness.depolarizing_0_01.macro_f1.mean",
                "financial.robustness.measurement_flip_0_02.macro_f1.mean",
            ],
        ),
    ]
    return claims


def _write_factsheets(
    config: PublicationConfig,
    registry: FactRegistry,
) -> list[Path]:
    output = config.tracked_output_root
    qrc_f1 = registry.get("financial.test.qrc_mean.macro_f1")["display_rounded_value"]
    esn_f1 = registry.get("financial.test.esn.macro_f1")["display_rounded_value"]
    logistic_f1 = registry.get("financial.test.logistic_regression.macro_f1")[
        "display_rounded_value"
    ]
    qrc_qlike = registry.get("financial.test.qrc_mean.qlike")["display_rounded_value"]
    garch_qlike = registry.get("financial.test.garch_1_1.qlike")["display_rounded_value"]
    esn_rmse = registry.get("financial.test.esn.rmse")["display_rounded_value"]
    qrc_rmse = registry.get("financial.test.qrc_mean.rmse")["display_rounded_value"]
    mnist_qrc = registry.get("mnist.test.exact_qrc_mean.accuracy")["display_rounded_value"]
    mnist_esn = registry.get("mnist.test.size_controlled_esn.accuracy")["display_rounded_value"]
    mnist_shots = registry.get("mnist.test.2_048_shots_seed_2026.accuracy")["display_rounded_value"]
    results_path = output / "results_factsheet.md"
    results_path.write_text(
        "\n".join(
            [
                "# Results factsheet",
                "",
                (
                    f"- Logistic regression led financial test macro-F1 ({logistic_f1}); "
                    f"the frozen QRC architecture mean was {qrc_f1}, and ESN was {esn_f1}."
                ),
                (
                    "- The QRC-minus-logistic macro-F1 difference was Holm-significant; "
                    "no QRC-versus-ESN financial classification metric was Holm-significant."
                ),
                (
                    f"- GARCH had the lowest test QLIKE ({garch_qlike}) versus "
                    f"{qrc_qlike} for the QRC mean."
                ),
                (
                    f"- ESN had the lowest test RMSE ({esn_rmse}) versus "
                    f"{qrc_rmse} for the QRC mean."
                ),
                (
                    "- QRC significantly exceeded regime persistence on transition PR-AUC "
                    "after Holm correction."
                ),
                (
                    f"- On the balanced MNIST test subset, exact QRC mean accuracy was "
                    f"{mnist_qrc}; ESN reached {mnist_esn}."
                ),
                (
                    f"- QRC seed-2026 MNIST accuracy fell to {mnist_shots} at 2,048 shots; "
                    "the controlled noise rows remained substantially below the analytic result."
                ),
                "- No physical QPU was executed and no quantum-advantage conclusion is supported.",
                "",
                "All numeric sentences trace to `final_results_manifest.json`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    limitations_path = output / "limitations_factsheet.md"
    limitations_path.write_text(
        "\n".join(
            [
                "# Limitations factsheet",
                "",
                "- No physical-QPU execution was performed, and no quantum advantage is claimed.",
                "- QRC states were generated by classical exact density-matrix simulation.",
                (
                    "- Finite-shot, depolarising and measurement-flip conditions are "
                    "controlled simulations, not hardware-calibrated noise models."
                ),
                "- Only three frozen QRC reservoir seeds were available.",
                (
                    "- Financial targets are overlapping five-trading-day realised-variance "
                    "targets, so serial dependence remains."
                ),
                (
                    "- High-volatility test tails are small: 29 observations above the "
                    "training P90 threshold and 13 above P95."
                ),
                (
                    "- QRC probability calibration is limited; low top-label ECE does not "
                    "imply higher classification accuracy."
                ),
                (
                    "- Seed 2027 has a near-singular training-feature matrix, although "
                    "frozen predictions and coefficients remained finite."
                ),
                "- MNIST QRC performance is sensitive to the frozen finite-shot conditions.",
                (
                    "- Five-band MNIST compression discards within-band column detail "
                    "before reservoir processing."
                ),
                (
                    "- Transition lead time is not identifiable from the frozen aggregate "
                    "target and date fields."
                ),
                (
                    "- Post-hoc Stage 2B diagnostics are descriptive and do not replace "
                    "Stage 2A inference."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    architecture = _load_json(config.sources["financial_architecture"].path)
    mnist_environment = _load_json(config.sources["mnist_environment"].path)
    mnist_runtime = registry.get("reproducibility.mnist_exact_total_runtime_seconds_mean")[
        "display_rounded_value"
    ]
    diagnostics_runtime = registry.get("reproducibility.stage2b_runtime_seconds")[
        "display_rounded_value"
    ]
    reproduction_path = output / "reproducibility_factsheet.md"
    reproduction_path.write_text(
        "\n".join(
            [
                "# Reproducibility factsheet",
                "",
                "## Frozen identities",
                "",
                f"- Stage 2C source commit: `{config.frozen_source_commit}`.",
                f"- Final financial architecture commit: `{architecture['git']['commit']}`.",
                f"- MNIST benchmark commit: `{mnist_environment['git']['commit']}`.",
                (
                    "- Financial architecture manifest SHA-256: "
                    f"`{config.sources['financial_architecture'].sha256}`."
                ),
                (
                    "- Public-data manifest SHA-256: "
                    f"`{config.sources['public_data_manifest'].sha256}`."
                ),
                (
                    "- MNIST balanced-subset checksum: "
                    "`06c1ee9c6db87efc13ebaacc7f4406297d061d10d573731434c3f957e7c0574e`."
                ),
                "",
                "## Runtime expectations",
                "",
                (
                    "- Publication generation reads frozen artifacts and normally "
                    "completes in seconds; it invokes no model runner."
                ),
                (
                    "- The frozen two-qubit scaling row reports about 0.276 seconds "
                    "of state generation."
                ),
                (
                    f"- Frozen exact MNIST QRC execution averaged {mnist_runtime} seconds "
                    "per seed in its recorded environment."
                ),
                (
                    "- The frozen full Stage 2B diagnostic compiler completed in "
                    f"{diagnostics_runtime} seconds."
                ),
                "",
                "Runtime values are environment-specific and are not performance guarantees.",
                "",
                "## qBraid and environment assumptions",
                "",
                (
                    "- The frozen final financial and MNIST manifests record local "
                    "classical simulation and null qBraid environment identifiers."
                ),
                "- Reproducing the publication assets does not require qBraid or QPU access.",
                (
                    "- A later Stage 3 clean-room run should use the repository lock file "
                    "on a supported Python environment and record any qBraid image or "
                    "environment identifier if one is used."
                ),
                "",
                "## Stage 3 clean-room commands",
                "",
                "```bash",
                "uv sync --frozen",
                "uv run python scripts/freeze_publication_assets.py",
                "uv run python scripts/freeze_publication_assets.py",
                "uv run ruff format --check .",
                "uv run ruff check .",
                "uv run mypy src tests",
                "uv run pytest",
                "```",
                "",
                (
                    "The second generation must reproduce all tracked scientific tables, "
                    "manifests and figures byte-for-byte."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [results_path, limitations_path, reproduction_path]


def _write_caption_and_layout_files(config: PublicationConfig) -> list[Path]:
    captions = config.tracked_output_root / "figure_captions.md"
    captions.write_text(
        "\n".join(
            [
                "# Main-paper figure captions",
                "",
                (
                    "1. **Validation-based architecture selection.** Validation macro-F1, "
                    "transition PR-AUC and QLIKE are shown with computational-cost "
                    "diagnostics for the qubit and virtual-node grids. Stars identify the "
                    "validation/performance-cost choice of two qubits and two virtual nodes. "
                    "Architecture choices used validation evidence only; test results were "
                    "inspected only after selection. The curves are not evidence of "
                    "monotonic quantum scaling."
                ),
                (
                    "2. **Financial comparison and uncertainty.** Frozen test point "
                    "estimates are paired with selected Stage 2A architecture-level 95% "
                    "intervals for QRC-minus-baseline metric or loss differences. Filled "
                    "interval markers indicate Holm-adjusted support; hollow markers do not. "
                    "Logistic regression leads macro-F1, GARCH leads QLIKE, ESN leads RMSE, "
                    "and QRC is competitive on selected metrics without dominating."
                ),
                (
                    "3. **Final financial QRC robustness.** Analytic, finite-shot, "
                    "depolarising and measurement-bit-flip results are classical simulations. "
                    "Noise models are controlled and not hardware calibrated. Non-monotonic "
                    "changes are treated as stochastic/model variability; no claim is made "
                    "that noise improves performance."
                ),
                (
                    "4. **MNIST benchmark and robustness.** Exact QRC means show population "
                    "variation across three seeds; robustness rows use frozen seed 2026. ESN "
                    "outperformed QRC, and the 2,048-shot and controlled-noise conditions show "
                    "substantial degradation from analytic QRC."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    footprint = config.tracked_output_root / "page_footprint.md"
    footprint.write_text(
        "\n".join(
            [
                "# Estimated five-page footprint",
                "",
                "| Asset | Intended placement | Estimated page fraction |",
                "|---|---|---:|",
                "| Table 1 | Methods/design, one column | 0.30 |",
                "| Figure 1 | Methods/selection, page width | 0.45 |",
                "| Table 2 | Financial results, page width | 0.35 |",
                "| Figure 2 | Financial results, page width | 0.55 |",
                "| Figure 3 | Financial robustness, page width | 0.45 |",
                "| Table 3 | MNIST results, page width | 0.35 |",
                "| Figure 4 | MNIST results, page width | 0.45 |",
                "| **Total main-body asset estimate** | | **2.90 pages** |",
                "",
                (
                    "This leaves approximately 2.10 pages for problem framing, "
                    "architecture/data prose, limitations, reproducibility and conclusion. "
                    "References are excluded from the five-page limit. Appendix assets are "
                    "not counted."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [captions, footprint]


def _claim_evidence_sources(
    claim: Mapping[str, Any],
    registry: FactRegistry,
) -> list[dict[str, str]]:
    evidence_fact_ids = claim.get("evidence_fact_ids")
    if not isinstance(evidence_fact_ids, list):
        raise ValueError("claim evidence_fact_ids must be a list")
    sources = {
        (
            str(registry.get(fact_id)["source_artifact_path"]),
            str(registry.get(fact_id)["source_artifact_sha256"]),
        )
        for fact_id in evidence_fact_ids
    }
    return [{"path": path, "sha256": sha256} for path, sha256 in sorted(sources)]


def _write_final_results_manifests(
    config: PublicationConfig,
    registry: FactRegistry,
    claims: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
) -> list[Path]:
    json_path = config.tracked_output_root / "final_results_manifest.json"
    markdown_path = config.tracked_output_root / "final_results_manifest.md"
    enriched_claims = [
        {**claim, "evidence_sources": _claim_evidence_sources(claim, registry)} for claim in claims
    ]
    facts = registry.records()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "frozen_source_commit": config.frozen_source_commit,
        "selection_contract": {
            "architecture_selected_from_validation_only": True,
            "test_inspected_after_selection": True,
            "new_hypothesis_tests_conducted": False,
            "significance_source": "Stage 2A architecture-level Holm-adjusted results only",
        },
        "sources": source_records,
        "facts": facts,
        "claims": enriched_claims,
        "metric_directions": {
            "accuracy": "higher",
            "balanced_accuracy": "higher",
            "macro_f1": "higher",
            "macro_roc_auc": "higher",
            "transition_pr_auc": "higher",
            "correlation": "higher",
            "qlike": "lower",
            "rmse": "lower",
            "mae": "lower",
        },
    }
    _write_json(json_path, payload)

    lines = [
        "# Final frozen results and claims manifest",
        "",
        f"- Frozen source commit: `{config.frozen_source_commit}`",
        f"- Traced facts: {len(facts)}",
        f"- Classified claims: {len(enriched_claims)}",
        "- Significance marks: Stage 2A architecture-level Holm-adjusted evidence only",
        "",
        "## Claims",
        "",
        "| ID | Claim | Status | Qualification or rationale |",
        "|---:|---|---|---|",
    ]
    for claim in enriched_claims:
        lines.append(
            f"| {claim['claim_id']} | {claim['claim']} | {claim['status']} | {claim['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Numeric and design facts",
            "",
            (
                "| Fact ID | Exact value | Display value | Split | Scope | "
                "Holm-adjusted? | Preference | Frozen source and locator |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for fact in facts:
        exact = json.dumps(fact["exact_value"], sort_keys=True, separators=(",", ":"))
        lines.append(
            f"| `{fact['fact_id']}` | `{exact}` | {fact['display_rounded_value']} | "
            f"{fact['split']} | {fact['value_scope']} | "
            f"{str(fact['significance_adjusted']).lower()} | "
            f"{fact['metric_preference']} | `{fact['source_artifact_path']}` → "
            f"`{fact['source_column_or_key']}` |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, markdown_path]


def _copy_appendix_assets(config: PublicationConfig) -> list[Path]:
    destinations: list[Path] = []
    appendix_root = config.tracked_output_root / "appendix"
    appendix_root.mkdir(parents=True, exist_ok=True)
    for asset_id, formats in sorted(config.appendix_sources.items()):
        for extension, source in sorted(formats.items()):
            destination = appendix_root / f"{asset_id}.{extension}"
            shutil.copyfile(source.path, destination)
            destinations.append(destination)
    index = appendix_root / "README.md"
    index.write_text(
        """# Optional appendix assets

These checksum-preserved figures are excluded from the five-page main-body estimate:

- variance calibration;
- rolling QLIKE;
- Mincer-Zarnowitz diagnostics;
- frozen regime confusion matrices;
- transition-type diagnostics;
- MNIST per-digit F1;
- QRC numerical conditioning.

Each asset is copied without modification from the source listed in
`../publication_assets_manifest.json`.
""",
        encoding="utf-8",
    )
    destinations.append(index)
    return destinations


def _asset_record(
    config: PublicationConfig,
    path: Path,
    *,
    asset_id: str,
    role: str,
) -> dict[str, Any]:
    relative = path.relative_to(config.project_root).as_posix()
    return {
        "asset_id": asset_id,
        "role": role,
        "path": relative,
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _write_publication_manifest(
    config: PublicationConfig,
    *,
    selected_paths: Mapping[str, tuple[Path, str]],
    source_records: list[dict[str, Any]],
) -> Path:
    manifest_path = config.tracked_output_root / "publication_assets_manifest.json"
    selected_assets = [
        _asset_record(config, path, asset_id=asset_id, role=role)
        for asset_id, (path, role) in sorted(selected_paths.items())
    ]
    runtime = runtime_metadata()
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "publication_assets_frozen": True,
            "no_new_model_execution": True,
            "no_test_based_selection": True,
            "no_new_hypothesis_tests": True,
            "no_recalibration_or_ensembling": True,
            "quantum_advantage_claim": False,
            "generation_command": config.generation_command,
            "git_commit": config.frozen_source_commit,
            "python_version": runtime["python_version"],
            "package_versions": runtime["package_versions"],
            "operating_system": runtime["operating_system"],
            "source_artifacts": source_records,
            "selected_assets": selected_assets,
            "selected_asset_count": len(selected_assets),
            "manifest_self_checksum_excluded": True,
        },
    )
    return manifest_path


def _assert_no_absolute_local_paths(root: Path) -> None:
    text_suffixes = {".csv", ".json", ".md", ".tex"}
    forbidden = ("/" + "Users/", "\\" + "Users\\", str(root.resolve().parent))
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker and marker in text:
                raise ValueError(f"absolute local path found in tracked asset {path}: {marker}")


def _selected_path_map(
    config: PublicationConfig,
    *,
    table_paths: Mapping[str, Mapping[str, Path]],
    figure_paths: Mapping[str, Mapping[str, Path]],
    support_paths: Sequence[Path],
    appendix_paths: Sequence[Path],
) -> dict[str, tuple[Path, str]]:
    selected: dict[str, tuple[Path, str]] = {}
    for table_name, formats in table_paths.items():
        for extension, path in formats.items():
            selected[f"{table_name}_{extension}"] = (path, "main_table")
    for figure_name, formats in figure_paths.items():
        for extension, path in formats.items():
            selected[f"{figure_name}_{extension}"] = (path, "main_figure")
    for path in support_paths:
        selected[path.relative_to(config.tracked_output_root).as_posix()] = (
            path,
            "paper_support",
        )
    for path in appendix_paths:
        selected[path.relative_to(config.tracked_output_root).as_posix()] = (
            path,
            "appendix",
        )
    return selected


def freeze_publication_assets(config_path: Path) -> Path:
    """Freeze tracked publication assets without invoking any model experiment."""

    config = load_publication_config(config_path)
    sources_before = verify_publication_sources(config)
    config.tracked_output_root.mkdir(parents=True, exist_ok=True)
    config.intermediate_output_root.mkdir(parents=True, exist_ok=True)
    registry = FactRegistry(config)

    design_rows = _build_design_table(config, registry)
    financial_rows, financial_inference, _ = _build_financial_table(config, registry)
    mnist_rows, _ = _build_mnist_table(config, registry)
    qubit_rows, virtual_rows = _build_architecture_figure_data(config, registry)
    shot_rows, noise_rows = _build_financial_robustness_figure_data(config, registry)
    constraint_facts = _register_constraint_and_reproducibility_facts(config, registry)
    claims = _build_claims(registry, constraint_facts)

    table_root = config.tracked_output_root / "tables"
    table_paths = {
        TABLE_NAMES[0]: _write_table_bundle(
            directory=table_root,
            name=TABLE_NAMES[0],
            title="Frozen architecture and experiment design",
            columns=(
                ("section", "Application"),
                ("item", "Design item"),
                ("value", "Frozen value"),
            ),
            rows=design_rows,
            footnotes=(
                "All architecture and threshold values were frozen before final test inspection.",
                "QRC execution used classical simulation; no physical QPU was used.",
            ),
        ),
        TABLE_NAMES[1]: _write_table_bundle(
            directory=table_root,
            name=TABLE_NAMES[1],
            title="Frozen financial test benchmark",
            columns=(
                ("model", "Model"),
                ("macro_f1", "Macro-F1 ↑"),
                ("balanced_accuracy", "Balanced acc. ↑"),
                ("transition_pr_auc", "Transition PR-AUC ↑"),
                ("qlike", "QLIKE ↓"),
                ("rmse", "RMSE ↓"),
                ("mae", "MAE ↓"),
                ("correlation", "Correlation ↑"),
                ("holm_annotation", "Holm-adjusted evidence vs QRC"),
            ),
            rows=financial_rows,
            footnotes=(
                (
                    "Lower QLIKE, RMSE and MAE is better; higher classification metrics "
                    "and correlation is better."
                ),
                "QRC values are arithmetic means across the three frozen reservoir seeds.",
                (
                    "Significance statements use paired time-series-aware Stage 2A tests "
                    "after Holm adjustment; raw-p-only findings are not marked significant."
                ),
            ),
        ),
        TABLE_NAMES[2]: _write_table_bundle(
            directory=table_root,
            name=TABLE_NAMES[2],
            title="Frozen MNIST benchmark and QRC robustness",
            columns=(
                ("section", "Section"),
                ("model_or_condition", "Model or condition"),
                ("accuracy", "Accuracy ↑"),
                ("macro_f1", "Macro-F1 ↑"),
                ("balanced_accuracy", "Balanced acc. ↑"),
                ("macro_roc_auc", "Macro ROC-AUC ↑"),
                ("holm_annotation", "Inference or scope"),
            ),
            rows=mnist_rows,
            footnotes=(
                "Exact QRC is mean ± population SD across seeds 2026-2028.",
                (
                    "QRC versus flattened logistic was not Holm-significant; ESN was "
                    "significantly higher than QRC."
                ),
                (
                    "Robustness rows use frozen seed-2026 predictions and were not "
                    "significance-tested."
                ),
            ),
        ),
    }

    figure_root = config.tracked_output_root / "figures"
    comparison_rows = [
        {
            "model": row["model_or_condition"],
            "accuracy": row["accuracy"],
            "accuracy_sd": row["accuracy_sd"],
            "macro_f1": row["macro_f1"],
            "macro_f1_sd": row["macro_f1_sd"],
        }
        for row in mnist_rows
        if row["section"] == "Model comparison"
    ]
    robustness_labels = {
        "Analytic (seed 2026)": "analytic",
        "2,048 shots (seed 2026)": "2,048 shots",
        "2,048 shots + depolarising 0.01": "depol. 0.01",
        "2,048 shots + measurement flip 0.02": "bit flip 0.02",
    }
    mnist_robustness_rows = [
        {
            "condition": robustness_labels[str(row["model_or_condition"])],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
        }
        for row in mnist_rows
        if row["section"] == "QRC robustness"
    ]
    figure_paths = {
        FIGURE_NAMES[0]: architecture_selection_figure(
            qubit_rows=qubit_rows,
            virtual_rows=virtual_rows,
            destination=figure_root / FIGURE_NAMES[0],
            dpi=config.dpi,
        ),
        FIGURE_NAMES[1]: financial_comparison_figure(
            benchmark_rows=financial_rows,
            inference=financial_inference,
            destination=figure_root / FIGURE_NAMES[1],
            dpi=config.dpi,
        ),
        FIGURE_NAMES[2]: financial_robustness_figure(
            shot_rows=shot_rows,
            noise_rows=noise_rows,
            destination=figure_root / FIGURE_NAMES[2],
            dpi=config.dpi,
        ),
        FIGURE_NAMES[3]: mnist_benchmark_figure(
            comparison_rows=comparison_rows,
            robustness_rows=mnist_robustness_rows,
            destination=figure_root / FIGURE_NAMES[3],
            dpi=config.dpi,
        ),
    }
    appendix_paths = _copy_appendix_assets(config)
    support_paths = [
        *_write_factsheets(config, registry),
        *_write_caption_and_layout_files(config),
        *_write_final_results_manifests(
            config,
            registry,
            claims,
            sources_before,
        ),
    ]

    sources_after = verify_publication_sources(config)
    if sources_after != sources_before:
        raise RuntimeError("a frozen publication source changed during asset generation")
    _write_json(
        config.intermediate_output_root / "source_verification.json",
        {
            "schema_version": 1,
            "sources": sources_after,
            "unchanged_during_generation": True,
        },
    )
    _write_json(
        config.intermediate_output_root / "resolved_facts.json",
        {"schema_version": 1, "facts": registry.records()},
    )
    _write_json(
        config.intermediate_output_root / "claims.json",
        {"schema_version": 1, "claims": claims},
    )

    selected = _selected_path_map(
        config,
        table_paths=table_paths,
        figure_paths=figure_paths,
        support_paths=support_paths,
        appendix_paths=appendix_paths,
    )
    manifest = _write_publication_manifest(
        config,
        selected_paths=selected,
        source_records=sources_after,
    )
    _assert_no_absolute_local_paths(config.tracked_output_root)
    _write_json(
        config.intermediate_output_root / "generation_summary.json",
        {
            "schema_version": 1,
            "status": "success",
            "study_id": STUDY_ID,
            "tracked_asset_count_excluding_self_manifest": len(selected),
            "table_count": len(table_paths),
            "figure_count": len(figure_paths),
            "claim_count": len(claims),
            "fact_count": len(registry.records()),
            "publication_manifest": manifest.relative_to(config.project_root).as_posix(),
            "no_new_model_execution": True,
        },
    )
    return manifest
