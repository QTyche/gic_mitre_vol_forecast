"""Small dependency-free loader for the genuine official MNIST IDX partitions."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.encoding import array_checksum

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
MNIST_TRAIN_SIZE = 60_000
MNIST_TEST_SIZE = 10_000
MNIST_IMAGE_SHAPE = (28, 28)
DIGITS = tuple(range(10))
COLUMN_BANDS = ((0, 6), (6, 12), (12, 18), (18, 23), (23, 28))
TEMPORAL_WINDOWS = ((0, 7), (7, 14), (14, 21), (21, 28))


@dataclass(frozen=True)
class MNISTSourceFile:
    """One checksum-pinned compressed official IDX source."""

    key: str
    filename: str
    url: str
    sha256: str
    md5: str
    bytes: int


@dataclass(frozen=True)
class MNISTOfficialPartitions:
    """The unchanged official training and test partitions."""

    train_images: NDArray[np.uint8]
    train_labels: NDArray[np.uint8]
    test_images: NDArray[np.uint8]
    test_labels: NDArray[np.uint8]
    source_manifest: dict[str, Any]


@dataclass(frozen=True)
class MNISTSelectedSplit:
    """One balanced subset with official partition indices and row sequences."""

    name: str
    source_partition: str
    official_indices: NDArray[np.int64]
    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    sequences: NDArray[np.float64]


@dataclass(frozen=True)
class MNISTBenchmarkData:
    """Train, validation, and test subsets plus complete deterministic provenance."""

    train: MNISTSelectedSplit
    validation: MNISTSelectedSplit
    test: MNISTSelectedSplit
    source_manifest: dict[str, Any]
    index_manifest: dict[str, Any]
    preprocessing_manifest: dict[str, Any]
    subset_checksum: str
    is_synthetic: bool = False


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_path(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source_files(raw: object) -> tuple[MNISTSourceFile, ...]:
    """Parse the checksum-pinned file mapping from the benchmark configuration."""

    if not isinstance(raw, dict) or set(raw) != {
        "train_images",
        "train_labels",
        "test_images",
        "test_labels",
    }:
        raise ValueError("MNIST configuration must define all four official IDX files")
    files: list[MNISTSourceFile] = []
    for key in ("train_images", "train_labels", "test_images", "test_labels"):
        record = raw[key]
        if not isinstance(record, dict):
            raise ValueError(f"dataset.files.{key} must be a mapping")
        files.append(
            MNISTSourceFile(
                key=key,
                filename=str(record["filename"]),
                url=str(record["url"]),
                sha256=str(record["sha256"]),
                md5=str(record["md5"]),
                bytes=int(record["bytes"]),
            )
        )
    return tuple(files)


def _verified_file(path: Path, source: MNISTSourceFile) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing official MNIST file {path}; run "
            "`python scripts/run_qrc_mnist.py --download-only`"
        )
    size = path.stat().st_size
    sha256 = sha256_path(path)
    md5 = md5_path(path)
    if size != source.bytes or sha256 != source.sha256 or md5 != source.md5:
        raise ValueError(
            f"MNIST source checksum mismatch for {path}: bytes={size}, sha256={sha256}, md5={md5}"
        )
    return {
        **asdict(source),
        "cache_filename": path.name,
        "verified": True,
    }


def download_mnist(cache_dir: Path, sources: tuple[MNISTSourceFile, ...]) -> dict[str, Any]:
    """Download missing official files atomically and verify every pinned checksum."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for source in sources:
        destination = cache_dir / source.filename
        if not destination.is_file():
            temporary = cache_dir / f".{source.filename}.partial"
            request = urllib.request.Request(
                source.url,
                headers={"User-Agent": "qtyche-qrc-mnist/1.0"},
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as handle,
            ):
                while block := response.read(1024 * 1024):
                    handle.write(block)
            os.replace(temporary, destination)
        records[source.key] = _verified_file(destination, source)
    return {
        "schema_version": 1,
        "dataset": "MNIST",
        "provider": "Google-hosted mirror of original Yann LeCun MNIST IDX files",
        "official_partitions_preserved": True,
        "files": records,
    }


def verify_mnist_sources(cache_dir: Path, sources: tuple[MNISTSourceFile, ...]) -> dict[str, Any]:
    """Verify an already-cached official MNIST download without network access."""

    return {
        "schema_version": 1,
        "dataset": "MNIST",
        "provider": "Google-hosted mirror of original Yann LeCun MNIST IDX files",
        "official_partitions_preserved": True,
        "files": {
            source.key: _verified_file(cache_dir / source.filename, source) for source in sources
        },
    }


def read_idx_images(path: Path, *, expected_count: int | None = None) -> NDArray[np.uint8]:
    """Read a gzip-compressed IDX image file with strict header and length checks."""

    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        payload = handle.read()
    if magic != MNIST_IMAGE_MAGIC or (rows, columns) != MNIST_IMAGE_SHAPE:
        raise ValueError(f"invalid MNIST IDX image header in {path}")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"MNIST image count changed in {path}: {count}")
    expected_bytes = count * rows * columns
    if len(payload) != expected_bytes:
        raise ValueError(f"MNIST IDX image payload length mismatch in {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns).copy()


def read_idx_labels(path: Path, *, expected_count: int | None = None) -> NDArray[np.uint8]:
    """Read a gzip-compressed IDX label file and require labels 0 through 9."""

    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {path}")
        magic, count = struct.unpack(">II", header)
        payload = handle.read()
    if magic != MNIST_LABEL_MAGIC:
        raise ValueError(f"invalid MNIST IDX label header in {path}")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"MNIST label count changed in {path}: {count}")
    if len(payload) != count:
        raise ValueError(f"MNIST IDX label payload length mismatch in {path}")
    labels = np.frombuffer(payload, dtype=np.uint8).copy()
    if set(np.unique(labels).tolist()) != set(DIGITS):
        raise ValueError("official MNIST labels must cover digits 0 through 9")
    return labels


def load_official_mnist(
    cache_dir: Path,
    sources: tuple[MNISTSourceFile, ...],
    *,
    download: bool = False,
) -> MNISTOfficialPartitions:
    """Load checksum-verified genuine MNIST without alternate-dataset fallback."""

    manifest = (
        download_mnist(cache_dir, sources) if download else verify_mnist_sources(cache_dir, sources)
    )
    paths = {source.key: cache_dir / source.filename for source in sources}
    train_images = read_idx_images(paths["train_images"], expected_count=MNIST_TRAIN_SIZE)
    train_labels = read_idx_labels(paths["train_labels"], expected_count=MNIST_TRAIN_SIZE)
    test_images = read_idx_images(paths["test_images"], expected_count=MNIST_TEST_SIZE)
    test_labels = read_idx_labels(paths["test_labels"], expected_count=MNIST_TEST_SIZE)
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("official MNIST images and labels do not align")
    return MNISTOfficialPartitions(
        train_images,
        train_labels,
        test_images,
        test_labels,
        manifest,
    )


def deterministic_stratified_indices(
    train_labels: NDArray[np.integer[Any]],
    test_labels: NDArray[np.integer[Any]],
    *,
    train_per_digit: int,
    validation_per_digit: int,
    test_per_digit: int,
    seed: int,
) -> dict[str, NDArray[np.int64]]:
    """Select balanced, disjoint subsets from the unchanged official partitions."""

    if min(train_per_digit, validation_per_digit, test_per_digit) <= 0:
        raise ValueError("MNIST per-digit subset sizes must be positive")
    train_values = np.asarray(train_labels, dtype=int).reshape(-1)
    test_values = np.asarray(test_labels, dtype=int).reshape(-1)
    if set(np.unique(train_values)) != set(DIGITS) or set(np.unique(test_values)) != set(DIGITS):
        raise ValueError("MNIST subset selection requires all ten genuine digit labels")
    rng = np.random.default_rng(seed)
    selected_train: list[NDArray[np.int64]] = []
    selected_validation: list[NDArray[np.int64]] = []
    selected_test: list[NDArray[np.int64]] = []
    for digit in DIGITS:
        official_train = np.flatnonzero(train_values == digit)
        if len(official_train) < train_per_digit + validation_per_digit:
            raise ValueError(f"official training partition has too few digit-{digit} rows")
        ordered_train = np.asarray(rng.permutation(official_train), dtype=np.int64)
        selected_train.append(ordered_train[:train_per_digit])
        selected_validation.append(
            ordered_train[train_per_digit : train_per_digit + validation_per_digit]
        )
        official_test = np.flatnonzero(test_values == digit)
        if len(official_test) < test_per_digit:
            raise ValueError(f"official test partition has too few digit-{digit} rows")
        selected_test.append(
            np.asarray(rng.permutation(official_test)[:test_per_digit], dtype=np.int64)
        )
    return {
        "train": np.sort(np.concatenate(selected_train)),
        "validation": np.sort(np.concatenate(selected_validation)),
        "test": np.sort(np.concatenate(selected_test)),
    }


def compress_image_rows(
    images: NDArray[np.integer[Any]],
    bands: tuple[tuple[int, int], ...] = COLUMN_BANDS,
) -> NDArray[np.float64]:
    """Scale to [0,1] and average each image row over five contiguous bands."""

    values = np.asarray(images)
    if values.ndim != 3 or tuple(values.shape[1:]) != MNIST_IMAGE_SHAPE:
        raise ValueError("MNIST images must have shape (n, 28, 28)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("MNIST source images must be integer pixels")
    if values.min(initial=0) < 0 or values.max(initial=0) > 255:
        raise ValueError("MNIST source pixels must lie in [0,255]")
    if bands != COLUMN_BANDS:
        raise ValueError(f"MNIST five-band boundaries must remain {COLUMN_BANDS}")
    scaled = np.asarray(values, dtype=float) / 255.0
    compressed = np.stack(
        [scaled[:, :, start:end].mean(axis=2) for start, end in bands],
        axis=2,
    )
    if compressed.shape != (len(values), 28, 5):
        raise ValueError("MNIST row compression produced the wrong sequence shape")
    if not np.isfinite(compressed).all() or np.any((compressed < 0) | (compressed > 1)):
        raise ValueError("MNIST row compression produced values outside [0,1]")
    return np.asarray(compressed, dtype=float)


def class_counts(labels: NDArray[np.integer[Any]]) -> dict[str, int]:
    values = np.asarray(labels, dtype=int).reshape(-1)
    return {str(digit): int(np.sum(values == digit)) for digit in DIGITS}


def _selected_split(
    name: str,
    source_partition: str,
    indices: NDArray[np.int64],
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
) -> MNISTSelectedSplit:
    selected_images = np.asarray(images[indices], dtype=np.uint8)
    selected_labels = np.asarray(labels[indices], dtype=np.int64)
    return MNISTSelectedSplit(
        name=name,
        source_partition=source_partition,
        official_indices=indices,
        images=selected_images,
        labels=selected_labels,
        sequences=compress_image_rows(selected_images),
    )


def _json_checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_mnist_benchmark_data(
    official: MNISTOfficialPartitions,
    *,
    train_per_digit: int,
    validation_per_digit: int,
    test_per_digit: int,
    seed: int,
) -> MNISTBenchmarkData:
    """Build the deterministic balanced benchmark and all selection checksums."""

    indices = deterministic_stratified_indices(
        official.train_labels,
        official.test_labels,
        train_per_digit=train_per_digit,
        validation_per_digit=validation_per_digit,
        test_per_digit=test_per_digit,
        seed=seed,
    )
    if np.intersect1d(indices["train"], indices["validation"]).size:
        raise ValueError("MNIST training and validation official indices overlap")
    train = _selected_split(
        "train",
        "official_train",
        indices["train"],
        official.train_images,
        official.train_labels,
    )
    validation = _selected_split(
        "validation",
        "official_train",
        indices["validation"],
        official.train_images,
        official.train_labels,
    )
    test = _selected_split(
        "test",
        "official_test",
        indices["test"],
        official.test_images,
        official.test_labels,
    )
    index_manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection_seed": seed,
        "selection_algorithm": (
            "digit-ordered NumPy PCG64 permutation; train then validation from "
            "official training partition, test from official test partition"
        ),
        "official_partition_identity": {
            "train_and_validation": "official_train",
            "test": "official_test",
        },
        "splits": {},
    }
    for split in (train, validation, test):
        index_manifest["splits"][split.name] = {
            "source_partition": split.source_partition,
            "official_indices": split.official_indices.tolist(),
            "official_indices_checksum": array_checksum(split.official_indices),
            "labels_checksum": array_checksum(split.labels),
            "images_checksum": array_checksum(split.images),
            "sequences_checksum": array_checksum(split.sequences),
            "class_counts": class_counts(split.labels),
            "rows": len(split.labels),
        }
    index_manifest["checksum"] = _json_checksum(index_manifest)
    preprocessing_manifest: dict[str, Any] = {
        "schema_version": 1,
        "learned_parameters": False,
        "validation_or_test_fit": False,
        "pixel_scaling": "uint8 / 255.0",
        "output_range": [0.0, 1.0],
        "image_shape": list(MNIST_IMAGE_SHAPE),
        "sequence_shape_per_image": [28, 5],
        "column_bands_half_open": [list(value) for value in COLUMN_BANDS],
        "band_widths": [end - start for start, end in COLUMN_BANDS],
        "temporal_windows_half_open": [list(value) for value in TEMPORAL_WINDOWS],
    }
    preprocessing_manifest["checksum"] = _json_checksum(preprocessing_manifest)
    subset_checksum = _json_checksum(
        {
            "source_files": {
                key: value["sha256"] for key, value in official.source_manifest["files"].items()
            },
            "selection_checksum": index_manifest["checksum"],
            "preprocessing_checksum": preprocessing_manifest["checksum"],
        }
    )
    return MNISTBenchmarkData(
        train,
        validation,
        test,
        official.source_manifest,
        index_manifest,
        preprocessing_manifest,
        subset_checksum,
    )
