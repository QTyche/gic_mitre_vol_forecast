"""Isolated feature generation for finite-shot and simulated-noise QRC studies."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.backends import BACKEND_NAME, BACKEND_VERSION
from qtyche_qrc.models.qrc.encoding import (
    array_checksum,
    initial_density_matrix,
    input_projection,
    reset_and_encode_input,
)
from qtyche_qrc.models.qrc.features import (
    FeatureCacheIntegrityError,
    QRCFeatureBundle,
    feature_column_checksum,
)
from qtyche_qrc.models.qrc.hamiltonian import HamiltonianDefinition, generate_hamiltonian
from qtyche_qrc.models.qrc.noise import (
    QRCMeasurementConfig,
    apply_local_depolarizing_channel,
    measurement_rng,
    sample_commuting_observables,
)
from qtyche_qrc.models.qrc.observables import ObservableSet
from qtyche_qrc.models.qrc.reservoir import QRCConfig

ROBUST_FEATURE_VERSION = "1.0"


@dataclass(frozen=True)
class RobustFeatureCacheKey:
    """Collision-proof identity for stochastic/noisy reservoir features."""

    processed_data_manifest_checksum: str
    feature_column_checksum: str
    qrc_configuration_checksum: str
    measurement_configuration_checksum: str
    reservoir_seed: int
    measurement_seed: int | None
    backend_version: str
    robust_feature_version: str

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_robust_feature_cache_key(
    *,
    processed_data_manifest_checksum: str,
    feature_names: tuple[str, ...],
    qrc_config: QRCConfig,
    measurement_config: QRCMeasurementConfig,
) -> RobustFeatureCacheKey:
    """Build a deterministic cache key including both RNG and channel settings."""

    qrc_config.validate()
    measurement_config.validate()
    return RobustFeatureCacheKey(
        processed_data_manifest_checksum=processed_data_manifest_checksum,
        feature_column_checksum=feature_column_checksum(feature_names),
        qrc_configuration_checksum=qrc_config.checksum,
        measurement_configuration_checksum=measurement_config.checksum,
        reservoir_seed=qrc_config.reservoir_seed,
        measurement_seed=measurement_config.measurement_seed,
        backend_version=BACKEND_VERSION,
        robust_feature_version=ROBUST_FEATURE_VERSION,
    )


class RobustQuantumReservoir:
    """Exact density-matrix QRC with optional simulated channels and joint shots."""

    def __init__(
        self,
        input_size: int,
        qrc_config: QRCConfig,
        measurement_config: QRCMeasurementConfig,
    ) -> None:
        qrc_config.validate()
        measurement_config.validate()
        if input_size <= 0:
            raise ValueError("QRC input_size must be positive")
        self.input_size = input_size
        self.qrc_config = qrc_config
        self.measurement_config = measurement_config
        self.hamiltonian: HamiltonianDefinition = generate_hamiltonian(
            qrc_config.n_qubits,
            seed=qrc_config.reservoir_seed,
            graph=qrc_config.graph,
            chords=qrc_config.chords,
            j_strength=qrc_config.j_strength,
            h_strength=qrc_config.h_strength,
            h_min_factor=qrc_config.h_min_factor,
            h_max_factor=qrc_config.h_max_factor,
        )
        from qtyche_qrc.models.qrc.backends import ExactDensityMatrixBackend

        self.backend = ExactDensityMatrixBackend(
            self.hamiltonian.matrix,
            qrc_config.n_qubits,
            qrc_config.delta_tau,
        )
        self.input_projection = input_projection(
            qrc_config.virtual_nodes,
            input_size,
            qrc_config.reservoir_seed,
        )
        self.observables = ObservableSet.build(
            qrc_config.n_qubits,
            self.hamiltonian.edges,
            qrc_config.virtual_nodes,
        )
        self._state = initial_density_matrix(qrc_config.n_qubits)
        self._rng = (
            measurement_rng(measurement_config.measurement_seed, stream=0)
            if measurement_config.measurement_seed is not None
            else None
        )
        self._bit_flip_rng = (
            measurement_rng(measurement_config.measurement_seed, stream=1)
            if measurement_config.measurement_seed is not None
            else None
        )
        self._state_generation_seconds = 0.0
        self._sampling_seconds = 0.0
        self._sample_batches = 0
        self._sampled_bitstrings = 0
        self._connected_absolute_sum = 0.0
        self._connected_count = 0

    def reset_state(self) -> None:
        """Reset the density matrix without rewinding the measurement RNG."""

        self._state = initial_density_matrix(self.qrc_config.n_qubits)

    def get_state(self) -> NDArray[np.complex128]:
        return self._state.copy()

    def step(self, input_row: NDArray[np.float64]) -> NDArray[np.float64]:
        """Process one input using a single joint bitstring batch per virtual node."""

        row = np.asarray(input_row, dtype=float).reshape(-1)
        if row.shape != (self.input_size,) or not np.isfinite(row).all():
            raise ValueError("QRC input row has the wrong shape or non-finite values")
        features: list[float] = []
        for projection_row in self.input_projection:
            state_started = time.perf_counter()
            theta = float(self.qrc_config.input_scaling * np.dot(projection_row, row))
            reset_state = reset_and_encode_input(
                self._state,
                theta,
                self.qrc_config.n_qubits,
                q_in=0,
            )
            self._state = self.backend.validate_state(reset_state, context="reset state")
            self._state = self.backend.evolve(self._state)
            if self.measurement_config.depolarizing_probability > 0.0:
                self._state = apply_local_depolarizing_channel(
                    self._state,
                    n_qubits=self.qrc_config.n_qubits,
                    probability=self.measurement_config.depolarizing_probability,
                )
                self._state = self.backend.validate_state(
                    self._state,
                    context="locally depolarized state",
                )
            self._state_generation_seconds += time.perf_counter() - state_started

            if self.measurement_config.shots is None:
                values, connected = self.observables.expectations(self._state)
            else:
                if self._rng is None:
                    raise RuntimeError("finite-shot reservoir has no measurement RNG")
                sampling_started = time.perf_counter()
                values, connected = sample_commuting_observables(
                    self._state,
                    n_qubits=self.qrc_config.n_qubits,
                    edges=self.hamiltonian.edges,
                    shots=self.measurement_config.shots,
                    rng=self._rng,
                    measurement_bit_flip_probability=(
                        self.measurement_config.measurement_bit_flip_probability
                    ),
                    bit_flip_rng=self._bit_flip_rng,
                )
                self._sampling_seconds += time.perf_counter() - sampling_started
                self._sample_batches += 1
                self._sampled_bitstrings += self.measurement_config.shots
            features.extend(float(value) for value in values)
            self._connected_absolute_sum += float(np.sum(np.abs(connected)))
            self._connected_count += len(connected)
        return np.asarray(features, dtype=float)

    def transform(
        self,
        inputs: NDArray[np.float64],
        *,
        reset: bool = False,
        reset_each_input: bool = False,
    ) -> NDArray[np.float64]:
        """Generate chronological features without consuming labels."""

        values = np.asarray(inputs, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError("QRC input matrix has the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError("QRC inputs must be finite")
        if reset:
            self.reset_state()
        output = np.empty((len(values), self.observables.raw_feature_dimension), dtype=float)
        for index, row in enumerate(values):
            if reset_each_input:
                self.reset_state()
            output[index] = self.step(row)
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.backend.diagnostics.as_dict(),
            "mean_absolute_connected_correlation": (
                self._connected_absolute_sum / self._connected_count
                if self._connected_count
                else None
            ),
            "sample_batches": self._sample_batches,
            "sampled_bitstrings": self._sampled_bitstrings,
        }

    def resource_metadata(self) -> dict[str, Any]:
        backend = self.backend.metadata()
        exact_noiseless = (
            self.measurement_config.analytic_expectations
            and self.measurement_config.depolarizing_probability == 0.0
            and self.measurement_config.measurement_bit_flip_probability == 0.0
        )
        return {
            **backend,
            "backend": f"{BACKEND_NAME}_controlled_measurement",
            "backend_version": BACKEND_VERSION,
            "exact_state_evolution": True,
            "analytic_expectations": self.measurement_config.analytic_expectations,
            "exact_noiseless": exact_noiseless,
            "finite_shots": self.measurement_config.shots is not None,
            "noiseless": (
                self.measurement_config.depolarizing_probability == 0.0
                and self.measurement_config.measurement_bit_flip_probability == 0.0
            ),
            "physical_noise": False,
            "hardware_execution": False,
            "controlled_noise_simulation": (
                self.measurement_config.depolarizing_probability > 0.0
                or self.measurement_config.measurement_bit_flip_probability > 0.0
            ),
            "measurement_configuration": self.measurement_config.metadata(),
            "measurement_configuration_checksum": self.measurement_config.checksum,
            "reservoir_seed": self.qrc_config.reservoir_seed,
            "state_policy": self.qrc_config.state_policy,
            "configuration_checksum": self.qrc_config.checksum,
            "hamiltonian_checksum": self.hamiltonian.checksum,
            "input_projection_shape": list(self.input_projection.shape),
            "input_projection_checksum": array_checksum(self.input_projection),
            "raw_feature_dimension": self.observables.raw_feature_dimension,
            "observable_checksum": self.observables.checksum,
            "state_generation_seconds": self._state_generation_seconds,
            "sampling_seconds": self._sampling_seconds,
            "sample_batches": self._sample_batches,
            "sampled_bitstrings": self._sampled_bitstrings,
            "labels_consumed": False,
            "explicit_feedback": False,
        }


def split_robust_qrc_features(
    reservoir: RobustQuantumReservoir,
    X_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    X_test: NDArray[np.float64],
    state_policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply the frozen temporal policy while preserving one RNG stream."""

    if state_policy == "carry_inputs":
        train = reservoir.transform(X_train, reset=True)
        validation = reservoir.transform(X_validation, reset=False)
        test = reservoir.transform(X_test, reset=False)
    elif state_policy == "reset":
        train = reservoir.transform(X_train, reset=True)
        validation = reservoir.transform(X_validation, reset=True)
        test = reservoir.transform(X_test, reset=True)
    elif state_policy == "reset_each_input":
        train = reservoir.transform(X_train, reset_each_input=True)
        validation = reservoir.transform(X_validation, reset_each_input=True)
        test = reservoir.transform(X_test, reset_each_input=True)
    else:
        raise ValueError("state_policy must be reset, carry_inputs, or reset_each_input")
    return train, validation, test


def _cache_paths(
    cache_root: Path,
    key: RobustFeatureCacheKey,
) -> tuple[Path, Path, Path]:
    directory = cache_root / key.checksum
    return directory, directory / "qrc_features.npz", directory / "metadata.json"


def load_robust_feature_cache(
    cache_root: Path,
    key: RobustFeatureCacheKey,
) -> QRCFeatureBundle | None:
    """Load and checksum-verify an isolated robustness feature bundle."""

    directory, arrays_path, metadata_path = _cache_paths(cache_root, key)
    if not directory.exists():
        return None
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FeatureCacheIntegrityError(f"incomplete robustness feature cache: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("cache_key") != asdict(key)
        or metadata.get("cache_key_checksum") != key.checksum
    ):
        raise FeatureCacheIntegrityError("robustness feature cache key checksum mismatch")
    with np.load(arrays_path) as values:
        arrays = {
            split: np.asarray(values[split], dtype=float)
            for split in ("train", "validation", "test")
        }
    expected = metadata.get("array_checksums")
    if not isinstance(expected, dict):
        raise FeatureCacheIntegrityError("robustness feature cache omits array checksums")
    for split, values in arrays.items():
        actual = array_checksum(values)
        if actual != expected.get(split):
            raise FeatureCacheIntegrityError(
                f"robustness feature checksum mismatch for {split}: "
                f"{actual} != {expected.get(split)}"
            )
    return QRCFeatureBundle(
        arrays["train"],
        arrays["validation"],
        arrays["test"],
        metadata,
        directory,
        True,
    )


def generate_or_load_robust_features(
    *,
    cache_root: Path,
    key: RobustFeatureCacheKey,
    feature_names: tuple[str, ...],
    qrc_config: QRCConfig,
    measurement_config: QRCMeasurementConfig,
    X_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> QRCFeatureBundle:
    """Generate one label-free feature point and persist timing and checksums."""

    cached = load_robust_feature_cache(cache_root, key)
    if cached is not None:
        return cached
    reservoir = RobustQuantumReservoir(
        len(feature_names),
        qrc_config,
        measurement_config,
    )
    train, validation, test = split_robust_qrc_features(
        reservoir,
        X_train,
        X_validation,
        X_test,
        qrc_config.state_policy,
    )
    directory, arrays_path, metadata_path = _cache_paths(cache_root, key)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(arrays_path, train=train, validation=validation, test=test)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "cache_key": asdict(key),
        "cache_key_checksum": key.checksum,
        "label_free_generation": True,
        "target_columns_consumed": [],
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "array_checksums": {
            "train": array_checksum(train),
            "validation": array_checksum(validation),
            "test": array_checksum(test),
        },
        "observable_metadata": reservoir.observables.metadata(),
        "observable_checksum": reservoir.observables.checksum,
        "resource_metadata": reservoir.resource_metadata(),
        "numerical_diagnostics": reservoir.diagnostics(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return QRCFeatureBundle(train, validation, test, metadata, directory, False)
