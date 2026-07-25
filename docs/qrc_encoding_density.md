# Exact QRC temporal-multiplexing and encoding-density study

## Purpose and controlled variable

This experiment asks how temporal readout density affects forecasting quality,
feature dimension, numerical conditioning, and runtime for a fixed two-qubit
reservoir. The only architectural variable is `virtual_nodes`:

```text
V = 1, 2, 4, 8
```

Each density is evaluated with independent reservoir seeds 2026, 2027, and
2028. The public snapshot, chronological splits, training-only preprocessing,
forecasting targets, validation-only ridge selection, Hamiltonian family,
two-qubit ring, interaction and field strengths, input scaling, partial
input-qubit reset, `carry_inputs` state policy, and exact noiseless backend
remain fixed. Classifier and regressor readouts share one label-free feature
cache for every `(V, reservoir_seed)` point. Test data are evaluated only after
the readout is selected and frozen.

## Temporal-multiplexing contract

The total evolution interval for one market input is always `tau=1.0`. For `V`
virtual nodes, the exact backend uses

```text
delta_tau = tau / V
sampling times = delta_tau, 2 delta_tau, ..., V delta_tau = tau
```

The current encoding semantics apply one seeded projection row and the same
partial reset/reinjection channel before each substep, then evolve for
`delta_tau` and record the same observables. The state after the final substep
is carried to the next market input. Increasing `V` therefore increases both
the within-interval encoding/readout density and the number of recorded
features; it does not lengthen the physical evolution interval.

For the two-qubit ring there are two `Z_i` values and one unique `Z_0 Z_1`
value at every temporal point. The expected raw dimensions are therefore 3, 6,
12, and 24 for `V=1,2,4,8`.

## Commands

Run the compact one-seed workflow:

```bash
python scripts/run_qrc_encoding_density.py \
  --virtual-nodes 1 2 8 \
  --seeds 2026 \
  --smoke
```

Run the complete four-density, three-seed experiment:

```bash
python scripts/run_qrc_encoding_density.py \
  --virtual-nodes 1 2 4 8 \
  --seeds 2026 2027 2028
```

The thin shell wrapper accepts the same arguments:

```bash
./scripts/run_qrc_encoding_density.sh --smoke
```

Runs resume by default. A complete matching task is reused; an incomplete or
missing classifier/regressor partner is run without discarding completed
work. `--no-resume` creates new readout runs while reusing the checksum-keyed
features. Smoke mode permits at most three densities and one seed.

Before any model is fitted, the runner verifies the frozen readout
configuration hashes, every raw snapshot checksum, every processed-data
checksum, the public snapshot identity, and the non-synthetic data marker.

## Output contract

Generated artifacts are isolated below `results/qrc_encoding_density/` and are
ignored by Git:

```text
results/qrc_encoding_density/
├── encoding_density_run_summary.json
├── encoding_density_state.json
├── condition_diagnostics/
├── feature_cache/
├── generated_configs/
├── runs/
│   └── <timestamp>_<model>_<task>_seed<seed>/
├── tables/
│   ├── qrc_encoding_density_per_run.{json,csv}
│   ├── qrc_encoding_density_aggregate.{json,csv}
│   ├── qrc_encoding_density_validation_candidates.{json,csv}
│   └── qrc_encoding_density_resources.{json,csv}
└── figures/
    ├── validation_macro_f1_vs_virtual_nodes.{png,pdf}
    ├── validation_transition_pr_auc_vs_virtual_nodes.{png,pdf}
    ├── validation_qlike_vs_virtual_nodes.{png,pdf}
    ├── test_macro_f1_vs_virtual_nodes.{png,pdf}
    ├── test_transition_pr_auc_vs_virtual_nodes.{png,pdf}
    ├── test_qlike_vs_virtual_nodes.{png,pdf}
    ├── feature_dimension_vs_virtual_nodes.{png,pdf}
    ├── feature_generation_time_vs_virtual_nodes.{png,pdf}
    └── condition_number_vs_virtual_nodes.{png,pdf}
```

The split-level per-run tables include all task metrics; selected ridge alpha;
feature and readout shapes; state-generation, feature-generation, readout-fit,
and attributed total runtime; feature-cache and configuration checksums; raw
and processed data checksums; training-feature rank and condition diagnostics;
Git, Python, package, and platform provenance; and the temporal-policy
assertions. In the exact generator, the state-generation timer covers the
complete state evolution plus observable feature construction, so
`feature_generation_seconds` and `state_generation_seconds` intentionally
refer to the same measured operation.

The aggregate table reports mean, population standard deviation, minimum,
maximum, contributing seeds, and seed count for each density, task, split, and
metric. Figures show every seed plus the mean and population-standard-deviation
uncertainty. A smoke run has one seed and therefore zero displayed seed
uncertainty.

The resource table estimates:

- dense exact-state workspace for three simultaneous complex128 matrices;
- the uncompressed float64 cache for all selected rows; and
- the float64 training matrix used for conditioning diagnostics.

These estimates exclude Python/SciPy overhead, eigensolver workspace, compressed
file overhead, predictions, plots, and fitted readouts. Practical process
memory is higher.

## Reference agreement and architecture evidence

The `V=2` QRC configuration has the same cache key as the existing exact
two-qubit qubit-scaling point. When that prior cache is available, the runner
compares train, validation, and test arrays and records checksums, maximum
absolute difference, and tolerance status. The unit tests also compare the two
independently derived configurations and their exact feature arrays.

The candidate table contains validation macro-F1, validation transition
PR-AUC, validation QLIKE, feature dimension, feature-generation time,
conditioning, and across-seed variability. It contains no test-driven
selection and explicitly marks `architecture_frozen=false`, because the
state-memory ablation remains outstanding.

## Interpretation

The experiment supports descriptive statements about sensitivity to the
implemented temporal encoding/readout density. Because increasing `V` changes
the number of substep input projections and partial reinjections as well as
the number of readout times, it should not be interpreted as a pure
measurement-only intervention. The controlled claim is narrower: all existing
virtual-node semantics are held constant while their density changes within a
fixed total interval.

Timing is machine- and cache-dependent. Three reservoir seeds provide only a
small uncertainty sample. Conditioning is computed from centered training
features without labels and is a numerical diagnostic, not a forecast-quality
metric.

This study is exact classical density-matrix simulation of a mathematical
quantum reservoir. It does not execute on a physical QPU and does not support a
quantum-advantage claim.
