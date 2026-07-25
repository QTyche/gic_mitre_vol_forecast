# Exact-noiseless QRC qubit-scaling study

## Purpose and controlled variable

This experiment measures how forecasting metrics and computational resource use
change as the exact quantum reservoir grows from two to six qubits. Its only
architectural experimental variable is `n_qubits`. Each size is evaluated with
independent reservoir seeds 2026, 2027, and 2028.

The following contracts remain frozen at the public-pilot values:

- public snapshot `yahoo_chart_20100101_20251231_v1` and its temporal splits;
- two virtual nodes;
- input encoding, input scaling, reset channel, and `carry_inputs` state policy;
- transverse-field Ising Hamiltonian family and seed construction;
- single-qubit Z and graph-edge ZZ observables;
- classifier and regressor targets;
- training-only preprocessing and validation-only ridge selection; and
- exact, noiseless `numpy_density_matrix_exact` simulation.

Classifier and regressor readouts consume the same label-free feature cache for
each `(n_qubits, reservoir_seed)` point. Test data are evaluated only after
readout selection and fitting have been frozen.

## Commands

The direct commands requested by the experiment protocol are:

```bash
# Small 2/4/6-qubit, one-seed smoke grid
python scripts/run_qrc_qubit_scaling.py \
  --qubits 2 4 6 \
  --seeds 2026 \
  --smoke

# Complete 2/3/4/5/6-qubit, three-seed grid
python scripts/run_qrc_qubit_scaling.py \
  --qubits 2 3 4 5 6 \
  --seeds 2026 2027 2028
```

The thin `uv` wrapper accepts the same arguments:

```bash
./scripts/run_qrc_qubit_scaling.sh --qubits 2 4 6 --seeds 2026 --smoke
```

Both forms resume by default. A complete matching task is reused, while a
missing or incomplete classifier/regressor task is run without discarding its
completed partner. `--no-resume` explicitly creates new task runs. `--smoke`
restricts execution to at most three configured qubit sizes and one configured
seed; omitting explicit grids selects the configured smoke or full grid.

Before execution, the driver verifies:

1. the frozen classifier and regressor configuration checksums;
2. the raw public snapshot manifest and every raw-file checksum;
3. the processed data manifest and file checksums;
4. the expected public snapshot identity and non-synthetic source marker; and
5. agreement of all frozen QRC fields across both reference readouts.

Failure of any check stops the study before fitting.

## Output contract

Generated artifacts live below `results/qrc_qubit_scaling/` and remain ignored
by Git:

```text
results/qrc_qubit_scaling/
├── scaling_run_summary.json
├── scaling_state.json
├── feature_cache/
├── generated_configs/
├── runs/
│   └── <timestamp>_<model>_<task>_seed<seed>/
├── tables/
│   ├── qrc_qubit_scaling_per_run.{json,csv}
│   ├── qrc_qubit_scaling_aggregate.{json,csv}
│   └── qrc_qubit_scaling_resources.{json,csv}
└── figures/
    ├── test_macro_f1_vs_qubits.{png,pdf}
    ├── test_transition_pr_auc_vs_qubits.{png,pdf}
    ├── test_qlike_vs_qubits.{png,pdf}
    ├── state_generation_time_vs_qubits.{png,pdf}
    └── qrc_feature_dimension_vs_qubits.{png,pdf}
```

The per-run table has separate validation and test rows. It records the qubit
count, fixed virtual-node count, seed, task, split, selected ridge alpha,
feature and readout dimensions, trainable parameter count, state-generation and
readout-fit times, feature-cache status and checksum, frozen data identity,
backend, exact/noiseless marker, task metrics, Git state, package versions, and
the temporal evaluation assertions.

The aggregate table groups each metric by qubit count, task, and split. It
reports the mean, population standard deviation, minimum, maximum, contributing
seeds, and seed count. The figures show individual seeds and the mean plus or
minus population standard deviation. A one-seed smoke run therefore has zero
displayed uncertainty; it tests the workflow rather than estimating
between-reservoir variability.

## Resource estimates

For a two-virtual-node ring, the raw feature dimension is
`V (N + |E|)`. The two-qubit ring has one unique undirected edge and therefore
six features; rings with three or more qubits have `|E| = N` and therefore
`4N` features.

The resource table estimates the memory for three simultaneous dense
complex128 matrices (Hamiltonian/state/workspace) as
`3 × 16 × (2^N)^2` bytes. It separately estimates an uncompressed float64
feature matrix over the selected train, validation, and test rows. These are
transparent analytical estimates, not process-level peak-RSS measurements.
They omit Python/SciPy overhead, eigensolver workspace, saved predictions, and
readout arrays, so practical memory use is higher.

## Interpretation

The study supports descriptive statements about performance stability,
feature dimension, and exact-simulator runtime as reservoir size changes.
Validation metrics remain model-selection evidence; test metrics are held-out
evaluation evidence. Three reservoir seeds give only a small uncertainty
sample, so min/max and individual points should be inspected alongside means.
Wall-clock timing is machine- and cache-dependent; `cache_hit` distinguishes
new state generation from feature reuse.

This experiment executes an exact classical density-matrix simulation of a
mathematical quantum reservoir. It does not execute on a physical QPU, include
shots or hardware noise, or establish quantum computational or forecasting
advantage.
