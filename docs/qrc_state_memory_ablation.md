# Exact QRC state-memory ablation

## Purpose and controlled variable

This experiment isolates the role of cross-observation reservoir memory for the
validation/performance-cost QRC candidate. It compares:

- `carry_inputs`: retain the full post-observation reservoir state before the
  next market observation; and
- `reset_each_input`: restore the full configured initial density matrix before
  every market observation.

Both policies retain the established partial input-qubit reset and reinjection
before every virtual substep. They also retain the within-observation state
across the two virtual nodes. The reset condition therefore removes
cross-observation reservoir memory without changing the input projection,
Hamiltonian, substep duration, observables, readouts, or targets.

The selected architecture is fixed at two qubits and `V=2`. The driver verifies
the checksum of the prior encoding-density validation candidate table and
rejects evidence involving test selection. Reservoir seeds are 2026, 2027, and
2028. All data, chronological splits, training-only preprocessing,
validation-only ridge selection, and exact noiseless simulation controls remain
frozen.

The older QRC `reset` policy means reset at split boundaries. It is retained
unchanged for compatibility and is not one of this ablation's conditions.

## Commands

Run the one-seed smoke grid:

```bash
python scripts/run_qrc_state_memory_ablation.py \
  --state-policies carry_inputs reset_each_input \
  --seeds 2026 \
  --smoke
```

Run both policies over all three reservoir seeds:

```bash
python scripts/run_qrc_state_memory_ablation.py \
  --state-policies carry_inputs reset_each_input \
  --seeds 2026 2027 2028
```

The shell wrapper accepts the same arguments:

```bash
./scripts/run_qrc_state_memory_ablation.sh --smoke
```

Complete matching task runs resume by default. `--no-resume` creates new
readout runs while retaining deterministic checksum-keyed feature reuse.

Before fitting, the runner verifies:

1. both frozen public-pilot readout configuration hashes;
2. the encoding-density validation candidate evidence and selected `V=2`;
3. the raw public snapshot manifest and every source-file checksum;
4. all processed-data checksums and public-data markers; and
5. agreement of every fixed QRC control across readouts.

Synthetic fixture data stop the experiment.

## State and timing semantics

For both policies, the first observation starts from the same fixed initial
state. Every market observation contains two virtual substeps of duration
`tau/V=0.5`, giving total evolution time `tau=1.0`. At each substep the current
input is projected, the input qubit is partially reset and reinjected, the
whole reservoir evolves, and the same `Z_i` and unique `Z_i Z_j` observables are
recorded.

Under `carry_inputs`, the state after the second substep becomes the state
before the next observation's partial reinjection. Under `reset_each_input`,
the full state is first restored before that next observation. No measurement
feedback is added.

The exact feature timer includes state evolution and observable construction,
so `feature_generation_seconds` describes that combined operation.

## Output contract

Generated artifacts are isolated below
`results/qrc_state_memory_ablation/` and ignored by Git:

```text
results/qrc_state_memory_ablation/
├── state_memory_run_summary.json
├── state_memory_state.json
├── feature_cache/
├── generated_configs/
├── memory_diagnostics/
├── runs/
├── tables/
│   ├── qrc_state_memory_per_run.{json,csv}
│   ├── qrc_state_memory_aggregate.{json,csv}
│   ├── qrc_state_memory_paired_differences.{json,csv}
│   ├── qrc_state_memory_paired_summary.{json,csv}
│   ├── qrc_state_memory_validation_policy_selection.{json,csv}
│   ├── qrc_state_memory_resources.{json,csv}
│   ├── qrc_state_memory_feature_autocorrelation.{json,csv}
│   ├── qrc_state_memory_lagged_input_feature_correlation.{json,csv}
│   └── qrc_state_memory_perturbation_decay.{json,csv}
└── figures/
    ├── validation_headline_metrics_by_state_policy.{png,pdf}
    ├── test_headline_metrics_by_state_policy.{png,pdf}
    ├── per_seed_paired_metric_differences.{png,pdf}
    ├── feature_conditioning_by_state_policy.{png,pdf}
    └── feature_generation_runtime_by_state_policy.{png,pdf}
```

The per-run tables retain all scalar and structured task metrics, ridge alpha,
feature/readout dimensions, timing, conditioning, memory diagnostics, cache and
configuration checksums, data provenance, Git state, Python/package/platform
metadata, and temporal-evaluation assertions. Aggregates report mean,
population standard deviation, minimum, maximum, contributing seeds, and seed
count.

Paired rows align each policy by reservoir seed and define every difference as:

```text
carry_inputs minus reset_each_input
```

Positive differences favor carry for macro-F1, balanced accuracy, and
transition PR-AUC. QLIKE and RMSE are lower-is-better metrics, so negative
differences favor carry. The table includes a direction-normalized improvement
column and explicit QLIKE interpretation.

## Validation-only policy choice

The policy choice reads validation files only and applies this lexicographic
order:

1. higher mean validation macro-F1;
2. higher mean validation transition PR-AUC;
3. lower mean validation QLIKE;
4. lower seed variability in the same metric order; and
5. lower feature-generation time.

An exact tie defaults to `reset_each_input` as the simpler no-memory condition.
The decision and its complete trace are saved before test metrics are collected
for reporting. Test results do not influence the choice.

## Memory diagnostics

All memory diagnostics consume training inputs and features without labels:

- feature autocorrelation reports mean absolute per-feature autocorrelation by
  lag;
- lagged input–feature correlation summarizes all nonconstant input/feature
  pairs by lag;
- centered training-feature singular values provide effective rank, numerical
  rank, and retained-spectrum conditioning; and
- perturbation decay drives two valid initial density matrices with the same
  observed inputs and measures trace distance.

For `reset_each_input`, perturbation distance must be zero after the first full
reset and identically driven observation. Nonzero feature autocorrelation may
remain because market inputs themselves are temporally correlated; it is not
proof of internal reservoir memory. The perturbation curve is empirical for the
tested inputs and is not a formal global contractivity result.

## Reference and interpretation

The `carry_inputs` cache key is identical to the selected two-qubit `V=2`
encoding-density point. When that prior cache is available, train, validation,
and test arrays are compared directly and the maximum absolute difference is
recorded.

Three seeds provide limited uncertainty evidence. Timing is machine- and
cache-dependent. Very large condition numbers indicate near-collinear features
under the stated tolerance and should be interpreted alongside effective and
numerical rank.

This is exact classical density-matrix simulation of a mathematical quantum
reservoir. It is not physical-QPU execution and does not support a
quantum-advantage claim.
