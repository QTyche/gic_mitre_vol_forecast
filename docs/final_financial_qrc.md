# Frozen final financial QRC

## Stage 1D purpose and freeze

Stage 1D freezes the financial QRC selected by the preceding controlled
experiments. The final architecture is:

- two qubits and two virtual nodes;
- `reset_each_input`, while retaining partial input-qubit reinjection at each
  virtual substep;
- total per-input evolution time `tau=1.0`, a ring graph, `J=1.0`, `h=1.0`,
  and input scaling `0.5`;
- analytic `Z_i` plus unique `Z_i Z_j` observables, giving six raw features;
- exact noiseless NumPy density-matrix simulation; and
- reservoir seeds 2026, 2027, and 2028.

The classifier and regressor retain the ridge grid
`[1e-5, 1e-3, 1e-1, 1.0]`. Preprocessing is fitted on training data, ridge
selection uses validation data only, and test evaluation occurs only after the
readout is frozen.

The selection lineage is explicit: two qubits came from the qubit-scaling
performance–cost evaluation, `V=2` from validation-only encoding-density
evidence, and `reset_each_input` from validation-only state-memory evidence.
Test metrics were inspected only after each validation decision. The
architecture is now frozen, and no further test-informed tuning is permitted.

## Commands

Run the deterministic smoke experiment, including its small robustness grid:

```bash
python scripts/run_final_financial_qrc.py --smoke
python scripts/run_final_financial_qrc.py --smoke
```

The repeated command should resume every complete run and reproduce the exact
and robustness tables byte for byte. Run the full three-seed exact benchmark
and complete robustness grid with:

```bash
python scripts/run_final_financial_qrc.py
```

The shell wrapper accepts the same options:

```bash
./scripts/run_final_financial_qrc.sh --smoke
./scripts/run_final_financial_qrc.sh
```

Complete matching runs resume by default. `--no-resume` creates new readout
runs while retaining checksum-keyed feature caches. `--skip-robustness` runs
only the exact benchmark and rebuilds reports from any robustness table already
present.

Before execution, the driver verifies the frozen data configuration, raw
snapshot and files, processed manifest and splits, final readout configs, prior
validation evidence, and robustness configuration against pinned SHA-256
values. Synthetic fixture data stop the run.

## Output contract

All generated artifacts are ignored by Git and isolated below:

```text
results/final_financial_qrc/
├── final_architecture_manifest.json
├── final_validation_selection.json
├── final_run_summary.json
├── exact/
│   ├── feature_cache/
│   └── runs/
├── robustness/
│   ├── feature_cache/
│   ├── generated_configs/
│   ├── runs/
│   ├── tables/
│   └── figures/
├── tables/
│   ├── final_qrc_exact_per_run.{json,csv}
│   ├── final_qrc_exact_aggregate.{json,csv}
│   └── final_financial_benchmark.{json,csv}
└── figures/
    ├── final_classification_benchmark.{png,pdf}
    ├── final_regression_benchmark.{png,pdf}
    ├── final_qrc_shot_convergence.{png,pdf}
    ├── final_qrc_depolarizing_robustness.{png,pdf}
    ├── final_qrc_measurement_robustness.{png,pdf}
    └── final_numerical_conditioning.{png,pdf}
```

The manifest records every fixed QRC parameter, ridge grid and selected ridge
per seed/task, source and split checksums, configuration checksum, Git state,
Python and package versions, platform provenance, selection lineage, and the
two freeze flags.

Exact per-run tables report validation and test macro-F1, balanced accuracy,
transition PR-AUC, confusion matrices, QLIKE, RMSE, MAE, prediction
correlation, floor/non-finite counts, effective rank, condition number,
coefficient norms, finiteness checks, timing, and cache provenance. Aggregate
tables give mean, population standard deviation, minimum, and maximum across
reservoir seeds.

## Final robustness refresh

The refresh reuses the established controlled computational-basis sampling and
noise implementations but changes the frozen state policy to
`reset_each_input`. It is a separate study under
`results/final_financial_qrc/robustness/`; the earlier
`results/qrc_noise_robustness/` carry-input study is never read as a resumable
final run or overwritten.

The full one-factor-at-a-time grid contains:

- analytic references for each reservoir seed;
- shots 128, 512, 2048, and 8192;
- local depolarising probabilities 0, 0.0001, 0.001, and 0.01 at 2048 shots;
- measurement bit-flip probabilities 0, 0.005, 0.01, and 0.02 at 2048 shots;
  and
- measurement seeds 0, 1, and 2, matching the prior study.

The tables show seed-level outcomes, mean and population uncertainty, error
relative to each analytic reference, reservoir-seed and measurement-seed
variation, runtime, cache size, dense-state memory, and sampling-work
estimates. Small non-monotonic changes under noise are sampling/model
variability and must not be described as noise improving performance.

## Comparison and interpretation

The final benchmark contains majority and regime-persistence classifiers,
logistic regression, realized-variance persistence, ESN classifier/regressor,
Gaussian GARCH(1,1), and the frozen QRC. Only metrics with aligned dates and
definitions are compared. GARCH transition PR-AUC remains not applicable
because its deterministic thresholded forecast supplies no calibrated
transition score. Stage 1D runs no formal significance tests.

The reset seed 2027 feature matrix is known to be near singular, with a
condition number around `5.77e12`. This is recorded as a numerical limitation;
neither the architecture nor ridge grid is changed. Coefficient and prediction
finiteness demonstrate that the validation-selected regularisation remains
numerically usable for these runs.

All QRC results here come from exact classical density-matrix or controlled
finite-shot/noise simulation of a quantum reservoir. No physical QPU is
executed, and neither the benchmark nor robustness refresh supports a
quantum-advantage claim.
