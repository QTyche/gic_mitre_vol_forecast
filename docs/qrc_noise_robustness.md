# QRC finite-shot and simulated-noise robustness study

## Scientific question and controls

This one-factor-at-a-time experiment measures how finite computational-basis
measurement budgets and two simple simulated channels affect QRC forecast
metrics, prediction stability, and runtime. The default selected reservoir has
two qubits, following the qubit-scaling study. `n_qubits` remains configurable
for controlled follow-up work.

All studies retain:

- public snapshot `yahoo_chart_20100101_20251231_v1` and its chronological,
  purged train/validation/test splits;
- two virtual nodes, input projection and angle encoding;
- the reset-and-reinject channel and `carry_inputs` temporal state policy;
- the disordered transverse-field Ising Hamiltonian family;
- Z and graph-edge ZZ observables;
- classifier/regressor targets and readout families;
- training-only preprocessing and validation-only ridge selection; and
- test evaluation only after readout selection and fitting are frozen.

The classifier and regressor share one label-free feature cache for every
reservoir/measurement condition. Robustness caches and runs are isolated from
the exact pilot and qubit-scaling result trees.

## One-factor grids

The configured full experiment contains:

| Study | Varied factor | Fixed factors |
|---|---|---|
| Analytic reference | none | exact expectations, no simulated noise |
| A: finite shot | `128, 512, 2048, 8192` shots | exact noiseless state, zero bit flips |
| B: depolarising | `0, 1e-4, 1e-3, 1e-2` | 2048 shots, zero bit flips |
| C: measurement noise | `0, 0.005, 0.01, 0.02` | 2048 shots, zero depolarisation |

Finite points use reservoir seeds 2026, 2027, and 2028 and measurement seeds 0,
1, and 2. The analytic expectation has one deterministic result per reservoir
seed because it has no sampling RNG.

The zero-noise 2048-shot conditions are retained under each named study for
clear one-factor comparisons. Their feature cache identity is shared, avoiding
duplicate state generation and sampling.

## Joint finite-shot measurement

The observables commute and are diagonal in the computational basis. For every
virtual-node density matrix, the implementation:

1. validates and normalises the real diagonal probabilities;
2. samples `shots` computational-basis indices from one deterministic NumPy RNG;
3. decodes q0 as the most-significant output bit;
4. optionally flips each sampled output bit independently; and
5. estimates all Z and ZZ observables from that same bitstring batch.

Sampling each observable independently would discard the joint correlations and
is explicitly not used. Domain-separated RNG streams for basis sampling and
optional bit flips continue chronologically through train, validation, and
test. Both are derived only from `measurement_seed`. Separating them keeps the
underlying basis-sampling stream fixed when measurement-noise probability
changes; labels never enter feature generation.

The analytic-expectation path does not instantiate a measurement RNG and is
bit-for-bit consistent with the existing exact reservoir for the same
configuration and inputs.

## Simulated channels

### Local depolarising channel

After each exact virtual-node unitary and before measuring that state, the
channel is applied independently and sequentially to every reservoir qubit:

```text
E_i,p(rho) = (1-p) rho + p [I_i/2 tensor Tr_i(rho)].
```

The code uses the equivalent Pauli mixture with weights
`(1 - 3p/4, p/4, p/4, p/4)` for I, X, Y, and Z on the selected qubit. The
configured probability is therefore the probability of replacing that qubit
with the maximally mixed state, per qubit and per virtual node. The resulting
noisy density matrix is propagated to the next reset/evolution step.

### Measurement bit-flip channel

After computational-basis sampling, each output bit is flipped independently
with probability `p_readout`. Z and ZZ estimates are then calculated from the
flipped joint batch. This channel changes observed features only; it does not
modify the density matrix propagated through time.

Both channels are traceable mathematical simulation controls. They are not
derived from device calibration, crosstalk, gate duration, topology, thermal
relaxation, or any other physical-QPU characterization.

## Commands and resumption

Run the compact grid:

```bash
python scripts/run_qrc_noise_robustness.py --smoke
```

Run all configured studies:

```bash
python scripts/run_qrc_noise_robustness.py
```

Override the selected architecture while keeping the study design fixed:

```bash
python scripts/run_qrc_noise_robustness.py --n-qubits 3 --smoke
```

The `uv` wrapper accepts the same options:

```bash
./scripts/run_qrc_noise_robustness.sh --smoke
```

Runs resume by default. A feature cache is checksum-validated before reuse, and
each classifier/regressor task is skipped only when its manifest, metrics, and
predictions are complete and provenance-valid. `--no-resume` refits requested
readouts while continuing to reuse valid feature caches.

Before feature generation, the runner verifies both frozen reference
configuration hashes, the public raw snapshot manifest and every raw checksum,
the processed-data manifest and checksums, the public snapshot identity, and
the non-synthetic source marker.

## Outputs

Generated artifacts are ignored by Git and isolated below:

```text
results/qrc_noise_robustness/
├── robustness_run_summary.json
├── robustness_state.json
├── feature_cache/
├── generated_configs/
├── runs/
├── tables/
│   ├── qrc_noise_robustness_per_run.{json,csv}
│   ├── qrc_noise_robustness_aggregate.{json,csv}
│   └── qrc_noise_robustness_resources.{json,csv}
└── figures/
```

The per-run table has separate validation/test rows and retains the complete
metric object plus scalar metrics. It records both seed types, all channel
settings, timings, dimensions, selected ridge alpha, prediction deviations
from the matching analytic reservoir, data/configuration/cache checksums, Git
state, Python/platform metadata, and package versions.

The aggregate table reports population mean, standard deviation, minimum, and
maximum at three levels:

1. measurement seeds within each reservoir seed;
2. reservoir seeds after averaging measurement repetitions; and
3. every reservoir/measurement repetition together.

Eleven publication figures are written as PNG and PDF: three shot/performance
figures, three metrics for each noise channel, runtime versus shots, and a
three-panel measurement-seed variance figure. Individual repetitions, mean
plus population standard deviation, and the analytic-expectation reference are
shown.

## Resource interpretation

At two qubits and 3,989 rows, every feature cache contains six float64 features
and is 191,472 bytes uncompressed. The full grid has 111 named experimental
points, 93 unique feature-cache identities after zero-noise reuse, and 222
readout tasks. It represents 741,954 virtual-node state evolutions and
1,663,508,736 sampled bitstrings. The estimated total uncompressed feature
cache is 17,806,896 bytes.

The three-dense-matrix state estimate is only 768 bytes for two qubits, while
the largest 8192-shot bit matrix is approximately 16,384 bytes. These values
exclude NumPy/SciPy workspace, sampled basis indices, random masks, Python
objects, compressed artifacts, predictions, model files, and plotting
overhead. Actual peak resident memory and disk use are therefore higher.

## Interpretation limits

Measurement-seed variability quantifies Monte Carlo stability only. Reservoir
seeds quantify a separate source of dynamical variability and are aggregated
separately before the all-repetition summary. Timing is machine- and
cache-dependent. The compact smoke grid cannot replace the full three-by-three
seed analysis.

This is exact classical density-matrix evolution plus controlled classical
sampling and mathematically specified simulated channels. It is not physical
QPU execution, not a hardware-faithful noise model, and not evidence of quantum
forecasting or computational advantage.
