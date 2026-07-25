# Common MNIST QRC benchmark

## Purpose and isolation

Stage 1E supplies the challenge organisers' common MNIST digit-classification
benchmark. It tests whether a fixed QRC can supply nonlinear features for a
standard ten-class task. It is independent of the financial study: it has its
own configuration, data cache, feature caches, run identities, and
`results/qrc_mnist/` output tree. It neither changes nor retunes the frozen
financial QRC.

The QRC calculations use NumPy exact density-matrix evolution on a classical
CPU. The robustness conditions simulate finite computational-basis sampling and
simple noise channels. No condition is physical-QPU execution, and the
benchmark does not establish or claim quantum advantage.

## Data contract

The source is the original MNIST IDX data served by Google's
`cvdf-datasets/mnist` mirror. The four compressed files total 11,504,722 bytes,
about 11.0 MiB (11.5 MB). Their URLs, expected sizes, SHA-256 hashes, and legacy
MD5 hashes are pinned in `configs/qrc_mnist_benchmark.yaml`.

Download or verify the cache explicitly:

```bash
python scripts/run_qrc_mnist.py --download-only
```

The files are cached at `data/raw/mnist/` and are excluded from Git. An
interrupted download uses a temporary file and is not accepted as a source
file. Every normal run verifies all four SHA-256 hashes before reading the IDX
headers. A missing file fails with a download instruction unless `--download`
is supplied:

```bash
python scripts/run_qrc_mnist.py --download --smoke
```

The loader requires exactly 60,000 official training images and labels and
10,000 official test images and labels. It refuses unexpected shapes, labels,
headers, checksums, or synthetic/substituted data.

Selection uses NumPy `PCG64` with seed 2026 and independent within-class
permutations. The full split is:

| Benchmark split | Official source | Per digit | Total |
| --- | --- | ---: | ---: |
| training | official training partition | 600 | 6,000 |
| validation | official training partition | 100 | 1,000 |
| test | official test partition | 100 | 1,000 |

Training and validation indices are disjoint. Test indices belong to the
separate official test namespace. Smoke mode uses the same selection procedure
with 20/5/5 examples per digit, giving 200/50/50 images.

The dataset, selected-index, and preprocessing manifests record source and
subset checksums, exact indices, class counts, partition namespaces, shapes,
and preprocessing checksum.

## Fixed image representation

Each 28-by-28 `uint8` image is divided by 255 and treated as a top-to-bottom
sequence of 28 rows. Each row is compressed without fitted parameters by taking
five contiguous column means:

```text
[0,6), [6,12), [12,18), [18,23), [23,28)
```

The result has shape `28 x 5` and range `[0,1]`. No validation or test statistic
is used by this transformation.

## QRC and image features

The fixed QRC uses:

- five qubits and two virtual nodes;
- total row evolution time `tau=1.0`, split into two equal virtual substeps;
- ring couplings with `j_strength=1.0` and `h_strength=1.0`;
- input scaling 0.5 and `Ry` partial input-qubit reinjection;
- exact NumPy density-matrix state evolution;
- the established financial-QRC Hamiltonian construction;
- five `Z_i` and five unique ring-edge `Z_i Z_j` observables;
- reservoir seeds 2026, 2027, and 2028.

The reservoir is reset to its configured initial state at the start of every
image, carried through that image's 28 rows, and never carried to the next
image. This image-specific state policy does not modify the financial state
policy.

At every row, each of the two virtual nodes yields ten observables, so the row
feature dimension is `2 x (5 + 5) = 20`. Seven deterministic temporal summaries
are concatenated:

1. final row;
2. mean over all 28 rows;
3. population standard deviation over all 28 rows;
4. means over `[0,7)`, `[7,14)`, `[14,21)`, and `[21,28)`.

The final QRC image feature dimension is therefore `20 x 7 = 140`. These
summaries are fixed in configuration and were not selected using test results.

## Readouts and controls

All ten-class readouts are multinomial logistic regressions. A
`StandardScaler` is fitted on training features only. Candidate inverse
regularisation strengths `C = 0.01, 0.1, 1, 10` are fitted on training data and
selected by validation macro-F1. The chosen model is frozen before test
evaluation. Runs store each selection trial, selected `C`, coefficient norms,
convergence diagnostics, and finite coefficient/probability checks.

Two classical controls use exactly the same selected images:

- flattened logistic regression on the `28 x 5 = 140` compressed sequence;
- a deterministic 32-state ESN followed by the same seven summaries, producing
  `32 x 7 = 224` features and the same validation-selected readout.

The exact QRC is run for all three reservoir seeds. Seed 2026 is also evaluated
under four clearly labelled measurement conditions:

| Condition | Shots | Simulated noise |
| --- | ---: | --- |
| `analytic` | none | none |
| `shots_2048` | 2,048 | none |
| `depolarizing_0_01` | 2,048 | local probability 0.01 |
| `measurement_flip_0_02` | 2,048 | independent bit-flip probability 0.02 |

The finite-shot paths reuse the project's shared computational-basis sampling
and simulated-noise implementation.

## Commands, resumption, and runtime

From the repository root:

```bash
# Small deterministic 200/50/50 split and one exact reservoir seed
python scripts/run_qrc_mnist.py --smoke

# Repeat to exercise feature-cache and completed-readout resumption
python scripts/run_qrc_mnist.py --smoke

# Complete 6,000/1,000/1,000 split and three exact reservoir seeds
python scripts/run_qrc_mnist.py

# Recompute readouts while retaining deterministic feature caches
python scripts/run_qrc_mnist.py --no-resume
```

The equivalent thin wrapper is:

```bash
./scripts/run_qrc_mnist.sh --smoke
./scripts/run_qrc_mnist.sh
```

A fresh smoke took about 22 seconds on the reference Apple-silicon workstation;
a repeated smoke is primarily table/figure regeneration and is much faster. The
uncached full workflow took 694.6 seconds (11.6 minutes) on that workstation;
allow approximately 12–30 CPU minutes on other workstations or shared qBraid
resources. Actual observed wall times and analytical resource estimates are
written to `run_summary.json`. Host load, BLAS implementation, and filesystem
speed can materially affect elapsed time.

Run directories are content-addressed by the model, condition, reservoir seed,
selected-subset checksum, and study-configuration checksum. A run resumes only
if all required artifacts exist and its manifest matches the genuine dataset,
configuration, validation-only selection rule, and post-freeze test policy.
Incomplete runs are refitted. Feature caches have separate smoke/full
namespaces, checksum their arrays, and never consume labels.

For qBraid with Python 3.12:

```bash
qbraid envs create -n qtyche-qrc-phase3 -f environment-qbraid.yaml -y
qbraid envs activate qtyche-qrc-phase3
./scripts/setup_qbraid.sh
python scripts/run_qrc_mnist.py --download-only
python scripts/run_qrc_mnist.py --smoke
python scripts/run_qrc_mnist.py
```

The downloader uses only the Python standard library; Stage 1E adds no
heavyweight dataset dependency.

## Output contract

All generated files stay below `results/qrc_mnist/`, which is excluded from
Git:

```text
results/qrc_mnist/
├── dataset/
│   ├── dataset_manifest.json
│   ├── preprocessing_manifest.json
│   └── selected_indices.json
├── feature_cache/{smoke,full}/<checksum>/
├── baseline_cache/{smoke,full}/<checksum>/
├── runs/{smoke,full}/<run-id>/
│   ├── manifest.json
│   ├── model.pkl
│   ├── result.json
│   ├── selection_results.json
│   ├── {validation,test}_metrics.json
│   └── {validation,test}_predictions.csv
├── tables/
│   ├── mnist_qrc_exact_per_run.{csv,json}
│   ├── mnist_qrc_exact_aggregate.{csv,json}
│   ├── mnist_qrc_finite_shot_noise.{csv,json}
│   ├── mnist_classical_baselines.{csv,json}
│   └── mnist_final_benchmark.{csv,json}
├── figures/
│   ├── example_compressed_mnist_sequences.{png,pdf}
│   ├── qrc_test_confusion_matrix.{png,pdf}
│   ├── accuracy_macro_f1_comparison.{png,pdf}
│   ├── per_digit_f1_comparison.{png,pdf}
│   ├── exact_finite_shot_noise_comparison.{png,pdf}
│   ├── runtime_comparison.{png,pdf}
│   └── feature_rank_conditioning.{png,pdf}
├── environment_manifest.json
├── mnist_download_manifest.json
└── run_summary.json
```

The per-run tables report validation and test accuracy, macro-F1, balanced
accuracy, per-digit precision/recall/F1, 10-by-10 confusion matrix, one-vs-rest
macro ROC-AUC, feature generation, fitting, selection, inference and total
times, selected regularisation, coefficient norms, convergence/finite checks,
feature rank/conditioning, cache identity, dataset checksum, and provenance.
Exact QRC aggregates contain mean, population standard deviation, minimum, and
maximum over reservoir seeds. The final table puts the three-seed exact QRC
aggregate beside both classical baselines without claiming statistical
significance.

## Interpretation and limitations

The exact QRC seed distribution is the primary MNIST QRC result. Validation
metrics explain model selection; test metrics are held-out evaluation only.
Classical baselines are directly comparable because all methods use the same
selected images and validation policy. Robustness results are a deliberately
small, one-seed controlled diagnostic, not a hardware-noise study.

The subset is balanced but smaller than the customary full 60,000/10,000 MNIST
benchmark, the five-band projection discards horizontal detail, and only three
QRC reservoir seeds are measured. The 32-state ESN is a simple size-controlled
reference rather than an exhaustive classical model search. No CNN, pretrained
model, physical QPU, hardware-calibrated noise, formal significance test, or
quantum-advantage analysis is part of Stage 1E.
