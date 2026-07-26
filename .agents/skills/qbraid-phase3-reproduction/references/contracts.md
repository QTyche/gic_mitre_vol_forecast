# Frozen reproduction contract

## Identities

- Repository: `https://github.com/QTyche/gic_mitre_vol_forecast.git`
- Branch: `main`
- Submission ancestor:
  `cd9fa988f854009e408af1774a97ed663b0e8b86`
- Financial snapshot: `yahoo_chart_20100101_20251231_v1`
- Exact backend: `numpy_density_matrix_exact`
- Evidence root: `qbraid_evidence/final_clean_room/`

## Tier outputs

- Verify: `fast_verification_report.json`, `execution_report.json`,
  environment/Git reports, command log, and transcript.
- Headline: verify outputs plus `headline_reproduction_report.json`,
  `garch_portability_report.json`, and normal ignored financial, GARCH,
  baseline, and MNIST-smoke outputs.
- Full: headline dependencies plus full robustness, genuine-MNIST, Stage 2A,
  Stage 2B, isolated regenerated publication assets, and
  `full_reproduction_report.json`.
- Artifact-reuse finalization: the original full schema-v1 failure preserved as
  `execution_report.pre_artifact_reuse.json`, strict
  `artifact_reuse_validation_report.json`, passing comparison reports and a
  successful schema-v2 `execution_report.json`.

## Equality policy

Require exact SHA-256 equality for tracked configs, dataset declarations, the
processed-data semantic reference, publication-manifest-selected files, and
generated evidence artifacts. Require each downloaded raw market file to match
its per-download immutable manifest exactly; record historical provider hashes
separately because Yahoo may revise unused `adjusted_close` digits.

For the six generated processed financial files, record historical and current
byte hashes and require the tracked semantic commitments. Columns, row order,
dates, split membership, labels, missingness, JSON structure, and non-float
values are exact. Finite numeric values are normalized at 10 significant
decimal digits, with maximum relative quantization width at most `1e-9`; signed
zero is normalized and non-finite data are rejected. This covers platform
last-bit differences in `log` and reductions without accepting material data
changes.

The global regenerated-metric contract remains absolute tolerance `1e-10` plus
relative tolerance `1e-9`. It applies to QRC, GARCH classification, and
publication comparisons.

GARCH test QLIKE, RMSE, and MAE may use the separate
`garch_optimizer_portability_v1` absolute tolerance of `2.5e-7` only after
`garch_portability_report.json` passes. That report requires successful
convergence, a frozen deterministic start or equivalent optimum, likelihood
within `1e-7`, bounded optimizer iterations and parameters, independent
candidate-path reconstruction within `1e-12`, exact dates/labels/regime
assignments/threshold crossings/floor counts, bounded forecast differences,
and unchanged displayed values and rankings. The parameter and forecast bounds,
observed qBraid deltas, equivalent-start envelope, and safety margins are
checksum-pinned in
`configs/reproduction/garch_portability_reference.json`. Never change a
tolerance or semantic reference without documenting the observed delta and
scientific justification.

Exact-QRC MNIST metrics do not receive a wider numeric tolerance. The
checksum-pinned `mnist_exact_lbfgs_portability_v1` contract may accept the
frozen prediction path or the observed qBraid e0l4 Linux/x86 prediction path
only. It requires exact MNIST source/subset/index/label/preprocessing/config
identities; seed-specific feature commitments and scaler summaries; full
coefficient, intercept, optimizer-objective, score and probability comparisons;
exact changed indices, predictions, confusion matrices and profile metrics;
the pinned Linux/x86 package environment; unchanged rankings; and explicit
retention of the frozen displayed accuracy. Its report is
`mnist_exact_portability_report.json`. Eight-decimal feature canonicalization
only neutralizes `<=6.44e-15` state-generation variation and is paired with the
downstream full-path gates; it is never applied to model inputs.

The checksum-pinned failed-at-comparison qBraid run may be finalized without
scientific recomputation only when all prior task records are successful, every
recorded artifact is rehashed exactly (using the last recorded identity when a
later ordered task intentionally overwrote a watched output), the source commit matches the pinned
artifact commit, the clean validation commit changes only enumerated
reproduction/validation files, and fresh frozen verification, focused tests and
full comparison pass. Preserve the original failed execution report
byte-for-byte. The successful schema-v2 report must state that scientific-model
and MNIST-reservoir recomputation are false. Evidence packaging independently
revalidates the source report, validation report, full task sequence and every
recorded artifact.

## Runtime status

Treat ranges displayed before execution as planning guidance until actual
qBraid measurements replace them in the final evidence. Record installation,
verification, financial, MNIST, statistics/publication, total wall time, disk
usage, and peak child RSS separately.
