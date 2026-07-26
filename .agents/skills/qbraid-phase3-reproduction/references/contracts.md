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
- Headline: verify outputs plus `headline_reproduction_report.json` and normal
  ignored financial, GARCH, baseline, and MNIST-smoke outputs.
- Full: headline dependencies plus full robustness, genuine-MNIST, Stage 2A,
  Stage 2B, isolated regenerated publication assets, and
  `full_reproduction_report.json`.

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
decimal digits, with maximum relative quantization width below `1e-9`; signed
zero is normalized and non-finite data are rejected. This covers platform
last-bit differences in `log` and reductions without accepting material data
changes.

Regenerated metrics use absolute tolerance `1e-10` plus relative tolerance
`1e-9` because BLAS, SciPy, and CPU implementations can change final floating
digits. Never change a tolerance or semantic reference without documenting the
observed delta and scientific justification.

## Runtime status

Treat ranges displayed before execution as planning guidance until actual
qBraid measurements replace them in the final evidence. Record installation,
verification, financial, MNIST, statistics/publication, total wall time, disk
usage, and peak child RSS separately.
