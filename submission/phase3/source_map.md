# Phase 3 paper source map

The paper base is the final Stage 3A orchestration commit
`49c1fde09b56df9a5d9c1bef04dfa039275926b0`; scientific facts remain frozen at
`3b562e72c655e6e4fb38b45ec22cfb4a1f96b530`. Generated `results/` trees are not
treated as primary sources because they are intentionally ignored by Git. The
tracked Stage 2C manifests, frozen assets, configurations, final qBraid reports
and portability references below are the checkout-reproducible evidence.

## Sections and headline claims

| Paper location | Claim or content | Frozen source |
| --- | --- | --- |
| Title/rules | Challenge year, phase, track and page rule | Official Aqora challenge page linked in `README.md`; `paper_assets/page_footprint.md` |
| Abstract | Financial QRC macro-F1 0.454, transition PR-AUC 0.640 and QLIKE -2.814 | `paper_assets/tables/publication_table_2_financial_benchmark.json` (`rows[model=QRC mean]`) |
| Abstract | MNIST exact-QRC accuracy 0.874 and ESN accuracy 0.934 | `paper_assets/tables/publication_table_3_mnist_benchmark.json` (`rows[model_or_condition=Exact QRC mean]`, `rows[model_or_condition=Size-controlled ESN]`) |
| Abstract | Independent qBraid reproduction and narrowly gated solver variation | `qbraid_evidence/final_clean_room/headline_reproduction_report.json`; `qbraid_evidence/final_clean_room/execution_report.json`; `configs/reproduction/garch_portability_reference.json`; `configs/reproduction/mnist_exact_portability_reference.json` |
| Sec. 1 | Benchmark scope; no QPU and no advantage claim | `paper_assets/results_factsheet.md`; `paper_assets/limitations_factsheet.md` |
| Sec. 2 | Snapshot, symbols, dates, 16 inputs, split rows/dates, target and thresholds | `data/processed/public_market/data_manifest.json`; `configs/data_public_market.yaml`; `src/qtyche_qrc/data/features.py`; `src/qtyche_qrc/data/targets.py`; `src/qtyche_qrc/data/splits.py` |
| Sec. 2 | Leakage controls and five-day purge | `data/processed/public_market/data_manifest.json`; `docs/public_data_provenance.md`; `docs/data_contract.md`; `configs/phase3_reproduction.yaml` |
| Sec. 3 | Two qubits, two virtual nodes, reset policy, Hamiltonian, encoding and six observables | `configs/reproduction/final_financial_qrc.yaml`; `docs/final_financial_qrc.md`; `src/qtyche_qrc/models/qrc/reservoir.py` |
| Sec. 3 | Ridge families, validation selection and seeds | `configs/reproduction/final_financial_qrc.yaml`; `paper_assets/final_results_manifest.json` |
| Sec. 3 | Broad set of strong classical controls | `paper_assets/tables/publication_table_2_financial_benchmark.json`; `configs/models/public_market/`; `configs/baseline_esn.yaml`; `configs/reproduction/garch_baseline.yaml` |
| Sec. 4 | Financial point estimates and rankings | `paper_assets/final_results_manifest.json` (`financial.test`); `paper_assets/tables/publication_table_2_financial_benchmark.json` |
| Sec. 4 | HAC, block bootstrap, Holm tests and selected intervals | `paper_assets/final_results_manifest.json` (`financial.inference`); `configs/statistical_validation.yaml`; `src/qtyche_qrc/statistics/bootstrap.py`; `docs/statistical_validation.md` |
| Sec. 4 | Former stored empirical zero displayed as \(p<2\times10^{-4}\) | `paper_assets/final_results_manifest.json` (`financial.inference.transition_pr_auc.qrc_vs_regime_persistence.holm_p`); `src/qtyche_qrc/statistics/bootstrap.py::bootstrap_interval`; `configs/statistical_validation.yaml` (`full_repetitions=10000`). The strict two-sided rule is \(2\min\{\Pr(\Delta<0),\Pr(\Delta>0)\}\), gives stored zero when no opposing-sign draw occurs and has smallest non-zero value \(2/10000\). The paper reports that finite grid resolution rather than exactly zero; estimate, interval and Holm support are unchanged. |
| Sec. 4 | Calibration, tail and conditioning limitations | `paper_assets/limitations_factsheet.md`; `docs/benchmark_diagnostics.md`; `configs/benchmark_diagnostics.yaml` |
| Sec. 5 | Genuine-MNIST source, partition, preprocessing and architecture | `configs/qrc_mnist_benchmark.yaml`; `docs/qrc_mnist_benchmark.md`; `paper_assets/final_results_manifest.json` |
| Sec. 5 | MNIST point estimates, seed uncertainty, robustness and inference | `paper_assets/tables/publication_table_3_mnist_benchmark.json`; `paper_assets/final_results_manifest.json` (`mnist`) |
| Sec. 5 | qBraid Linux/x86 result, changed predictions and L-BFGS root cause | `configs/reproduction/mnist_exact_portability_reference.json`; `docs/qbraid_reproduction.md` |
| Sec. 6 | Pinned qBraid Linux/x86 clean-room execution and regenerated artefacts | `configs/phase3_reproduction.yaml`; `qbraid_evidence/final_clean_room/execution_report.json`; `docs/qbraid_reproduction.md` |
| Sec. 6 | Checksum-exact immutable sources and semantic processed-data verification | `configs/reproduction/processed_data_semantic_reference.json`; `qbraid_evidence/final_clean_room/processed_data_semantic_verification.json`; `docs/qbraid_reproduction.md` |
| Sec. 6 | Explicitly gated GARCH and MNIST numerical portability | `configs/reproduction/garch_portability_reference.json`; `configs/reproduction/mnist_exact_portability_reference.json`; `qbraid_evidence/final_clean_room/garch_portability_report.json`; `docs/qbraid_reproduction.md` |
| Sec. 6 | Limitations and conclusion | `paper_assets/limitations_factsheet.md`; `paper_assets/reproducibility_factsheet.md`; `paper_assets/results_factsheet.md` |

## Tables

| Paper asset | Frozen source | Transformation |
| --- | --- | --- |
| Table 1, `tables/financial_results.tex` | `paper_assets/tables/publication_table_2_financial_benchmark.json` | Deterministic field selection and two side-by-side LaTeX panels by `prepare_assets.py`; no numeric recomputation |
| Table 2, `tables/mnist_results.tex` | `paper_assets/tables/publication_table_3_mnist_benchmark.json` | Deterministic selection of three model-comparison rows by `prepare_assets.py`; no numeric recomputation |

## Figures

| Paper figure | Submission asset | Frozen evidence inputs | Inclusion |
| --- | --- | --- | --- |
| Figure 1, architecture | TikZ in `main.tex` | `configs/reproduction/final_financial_qrc.yaml`; `src/qtyche_qrc/models/qrc/reservoir.py` | Reproducible vector schematic; no scientific result |
| Figure 2, financial comparison | `figures/financial_comparison.pdf` | Original: `paper_assets/figures/publication_figure_2_financial_comparison.pdf`; values: `paper_assets/tables/publication_table_2_financial_benchmark.json`; intervals/adjusted p-values: `paper_assets/final_results_manifest.json` | Deterministic paper-local vector regeneration by `prepare_assets.py`. All eight panels, estimates and intervals are preserved; fonts, markers and line weights are enlarged. The sole statistical label change applies the documented finite two-sided reporting resolution. The frozen original is untouched. |
| Figure 3, MNIST and robustness | `figures/mnist_benchmark.pdf` | Original: `paper_assets/figures/publication_figure_4_mnist_benchmark.pdf`; values: `paper_assets/tables/publication_table_3_mnist_benchmark.json` | Deterministic paper-local vector regeneration by `prepare_assets.py`; values and uncertainty bars are preserved while dimensions, typography, markers and line weights are enlarged. The frozen original is untouched. |

`asset_manifest.json` contains the pinned original/source SHA-256 values,
paper-local generated SHA-256 values, transformation source lists, purpose and
statistical display contract for all four transformed assets.

## Bibliography provenance

The bibliography uses primary publication metadata (title, venue, year and
DOI) for Fujii--Nakajima QRC, Jaeger's ESN report, Bollerslev GARCH,
Patton QLIKE, Diebold--Mariano forecast comparison, Newey--West HAC, the
original MNIST paper, and Chen--Nurdin--Yamamoto noisy-QRC analysis. No
placeholder citations from the earlier draft are retained.
