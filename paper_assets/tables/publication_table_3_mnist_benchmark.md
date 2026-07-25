# Frozen MNIST benchmark and QRC robustness

| Section | Model or condition | Accuracy ↑ | Macro-F1 ↑ | Balanced acc. ↑ | Macro ROC-AUC ↑ | Inference or scope |
| --- | --- | --- | --- | --- | --- | --- |
| Model comparison | Flattened logistic | 0.889 | 0.888 | 0.889 | 0.987 | No Holm-adjusted QRC difference |
| Model comparison | Size-controlled ESN | 0.934 | 0.934 | 0.934 | 0.995 | ESN higher than QRC on all metrics (Holm-adjusted) |
| Model comparison | Exact QRC mean | 0.874 ± 0.005 | 0.874 ± 0.005 | 0.874 ± 0.005 | 0.987 ± 0.001 | Mean ± population SD across seeds 2026-2028 |
| QRC robustness | Analytic (seed 2026) | 0.871 | 0.870 | 0.871 | 0.987 | Frozen seed-2026 prediction condition; no significance test |
| QRC robustness | 2,048 shots (seed 2026) | 0.504 | 0.501 | 0.504 | 0.869 | Frozen seed-2026 prediction condition; no significance test |
| QRC robustness | 2,048 shots + depolarising 0.01 | 0.492 | 0.488 | 0.492 | 0.860 | Frozen seed-2026 prediction condition; no significance test |
| QRC robustness | 2,048 shots + measurement flip 0.02 | 0.485 | 0.482 | 0.485 | 0.863 | Frozen seed-2026 prediction condition; no significance test |

- Exact QRC is mean ± population SD across seeds 2026-2028.
- QRC versus flattened logistic was not Holm-significant; ESN was significantly higher than QRC.
- Robustness rows use frozen seed-2026 predictions and were not significance-tested.
