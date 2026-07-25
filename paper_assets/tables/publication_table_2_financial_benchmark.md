# Frozen financial test benchmark

| Model | Macro-F1 ↑ | Balanced acc. ↑ | Transition PR-AUC ↑ | QLIKE ↓ | RMSE ↓ | MAE ↓ | Correlation ↑ | Holm-adjusted evidence vs QRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority classifier | 0.151 | 0.333 | 0.506 | — | — | — | — | QRC higher on all three classification metrics (Holm-adjusted) |
| Regime persistence | 0.485 | 0.485 | 0.511 | — | — | — | — | QRC higher on transition PR-AUC only (Holm-adjusted) |
| Logistic regression | 0.526 | 0.529 | 0.683 | — | — | — | — | Logistic regression higher on macro-F1 (Holm-adjusted) |
| ESN | 0.446 | 0.478 | 0.645 | -2.865 | 0.0662 | 0.0183 | 0.532 | No Holm-adjusted QRC difference |
| RV persistence | — | — | — | -2.397 | 0.0866 | 0.0255 | 0.335 | No Holm-adjusted QRC difference |
| GARCH(1,1) | 0.373 | 0.442 | — | -2.895 | 0.0740 | 0.0233 | 0.356 | No Holm-adjusted regression difference; transition PR-AUC undefined |
| QRC mean | 0.454 | 0.491 | 0.640 | -2.814 | 0.0894 | 0.0216 | 0.543 | Reference: mean of three frozen reservoir seeds |

- Lower QLIKE, RMSE and MAE is better; higher classification metrics and correlation is better.
- QRC values are arithmetic means across the three frozen reservoir seeds.
- Significance statements use paired time-series-aware Stage 2A tests after Holm adjustment; raw-p-only findings are not marked significant.
