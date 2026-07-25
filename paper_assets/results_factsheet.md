# Results factsheet

- Logistic regression led financial test macro-F1 (0.526); the frozen QRC architecture mean was 0.454, and ESN was 0.446.
- The QRC-minus-logistic macro-F1 difference was Holm-significant; no QRC-versus-ESN financial classification metric was Holm-significant.
- GARCH had the lowest test QLIKE (-2.895) versus -2.814 for the QRC mean.
- ESN had the lowest test RMSE (0.0662) versus 0.0894 for the QRC mean.
- QRC significantly exceeded regime persistence on transition PR-AUC after Holm correction.
- On the balanced MNIST test subset, exact QRC mean accuracy was 0.874; ESN reached 0.934.
- QRC seed-2026 MNIST accuracy fell to 0.504 at 2,048 shots; the controlled noise rows remained substantially below the analytic result.
- No physical QPU was executed and no quantum-advantage conclusion is supported.

All numeric sentences trace to `final_results_manifest.json`.
