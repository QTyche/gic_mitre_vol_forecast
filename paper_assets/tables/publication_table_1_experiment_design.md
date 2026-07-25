# Frozen architecture and experiment design

| Application | Design item | Frozen value |
| --- | --- | --- |
| Financial application | Dataset and date range | SPY, QQQ and VIX public-market snapshot (2010-01-04 to 2025-12-31) |
| Financial application | Temporal split sizes | 2744 train / 748 validation / 497 test |
| Financial application | Forecast target | 5-trading-day realised variance |
| Financial application | Frozen regime thresholds | low/medium=0.006637; medium/high=0.019546 |
| Financial application | Frozen QRC architecture | 2 qubits; V=2; reset_each_input; feature dimension 6 |
| Financial application | Reservoir seeds and exact backend | 2026, 2027, 2028; numpy_density_matrix_exact |
| Financial application | Controlled robustness conditions | analytic; 128/512/2,048/8,192 shots; depolarising 0.01; bit flip 0.02 |
| Financial application | Final architecture-manifest SHA-256 | 10a431b7d047f5e0b18b657815492560a717aca6588067402bf47cb64983190f |
| MNIST benchmark | Genuine balanced split sizes | 6000 train / 1000 validation / 1000 test; 100 test images per digit |
| MNIST benchmark | QRC representation | 5 qubits; 28-row, five-band sequence; image feature dimension 140 |
| MNIST benchmark | Reservoir seeds and backend | 2026, 2027, 2028; numpy_density_matrix_exact |
| MNIST benchmark | Balanced-subset checksum | 06c1ee9c6db87efc13ebaacc7f4406297d061d10d573731434c3f957e7c0574e |
| MNIST benchmark | Analytic and robustness conditions | analytic; 2,048 shots; depolarising 0.01; measurement flip 0.02 |

- All architecture and threshold values were frozen before final test inspection.
- QRC execution used classical simulation; no physical QPU was used.
