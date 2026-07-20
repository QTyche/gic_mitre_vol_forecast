# Team QTyche data contract v1

All numeric market inputs must be finite. Prices are strictly positive, volumes
are non-negative, and dates are unique and increasing. “No missingness” below
means a row cannot enter a processed split with that value missing.

| Column | Type | Definition | Information availability | Allowed missingness | Role |
|---|---|---|---|---|---|
| `date` | date | SPY trading-session date | At session close | None | Identifier |
| `spy_open` | float | SPY unadjusted session open | By session close | None | Canonical input |
| `spy_high` | float | SPY unadjusted session high | By session close | None | Canonical input |
| `spy_low` | float | SPY unadjusted session low | By session close | None | Canonical input |
| `spy_close` | float | SPY unadjusted session close | At session close | None | Canonical input and target basis |
| `spy_adjusted_close` | float | Vendor-adjusted SPY close | At/after session close | None | Provenance input; not used in v1 returns |
| `spy_volume` | integer | SPY session volume | At session close | None | Canonical input |
| `vix_close` | float | VIX close aligned to the SPY date | At session close | Secondary-source gap removes row and is reported | Canonical input |
| `qqq_close` | float | QQQ close aligned to the SPY date | At session close | Secondary-source gap removes row and is reported | Canonical input |
| `qqq_volume` | integer | QQQ session volume aligned to SPY | At session close | Secondary-source gap removes row and is reported | Canonical input |
| `spy_log_return_1d` | float | `log(spy_close_t / spy_close_{t-1})` | At `t` | Warm-up rows only | Feature |
| `spy_return_5d` | float | `log(spy_close_t / spy_close_{t-5})` | At `t` | Warm-up rows only | Feature |
| `spy_return_20d` | float | `log(spy_close_t / spy_close_{t-20})` | At `t` | Warm-up rows only | Feature |
| `spy_rv_5d` | float | `(252/5)` times trailing 5 squared SPY log returns | At `t` | Warm-up rows only | Feature and current-regime basis |
| `spy_rv_10d` | float | `(252/10)` times trailing 10 squared SPY log returns | At `t` | Warm-up rows only | Feature |
| `spy_rv_20d` | float | `(252/20)` times trailing 20 squared SPY log returns | At `t` | Warm-up rows only | Feature |
| `spy_parkinson_vol_5d` | float | Square root of annualized trailing five-day Parkinson estimator | At `t` | Warm-up rows only | Feature |
| `spy_volume_zscore_20d` | float | Trailing 20-day population z-score of SPY volume | At `t` | Warm-up or zero trailing variance | Feature |
| `spy_high_low_range` | float | `log(spy_high_t / spy_low_t)` | At `t` | None after canonical validation | Feature |
| `vix_log_level` | float | `log(vix_close_t)` | At `t` | None after alignment | Feature |
| `vix_change_1d` | float | `vix_close_t - vix_close_{t-1}` | At `t` | Warm-up rows only | Feature |
| `vix_change_5d` | float | `vix_close_t - vix_close_{t-5}` | At `t` | Warm-up rows only | Feature |
| `qqq_log_return_1d` | float | `log(qqq_close_t / qqq_close_{t-1})` | At `t` | Warm-up rows only | Feature |
| `qqq_rv_5d` | float | `(252/5)` times trailing 5 squared QQQ log returns | At `t` | Warm-up rows only | Feature |
| `spy_qqq_return_spread` | float | SPY one-day log return minus QQQ one-day log return | At `t` | Warm-up rows only | Feature |
| `day_of_week` | float | Monday `0` through Friday `4` | Known at `t` | None | Feature |
| `target_rv_5d` | float | `(252/5)` times squared SPY returns at `t+1,...,t+5` | Only after `t+5` | Target-tail rows are removed | Secondary regression target |
| `target_regime_5d` | integer | Forward RV labeled by frozen training quantiles | Only after `t+5` | None in processed data | Primary classification target |
| `current_regime` | integer | `spy_rv_5d` labeled by the same frozen training quantiles | At `t` after training fit | None in processed data | Transition reference |
| `target_transition` | integer | `1[target_regime_5d != current_regime]` | Only after `t+5` | None | Primary transition target |
| `target_upward_transition` | integer | `1[target_regime_5d > current_regime]` | Only after `t+5` | None | Diagnostic target |
| `target_downward_transition` | integer | `1[target_regime_5d < current_regime]` | Only after `t+5` | None | Diagnostic target |

