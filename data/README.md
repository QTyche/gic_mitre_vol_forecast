# Data preparation and leakage contract

The committed repository contains no claimed market results. The smoke workflow
creates deterministic market-shaped fixtures under `data/raw/`; these files are
synthetic test inputs and are not historical financial observations. Real inputs
must be saved as immutable CSV snapshots with their paths and SHA-256 checksums
recorded in `data/processed/data_manifest.json`.

## Canonical input variables

The pipeline aligns VIX and QQQ to the SPY trading calendar without filling
missing dates. Its canonical columns are `date`, SPY open/high/low/close/adjusted
close/volume, VIX close, and QQQ close/volume. SPY OHLC and volume are required on
every SPY row. A missing secondary observation is either an error or causes the
SPY-calendar row to be removed and listed in the quality report, according to
`missing_data_policy`. There is no forward fill, backward fill, or interpolation.

## Returns and targets

For close price `close_t`, the daily log return is:

```text
r_t = log(close_t / close_{t-1})
```

The current five-day annualized realized variance uses only information through
the observation date:

```text
rv_current_t = (252 / 5) * sum(r_{t-i}^2, i=0,...,4)
```

The regression target uses exactly the next five trading-day returns:

```text
target_rv_5d_t = (252 / 5) * sum(r_{t+i}^2, i=1,...,5)
```

The 33rd and 66th percentiles of `target_rv_5d` are fitted on retained training
rows only. Those frozen thresholds label both current and forward variance:

- `0`: value less than or equal to the 33rd-percentile threshold
- `1`: above the 33rd and less than or equal to the 66th-percentile threshold
- `2`: above the 66th-percentile threshold

`target_transition` is one exactly when the target regime differs from the
current regime. Upward and downward transition labels use strict greater-than
and less-than comparisons. The target definition is versioned as
`qtyche_volatility_regime_v1`; changes require a new version and documentation.

## Features

All rolling windows end at date `t`; centered windows and future backfilling are
forbidden.

- `spy_log_return_1d`: `log(spy_close_t / spy_close_{t-1})`
- `spy_return_5d`, `spy_return_20d`: close-to-close log returns over 5 and 20 days
- `spy_rv_5d`, `spy_rv_10d`, `spy_rv_20d`: `(252 / n)` times the trailing sum of squared SPY log returns
- `spy_parkinson_vol_5d`: square root of the annualized five-day Parkinson range-variance estimate
- `spy_volume_zscore_20d`: current volume minus trailing 20-day mean, divided by trailing population standard deviation
- `spy_high_low_range`: `log(spy_high_t / spy_low_t)`
- `vix_log_level`: natural logarithm of VIX close
- `vix_change_1d`, `vix_change_5d`: VIX close difference in index points over 1 and 5 days
- `qqq_log_return_1d`: one-day QQQ close log return
- `qqq_rv_5d`: annualized trailing five-day QQQ realized variance
- `spy_qqq_return_spread`: one-day SPY log return minus one-day QQQ log return
- `day_of_week`: Monday `0` through Friday `4`

## Temporal splits and purge

The default observations are training 2010–2020, validation 2021–2023, and test
2024–2025, inclusive. Dates are strictly chronological and never shuffled. A
row is retained only if its saved fifth-forward-trading-day window end is within
the same split. This removes the last five observations from every split and
prevents a target from crossing a split boundary. Trailing features may use
earlier historical context because that information was already available at
the observation date.

## Normalization and leakage prevention

Feature means and population standard deviations (`ddof=0`) are fitted on the
training split only, saved to `preprocessing.json`, and reused unchanged for
validation and test. Zero-variance training columns are centered and divided by
one, and are listed explicitly. Transformation fails if feature names or order
differ from the fitted contract.

Leakage is prevented by causal feature operations, explicit forward shifts for
targets, training-only threshold and scaler fitting, complete target-window
purging, immutable source checksums, and network-independent tests that mutate
future/validation/test values and verify fitted past quantities do not change.

## Using fixture or public data

Run the complete offline fixture workflow with:

```bash
make smoke
```

To replace fixtures with public snapshots, change `data.mode` to `download`, set
date-stamped raw paths and the symbol mapping in `configs/data.yaml`, then run:

```bash
uv run python -m qtyche_qrc.cli prepare-data --config configs/data.yaml
```

The downloader uses the public Yahoo chart endpoint only for missing raw paths
and never overwrites an existing snapshot. Review licensing and redistribution
terms before committing any downloaded data. For a fully frozen submission,
switch back to `cached_csv` after acquisition and reproduce from the checksummed
snapshots.

