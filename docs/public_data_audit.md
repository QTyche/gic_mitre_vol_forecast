# Public-market data audit

The audit uses SPY as the canonical trading calendar and never forward-fills,
backfills, or interpolates VIX or QQQ.

## Raw observations

- Requested range: 2010-01-01 through 2025-12-31.
- Actual range for SPY, QQQ, and VIX: 2010-01-04 through 2025-12-31.
- SPY and QQQ: 4,024 rows each; VIX: 4,173 provider rows.
- Duplicate or non-increasing dates: zero.
- Weekend observations: zero.
- SPY dates missing QQQ or VIX: zero; rows removed during alignment: zero.
- VIX-only dates: 149. They are exchange-holiday placeholders with missing
  close values and are absent from the SPY calendar. They are reported and
  excluded by alignment, not filled.
- Zero or negative prices, non-positive volume, and SPY OHLC violations: zero.
- Absolute one-day SPY close moves above the configured 20% threshold: zero.
- Adjusted close differs from unadjusted close on all 4,024 SPY rows, with a
  maximum absolute relative difference of 25.37%. Returns intentionally use
  unadjusted `spy_close`, preserving the frozen scientific definition.

## Construction and splits

The canonical aligned source has 4,024 observations. Twenty initial rows are
removed for causal feature warm-up, and five final rows are removed because a
complete t+1 through t+5 target cannot be formed. The earliest valid feature
date is 2010-02-02; the latest valid target observation is 2025-12-23 and its
target window ends 2025-12-31.

Five rows are purged at the training boundary and five at the validation
boundary. Test purge is zero because the final five incomplete target rows were
already removed during target construction. No retained observation has an
incomplete or cross-boundary forward window.

| Split | Rows | Start | End |
|---|---:|---|---|
| Train | 2,744 | 2010-02-02 | 2020-12-23 |
| Validation | 748 | 2021-01-04 | 2023-12-21 |
| Test | 497 | 2024-01-02 | 2025-12-23 |

Training-only target thresholds are 0.006637091876978361 and
0.019546076157892393. The training-only scaler has no zero-variance features.
The processed manifest records every source and processed checksum, purged
date, class distribution, transition rate, target summary, Git commit, and
dirty status.
