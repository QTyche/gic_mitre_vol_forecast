# Public-market data provenance

The canonical snapshot is `yahoo_chart_20100101_20251231_v1`, retrieved from
the public Yahoo chart endpoint on 2026-07-21. The requested daily range is
2010-01-01 through 2025-12-31. All three instruments begin on the first trading
day, 2010-01-04, and end on 2025-12-31.

| Instrument | Rows | SHA-256 |
|---|---:|---|
| SPY | 4,024 | `4082fd426a19de89f6f77d8b0927d07cace1952117abd7c4a9167a9d4b2cbce8` |
| QQQ | 4,024 | `061564a4e18b7d92e95921d593d4bd7e876e687b0aa6d944312b98e38f298136` |
| VIX | 4,173 | `0a1ede70191c6383af273cb418c779f7a707b169404c889152ecc2a7e1b09558` |

The raw canonical CSV files and generated `snapshot_manifest.json` live below
`data/raw/public_market/yahoo_chart_20100101_20251231_v1/`. The manifest records
the snapshot ID, provider, retrieval timestamp, requested range, symbol,
provider URL, timezone, row count, actual range, adjustment availability, and
file checksum. Its SHA-256 is
`5f8c04b1fd12315a0367ea98be325e66f5881bf47bbe049ebb27b28d4b7efb2f`.

Raw provider files are intentionally ignored by Git. The committed
`data/public_market_snapshot.json`, configuration, downloader, schema
documentation, and checksums provide the acquisition recipe. Provider terms
must be reviewed before redistributing raw files; this project makes no legal
claim about redistribution rights.

Download or verify the snapshot explicitly:

```bash
uv run python -m qtyche_qrc.cli download-public-data \
  --config configs/data_public_market.yaml
```

Existing valid snapshots are checksum-verified and never overwritten. A
partial or changed snapshot fails. Replacement requires the explicit `--force`
flag and redownloads every instrument. After acquisition, `prepare-data
--cached` performs no network request.
