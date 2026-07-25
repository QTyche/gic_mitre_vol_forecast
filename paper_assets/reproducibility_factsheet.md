# Reproducibility factsheet

## Frozen identities

- Stage 2C source commit: `a5c9f73ddee00a483299f22a5887a3dab6a7819c`.
- Final financial architecture commit: `b2a21c4defad2ae8b150cc1b43aae2399188b997`.
- MNIST benchmark commit: `7cbc1945ca4a98f9b1adbcf433e91b40867ace1e`.
- Financial architecture manifest SHA-256: `10a431b7d047f5e0b18b657815492560a717aca6588067402bf47cb64983190f`.
- Public-data manifest SHA-256: `97c280252b0aeabc2e524d97da4c152f5a3f11429a223044600dc99a1edafd33`.
- MNIST balanced-subset checksum: `06c1ee9c6db87efc13ebaacc7f4406297d061d10d573731434c3f957e7c0574e`.

## Runtime expectations

- Publication generation reads frozen artifacts and normally completes in seconds; it invokes no model runner.
- The frozen two-qubit scaling row reports about 0.276 seconds of state generation.
- Frozen exact MNIST QRC execution averaged 79.1 seconds per seed in its recorded environment.
- The frozen full Stage 2B diagnostic compiler completed in 4.4 seconds.

Runtime values are environment-specific and are not performance guarantees.

## qBraid and environment assumptions

- The frozen final financial and MNIST manifests record local classical simulation and null qBraid environment identifiers.
- Reproducing the publication assets does not require qBraid or QPU access.
- A later Stage 3 clean-room run should use the repository lock file on a supported Python environment and record any qBraid image or environment identifier if one is used.

## Stage 3 clean-room commands

```bash
uv sync --frozen
uv run python scripts/freeze_publication_assets.py
uv run python scripts/freeze_publication_assets.py
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The second generation must reproduce all tracked scientific tables, manifests and figures byte-for-byte.
