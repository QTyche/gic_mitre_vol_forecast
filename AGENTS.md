# Agent working agreement

This repository supports a reproducible scientific benchmark. Preserve the
scientific contract while making changes.

- Never change target definitions silently. Document and version every target change.
- Never use shuffled train/test splits for time series.
- Never fit preprocessing on validation or test data.
- Never report a result without a saved configuration and experiment manifest.
- Never claim quantum advantage from one random seed.
- All headline results must run from a script, not only a notebook.
- Prefer small, testable changes.
- Run tests and linting after each change.
- Do not commit credentials, proprietary data, or generated benchmark claims.
- Keep simulator and hardware backends behind explicit interfaces so experiments remain portable.

