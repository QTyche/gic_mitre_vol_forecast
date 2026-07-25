# Frozen Phase 3 publication assets

Stage 2C compiles a compact five-page-paper asset package from checksum-pinned
Stage 1 and Stage 2 outputs. It is a read-only scientific reporting step: it
does not fit or rerun a model, change an architecture or threshold, recalibrate
predictions, tune on test results, create an ensemble, or conduct a new
hypothesis test.

## Reproduce the freeze

From the repository root, run:

```bash
uv run python scripts/freeze_publication_assets.py
uv run python scripts/freeze_publication_assets.py
```

The second command is intentional: all tracked assets must be byte-identical
after regeneration. The equivalent thin wrapper is:

```bash
./scripts/freeze_publication_assets.sh
```

The compiler first verifies every configured SHA-256 digest and rejects a
synthetic financial-data manifest. Source identities, the pre-Stage-2C commit,
output roots, safeguards, precision, and appendix inputs are frozen in
`configs/publication_assets.yaml`.

## Main-paper selection

The three tables are:

1. frozen financial and MNIST architecture and experiment design;
2. directly comparable financial test benchmarks plus Stage 2A
   Holm-adjusted annotations;
3. MNIST model comparison and seed-2026 finite-shot/noise robustness.

The four figures are:

1. validation-only qubit and temporal-multiplexing selection with cost
   diagnostics;
2. financial test comparison and selected architecture-level Stage 2A
   intervals;
3. final financial QRC analytic, shot-count and controlled-noise robustness;
4. MNIST model comparison and seed-2026 robustness.

All tables are emitted as CSV, JSON, LaTeX and Markdown. All figures are
emitted as 300-DPI PNG and deterministic PDF. Seven existing diagnostics are
copied without modification to `paper_assets/appendix/` and are not part of
the estimated five-page main-body footprint.

## Output contract

```text
paper_assets/
├── tables/                         # 3 x CSV/JSON/LaTeX/Markdown
├── figures/                        # 4 x PNG/PDF
├── appendix/                       # 7 preserved PNG/PDF pairs plus index
├── final_results_manifest.{json,md}
├── publication_assets_manifest.json
├── results_factsheet.md
├── limitations_factsheet.md
├── reproducibility_factsheet.md
├── figure_captions.md
└── page_footprint.md

results/publication_assets/
├── source_verification.json
├── resolved_facts.json
├── claims.json
└── generation_summary.json
```

`paper_assets/final_results_manifest.json` traces every selected numerical or
design fact to an exact value, display value, repository-relative artifact,
key or row locator, source digest, split, scope, adjustment status, and metric
direction. Its claims section classifies paper-language candidates as
supported, qualified, unsupported or prohibited.

`paper_assets/publication_assets_manifest.json` freezes the final selection,
asset digests, source digests, generation command, source commit, environment,
and the no-model-execution and no-test-selection safeguards. Its own digest is
explicitly excluded to avoid self-reference.

## Interpretation limits

The QRC performance values come from classical exact density-matrix
simulation, with separately labelled finite-shot and controlled-noise
simulations. No physical QPU was executed. The noise models are not
hardware-calibrated, non-monotonic changes are not evidence that noise helps,
and the qubit-selection curves are not evidence of monotonic quantum scaling.

Only three frozen reservoir seeds are available. The five-day financial
targets overlap, tail samples are small, seed 2027 is near-singular, MNIST
compression discards within-band detail, and transition lead time is not
identifiable from the frozen aggregate rows. Failure to reject a paired
difference is not an equivalence result. No quantum-advantage claim is made.

## Validation

The Stage 2C audit is:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The tests verify source and selected-asset checksums, fact tracing, split and
scope isolation, Holm-adjusted annotations, claim status, relative paths,
format pairs, PNG dimensions, deterministic regeneration, preservation of all
prior result trees, and absence of a model-execution path.
