# Team QTyche Phase 3 paper

This directory is the self-contained Track A technical paper built from the
frozen evidence at commit
`3b562e72c655e6e4fb38b45ec22cfb4a1f96b530`. It does not run or refit a
scientific model.

## Build

From the repository root:

```bash
./submission/phase3/build.sh
```

Or from this directory:

```bash
make
```

The script first verifies the SHA-256 of every frozen figure/table source,
copies the selected figures byte-for-byte, regenerates the two compact LaTeX
tables, and writes `asset_manifest.json`. It then prefers `latexmk`; where that
is unavailable it uses a `pdflatex`/`bibtex` sequence, with Tectonic as a final
portable fallback. The build finishes by checking asset hashes, missing files,
LaTeX diagnostics, the five-page ceiling, and the PDF SHA-256. Generated
compiler files live in `build/`; the submission PDF is
`Team_QTyche_Phase3.pdf`.

PDF metadata uses `SOURCE_DATE_EPOCH=1785068776`, the validated source
commit's timestamp, so repeated builds with the same LaTeX toolchain are
byte-stable.

The supplied preamble is unchanged and no additional LaTeX packages were
added. Figures and tables use ordered non-floating blocks so they cannot
interrupt unrelated result or portability paragraphs. The only non-LaTeX
build dependency is Python 3 from the repository environment; the asset and
validation scripts use only the standard library.

## Submission rule interpretation

The official
[Global Industry Challenge 2026 Track page](https://aqora.io/challenges/global-industry-challenge-2026/tracks/gic-2026-qBraid-MITRE-JonesTrading)
states that the written report is a maximum of five pages and that references
do not count toward that maximum. The build nevertheless enforces five pages
for the entire compiled PDF, including references, which is the more
conservative interpretation. It also uses the required 11-point Times style,
single spacing and A4 geometry specified in the paper brief.

The challenge page separately mentions an organiser cover-page template. The
technical paper uses the exact title block requested for this workstream; any
organiser-supplied administrative cover should be added only at final portal
packaging, without modifying this evidence-traced technical PDF.

## Provenance policy

- `prepare_assets.py` pins the source commit and expected source SHA-256 values.
- Direct PDF figures are byte-identical copies of Stage 2C assets.
- Compact tables are deterministic field selections from frozen JSON, with no
  recomputation or manual numeric editing.
- `asset_manifest.json` records source and destination hashes and whether each
  item was transformed.
- `source_map.md` maps each section, figure, table and headline claim to
  repository evidence.
- Every numerical claim in `main.tex` has a nearby `% SOURCE:` comment.

All reported QRC results are exact classical density-matrix simulations or
explicitly labelled controlled finite-shot/noise simulations. No physical QPU
execution or quantum advantage is claimed.
