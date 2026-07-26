# Team QTyche Phase 3 paper

This directory is the self-contained Track A technical paper based on the final
Stage 3A orchestration commit
`49c1fde09b56df9a5d9c1bef04dfa039275926b0`. Scientific facts remain frozen at
`3b562e72c655e6e4fb38b45ec22cfb4a1f96b530`; the paper does not run or refit a
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

The script first verifies the SHA-256 of every frozen figure, table and fact
source. It then regenerates two paper-local vector figures with presentation
changes only, regenerates the compact LaTeX tables, and writes
`asset_manifest.json`. The original Stage 2C figures remain untouched. The
build prefers `latexmk`; where unavailable it uses a `pdflatex`/`bibtex`
sequence, with Tectonic as a final portable fallback. It finishes by checking
asset hashes, missing files, LaTeX diagnostics, the five-page ceiling and the
PDF SHA-256. Generated compiler files live in `build/`; the submission PDF is
`Team_QTyche_Phase3.pdf`.

PDF metadata uses `SOURCE_DATE_EPOCH=1785078432`, the final Stage 3A base
commit's timestamp, so repeated builds with the same pinned toolchain are
byte-stable.

The supplied preamble is unchanged and no additional LaTeX packages were
added. Figures and tables use ordered non-floating blocks so they cannot
interrupt unrelated result or portability paragraphs. The only non-LaTeX
build dependencies are Python 3, NumPy and Matplotlib from the pinned
repository environment. The validation script uses only the standard library.

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
packaging, without modifying this evidence-traced technical PDF. Team-member
names, affiliations, contact details and a challenge identifier beyond the
verified Track A title block remain unresolved because no authoritative source
for them is present in the repository; they have intentionally not been
guessed.

## Provenance policy

- `prepare_assets.py` pins the source commit and expected source SHA-256 values.
- Original frozen PDFs are checksum-verified but never edited. The paper-local
  Figure 2 and Figure 3 PDFs are deterministic transformations with larger
  typography; all plotted values and intervals remain frozen.
- Compact tables are deterministic field selections from frozen JSON, with no
  recomputation or manual numeric editing.
- `asset_manifest.json` records source and destination hashes and whether each
  item was transformed.
- `source_map.md` maps each section, figure, table and headline claim to
  repository evidence.
- Every numerical claim in `main.tex` has a nearby `% SOURCE:` comment.

The strict empirical bootstrap implementation assigns a stored value of zero
when neither an opposing-sign draw nor a tie occurs; it adds no pseudocount.
The affected transition comparison had zero opposing-sign draws in 10,000
replicates. The paper therefore displays the Holm-adjusted result as
`p < 2e-4`, the smallest non-zero two-sided value on that resampling grid,
rather than printing `p = 0`. This display convention changes no estimate,
interval, support decision or scientific conclusion and is recorded in
`asset_manifest.json` and `source_map.md`.

All reported QRC results are exact classical density-matrix simulations or
explicitly labelled controlled finite-shot/noise simulations. No physical QPU
execution or quantum advantage is claimed.
