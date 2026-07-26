#!/usr/bin/env bash
set -euo pipefail

submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$submission_dir"

# Pin PDF metadata to the validated source commit time for byte-stable rebuilds.
export SOURCE_DATE_EPOCH=1785068776
export FORCE_SOURCE_DATE=1
export TZ=UTC

python3 prepare_assets.py
mkdir -p build

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  (cd build && BIBINPUTS="..:${BIBINPUTS:-}" bibtex main)
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile --keep-logs --outdir build main.tex
else
  echo "No supported LaTeX toolchain found (latexmk, pdflatex+bibtex, or tectonic)." >&2
  exit 1
fi

cp build/main.pdf Team_QTyche_Phase3.pdf
python3 validate_submission.py
