#!/usr/bin/env python3
"""Validate Phase 3 paper assets, LaTeX diagnostics, page count, and PDF hash."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PDF = HERE / "Team_QTyche_Phase3.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: Path) -> int:
    log = HERE / "build" / "main.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Output written on .+?\((\d+) pages?,", text)
        if matches:
            return int(matches[-1])
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        output = subprocess.check_output([pdfinfo, str(path)], text=True)
        match = re.search(r"^Pages:\s+(\d+)\s*$", output, re.MULTILINE)
        if match:
            return int(match.group(1))
    qpdf = shutil.which("qpdf")
    if qpdf:
        output = subprocess.check_output([qpdf, "--show-npages", str(path)], text=True)
        return int(output.strip())
    mdls = shutil.which("mdls")
    if mdls:
        output = subprocess.check_output(
            [mdls, "-raw", "-name", "kMDItemNumberOfPages", str(path)],
            text=True,
        ).strip()
        if output.isdigit():
            return int(output)
    raw = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page\b", raw))
    if count < 1:
        raise ValueError(f"could not determine page count for {path}")
    return count


def validate_assets() -> list[str]:
    manifest_path = HERE / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked: list[str] = []
    for asset in manifest["assets"]:
        source = REPOSITORY / asset["original_repository_path"]
        destination = HERE / asset["destination_filename"]
        for path in (source, destination):
            if not path.is_file():
                raise FileNotFoundError(f"manifest asset is missing: {path}")
        if sha256(source) != asset["original_sha256"]:
            raise ValueError(f"source checksum mismatch: {source}")
        if sha256(destination) != asset["copied_file_sha256"]:
            raise ValueError(f"destination checksum mismatch: {destination}")
        if not asset["transformed"] and source.read_bytes() != destination.read_bytes():
            raise ValueError(f"directly copied asset is not byte-identical: {destination}")
        checked.append(asset["destination_filename"])
    return checked


def latex_diagnostics() -> dict[str, Any]:
    log = HERE / "build" / "main.log"
    if not log.is_file():
        raise FileNotFoundError("build/main.log is missing; compile the paper first")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = [
        r"LaTeX Error:",
        r"Package .* Error:",
        r"Undefined control sequence",
        r"There were undefined references",
        r"Citation [`'][^`']+[`'] on page .* undefined",
    ]
    fatal = [pattern for pattern in fatal_patterns if re.search(pattern, text, re.IGNORECASE)]
    if fatal:
        raise ValueError(f"fatal LaTeX diagnostics found: {fatal}")
    warnings = sorted(set(re.findall(r"(?:LaTeX|Package \S+) Warning:.*", text)))
    overfull = re.findall(r"Overfull \\[hv]box.*", text)
    return {
        "log": "build/main.log",
        "warnings": warnings,
        "overfull_boxes": overfull,
    }


def main() -> None:
    required = [
        "main.tex",
        "references.bib",
        "source_map.md",
        "asset_manifest.json",
        "build.sh",
        "Makefile",
        "README.md",
    ]
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required submission files are missing: {missing}")
    if not PDF.is_file():
        raise FileNotFoundError(f"compiled PDF is missing: {PDF}")

    pages = page_count(PDF)
    if pages > 5:
        raise ValueError(f"paper has {pages} pages; the enforced maximum is 5")

    report = {
        "status": "pass",
        "pdf": PDF.relative_to(REPOSITORY).as_posix(),
        "pdf_pages": pages,
        "page_limit_enforced": 5,
        "pdf_sha256": sha256(PDF),
        "assets_checked": validate_assets(),
        "latex": latex_diagnostics(),
    }
    output = HERE / "build" / "validation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
