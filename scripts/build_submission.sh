#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec "${PYTHON_BIN:-python}" scripts/package_qbraid_evidence.py "$@"
