#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"
exec python scripts/run_qrc_state_memory_ablation.py "$@"
