#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# Dependencies are installed once during setup, not implicitly on each training run.
if [[ -n "${POI_PYTHON:-}" ]]; then
    python_bin="$POI_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    python_bin="$PROJECT_ROOT/.venv/bin/python"
else
    python_bin="python"
fi
exec "$python_bin" -m poi.train --config "${1:-configs/runs/baseline.yaml}"
