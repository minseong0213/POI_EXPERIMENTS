#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -n "${POI_PYTHON:-}" ]]; then
    python_bin="$POI_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    python_bin="$PROJECT_ROOT/.venv/bin/python"
else
    python_bin="python"
fi
exec "$python_bin" -m poi.eda --data-dir "${DATA_DIR:-data/poi_34k_seed42}" \
  --output-root "${1:-artifacts/eda-001/reports}"
