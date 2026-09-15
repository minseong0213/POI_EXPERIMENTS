#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -n "${POI_PYTHON:-}" ]]; then python_bin="$POI_PYTHON";
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then python_bin="$PROJECT_ROOT/.venv/bin/python";
else python_bin="python"; fi
args=(--config "${1:-configs/stages/06_region_models_tree.yaml}")
if [[ -n "${RESULT_DIR:-}" ]]; then args+=(--output "$RESULT_DIR/work/tree_benchmark"); fi
exec "$python_bin" -m poi.benchmark "${args[@]}"
