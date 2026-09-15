#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
output="${RESULT_DIR:-artifacts/poi-robustness-next}/work/tabpfn_benchmark"
exec python -m poi.tabpfn_benchmark --config configs/stages/06_region_models_tabpfn.yaml --output "$output"
