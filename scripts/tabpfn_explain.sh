#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output="${RESULT_DIR:-artifacts/poi-tabpfn-explain-001}/reports/07_explainability"
exec python -m poi.tabpfn_explain --config configs/tabpfn_explain.yaml --output "$output"
