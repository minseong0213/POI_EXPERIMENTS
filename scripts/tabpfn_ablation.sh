#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output="${RESULT_DIR:-artifacts/poi-tabpfn-ablation-001}/reports/06_model_selection_ablation"
exec python -m poi.tabpfn_ablation --config configs/tabpfn_ablation.yaml --output "$output"
