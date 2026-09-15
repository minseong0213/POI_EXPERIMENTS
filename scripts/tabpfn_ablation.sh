#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output="${RESULT_DIR:-artifacts/poi-robustness-next}/reports/07_backbone_ablation"
exec python -m poi.tabpfn_ablation --config configs/stages/07_backbone_ablation_tabpfn.yaml --output "$output"
