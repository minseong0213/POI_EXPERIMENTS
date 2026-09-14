#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
output="${RESULT_DIR:-artifacts/poi-tabpfn-validation-001}/tabpfn"
exec python -m poi.tabpfn_benchmark --config configs/tabpfn_benchmark.yaml --output "$output"
