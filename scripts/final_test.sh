#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output="${RESULT_DIR:-artifacts/poi-final-test-001}/final_test"
exec python -m poi.final_test --config "${1:-configs/final_test.yaml}" --output "$output"
