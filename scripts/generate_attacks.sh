#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output="${RESULT_DIR:-artifacts/poi-adversarial-20260915-001}/reports/01_attack_generation"
exec python -m poi.adversarial --config configs/adversarial.yaml --output "$output"
