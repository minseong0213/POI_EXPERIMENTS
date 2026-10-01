#!/usr/bin/env bash
set -Eeuo pipefail
python -m pip install tabpfn==8.1.0 numpy==2.0.2 pandas==2.3.3 pyarrow PyYAML==6.0.2 shap==0.46.0 lime==0.2.0.1 statsmodels==0.14.5 matplotlib==3.9.4
python -m pip install --no-deps --no-build-isolation .
export TABPFN_TOKEN_FILE=/root/.config/poi/tabpfn_token
python -m pip freeze > "$RESULT_DIR/runtime_packages.txt"
python scripts/smoke_proposed_defense.py --device cuda --context-rows 256 --query-rows 64 --mixed-precision --data-dir "$DATA_DIR" --output "$RESULT_DIR/reports/08_proposed_defense/gpu_smoke.json"
python -m poi.feature_ablation --config configs/stages/07_clean_feature_ablation.yaml
