#!/usr/bin/env bash
set -Eeuo pipefail
python -m pip install tabpfn==8.1.0 pandas==2.3.3 pyarrow PyYAML==6.0.2 shap==0.46.0 lime==0.2.0.1 statsmodels==0.14.5 matplotlib==3.9.4
python -m pip install --no-deps --no-build-isolation .
export TABPFN_TOKEN_FILE=/root/.config/poi/tabpfn_token
python -m poi.feature_ablation --config configs/stages/07_clean_feature_ablation.yaml
