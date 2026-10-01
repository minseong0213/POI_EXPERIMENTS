#!/usr/bin/env bash
set -Eeuo pipefail
python -m pip install numpy==2.0.2 pandas==2.3.3 scikit-learn==1.6.1 PyYAML==6.0.2 matplotlib==3.9.4 seaborn==0.13.2 pyarrow pytest==8.3.5
python -m pip install --no-deps -e .
python -c 'import poi.adversarial, torch; assert torch.cuda.is_available(), "CUDA required"'
mkdir -p "$RESULT_DIR/preflight"
python -m pip freeze > "$RESULT_DIR/preflight/requirements-resolved.txt"
python -m pytest tests/test_adversarial.py tests/test_data.py tests/test_geospatial.py -q > "$RESULT_DIR/preflight/tests.txt" 2>&1
bash scripts/generate_attacks.sh
