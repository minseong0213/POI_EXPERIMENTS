#!/usr/bin/env python3
"""Fail closed unless the experiment is complete through adversarial evaluation."""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


REQUIRED = [
    "00_protocol", "01_attack_generation", "01_descriptive", "02_distribution",
    "03_embedding", "04_statistics", "06_region_models", "06_attack_evaluation",
    "07_explainability",
]
MODELS = {"decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
          "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"}


def run(root):
    root = Path(root); reports = root / "reports"; failures = []
    for stage in REQUIRED:
        directory = reports / stage
        if not directory.is_dir():
            failures.append(f"missing stage: {stage}")
        if not (directory / "report.md").is_file():
            failures.append(f"missing report: {stage}")
        if not (directory / "metadata.json").is_file():
            failures.append(f"missing metadata: {stage}")
    svg = list(root.rglob("*.svg"))
    if svg:
        failures.append(f"SVG files forbidden: {len(svg)}")
    pngs = list(reports.rglob("*.png"))
    low_dpi = []
    for path in pngs:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(dpi) < 299:
                low_dpi.append(str(path))
    if low_dpi:
        failures.append(f"PNG below 300 dpi tolerance: {len(low_dpi)}")

    attacks_path = reports / "01_attack_generation/tables/attacks.parquet"
    if attacks_path.is_file():
        attacks = pd.read_parquet(attacks_path)
        if len(attacks) != 870400 or attacks.attack_condition.nunique() != 32:
            failures.append("attack row/condition cardinality mismatch")
        if set(attacks.split) != {"train", "validation"}:
            failures.append("attack data used an unexpected split")
        if (~attacks.constraints_valid.astype(bool)).any():
            failures.append("invalid constrained attacks found")
        if (~attacks.label_preserved.astype(bool)).any():
            failures.append("attack label changes found")
        if attacks.duplicated(["split", "POI_ID", "attack_condition"]).any():
            failures.append("duplicate attack keys found")
    else:
        failures.append("missing attacks.parquet")

    ranking_path = reports / "06_region_models/tables/model_ranking_validation.csv"
    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path)
        if set(ranking.model) != MODELS or ranking.model.duplicated().any() or ranking["mean"].isna().any():
            failures.append("eight-model clean ranking incomplete")
    else:
        failures.append("missing model ranking")
    model_metrics = reports / "06_region_models/tables/metrics_by_region.csv"
    if model_metrics.is_file():
        values = pd.read_csv(model_metrics)
        if len(values) != 408 or set(values.condition) != {"clean"}:
            failures.append("clean comparison cardinality mismatch")
    attack_metrics = reports / "06_attack_evaluation/tables/metrics_by_region.csv"
    attack_predictions = reports / "06_attack_evaluation/tables/all_predictions.parquet"
    if attack_metrics.is_file():
        values = pd.read_csv(attack_metrics)
        if len(values) != 1683 or values.condition.nunique() != 33 or set(values.split) != {"validation"}:
            failures.append("attack metric cardinality mismatch")
    else:
        failures.append("missing attack metrics")
    if attack_predictions.is_file():
        values = pd.read_parquet(attack_predictions, columns=["split"])
        if len(values) != 5722200 or set(values.split) != {"validation"}:
            failures.append("attack prediction cardinality mismatch")
    else:
        failures.append("missing attack predictions")
    shap_pngs = list((reports / "07_explainability/figures").glob("shap_beeswarm_*.png"))
    if len(shap_pngs) != 34:
        failures.append(f"expected 34 SHAP beeswarm PNGs, found {len(shap_pngs)}")
    shap_table = reports / "07_explainability/tables/shap_values.csv"
    if shap_table.is_file():
        values = pd.read_csv(shap_table)
        if values.region.nunique() != 17 or set(values.condition) != {"clean", "adversarial"}:
            failures.append("SHAP region/condition coverage mismatch")
    else:
        failures.append("missing SHAP values")
    for stage in ["00_protocol", "01_attack_generation", "01_descriptive", "02_distribution",
                  "03_embedding", "04_statistics", "06_region_models", "06_attack_evaluation",
                  "07_explainability"]:
        path = reports / stage / "metadata.json"
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text())
        test_value = metadata.get("test_used", metadata.get("test_split_used", False))
        if test_value is not False:
            failures.append(f"test split flag is not false: {stage}")
    result = {"status": "failed" if failures else "complete", "failures": failures,
              "required_stages": REQUIRED, "png_files": len(pngs), "svg_files": len(svg),
              "checks": 12}
    if failures:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    output = reports / "99_audit"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir()
    (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "report.md").write_text(
        "# 최종 공격 범위 감사\n\n필수 9개 단계, 공격 870,400행·32조건, clean 8개 모델 "
        "순위, 공격 평가 5,722,200개 예측, SHAP beeswarm 34개를 검사했다. 모든 PNG는 "
        "300dpi 허용 오차를 충족하며 SVG와 test 사용은 0건이다.\n")
    (output / "_SUCCESS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/poi-adversarial-20260915-001")
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
