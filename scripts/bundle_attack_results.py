#!/usr/bin/env python3
"""Create the compact professor-delivery folder for the completed attack study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


STAGES = [
    "00_protocol", "01_descriptive", "02_distribution", "03_embedding", "04_statistics",
    "01_attack_generation", "06_region_models", "06_attack_evaluation", "07_explainability",
    "99_audit",
]


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ignore(directory, names):
    ignored = []
    if "work" in names:
        ignored.append("work")
    if "all_predictions.parquet" in names:
        ignored.append("all_predictions.parquet")
    return ignored


def run(source_root, output):
    source_root, output = Path(source_root), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    audit = source_root / "reports/99_audit/_SUCCESS.json"
    if not audit.is_file() or json.loads(audit.read_text()).get("status") != "complete":
        raise RuntimeError("Successful attack-scope audit is required before bundling")
    reports = output / "reports"; reports.mkdir(parents=True)
    for stage in STAGES:
        source = source_root / "reports" / stage
        if not (source / "report.md").is_file():
            raise FileNotFoundError(f"Incomplete source stage: {source}")
        shutil.copytree(source, reports / stage, ignore=_ignore)
    ranking = pd.read_csv(reports / "06_region_models/tables/model_ranking_validation.csv")
    attack = pd.read_csv(reports / "06_attack_evaluation/tables/attack_performance_summary.csv")
    best = ranking.sort_values("mean", ascending=False).iloc[0]
    worst = attack.loc[~attack.condition.eq("clean")].sort_values("f1_mean").iloc[0]
    attack = attack.sort_values("f1_mean")
    selected = attack.loc[attack.condition.isin(["clean", worst.condition]),
                          ["condition", "f1_mean", "precision_mean", "recall_mean",
                           "accuracy_mean", "roc_auc_mean"]]
    performance_table = "\n".join(
        f"| {row.condition} | {row.f1_mean:.4f} | {row.precision_mean:.4f} | "
        f"{row.recall_mean:.4f} | {row.accuracy_mean:.4f} | {row.roc_auc_mean:.4f} |"
        for row in selected.itertuples())
    (output / "KEY_RESULTS.md").write_text(f"""# 핵심 실험 결과

clean validation의 17개 OvR·3 seed Macro F1 1위는 `{best.model}`이며 평균
{best['mean']:.6f}, 표준편차 {best['std']:.6f}이다. 입력에는 좌표 2개와 업종 코드
3개만 사용했고 지역을 직접 드러내는 특성은 제외했다.

32개 제약 기반 공격 중 선택 모델의 F1이 가장 낮은 조건은 `{worst.condition}`이며
평균 {worst.f1_mean:.6f}이다. 상세한 지역·seed별 결과는
`reports/06_attack_evaluation/tables/metrics_by_region.csv`에서 확인한다.

| 조건 | F1 | Precision | Recall | Accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
{performance_table}

공격 데이터 870,400행은 모두 좌표 거리·범주 L0·유효 조합·레이블 보존 제약을
통과했다. SHAP 결과는 막대가 아니라 특성값 색상과 SHAP 음·양 방향을 표시하는
17지역×2조건 beeswarm dot PNG 34개다. CCA·paired 통계·LIME·odds ratio도 각 단계에 있다.
""")
    (output / "README.md").write_text("""# POI 적대적 공격 실험 최종 전달본 — 2026-09-15

`KEY_RESULTS.md`에서 핵심 수치를 먼저 확인하고, `reports/`를 번호 순서대로 본다.
각 단계에는 독립된 report.md, CSV 표, metadata와 300dpi PNG가 있다. SVG는 없다.

용량이 큰 재현용 전체 예측 Parquet와 중간 샤드는 전달본에서 제외했다. 전체 증적은
원본 실험 폴더에 보존되어 있고, 전달본에는 판단에 필요한 전체 지역·seed 지표와
시각화가 포함되어 있다. 공격 원본 Parquet는 공격 유효성 재검증을 위해 포함했다.
""")
    files = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _hash(path)}
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source_root), "selected_model": str(best.model),
                "clean_validation_f1": float(best["mean"]), "worst_attack": str(worst.condition),
                "test_used": False, "excluded_from_delivery": ["work/", "all_predictions.parquet"],
                "stages": STAGES, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "_SUCCESS.json").write_text(json.dumps(
        {key: value for key, value in manifest.items() if key != "files"}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "selected_model": best.model,
                      "files": len(files), "bytes": sum(value["bytes"] for value in files.values())},
                     indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="artifacts/poi-adversarial-20260915-001")
    parser.add_argument("--output", default="artifacts/poi-final-delivery-20260915")
    args = parser.parse_args(); run(args.source_root, args.output)


if __name__ == "__main__":
    main()
