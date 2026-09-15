"""Write the immutable experiment-scope report before model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from .data import CATEGORICAL, NUMERIC, sha256


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_dir, attacks_file, tree_config, tabpfn_config, output):
    data_dir, attacks_file, output = Path(data_dir), Path(attacks_file), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True); figures.mkdir()
    manifest = pd.read_csv(data_dir / "sample_manifest.csv", dtype=str)
    attacks = pd.read_parquet(attacks_file)
    tree = yaml.safe_load(Path(tree_config).read_text())
    tabpfn = yaml.safe_load(Path(tabpfn_config).read_text())
    split = manifest.groupby(["split", "region"]).size().rename("count").reset_index()
    split.to_csv(tables / "split_counts_by_region.csv", index=False)
    split.groupby("split").agg(rows=("count", "sum"), regions=("region", "nunique")).reset_index().to_csv(
        tables / "split_counts.csv", index=False)
    included = [{"feature": name, "type": "numeric", "role": "model_input", "reason": ""}
                for name in NUMERIC]
    included += [{"feature": name, "type": "categorical", "role": "model_input", "reason": ""}
                 for name in CATEGORICAL]
    excluded = [
        {"feature": "region", "type": "label", "role": "excluded", "reason": "classification target"},
        {"feature": "POI_ID", "type": "identifier", "role": "excluded", "reason": "row identity only"},
        {"feature": "address/name/admin-area fields", "type": "leakage", "role": "excluded",
         "reason": "directly reveals region"},
    ]
    pd.DataFrame(included + excluded).to_csv(tables / "feature_policy.csv", index=False)
    model_rows = [{"family": "tree", "model": model, "seeds": len(tree["seeds"]), "regions": 17,
                   "selection_split": "validation", "selection_condition": "clean"}
                  for model in tree["models"]]
    model_rows += [{"family": "TabPFN", "model": f"tabpfn_{version.replace('.', '_')}",
                    "seeds": len(tabpfn["seeds"]), "regions": 17,
                    "selection_split": "validation", "selection_condition": "clean"}
                   for version in tabpfn["versions"]]
    pd.DataFrame(model_rows).to_csv(tables / "model_matrix.csv", index=False)
    attack_matrix = attacks[["attack_method", "attack_condition", "numeric_budget_m",
                             "categorical_budget_l0"]].drop_duplicates().sort_values("attack_condition")
    attack_matrix.to_csv(tables / "attack_matrix.csv", index=False)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=split, x="region", y="count", hue="split", ax=ax)
    ax.tick_params(axis="x", rotation=55); ax.set_title("Fixed split counts by region")
    fig.savefig(figures / "fixed_split_counts.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    counts = attacks.groupby(["attack_method", "attack_condition"]).size().rename("rows").reset_index()
    heat = counts.pivot(index="attack_method", columns="attack_condition", values="rows").fillna(0)
    fig, ax = plt.subplots(figsize=(17, 5))
    sns.heatmap(heat, cmap="Blues", cbar_kws={"label": "rows"}, ax=ax)
    ax.set_title("Generated attack-condition coverage (train + validation)")
    fig.savefig(figures / "attack_condition_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "status": "complete", "date": "2026-09-15", "git_commit": commit,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(), "sample_rows": len(manifest), "regions": manifest.region.nunique(),
        "attack_rows": len(attacks), "attack_conditions": attacks.attack_condition.nunique(),
        "selection": "mean clean validation OvR F1 across 3 seeds and 17 regions",
        "test_used": False,
        "sha256": {"poi_data_region.csv": sha256(data_dir / "poi_data_region.csv"),
                   "sample_manifest.csv": sha256(data_dir / "sample_manifest.csv"),
                   "attacks.parquet": _hash(attacks_file)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "report.md").write_text("""# 00. POI 지역분류 및 적대적 공격 실험 프로토콜

실험일은 2026-09-15이다. 고정된 34,000개 POI를 train 23,800개, validation
3,400개, test 6,800개로 나눴다. 이번 단계의 모델 선택·공격 생성·공격 평가는
train과 validation만 사용하며 test는 열지 않는다.

모델 입력은 좌표 2개와 업종 분류 코드 3개뿐이다. 지역명·주소·행정구역·POI 이름처럼
지역을 직접 드러내는 열과 식별자는 제외한다. 5개 트리 모델과 TabPFN v2.5·v2.6·v3를
같은 clean validation의 17개 OvR F1로 비교해 한 모델을 고정한다.

공격은 수치형 FGSM·PGD·CW-L2·CAPGD, 범주형 exact·PCAA, 혼합형 MOEVA·CAA의
32개 조건이다. 좌표 거리, 범주 L0, 유효 범주 조합, 레이블 보존 제약을 모두 통과한
행만 평가한다. 설정과 행 수는 tables, 흐름 확인용 그림은 300dpi PNG에 기록했다.
""")
    (output / "_SUCCESS.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/poi_34k_seed42")
    parser.add_argument("--attacks-file", default=(
        "artifacts/poi-adversarial-20260915-001/reports/01_attack_generation/tables/attacks.parquet"))
    parser.add_argument("--tree-config", default="configs/stages/06_region_models_tree.yaml")
    parser.add_argument("--tabpfn-config", default="configs/stages/06_region_models_tabpfn.yaml")
    parser.add_argument("--output", default="artifacts/poi-adversarial-20260915-001/reports/00_protocol")
    args = parser.parse_args()
    run(args.data_dir, args.attacks_file, args.tree_config, args.tabpfn_config, args.output)


if __name__ == "__main__":
    main()
