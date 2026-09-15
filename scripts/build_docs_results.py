"""Publish a small, reviewable result set for the repository documentation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "poi-adversarial-20260915-001"
SOURCE = ROOT / "artifacts" / EXPERIMENT_ID
DESTINATION = ROOT / "docs" / "assets" / EXPERIMENT_ID

PUBLISHED = [
    "workflow/state.json",
    "reports/00_protocol/report.md",
    "reports/00_protocol/tables/feature_policy.csv",
    "reports/00_protocol/tables/split_counts.csv",
    "reports/99_independent_review/review.md",
    "reports/99_independent_review/verdict.json",
    "reports/99_independent_review/_FAILED.json",
    "reports/99_audit/report.md",
    "reports/01_attack_generation/report.md",
    "reports/01_attack_generation/tables/attack_validity_summary.csv",
    "reports/01_attack_generation/figures/attack_success_by_condition.png",
    "reports/01_attack_generation/figures/attack_cost_vs_success.png",
    "reports/01_descriptive/report.md",
    "reports/01_descriptive/tables/table_one_numeric_by_region.csv",
    "reports/01_descriptive/tables/table_one_categorical_by_region.csv",
    "reports/01_descriptive/figures/class_balance_train.png",
    "reports/01_descriptive/figures/paired_change_rate.png",
    "reports/02_distribution/report.md",
    "reports/02_distribution/figures/boxplot_region_condition.png",
    "reports/02_distribution/figures/histograms_and_deltas.png",
    "reports/02_distribution/figures/paired_coordinate_scatter.png",
    "reports/03_embedding/report.md",
    "reports/03_embedding/figures/tsne_region_condition.png",
    "reports/03_embedding/figures/umap_region_condition.png",
    "reports/03_embedding/figures/dendrogram_region_condition.png",
    "reports/04_statistics/report.md",
    "reports/04_statistics/tables/statistical_tests_all_attacks.csv",
    "reports/04_statistics/tables/cca_correlations.csv",
    "reports/04_statistics/figures/cca_correlations.png",
    "reports/04_statistics/figures/correlation_heatmap.png",
    "reports/04_statistics/figures/paired_t_effects.png",
    "reports/06_region_models/report.md",
    "reports/06_region_models/tables/model_ranking_validation.csv",
    "reports/06_region_models/tables/metrics_summary.csv",
    "reports/06_region_models/tables/metrics_by_region.csv",
    "reports/06_region_models/figures/metric_summary.png",
    "reports/06_region_models/figures/clean_f1_heatmap.png",
    "reports/06_region_models/figures/confusion_matrices_normalized.png",
    "reports/06_region_models/figures/pr_summary_seed_42.png",
    "reports/06_region_models/figures/roc_summary_seed_42.png",
    "reports/06_attack_evaluation/report.md",
    "reports/06_attack_evaluation/tables/attack_performance_summary.csv",
    "reports/06_attack_evaluation/tables/metrics_by_region.csv",
    "reports/06_attack_evaluation/figures/f1_heatmap_all_attacks.png",
    "reports/06_attack_evaluation/figures/representative_confusion_pr_roc.png",
    "reports/07_explainability/report.md",
    "reports/07_explainability/tables/shap_values.csv",
    "reports/07_explainability/tables/shap_global_summary.csv",
    "reports/07_explainability/tables/lime_representative.csv",
    "reports/07_explainability/tables/odds_ratios.csv",
    "reports/07_explainability/figures/shap_beeswarm_seoul_clean.png",
    "reports/07_explainability/figures/shap_beeswarm_seoul_adversarial.png",
    "reports/07_explainability/figures/lime_representative.png",
    "reports/07_explainability/figures/odds_ratio_forest_representative.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [relative for relative in PUBLISHED if not (SOURCE / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    records = []
    for relative in PUBLISHED:
        source = SOURCE / relative
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_digest = sha256(source)
        if source.suffix in {".md", ".json"}:
            destination.write_bytes(source.read_bytes().rstrip(b"\r\n") + b"\n")
        else:
            shutil.copy2(source, destination)
        records.append({
            "path": relative,
            "bytes": destination.stat().st_size,
            "source_sha256": source_digest,
            "sha256": sha256(destination),
        })

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "source_root": str(SOURCE.relative_to(ROOT)),
        "publication_scope": "Documentation summaries and representative PNG figures only",
        "overall_verdict": "FAIL",
        "files": records,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Published {len(records)} files to {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
