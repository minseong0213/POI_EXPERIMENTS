# POI 강건성 실험 결과 허브

검수 기준일은 2026-09-16이다. 전체 계획 판정은 **FAIL**이다. 현재 artifact는 공격
생성, 기술통계, clean/공격 평가와 제한된 설명 가능성까지만 입증한다. 이름이
`poi-final-delivery-20260915`인 번들도 전체 00–15 계획의 완료본이 아니다.

원 요구사항은 [전체 실험 계획](experiments/2026-09-15_full_robustness_experiment_plan.md),
검수 방법과 상세 증거는 [독립 검수 보고서](assets/poi-adversarial-20260915-001/reports/99_independent_review/review.md),
기계 판독 판정은 [verdict.json](assets/poi-adversarial-20260915-001/reports/99_independent_review/verdict.json)에 있다.

## 핵심 검증 결과

- 데이터 34,000행: train 23,800, validation 3,400, test 6,800; 17개 지역이며 ID와 split 중복이 없다.
- 모델 입력은 `X_COORD`, `Y_COORD`, 세 범주 코드뿐이고 식별자·지역·split 누수 피처는 제외됐다.
- 공격 870,400행: train 761,600, validation 108,800, 8방법×32조건. test 공격 행은 0이다.
- 8개 clean 모델의 408개 지표행과 1,387,200개 예측행을 재계산했다. 최고 clean validation macro F1은 TabPFN v3의 0.9972673138369298이다.
- 33개 clean/공격 조건의 1,683개 지표행과 5,722,200개 예측행을 재계산했다. 최저 macro F1은 `pgd_m1000`의 0.9884092715646228이다.
- SHAP PNG 34개는 17지역×clean/adversarial의 실제 dot/beeswarm이다. LIME은 Seoul clean 1개 표본뿐이다.
- paired bootstrap CI, 제안 방어모델, URE-RF, 탐지 후 지역분류, 전 표본 복원, 앙상블, sensitivity/stability/구조 ablation, 잠금 test 증적은 없다.

## 핵심 결과 바로가기

| 구분 | 보고서·표 | 대표 그림 |
| --- | --- | --- |
| 공격 생성 | [보고서](assets/poi-adversarial-20260915-001/reports/01_attack_generation/report.md), [유효성·ASR 표](assets/poi-adversarial-20260915-001/reports/01_attack_generation/tables/attack_validity_summary.csv) | [조건별 성공률](assets/poi-adversarial-20260915-001/reports/01_attack_generation/figures/attack_success_by_condition.png), [비용과 성공률](assets/poi-adversarial-20260915-001/reports/01_attack_generation/figures/attack_cost_vs_success.png) |
| 기술통계 | [보고서](assets/poi-adversarial-20260915-001/reports/01_descriptive/report.md), [수치형 Table One](assets/poi-adversarial-20260915-001/reports/01_descriptive/tables/table_one_numeric_by_region.csv) | [지역별 클래스 균형](assets/poi-adversarial-20260915-001/reports/01_descriptive/figures/class_balance_train.png), [공격 전후 변화율](assets/poi-adversarial-20260915-001/reports/01_descriptive/figures/paired_change_rate.png) |
| 분포 | [보고서](assets/poi-adversarial-20260915-001/reports/02_distribution/report.md) | [지역·조건 box plot](assets/poi-adversarial-20260915-001/reports/02_distribution/figures/boxplot_region_condition.png), [histogram](assets/poi-adversarial-20260915-001/reports/02_distribution/figures/histograms_and_deltas.png), [paired scatter](assets/poi-adversarial-20260915-001/reports/02_distribution/figures/paired_coordinate_scatter.png) |
| 임베딩·군집 | [보고서](assets/poi-adversarial-20260915-001/reports/03_embedding/report.md) | [t-SNE](assets/poi-adversarial-20260915-001/reports/03_embedding/figures/tsne_region_condition.png), [UMAP](assets/poi-adversarial-20260915-001/reports/03_embedding/figures/umap_region_condition.png), [dendrogram](assets/poi-adversarial-20260915-001/reports/03_embedding/figures/dendrogram_region_condition.png) |
| CCA·통계 | [보고서](assets/poi-adversarial-20260915-001/reports/04_statistics/report.md), [검정 원자료](assets/poi-adversarial-20260915-001/reports/04_statistics/tables/statistical_tests_all_attacks.csv) | [CCA](assets/poi-adversarial-20260915-001/reports/04_statistics/figures/cca_correlations.png), [correlation](assets/poi-adversarial-20260915-001/reports/04_statistics/figures/correlation_heatmap.png), [paired t 효과](assets/poi-adversarial-20260915-001/reports/04_statistics/figures/paired_t_effects.png) |
| clean 모델 비교 | [보고서](assets/poi-adversarial-20260915-001/reports/06_region_models/report.md), [모델 순위](assets/poi-adversarial-20260915-001/reports/06_region_models/tables/model_ranking_validation.csv) | [성능 요약](assets/poi-adversarial-20260915-001/reports/06_region_models/figures/metric_summary.png), [F1 heatmap](assets/poi-adversarial-20260915-001/reports/06_region_models/figures/clean_f1_heatmap.png), [PR](assets/poi-adversarial-20260915-001/reports/06_region_models/figures/pr_summary_seed_42.png), [ROC](assets/poi-adversarial-20260915-001/reports/06_region_models/figures/roc_summary_seed_42.png) |
| 공격 후 지역분류 | [보고서](assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/report.md), [공격별 성능표](assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/tables/attack_performance_summary.csv) | [전체 F1 heatmap](assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/figures/f1_heatmap_all_attacks.png), [confusion·PR·ROC](assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/figures/representative_confusion_pr_roc.png) |
| 설명 가능성 | [보고서](assets/poi-adversarial-20260915-001/reports/07_explainability/report.md), [SHAP 원자료](assets/poi-adversarial-20260915-001/reports/07_explainability/tables/shap_values.csv), [LIME 원자료](assets/poi-adversarial-20260915-001/reports/07_explainability/tables/lime_representative.csv) | [Seoul clean SHAP](assets/poi-adversarial-20260915-001/reports/07_explainability/figures/shap_beeswarm_seoul_clean.png), [Seoul attack SHAP](assets/poi-adversarial-20260915-001/reports/07_explainability/figures/shap_beeswarm_seoul_adversarial.png), [LIME](assets/poi-adversarial-20260915-001/reports/07_explainability/figures/lime_representative.png) |

## 00–15 상태

| 단계 | 상태 | 문서 | 실제 증적과 판정 |
| --- | --- | --- | --- |
| 00 | PASS | [공통 프로토콜](experiments/00_protocol.md) | 데이터·split·피처·모델/지표 계약 확인 |
| 01 | FAIL | [공격 생성](experiments/01_attack_generation.md) | `artifacts/poi-adversarial-20260915-001/reports/01_attack_generation/tables/attacks.parquet`는 완전하나 범위 초과 X 12행/Y 15행, 공간 라벨 보존 미검증, 공격 seed 42만 존재 |
| 02 | PARTIAL | [기술통계](experiments/02_descriptive.md) | artifact의 옛 `01_descriptive`; 대표 공격 조건 중심 |
| 03 | PARTIAL | [분포](experiments/03_distribution.md) | artifact의 옛 `02_distribution`; 대표 공격 조건 중심 |
| 04 | PARTIAL | [임베딩·군집](experiments/04_embedding_clustering.md) | artifact의 옛 `03_embedding`; 계획 일부 수행 |
| 05 | PARTIAL | [통계](experiments/05_statistics.md) | artifact의 옛 `04_statistics`; CCA 160행, paired/Bowker 224행이나 Pearson·전체 조건·CCA q값 없음 |
| 06 | PARTIAL | [지역 모델](experiments/06_region_models.md) | [clean metrics](assets/poi-adversarial-20260915-001/reports/06_region_models/tables/metrics_by_region.csv), [attack metrics](assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/tables/metrics_by_region.csv)는 재계산 PASS; paired bootstrap CI 없음 |
| 07 | NOT_RUN | [backbone ablation](experiments/07_backbone_ablation.md) | 결과 없음 |
| 08 | NOT_RUN | [제안 모델](experiments/08_proposed_model.md) | 구조 구현과 결과 없음 |
| 09 | NOT_RUN | [공격 탐지](experiments/09_attack_detection.md) | URE-RF 포함 현 계획 결과 없음 |
| 10 | NOT_RUN | [탐지 후 지역분류](experiments/10_filtered_region_classification.md) | 결과 없음 |
| 11 | NOT_RUN | [전 표본 강건성](experiments/11_full_coverage_robustness.md) | 결과 없음 |
| 12 | NOT_RUN | [앙상블](experiments/12_ensemble.md) | 결과 없음 |
| 13 | PARTIAL | [설명 가능성](experiments/13_explainability.md) | [SHAP 표](assets/poi-adversarial-20260915-001/reports/07_explainability/tables/shap_values.csv)는 680행·34그룹; LIME 범위와 모델 범위 미달 |
| 14 | NOT_RUN | [강건성 분석](experiments/14_robustness_analysis.md) | sensitivity, stability, ranking, 최종 ablation 없음 |
| 15 | NOT_RUN | [잠금 test](experiments/15_locked_test.md) | test 결과 0행 |

## 고정 artifact의 번호 대응

기존 증적의 내용과 manifest 해시를 보존하기 위해 디렉터리는 이름을 바꾸지 않았다.

| 고정 artifact 경로 | canonical 단계 |
| --- | --- |
| `reports/01_attack_generation` | 01 |
| `reports/01_descriptive` | 02 |
| `reports/02_distribution` | 03 |
| `reports/03_embedding` | 04 |
| `reports/04_statistics` | 05 |
| `reports/06_region_models`, `reports/06_attack_evaluation` | 06 |
| `reports/07_explainability` | 13 |
| `reports/99_audit`, `reports/99_independent_review` | 검수 증적 |

새 실행은 문서와 동일한 canonical 단계 번호를 사용한다. 단계별 설계 문서는 유지하며
[실험 문서 인덱스](experiments.md)에서 탐색할 수 있다.
