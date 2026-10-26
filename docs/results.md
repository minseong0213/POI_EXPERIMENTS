# POI 실험 결과 안내

전체 연구 판정은 **PARTIAL**입니다. clean 비교와 기존 공격 후 지역분류 수치는 독립 검수에서 PASS했지만, 새 방어 추론과 잠금 test는 **NOT_RUN**입니다. raw RF의 이전 source 검수에는 **FAIL** 항목이 남아 있으며 실제 scientific 결과도 완료되지 않았습니다.

| 확인할 내용 | 자료 |
|---|---|
| 피처 3조건 × 8모델 동일 기준 비교 | [성능 보고서](../new/poi-unified-20261020-001/reports/01_clean_feature_comparison/report.md) · [F1 신뢰구간 그림](../new/poi-unified-20261020-001/reports/01_clean_feature_comparison/figures/macro_f1_bootstrap_ci.png) · [seed 요약표](../new/poi-unified-20261020-001/reports/01_clean_feature_comparison/tables/macro_seed_summary.csv) |
| clean 성능 수치 검수와 한계 | [독립 검수: 수치 PASS / 전체 PARTIAL](../new/poi-unified-20261020-001/reports/99_independent_review/clean_aggregate_actual_revision1/review.md) |
| clean 그림 수정본 검수 | [독립 결과 검수 PASS](../new/poi-unified-20261020-001/reports/99_independent_review/clean_publication_actual_revision2/review.md) |
| 공격 후 지역분류 수치 | [32조건 요약·그림](../new/poi-unified-20261020-001/reports/03_attack_effect_summary_revision1/report.md) · [독립 수치 검수](../new/poi-unified-20261020-001/reports/99_independent_review/attack_terminal_repair_actual_revision14_20261023/report.md) |
| 방어·분석 후속 평가 | [현재 범위](../new/poi-unified-20261020-001/reports/02_attack_defense_readiness/current_scope_2026-10-23.csv) · [남은 실험](../new/poi-unified-20261020-001/reports/02_attack_defense_readiness/remaining_scope_2026-10-23.csv) |

공개 자료는 같은 calibration/evaluation 표본과 평가 규칙을 사용합니다. 공격은 MLP surrogate에서 생성해 TabPFN v3로 전이한 결과이며, 새 방어모델을 거친 지역분류 성능으로 해석하지 않습니다. 잠금 test도 평가하지 않았습니다.
