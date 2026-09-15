# POI 지역분류·적대적 공격 실험 진행 보고서

> 이 문서는 실행 당시의 진행 기록이다. `_SUCCESS.json` 존재만으로 부여한 완료 표시는
> 전체 계획의 독립 완료 판정이 아니다. 현재 판정과 재계산 근거는
> [결과 허브](../results.md)와 [독립 검수 보고서](../assets/poi-adversarial-20260915-001/reports/99_independent_review/review.md)를 우선한다.

- 기준 시각: 2026-09-16 00:13 KST
- 저장소 기준 커밋: `b476432`
- 통합 결과 경로: `artifacts/poi-adversarial-20260915-001`
- 전체 설계: [2026-09-15 전체 강건성 실험 계획](2026-09-15_full_robustness_experiment_plan.md)
- 실시간 자동 실행 상태: [workflow/state.json](../assets/poi-adversarial-20260915-001/workflow/state.json)

자동 관리자 로그 `workflow/manager.log`와 PID·lock 파일은 실행 중에만 유지한다. 전체
workflow와 GPU 삭제가 정상 완료되면 자동 삭제하고 최종 `state.json`만 남긴다. 실패한
경우에는 원인 진단을 위해 로그를 보존한다.

이 문서는 현재 산출물을 기준으로 완료된 실험과 남은 실험을 구분한다. 완료 판정은
해당 단계의 `_SUCCESS.json`, 보고서, 표와 그림이 모두 존재하는 경우에만 부여했다.
실행 중인 공격 평가는 완료 결과에 포함하지 않았다.

## 독립 검수 판정

2026-09-16 새 컨텍스트의 읽기 전용 검수 에이전트가 기존 완료 주장과 성공 마커를
증명으로 사용하지 않고 원자료와 생성 코드를 대조했다. **전체 원 계획 기준 판정은
FAIL**이다. 기존 workflow의 `complete`는 공격 평가 중심의 제한된 9개 단계가 끝났다는
뜻이며 전체 강건성 계획 완료를 뜻하지 않는다.

- PASS: 데이터 분할, 5개 입력 피처와 누수 제외, 공격 cardinality와 거리·L0·조합,
  clean 8모델 지표, 공격 후 5개 지표, SHAP 34개, PNG/SVG, 전달본 파일 무결성
- PARTIAL: 공격 seed, test 메모리 격리, Table One, 분포·임베딩 시각화 범위,
  CCA·correlation, 계획상 SHAP 모델 범위
- FAIL: 실제 수치 범위·label 보존·victim provenance, paired bootstrap CI,
  지역별 LIME, 전체 최종 전달 구성
- NOT_RUN: 제안 방어 모델, URE-RF, 공격 탐지 후 지역분류, full coverage, ensemble,
  sensitivity·stability, 최종 ablation, 잠금 test

전체 판정표와 원자료 수치는 [독립 검수 보고서](../assets/poi-adversarial-20260915-001/reports/99_independent_review/review.md),
기계 판독 결과는 [verdict.json](../assets/poi-adversarial-20260915-001/reports/99_independent_review/verdict.json)과
[_FAILED.json](../assets/poi-adversarial-20260915-001/reports/99_independent_review/_FAILED.json)에
있다. 앞으로는 [독립 검수 운영 규칙](independent_review_protocol.md)에 따라 독립 검수
PASS 전에는 전체 완료로 기록하지 않는다.

## 현재 상태 요약

| 실험 항목 | 상태 | 현재 결론 | 결과·증적 |
| --- | --- | --- | --- |
| 데이터 34,000건 고정·누수 피처 제외 | 완료 | train 23,800, validation 3,400, 잠금 test 6,800. 입력은 좌표 2개와 범주 코드 3개 | [프로토콜 보고서](../assets/poi-adversarial-20260915-001/reports/00_protocol/report.md), [피처 정책](../assets/poi-adversarial-20260915-001/reports/00_protocol/tables/feature_policy.csv), [분할표](../assets/poi-adversarial-20260915-001/reports/00_protocol/tables/split_counts.csv) |
| 공격 데이터 재생성·유효성 검증 | 완료 | 32개 조건, 870,400행. 제약 충족률과 label 보존율은 모든 조건에서 100% | [공격 생성 보고서](../assets/poi-adversarial-20260915-001/reports/01_attack_generation/report.md), [조건별 유효성·ASR](../assets/poi-adversarial-20260915-001/reports/01_attack_generation/tables/attack_validity_summary.csv), [공격 성공률 그림](../assets/poi-adversarial-20260915-001/reports/01_attack_generation/figures/attack_success_by_condition.png) |
| 기술통계·Table One | 완료 | train의 clean/attack paired 23,800개 분석 | [보고서](../assets/poi-adversarial-20260915-001/reports/01_descriptive/report.md), [수치형 Table One](../assets/poi-adversarial-20260915-001/reports/01_descriptive/tables/table_one_numeric_by_region.csv), [범주형 Table One](../assets/poi-adversarial-20260915-001/reports/01_descriptive/tables/table_one_categorical_by_region.csv) |
| Box plot·histogram·scatter | 완료 | 대표 공격 `caa_high` 시각화, 전체 32개 공격 수치는 CSV에 기록 | [보고서](../assets/poi-adversarial-20260915-001/reports/02_distribution/report.md), [box plot](../assets/poi-adversarial-20260915-001/reports/02_distribution/figures/boxplot_region_condition.png), [histogram](../assets/poi-adversarial-20260915-001/reports/02_distribution/figures/histograms_and_deltas.png), [paired scatter](../assets/poi-adversarial-20260915-001/reports/02_distribution/figures/paired_coordinate_scatter.png) |
| t-SNE·UMAP·dendrogram | 완료 | 지역당 100개 paired 표본, dendrogram은 34개 지역×조건 centroid 사용 | [보고서](../assets/poi-adversarial-20260915-001/reports/03_embedding/report.md), [t-SNE](../assets/poi-adversarial-20260915-001/reports/03_embedding/figures/tsne_region_condition.png), [UMAP](../assets/poi-adversarial-20260915-001/reports/03_embedding/figures/umap_region_condition.png), [dendrogram](../assets/poi-adversarial-20260915-001/reports/03_embedding/figures/dendrogram_region_condition.png) |
| CCA·correlation·통계 검정 | 완료 | paired t/Wilcoxon, Bowker, chi-square, Cramér's V와 BH-FDR 적용 | [보고서](../assets/poi-adversarial-20260915-001/reports/04_statistics/report.md), [CCA 표](../assets/poi-adversarial-20260915-001/reports/04_statistics/tables/cca_correlations.csv), [통계 검정표](../assets/poi-adversarial-20260915-001/reports/04_statistics/tables/statistical_tests_all_attacks.csv), [상관 그림](../assets/poi-adversarial-20260915-001/reports/04_statistics/figures/correlation_heatmap.png) |
| clean 17개 지역 OvR 8모델 비교 | 완료 | clean validation Macro F1 1위는 TabPFN v3 | [모델 비교 보고서](../assets/poi-adversarial-20260915-001/reports/06_region_models/report.md), [전체 순위](../assets/poi-adversarial-20260915-001/reports/06_region_models/tables/model_ranking_validation.csv), [전체 성능표](../assets/poi-adversarial-20260915-001/reports/06_region_models/tables/metrics_summary.csv), [혼동행렬](../assets/poi-adversarial-20260915-001/reports/06_region_models/figures/confusion_matrices_normalized.png), [PR 요약](../assets/poi-adversarial-20260915-001/reports/06_region_models/figures/pr_summary_seed_42.png), [ROC 요약](../assets/poi-adversarial-20260915-001/reports/06_region_models/figures/roc_summary_seed_42.png) |
| 선택 모델의 32개 공격 성능 평가 | 완료 | TabPFN v3, seed 3개, clean+32개 조건의 99개 조합 완료 | [공격 평가 보고서](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/report.md), [공격별 성능표](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/tables/attack_performance_summary.csv), [전체 F1 heatmap](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/figures/f1_heatmap_all_attacks.png), [confusion·PR·ROC](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/figures/representative_confusion_pr_roc.png) |
| 선택 모델 SHAP·LIME·odds ratio | 부분 완료 | SHAP은 17지역×clean/`caa_high` 34개, odds ratio는 17지역 완료. LIME은 Seoul clean 대표 사례 1개만 생성 | [설명 분석 보고서](../assets/poi-adversarial-20260915-001/reports/07_explainability/report.md), [SHAP 전역 요약](../assets/poi-adversarial-20260915-001/reports/07_explainability/tables/shap_global_summary.csv), [Seoul LIME](../assets/poi-adversarial-20260915-001/reports/07_explainability/tables/lime_representative.csv), [odds ratio 표](../assets/poi-adversarial-20260915-001/reports/07_explainability/tables/odds_ratios.csv) |

## 현재까지의 핵심 결과

### 공격 데이터

공격 생성에는 자체 PyTorch surrogate `poi_mlp_surrogate_v1`을 source model로
사용했다. holdout accuracy는 0.972735였다. FGSM·PGD·CW-L2·CAPGD는 좌표를,
category exact·PCAA는 범주를, MOEVA·CAA는 좌표와 범주를 함께 공격한다. test 데이터는
공격 생성이나 조건 선택에 사용하지 않았다.

대표적인 validation source-side 공격 성공률은 다음과 같다.

| 조건 | 전체 ASR | clean 정답 표본 기준 ASR | 제약 충족률 |
| --- | ---: | ---: | ---: |
| CAA low | 0.2350 | 0.2398 | 1.0000 |
| CAA medium | 0.5985 | 0.6107 | 1.0000 |
| CAA high | 0.7403 | 0.7554 | 1.0000 |
| CAPGD 1,000m | 0.0138 | 0.0141 | 1.0000 |

이 값은 공격 생성용 surrogate에 대한 성공률이다. 실제 victim인 TabPFN v3의 F1,
precision, recall, accuracy와 ROC-AUC는 아래의 별도 공격 평가에서 측정했다. source-side
ASR과 victim 성능 감소는 서로 다른 값이므로 혼용하지 않는다.

### clean 지역분류

5개 트리 모델과 TabPFN 3개를 동일한 train/validation, 17개 OvR, seed 3개로 비교했다.

| 순위 | 모델 | Macro F1 평균 | 표준편차 |
| ---: | --- | ---: | ---: |
| 1 | TabPFN v3 | 0.997267 | 0.003197 |
| 2 | TabPFN v2.5 | 0.995070 | 0.006647 |
| 3 | TabPFN v2.6 | 0.994489 | 0.005005 |
| 4 | LightGBM | 0.993638 | 0.006440 |
| 5 | XGBoost | 0.992396 | 0.007026 |
| 6 | CatBoost | 0.989996 | 0.010279 |
| 7 | Decision Tree | 0.987587 | 0.009572 |
| 8 | Random Forest | 0.980439 | 0.016180 |

TabPFN v3의 clean validation 평균은 precision 0.997474, recall 0.997073,
F1 0.997267, accuracy 0.999677, ROC-AUC 0.999996이다. 현재 공격 평가는 이 순위 규칙에
따라 TabPFN v3를 victim backbone으로 자동 선택했다.

### TabPFN v3 공격 성능

TabPFN v3를 clean train으로 학습한 뒤 validation의 clean과 32개 공격 조건을 seed
3개로 평가했다. 총 1,683개 지역별 지표 행과 5,722,200개 OvR 예측을 생성했으며 실행
시간은 16,350초였다. test는 사용하지 않았다.

| 조건 | Macro F1 | F1 감소 | Precision | Recall | Accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.997267 | 0.000000 | 0.997474 | 0.997073 | 0.999677 | 0.999996 |
| PGD 1,000m | 0.988409 | 0.008858 | 0.988361 | 0.988558 | 0.998627 | 0.999919 |
| CAPGD 1,000m | 0.988496 | 0.008772 | 0.988637 | 0.988451 | 0.998639 | 0.999919 |
| FGSM 1,000m | 0.988700 | 0.008568 | 0.988624 | 0.988860 | 0.998662 | 0.999917 |
| CAA high | 0.990976 | 0.006291 | 0.990580 | 0.991503 | 0.998933 | 0.999935 |
| CW-L2 1,000m | 0.992364 | 0.004904 | 0.992526 | 0.992372 | 0.999095 | 0.999978 |

32개 조건 중 Macro F1이 가장 낮은 공격은 PGD 1,000m였고 clean 대비 절대 감소는
0.008858이었다. 공격 제약은 모두 유효하지만 TabPFN v3로 전이됐을 때의 성능 감소는
1 percentage point보다 작았다. 이는 현재 공격의 제약 유효성과 별개로 victim 전이
강도가 제한적이라는 결과다. 세부 수치는 [공격별 성능표](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/tables/attack_performance_summary.csv)와
[지역별 전체 지표](../assets/poi-adversarial-20260915-001/reports/06_attack_evaluation/tables/metrics_by_region.csv)에 있다.

### CCA와 탐색 분석

대표 공격 `caa_high`의 첫 두 canonical correlation은 각각 0.999855와 0.999949다.
세 번째부터는 0.465196 이하로 낮아진다. 이 결과는 전체 표현의 공통 축 일부가 유지되는
동시에 다른 축에서 변화가 생겼다는 탐색적 근거이며, 공격 강건성의 결론은 모델 성능
평가와 함께 판단한다.

### SHAP·LIME·odds ratio

TabPFN v3의 17개 지역 OvR을 clean과 대표 공격 `caa_high`에서 설명했다. SHAP
permutation 방식의 beeswarm/dot PNG는 34개이며, 전 조건 평균 중요도 1위는
`Y_COORD`였다. `Y_COORD` 제거 조건의 SHAP도 다시 계산했다. 좌표 dependency,
대표 표본 waterfall·force와 별도 다변량 Binomial GLM의 odds ratio·95% CI·p-value·
FDR q-value를 함께 저장했다. 분석 시간은 1,941초였고 test는 사용하지 않았다.

LIME은 설정의 `representative_region: Seoul`에 따라 **Seoul clean 대표 표본 1개만**
생성했다. 다른 16개 지역과 adversarial 조건의 LIME은 없다. 이는 전체 계획의
“LIME 지역별 대표 사례” 완료 조건을 충족하지 않으며 별도 추가 실행이 필요하다. 현재
감사 스크립트는 SHAP의 17지역×2조건 cardinality만 검사하고 LIME 지역 수는 검사하지
않았기 때문에 이 누락을 탐지하지 못했다.

결과는 [설명 분석 보고서](../assets/poi-adversarial-20260915-001/reports/07_explainability/report.md),
[SHAP 원자료](../assets/poi-adversarial-20260915-001/reports/07_explainability/tables/shap_values.csv),
[LIME 결과](../assets/poi-adversarial-20260915-001/reports/07_explainability/tables/lime_representative.csv),
[odds ratio forest](../assets/poi-adversarial-20260915-001/reports/07_explainability/figures/odds_ratio_forest_representative.png)에서 확인한다.

## 완료된 자동 실행 범위

자동 workflow는 다음 순서를 모두 완료했다.

1. TabPFN v3의 clean+32개 공격 조건, seed 3개 평가
2. SHAP beeswarm/dot, dependency, force, Seoul 대표 LIME 1개와 logistic surrogate
   odds ratio·95% CI·p-value 생성
3. 공격 평가와 설명 결과를 로컬·R2에 보존
4. GPU 인스턴스 `51091286` 삭제 및 API 목록 부재 확인
5. 현재 범위 감사와 전달용 결과 묶음 생성
6. 통합 보고서를 R2에 게시

SHAP 1차 실행 `poi-selected-explain-20260915-001`은 결과를 만들기 전에 SHAP 0.46의
color 변환 코드가 최신 NumPy와 충돌해 종료 코드 1로 실패했다. 2차 실행
`poi-selected-explain-20260915-002`에서는 NumPy만 고정해 pandas 3.0.5와
statsmodels 0.14.5가 충돌했다. NumPy 2.0.2, pandas 2.3.3, scipy 1.13.1,
matplotlib 3.9.4, scikit-learn 1.6.1 등 프로젝트 분석 의존성 전체를 GPU에서 import
검증한 뒤 `poi-selected-explain-20260915-003`으로 재실행했다. 현재 작업 상태는
종료 코드 0의 완료다. 두 실패 모두 모델 계산 시작 전 환경 구성 단계에서 발생했으므로
공격 평가와 최종 설명 결과에는 영향을 주지 않았다.

SHAP은 bar chart가 아니라 SHAP value 0의 양쪽에 점이 분포하고 피처값의 높고 낮음이
색으로 표현되는 beeswarm/dot PNG로 생성하도록 구현돼 있다. SVG는 만들지 않는다.

## 남은 작업

### 현재 공격 실험을 닫기 위해 남은 작업

- [x] TabPFN v3 공격 평가 99개 조합 완료 및 `_SUCCESS.json` 확인
- [x] 공격 방법·강도·지역·seed별 F1, precision, recall, accuracy, ROC-AUC 표 생성
- [x] clean 대비 성능 감소량, PR·ROC·confusion matrix PNG 생성
- [x] 선택 모델의 17지역 clean/attack SHAP와 odds ratio 생성
- [ ] 선택 모델의 17지역별 LIME 대표 사례 생성
- [x] 로컬·R2 결과의 파일 수, 해시, PNG 규칙, test 미사용 감사
- [x] 교수님 전달용 공격 실험 묶음 갱신

현재 자동 실행 범위의 상태는 `complete`지만, 전체 계획 기준으로 지역별 LIME은
미완료다. 감사 결과는
[99_audit 보고서](../assets/poi-adversarial-20260915-001/reports/99_audit/report.md),
교수님에게 보여줄 현재 상태와 핵심 결과는 [결과 허브](../results.md)에서 확인한다.

### 전체 방어 실험에서 남은 작업

- [ ] 5개 피처 제거 조합과 좌표/범주 단독 사용에 대한 선택 backbone ablation
- [ ] 제안 방어 모델 구현: 수치·범주 encoder, 피처 오염 head, reconstruction layer,
  reliability gate, 공격 head와 지역 OvR head
- [ ] 제안 모델 학습: clean/attack consistency, 탐지, 복원과 지역분류 loss의 개별 기록
- [ ] 공격 탐지기 비교: raw-feature RF, URE-RF, 제안 모델 attack head
- [ ] 동일 clean retention 95%에서 감지 F1, precision, recall, accuracy, ROC-AUC,
  PR-AUC, attack rejection과 clean false positive 비교
- [ ] 각 감지기가 통과시킨 데이터에 고정 지역분류기를 적용하고 5개 지역분류 지표와
  coverage·risk-coverage·AURC 계산
- [ ] 제안 모델 full-coverage 복원 성능과 원본 TabPFN, attack augmentation,
  URE-RF 지역분류 기준 비교
- [ ] 단일 모델 고정 후 앙상블 아이디어 실험
- [ ] 중요 피처 제거 후 성능 및 제안 모델 SHAP·LIME 재분석
- [ ] sensitivity, stability, ranking과 최종 ablation
- [ ] 모든 선택을 고정한 뒤 잠금 test 1회 평가

현재 코드의 전체 계획 문서는 구조 변경 대상을 TabPFN v2.5로 적었지만, 새 clean 비교의
실제 1위는 TabPFN v3다. 제안 방어 모델 구현 전에 구조 변경 backbone을 v2.5로 유지할지,
사전 정의한 “clean 최고 성능 모델” 규칙에 따라 v3로 바꿀지 실험 프로토콜에 명시적으로
고정해야 한다. 어느 쪽도 아직 자체 방어 모델 성능 결과로 간주할 수 없다.

## 현재 사용할 수 없는 결론

- 제안 자체 방어 모델이 TabPFN 또는 URE-RF보다 강건하다는 결론
- 공격 감지 후 지역분류 성능과 coverage
- 선택 모델과 제안 방어 모델의 SHAP·LIME 비교 및 중요 피처 결론
- 앙상블, sensitivity, stability, 최종 ablation 결과
- 잠금 test 일반화 성능

위 항목은 해당 단계가 완료되고 결과 파일과 `_SUCCESS.json`이 생성되기 전까지 보고용
결론으로 사용하지 않는다.
