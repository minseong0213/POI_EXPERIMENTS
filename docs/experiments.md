# 실험 문서 인덱스

2026-09-15에 확정한 공격 데이터 재생성, TabPFN 2.5 구조 변경, URE-RF 감지기 비교와
잠금 test까지의 새 전체 계획은 [POI 지역분류 및 공격 강건성 전체 실험 계획](experiments/2026-09-15_full_robustness_experiment_plan.md)에 있다.

실험은 아래 순서를 따른다. 각 단계는 독립 문서, 설정 스냅샷, 표, 그림,
기계 판독 가능한 지표와 실행 메타데이터를 남긴다. 앞 단계의 완료 조건을 만족해야
다음 단계로 넘어간다.

| 단계 | 문서 | 주요 결과 |
| --- | --- | --- |
| 00 | [공통 프로토콜](experiments/00_protocol.md) | 17개 OvR 과제, 분할, 지표, 보고 규칙 |
| 01 | [기술통계와 Table One](experiments/01_descriptive.md) | Table One, 결측·분포·클래스 현황 |
| 02 | [분포 시각화](experiments/02_distribution.md) | box plot, histogram, scatter |
| 03 | [임베딩과 군집](experiments/03_embedding.md) | t-SNE, UMAP, dendrogram |
| 04 | [연관성과 통계 검정](experiments/04_statistics.md) | CCA, correlation, t/chi 검정, FDR |
| 05 | [지역 OvR 모델 비교](experiments/05_region_models.md) | 8개 모델, 17개 지역, PR/ROC/혼동행렬 |
| 06 | [모델 고정과 ablation](experiments/06_model_selection_ablation.md) | 최종 모델 선정, 피처군 제거 실험 |
| 07 | [설명 가능성과 odds ratio](experiments/07_explainability.md) | SHAP/LIME, dependency/force, OR/CI/p |
| 08 | [공격 탐지](experiments/08_attack_detection.md) | normal=0/attack=1 판별 성능과 해석 |
| 09 | [앙상블과 방어 파이프라인](experiments/09_ensemble_defense.md) | 새 방법, 탐지 후 지역분류, coverage |
| 10 | [강건성 분석](experiments/10_robustness.md) | sensitivity, stability, ranking, 최종 ablation |

01~10단계의 train/validation 결과와 11단계 잠금 test 평가는
`artifacts/poi-study-final-002/reports/`에 있다. 8개 모델 비교로 TabPFN v2.5를
고정한 뒤 test를 한 번의 평가 범위로 분리했다. 최종 상태는 `complete`이며 자동
감사 결과는 `artifacts/poi-study-final-002/audit.json`, TabPFN 전용 성능표는
`artifacts/poi-study-final-002/TABPFN_PERFORMANCE.md`, 원격 보존 위치는
`r2:ml-experiments/results/poi/poi-study-final-002/`이다.

생성 결과는 `artifacts/<EXP_ID>/reports/<NN_stage>/`에 저장한다. 각 단계는
`report.md`, `tables/`, `figures/`, `metrics.json`, `metadata.json`을 가져야 한다.
그림은 출판용 PNG(300 dpi)와 벡터 SVG를 함께 저장하고, 그래프의 원자료는 CSV 또는
Parquet으로 남긴다. README에는 링크와 현재 상태만 두며 결과 본문을 합치지 않는다.

보고서 작성에는 [EDA 템플릿](templates/eda_report.md)과
[모델 실험 템플릿](templates/model_report.md)을 사용한다.
