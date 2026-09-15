# 실험 문서 인덱스

2026-09-15에 확정한 공격 데이터 재생성, TabPFN 2.5 구조 변경, URE-RF 감지기 비교와
잠금 test까지의 새 전체 계획은 [POI 지역분류 및 공격 강건성 전체 실험 계획](experiments/2026-09-15_full_robustness_experiment_plan.md)에 있다.

기존 실험 증적은 2026-09-15에 로컬과 R2에서 삭제했다. 아래 순서로 모든 결과를 새로
생성한다. 각 단계는 독립 보고서, 설정 스냅샷, 표, 그림, 기계 판독 가능한 지표와 실행
메타데이터를 남긴다. 앞 단계의 완료 조건을 만족해야 다음 단계로 넘어간다.

| 단계 | 문서 | 주요 결과 |
| --- | --- | --- |
| 00 | [공통 프로토콜](experiments/00_protocol.md) | 피처, 17개 OvR 과제, 분할, 지표 고정 |
| 01 | 마스터 계획 §6 | 공격 생성과 제약·provenance 검증 |
| 02 | [기술통계와 Table One](experiments/01_descriptive.md) | Table One, 결측·분포·클래스 현황 |
| 03 | [분포 시각화](experiments/02_distribution.md) | box plot, histogram, scatter |
| 04 | [임베딩과 군집](experiments/03_embedding.md) | t-SNE, UMAP, dendrogram |
| 05 | [연관성과 통계 검정](experiments/04_statistics.md) | CCA, correlation, paired t/chi 검정, FDR |
| 06 | [지역 OvR 모델 비교](experiments/05_region_models.md) | 트리 5종·TabPFN 3종, PR/ROC/혼동행렬 |
| 07 | [모델 고정과 ablation](experiments/06_model_selection_ablation.md) | clean validation 최종 모델 선정, 피처 제거 |
| 08 | 마스터 계획 §6 | 제안 TabPFN 2.5 구조 구현과 학습 |
| 09 | [공격 탐지](experiments/08_attack_detection.md) | 제안 모델·URE-RF 등 normal=0/attack=1 비교 |
| 10 | [방어 파이프라인](experiments/09_ensemble_defense.md) | 탐지 후 고정 지역분류 성능과 coverage |
| 11 | 마스터 계획 §6 | 복원 포함 전 표본 지역분류 강건성 |
| 12 | [앙상블](experiments/09_ensemble_defense.md) | 단일 모델 고정 후 아이디어 실험 |
| 13 | [설명 가능성과 odds ratio](experiments/07_explainability.md) | SHAP dot/LIME, dependency/force, OR/CI/p |
| 14 | [강건성 분석](experiments/10_robustness.md) | sensitivity, stability, ranking, 최종 ablation |
| 15 | 마스터 계획 §6 | 모든 선택을 고정한 미사용 test 1회 평가 |

생성 결과는 `artifacts/<EXP_ID>/reports/<NN_stage>/`에 저장한다. 각 단계는
`report.md`, `tables/`, `figures/`, `metrics.json`, `metadata.json`을 가져야 한다.
그림은 출판용 PNG 300 dpi로만 저장하고 SVG는 생성하지 않는다. 그래프의 원자료는
CSV 또는 Parquet으로 남긴다. SHAP은 bar가 아닌 지역별 clean·공격 beeswarm/dot
plot으로 만든다. README에는 링크와 현재 상태만 두며 결과 본문을 합치지 않는다.

보고서 작성에는 [EDA 템플릿](templates/eda_report.md)과
[모델 실험 템플릿](templates/model_report.md)을 사용한다.
