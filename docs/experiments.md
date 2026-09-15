# 실험 문서 인덱스

현재 판정, 핵심 수치와 실제 artifact 링크는 [결과 허브](results.md)에서 확인한다.
실행 이력은 [2026-09-15 실험 진행 보고서](experiments/2026-09-15_experiment_progress_report.md)에 있다.

모든 완료 주장은 [독립 검수 에이전트 운영 규칙](experiments/independent_review_protocol.md)에
따라 원자료와 코드를 별도 검수한 뒤 확정한다.

현재 전체 계획의 독립 판정은 **FAIL**이다. 세부 근거는
[독립 검수 보고서](assets/poi-adversarial-20260915-001/reports/99_independent_review/review.md)에
있다.

2026-09-15에 확정한 공격 데이터 재생성, TabPFN 2.5 구조 변경, URE-RF 감지기 비교와
잠금 test까지의 새 전체 계획은 [POI 지역분류 및 공격 강건성 전체 실험 계획](experiments/2026-09-15_full_robustness_experiment_plan.md)에 있다.

검수된 기존 artifact는 경로와 manifest를 보존한다. 새 실행은 아래 canonical 번호를
사용한다. 각 단계는 독립 보고서, 설정 스냅샷, 표, 그림, 기계 판독 가능한 지표와 실행
메타데이터를 남긴다. 앞 단계의 완료 조건을 만족해야 다음 단계로 넘어간다.

| 단계 | 문서 | 주요 결과 |
| --- | --- | --- |
| 00 | [공통 프로토콜](experiments/00_protocol.md) | 피처, 17개 OvR 과제, 분할, 지표 고정 |
| 01 | [공격 생성](experiments/01_attack_generation.md) | 공격 생성과 제약·provenance 검증 |
| 02 | [기술통계와 Table One](experiments/02_descriptive.md) | Table One, 결측·분포·클래스 현황 |
| 03 | [분포 시각화](experiments/03_distribution.md) | box plot, histogram, scatter |
| 04 | [임베딩과 군집](experiments/04_embedding_clustering.md) | t-SNE, UMAP, dendrogram |
| 05 | [연관성과 통계 검정](experiments/05_statistics.md) | CCA, correlation, paired t/chi 검정, FDR |
| 06 | [지역 OvR 모델 비교](experiments/06_region_models.md) | 트리 5종·TabPFN 3종, PR/ROC/혼동행렬 |
| 07 | [backbone ablation](experiments/07_backbone_ablation.md) | clean validation 최종 모델 선정, 피처 제거 |
| 08 | [제안 모델](experiments/08_proposed_model.md) | 제안 TabPFN 2.5 구조 구현과 학습 |
| 09 | [공격 탐지](experiments/09_attack_detection.md) | 제안 모델·URE-RF 등 normal=0/attack=1 비교 |
| 10 | [탐지 후 지역분류](experiments/10_filtered_region_classification.md) | 고정 지역분류 성능과 coverage |
| 11 | [전 표본 강건성](experiments/11_full_coverage_robustness.md) | 복원 포함 전 표본 지역분류 강건성 |
| 12 | [앙상블](experiments/12_ensemble.md) | 단일 모델 고정 후 아이디어 실험 |
| 13 | [설명 가능성과 odds ratio](experiments/13_explainability.md) | SHAP dot/LIME, dependency/force, OR/CI/p |
| 14 | [강건성 분석](experiments/14_robustness_analysis.md) | sensitivity, stability, ranking, 최종 ablation |
| 15 | [잠금 test](experiments/15_locked_test.md) | 모든 선택을 고정한 미사용 test 1회 평가 |

생성 결과는 `artifacts/<EXP_ID>/reports/<NN_stage>/`에 저장한다. 각 단계는
`report.md`, `tables/`, `figures/`, `metrics.json`, `metadata.json`을 가져야 한다.
그림은 출판용 PNG 300 dpi로만 저장하고 SVG는 생성하지 않는다. 그래프의 원자료는
CSV 또는 Parquet으로 남긴다. SHAP은 bar가 아닌 지역별 clean·공격 beeswarm/dot
plot으로 만든다. README에는 링크와 현재 상태만 두며 결과 본문을 합치지 않는다.

보고서 작성에는 [EDA 템플릿](templates/eda_report.md)과
[모델 실험 템플릿](templates/model_report.md)을 사용한다.

파일 역할과 이름 규칙은 [저장소 구조 규칙](repository_layout.md), 설정의 canonical·legacy
구분은 [`configs/README.md`](../configs/README.md)에 명시한다.
