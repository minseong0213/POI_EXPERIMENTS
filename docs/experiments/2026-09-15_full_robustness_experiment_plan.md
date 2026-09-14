# POI 지역분류 및 공격 강건성 전체 실험 계획

- 작성일: **2026-09-15 (Asia/Seoul)**
- 프로토콜 버전: **2.0**
- 상태: **실험 설계 확정, 공격 데이터 재생성 및 제안 모델 구현 전**
- 프로젝트: `/home/mlops/orca/projects/poi`
- 데이터: `data/poi_34k_seed42/`
- R2 데이터: `r2:ml-experiments/datasets/poi/subsets/poi_34k_seed42_v1/`
- TabPFN 공식 소스: `vendor/tabpfn`, `v8.1.0`, commit
  `316663a3eb32c40afe1068744cc16cc9900b98a0`

이 문서는 기존 EDA·모델 비교·강건성 실험을 새 공격 데이터와 제안 방어 모델 기준으로
다시 수행하기 위한 마스터 계획이다. 세부 결과는 한 README에 합치지 않고 아래 단계별
폴더와 `report.md`에 나누어 저장한다.

## 1. 연구 목적

다음 질문에 순서대로 답한다.

1. 지역을 직접 나타내는 컬럼을 제외한 5개 피처로 17개 지역을 구분할 수 있는가?
2. 기존 clean 지역분류 성능이 가장 좋은 모델은 무엇인가?
3. 수치형·범주형·혼합형 공격이 유효한 제약 안에서 실제 지역분류 성능을 낮추는가?
4. 제안 공격 감지 모델이 URE 기반 Random Forest 감지기보다 공격을 더 잘 차단하는가?
5. 동일한 clean 보존율에서 감지 후 TabPFN 2.5의 지역분류 성능을 더 잘 보존하는가?
6. 제안한 TabPFN 2.5 구조 변경이 외부 감지기만 붙인 방식보다 강건성을 높이는가?
7. 결과가 공격 종류·강도·seed·지역·피처 제거 조건에서도 안정적인가?

## 2. 데이터와 입력 피처

원본 913,684건에서 지역 균형을 고려해 추출한 34,000개 POI를 사용한다. 동일한
`POI_ID`에서 만든 clean과 attack은 독립 표본으로 세지 않는다.

| split | POI 수 | 비율 | 용도 |
| --- | ---: | ---: | --- |
| train | 23,800 | 70% | 모델 및 감지기 학습, train 공격 생성 |
| validation | 3,400 | 10% | 모델 선택, threshold 및 hyperparameter 고정 |
| test | 6,800 | 20% | 모든 규칙을 고정한 뒤 최종 1회 평가 |

모델 입력은 다음 5개로 고정한다.

| 유형 | 피처 |
| --- | --- |
| 수치형 | `X_COORD`, `Y_COORD` |
| 범주형 | `ASORT_LCLASDC`, `ASORT_MLSFCDC`, `ASORT_SDASDC` |

`region`, 행정구역 코드, 주소, POI 이름, `POI_ID`, split, 공격 종류와 공격 강도는
모델 입력에서 제외한다. `POI_ID`는 짝지어진 평가와 누수 방지에만 사용한다.

기존 ablation에서 X/Y를 모두 제거한 지역분류 F1이 약 0.008이었으므로, 5개 피처를
모두 임의로 바꿀 수 있는 공격에서는 원래 지역을 식별할 정보가 부족하다. 강건성 주장은
좌표 이동 거리, 범주 변경 개수, 허용 전이 등 명시된 공격 예산 안으로 제한한다.

## 3. 지역분류 과제와 공통 지표

17개 지역 각각에 대해 `해당 지역=1`, `나머지 16개 지역=0`인 OvR 이진분류를
수행한다.

지역은 Busan, Chungbuk, Chungnam, Daegu, Daejeon, Gangwon, Gwangju,
Gyeongbuk, Gyeonggi, Gyeongnam, Incheon, Jeju, Jeonbuk, Jeonnam, Sejong,
Seoul, Ulsan이다.

모든 지역분류 단계에서 다음 지표를 동일하게 기록한다.

- F1-score
- Precision
- Recall
- Accuracy
- ROC-AUC
- Average Precision 및 PR curve
- confusion matrix: TN, FP, FN, TP와 행 정규화 결과
- 지역별 결과와 17개 지역 Macro 평균
- seed 평균·표준편차와 POI paired bootstrap 95% CI

주 모델 선택 지표는 **validation clean Macro F1**이다. Accuracy는 OvR 클래스
불균형의 영향을 크게 받으므로 모델 선택의 1차 지표로 사용하지 않는다. test는 모델,
피처, 공격 예산, 감지 threshold가 모두 고정되기 전까지 열지 않는다.

## 4. 최종 방어 비교의 정의

### 4.1 사용자가 지정한 주 비교

두 파이프라인은 같은 clean 데이터로 학습한 **동일한 고정 TabPFN 2.5**를 지역분류기로
사용한다.

1. `제안 공격 감지 모델 → 통과 표본 → TabPFN 2.5 지역분류`
2. `URE embedding + Random Forest 공격 감지기 → 통과 표본 → TabPFN 2.5 지역분류`

추가 기준선은 다음과 같다.

3. `감지기 없음 → TabPFN 2.5`
4. `5개 raw feature Random Forest 감지기 → 통과 표본 → TabPFN 2.5`

URE는 원 논문에서 robust embedding을 downstream 분류기에 전달하는 방법이다. 이
프로토콜에서는 embedding 위 Random Forest를 `normal=0`, `attack=1`로 학습하므로
결과표에는 **URE-RF attack detector**라고 기록한다. 이를 원 논문의 robust 지역분류
모델 결과로 표현하지 않는다.

### 4.2 모델 구조 변경 평가

제안 모델은 외부 감지기만 연결한 파이프라인과 구분해 평가한다. 공식 TabPFN 2.5
소스의 입력 encoder 앞 또는 내부에 다음 모듈을 추가한다.

- X/Y용 다중 거리 단위 수치 encoder
- 세 범주 코드와 조합 관계를 학습하는 범주 encoder
- 피처별 오염 확률 head
- 다른 피처로 공격 전 표현을 추정하는 masked-feature reconstruction layer
- 관측 표현과 복원 표현을 오염 확률로 결합하는 reliability gate
- 전체 공격 확률 head
- TabPFN 2.5 Transformer와 17개 OvR 출력 head

공식 TabPFN 원본은 submodule로 보존하고, 변경 코드는 POI 프로젝트 모듈과 명시적인
patch로 관리한다. 모델과 보고서에는 TabPFN 원 저작자, 버전, checkpoint, 변경 파일과
변경 내용을 표시한다.

구조 변경 모델은 두 방식으로 평가한다.

- **Selective mode:** 공격 head가 정상으로 통과시킨 표본의 지역분류 성능과 coverage
- **Full-coverage mode:** 내부 복원 후 모든 표본에 대해 지역을 출력한 성능

Selective mode는 4.1의 URE-RF 파이프라인과 비교하고, full-coverage mode는 감지기
없는 TabPFN 2.5 및 외부 robust 지역분류 모델과 별도로 비교한다.

## 5. 공격 데이터 재생성

기존 `poi_adversarial_data_final.csv`는 공격 방법, 예산, source/victim 모델과 생성
split이 기록되지 않았다. 기존 결과는 참고 자료로만 보존하고 새로운 최종 강건성 결론에는
사용하지 않는다.

### 5.1 공격군

| 공격군 | 방법 | 변경 피처 | 기본 예산 |
| --- | --- | --- | --- |
| 수치형 | FGSM | X/Y | 10, 50, 100, 500, 1,000m |
| 수치형 | PGD | X/Y | 같은 거리 예산, 반복 수 기록 |
| 수치형 | CW-L2 | X/Y | 거리 예산과 최적화 step 기록 |
| 수치형 | CAPGD | X/Y | 제약을 포함한 adaptive gradient 공격 |
| 범주형 | 유효 조합 완전 탐색 | 범주 3개 | L0 변경 1, 2, 3개 |
| 범주형 | PCAA | 범주 3개 | 허용 전이와 cost budget |
| 혼합형 | CAA | X/Y+범주 | 거리와 범주 cost 결합 예산 |
| 혼합형 | MOEVA | X/Y+범주 | 제약식과 query 수 기록 |

FGSM·PGD·CW는 수치 피처에 직접 적용한다. TabPFN과 트리처럼 직접 gradient를 쓰기
어려운 victim에는 미분 가능한 surrogate에서 공격을 생성해 transfer하고, MOEVA 및
범주 완전 탐색으로 black-box 공격을 추가한다.

### 5.2 공격 유효성 조건

공격 표본은 다음 조건을 모두 기록하고, 주 강건성 평가에는 조건을 만족한 표본만 쓴다.

- 원본과 `POI_ID` 및 실제 지역 label이 동일함
- 변경값이 수치 범위와 공식 범주 codebook을 만족함
- 대·중·소분류 조합이 허용된 관계를 만족함
- 좌표 이동과 범주 변경 cost가 설정 예산 이내임
- 공격 전 모델이 해당 POI를 올바르게 분류했는지 표시함
- 공격 후 예측 변화와 공격 성공 여부를 표시함
- 공격 생성 실패와 성공을 모두 저장해 성공 표본만 골라 남기지 않음

각 행에는 최소한 다음 provenance를 저장한다.

`POI_ID`, split, attack method, source model, victim model, targeted 여부, target region,
numeric budget, categorical budget, 원본·공격 피처, 이동 거리, 변경 피처 mask, 제약
충족 여부, label 보존 여부, clean·attack 예측, attack success, query 수, seed.

### 5.3 공격 데이터 자체 평가

공격의 품질은 지역분류 지표와 분리해 다음 보조 지표로 확인한다.

- Attack Success Rate
- 유효 공격 비율
- label 보존 비율
- 제약 위반률
- 평균·중앙 좌표 이동 거리
- 범주형 L0 변경 수
- query 수
- 모델 간 transfer ASR

## 6. 전체 실험 순서

### 00. 프로토콜 및 실행 환경 고정

- 본 문서, 데이터 manifest와 SHA-256 고정
- Git commit과 dirty 여부 기록
- Python, CUDA, GPU, 패키지 버전 고정
- TabPFN `v8.1.0` source와 checkpoint SHA-256 기록
- 공격 제약, 예산, seed, threshold 선택 규칙 고정

완료 조건: test를 열지 않은 상태에서 모든 설정 파일과 검증 코드가 존재한다.

### 01. 공격 데이터 재생성 및 검증

- split별 원본에서만 공격 생성
- train 공격은 학습에만, validation 공격은 선택에만, test 공격은 최종 평가에만 사용
- 공격 방법·예산·source/victim별 별도 파일과 manifest 생성
- 제약·label·pairing·hash 검증 수행

완료 조건: provenance 누락 0건, split 누수 0건, 유효성 보고서 생성.

### 02. 기술통계와 Table One

- 전체 및 지역별 clean/공격 조건 N
- 결측, 평균±표준편차, 중앙값[IQR], 최소·최대
- 범주 빈도와 비율
- 공격 방법·강도별 피처 변경률

결과: CSV/XLSX 표, `report.md`, 표 원자료.

### 03. Box plot, histogram, scatter

- 지역별 X/Y clean·공격 paired box plot
- 공격별 X/Y histogram과 paired delta histogram
- X-Y scatter/hexbin
- 범주 변경률 bar plot
- 모든 조건에 같은 축 범위와 색상 사용

결과: PNG 300dpi, SVG, 그림 원자료 CSV/Parquet.

### 04. t-SNE, UMAP, dendrogram

- clean과 공격을 같은 embedding 공간에 표시
- 지역 색상, clean/attack marker 구분
- t-SNE와 UMAP을 각각 지역별·공격별 시각화
- 지역 centroid 또는 고정 표본으로 dendrogram 작성
- embedding은 train에서 fit하고 validation/test에는 transform 가능한 방식 우선 사용

### 05. CCA, correlation, 통계 검정

- 같은 POI의 clean·attack 블록 CCA
- Pearson·Spearman correlation과 변화량
- 수치형 paired t-test 및 필요 시 Wilcoxon 검정
- 범주형 paired chi-square/Bowker 계열 검정
- effect size, 95% CI, 원 p-value와 BH-FDR q-value

### 06. 기존 지역분류 모델 비교

비교 모델은 다음 8개다.

1. XGBoost
2. LightGBM
3. Random Forest
4. CatBoost
5. Decision Tree
6. TabPFN 2.5
7. TabPFN 2.6
8. TabPFN 3

모든 모델은 clean train으로 학습하고 clean validation으로 backbone을 선택한다. 각 모델에
대해 17개 지역 OvR F1, precision, recall, accuracy, ROC-AUC, PR-AUC, confusion
matrix, PR curve와 ROC curve를 만든다.

이전 실험에서는 TabPFN 2.5가 가장 높았지만, 이는 과거 결과로 표기한다. 새 실험에서는
동일한 clean validation 규칙으로 확인한 뒤 backbone을 다시 고정한다.

### 07. Backbone 고정과 기본 ablation

- 전체 5개 피처
- X 제거, Y 제거
- 범주형 피처 각각 제거
- 좌표만 사용
- 범주만 사용
- X/Y 모두 제거
- 지역별·seed별 성능과 순위 안정성

완료 조건: backbone, 피처 구성과 모든 hyperparameter를 test 공개 전에 고정한다.

### 08. 제안 TabPFN 방어 구조 구현 및 학습

1. 새 input encoder 및 corruption/reconstruction module 구현
2. TabPFN 원 가중치를 동결하고 새 layer와 head부터 학습
3. validation 결과가 안정적일 때 마지막 Transformer block 일부만 낮은 learning rate로
   fine-tuning
4. clean·공격 paired consistency loss 적용
5. 변경 피처 mask로 피처별 공격 head 학습
6. 정상 성능 하락과 공격 성능 개선을 동시에 기록

학습 손실은 다음 항을 개별 기록한다.

```text
L = L_clean_region
  + lambda_adv * L_attack_region
  + lambda_consistency * L_clean_attack_consistency
  + lambda_detect * L_feature_attack_detection
  + lambda_reconstruct * L_feature_reconstruction
```

### 09. 공격 감지 모델 비교

- Raw-feature Random Forest detector
- URE-RF attack detector
- 제안 모델의 attack head

학습 label은 `normal=0`, `attack=1`이다. 같은 `POI_ID`의 clean·attack은 같은 split에
두고, train용 region model score는 out-of-fold 방식으로 만든다.

두 주 감지기는 validation clean 통과율이 **95%**가 되도록 threshold를 각각 고정한다.
추가 sensitivity 조건으로 clean 통과율 90%, 97.5%, 99%를 평가한다.

감지기 자체 지표:

- F1, precision, recall, accuracy, ROC-AUC, PR-AUC
- confusion matrix
- clean false positive rate
- attack rejection rate
- 공격 방법·강도·지역별 결과

### 10. 감지 후 TabPFN 2.5 지역분류

각 감지기가 정상으로 통과시킨 표본에 같은 TabPFN 2.5를 적용한다. 다음 세 조건을
분리한다.

1. 통과한 clean 데이터
2. 감지하지 못하고 통과한 attack 데이터
3. 두 조건을 합친 실제 통과 데이터

각 조건에서 지역별 및 Macro F1, precision, recall, accuracy, ROC-AUC를 동일하게
계산한다. 다음 선택 지표를 반드시 함께 둔다.

- Coverage: 전체 중 TabPFN으로 전달된 비율
- Clean retention
- Attack rejection
- 통과한 clean·attack 표본 수
- risk-coverage curve와 AURC

거부된 표본에는 지역 예측이 없으므로 accepted-only 지역분류 성능과 coverage를 분리해
보고한다. 거부 표본을 임의로 오답 처리한 F1은 최종 결과로 사용하지 않는다.

### 11. 모델 구조 변경의 full-coverage 평가

제안 모델의 복원 layer를 사용해 거부 없이 모든 표본을 분류한다. 다음을 비교한다.

- 원본 TabPFN 2.5
- TabPFN 2.5 단순 공격 데이터 augmentation
- 구조 변경 TabPFN 2.5 전체 모델
- URE+Random Forest가 지역 label을 직접 예측하는 외부 robust 기준
- 수치 공격 한정 GROOT Random Forest 보조 기준

이 단계도 동일한 지역분류 5개 지표를 clean, 공격 방법별, 공격 강도별로 계산한다.

### 12. 모델 고정 후 아이디어·ensemble 실험

- clean·공격 probability calibration
- 상위 모델 soft voting
- attack score를 사용한 가중 voting
- 제안 모델과 외부 robust 모델의 ensemble
- 단일 모델 대비 추론 시간·메모리와 성능 증가량

ensemble은 단일 제안 모델 결과가 고정된 뒤 별도 아이디어 실험으로 수행한다.

### 13. 설명 가능성, 중요 피처 제거, 통계적 해석

- 제안 모델과 고정 TabPFN 2.5의 SHAP summary
- 수치형 SHAP dependency plot
- 대표 지역 및 오류 표본 SHAP force plot
- LIME 지역별 대표 사례
- clean·공격 간 SHAP 순위 변화
- 중요 피처 제거 후 clean·공격 지역분류 성능
- 감지 결과에 대한 odds ratio, 95% CI, p-value와 FDR q-value

신경망 파라미터를 직접 odds ratio로 해석하지 않는다. 별도 logistic surrogate를 동일
피처와 split으로 적합하고, 완전분리된 계수는 `identifiable=false`로 표시한다.

### 14. Sensitivity, stability, ranking, 최종 ablation

- 공격 예산 변화
- 공격 데이터 혼합비 0, 10, 25, 50, 75, 100%
- clean retention threshold 90, 95, 97.5, 99%
- seed 42, 202, 340
- 지역별 모델 순위와 Kendall/Spearman 순위 상관
- leave-one-attack-out
- leave-one-source-model-out transfer
- 감지기까지 회피하는 adaptive attack
- encoder, corruption head, reconstruction, consistency loss, fine-tuning block 제거 ablation

### 15. 잠금 test와 최종 전달 자료

validation에서 다음을 모두 고정한 후 test 6,800개 POI를 한 번 평가한다.

- backbone과 checkpoint
- 공격 생성 코드와 예산
- 감지 threshold
- 모델 hyperparameter
- ablation 조건과 통계 규칙
- 그림과 표 생성 코드

최종 전달 폴더에는 단계별 핵심 결과, 종합 XLSX, 최종 보고서, 재현 설정, 실행 로그와
데이터·checkpoint SHA-256을 포함한다. 실패 결과와 가설에 반하는 결과도 제외하지 않는다.

## 7. 최종 판정표

### 7.1 공격 감지

| Detector | Clean retention | Attack rejection | F1 | Precision | Recall | Accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw RF | 0.95 |  |  |  |  |  |  |
| URE-RF detector | 0.95 |  |  |  |  |  |  |
| Proposed detector | 0.95 |  |  |  |  |  |  |

### 7.2 감지 후 지역분류

| Detector | 조건 | Coverage | Macro F1 | Precision | Recall | Accuracy | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 없음 | clean | 1.00 |  |  |  |  |  |
| 없음 | attack | 1.00 |  |  |  |  |  |
| URE-RF | accepted clean |  |  |  |  |  |  |
| URE-RF | accepted attack |  |  |  |  |  |  |
| Proposed | accepted clean |  |  |  |  |  |  |
| Proposed | accepted attack |  |  |  |  |  |  |

### 7.3 구조 변경 모델

| 모델 | 조건 | Macro F1 | Precision | Recall | Accuracy | ROC-AUC | Clean 대비 F1 변화 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN 2.5 | clean |  |  |  |  |  |  |
| TabPFN 2.5 | attack |  |  |  |  |  |  |
| Proposed TabPFN | clean |  |  |  |  |  |  |
| Proposed TabPFN | attack |  |  |  |  |  |  |
| URE-RF region classifier | clean |  |  |  |  |  |  |
| URE-RF region classifier | attack |  |  |  |  |  |  |

제안 감지 모델의 성공 조건은 동일한 95% clean retention에서 URE-RF보다 공격 차단율이
높고, 통과 데이터에서 TabPFN 2.5의 Macro F1과 ROC-AUC를 더 잘 보존하는 것이다.
구조 변경 TabPFN의 성공 조건은 clean 성능 감소를 함께 공개한 상태에서 공격 데이터의
Macro F1과 ROC-AUC가 원본 TabPFN 2.5 및 외부 robust 기준보다 높은 것이다.

## 8. 결과 저장 구조

새 실험 ID의 예시는 `poi-robustness-20260915-001`로 한다.

```text
artifacts/poi-robustness-20260915-001/
├── reports/
│   ├── 00_protocol/
│   ├── 01_attack_generation/
│   ├── 02_descriptive/
│   ├── 03_distribution/
│   ├── 04_embedding_clustering/
│   ├── 05_statistics/
│   ├── 06_region_models/
│   ├── 07_backbone_ablation/
│   ├── 08_proposed_model/
│   ├── 09_attack_detection/
│   ├── 10_filtered_region_classification/
│   ├── 11_full_coverage_robustness/
│   ├── 12_ensemble/
│   ├── 13_explainability/
│   ├── 14_robustness_analysis/
│   └── 15_locked_test/
├── audit.json
├── run_manifest.json
└── FINAL_REPORT.md
```

각 단계는 다음 파일을 가진다.

- `report.md`: 목적, 방법, 핵심 결과, 한계
- `tables/`: CSV, Parquet 및 필요한 XLSX
- `figures/`: PNG 300dpi와 SVG
- `metrics.json`: 기계 판독 가능한 핵심 지표
- `metadata.json`: 코드·데이터·모델·GPU·seed·시간 정보
- `config.yaml`: 실행 당시 설정 스냅샷

GPU 실행은 `gpu-orchestrator`를 사용하고 instance ID, GPU 종류, CUDA와 실행 시간을
metadata에 기록한다. 실험 종료 후 인스턴스가 삭제됐는지 확인한다. 토큰과 R2 자격
증명은 결과물, 로그, Git에 저장하지 않는다.

## 9. 현재 결과의 취급

기존 `artifacts/poi-study-final-002`와 교수님 전달 폴더는 이전 공격 CSV를 이용한
완료 결과로 보존한다. 다음 항목은 새 실험의 참고값일 뿐 최종 강건성 근거로 재사용하지
않는다.

- 공격 종류가 구분되지 않은 adversarial 성능
- 기존 Random Forest 공격 감지 성능
- 기존 detector gate 이후 accepted-only 및 강제 오답 end-to-end 결과
- 기존 공격에 대한 sensitivity·stability 분석

clean 데이터 EDA와 clean 모델 비교 결과는 데이터·코드·split hash가 동일하면 새
보고서에서 재현 확인 후 인용할 수 있다. 새 공격별 결론, 감지기 비교, 구조 변경 모델과
방어 후 지역분류 결과는 모두 새 experiment ID로 다시 생성한다.

