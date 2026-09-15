# 00. 공통 프로토콜

## 연구 과제

17개 지역마다 별도의 이진분류기를 만든다. 예를 들어 서울 과제는 `Seoul=1`,
나머지 16개 지역은 `0`이다. 같은 방식으로 17개 과제를 생성한다. 모든 모델은 동일한
POI_ID와 train/validation/test 분할을 사용한다. 원본과 공격 데이터의 같은 POI는
반드시 같은 split에 둔다.

지역 목록은 Busan, Chungbuk, Chungnam, Daegu, Daejeon, Gangwon, Gwangju,
Gyeongbuk, Gyeonggi, Gyeongnam, Incheon, Jeju, Jeonbuk, Jeonnam, Sejong,
Seoul, Ulsan이다.

## 모델 행렬

비교 모델은 XGBoost, LightGBM, Random Forest, CatBoost, Decision Tree,
TabPFN v2.5, TabPFN v2.6, TabPFN v3의 8개다. 앞의 다섯 개는 트리 계열이며
TabPFN 세 모델은 Transformer 기반 비교군이다. TabPFN 모델 가중치의 라이선스에
동의하고 headless GPU에 `TABPFN_TOKEN`을 공급해야 한다. 패키지 버전, 정확한
checkpoint 식별자, checkpoint SHA-256을 실행별로 기록한다.

TabPFN v1·v2는 현재 공식 패키지의 주 비교군으로 넣지 않는다. 꼭 필요하면 legacy
환경·행 수 제한·checkpoint 출처를 분리한 부록 실험으로 다룬다. 서로 다른 행 수로
학습한 결과를 같은 순위표에 놓지 않는다. 공식 저장소는 현재 v2, v2.5, v2.6, v3 및
이후 모델을 구분하며 v2.5는 최대 50,000행 규모를 대상으로 안내되어 이번 23,800행
학습셋을 포함한다.

## 데이터와 분할

- 데이터 버전: `datasets/poi/subsets/poi_34k_seed42_v1`
- 원본 POI 34,000개와 같은 ID의 공격 변형 34,000개
- 학습 23,800 / 검증 3,400 / 테스트 6,800 POI
- 튜닝과 모델 선택은 train/validation만 사용
- 테스트는 모든 모델·임계값·분석 규칙을 고정한 뒤 한 번 실행
- 기본 모델 입력은 좌표 2개와 업종 분류 코드 3개
- 지역과 직접 대응하는 행정구역 코드·주소·POI 식별자는 입력에서 제외

각 OvR 과제는 양성보다 음성이 약 16배 많다. 기본 비교는 데이터를 버리지 않고
전체 음성을 사용하며, 학습 데이터에서만 class weight를 계산한다. 임계값은 검증셋에서
고정하고 테스트에 그대로 적용한다. 음성 undersampling은 sensitivity 분석에서만 한다.

## 평가 단위와 지표

각 지역의 양성 클래스에 대해 precision, recall, F1, accuracy, ROC-AUC를 기록한다.
PR curve를 요구하므로 Average Precision(PR-AUC)도 함께 기록한다. 혼동행렬에는 TN,
FP, FN, TP와 정규화 버전을 모두 둔다. 17개 과제를 같은 비중으로 평균한 macro 지표,
표준편차, 95% bootstrap CI를 함께 제시한다.

모델 선택용 1차 점수는 17개 과제의 `validation clean macro F1`이다. 17개 지역 평균이
가장 높은 모델을 선택한다. 차이가 0.5%p 이내면 추론 시간이 짧은 모델을 고른다.
공격 성능은 선택 후 별도로 평가한다. 이 규칙은 테스트 공개 전에
고정한다. Accuracy는 불균형 때문에 선택 기준으로 쓰지 않는다.

## 반복과 불확실성

트리 모델은 seed 42, 202, 340으로 세 번 반복한다. TabPFN도 지원되는 모든 난수
설정을 고정하고 세 번 평가한다. 같은 POI에 대한 모델 간 차이는 paired bootstrap으로
95% CI를 계산한다. 여러 지역·피처·모델을 동시에 검정할 때 Benjamini–Hochberg FDR
보정을 적용하고 원 p-value와 q-value를 함께 저장한다.

## 보고 규칙

각 단계의 `metadata.json`에는 Git commit, dirty 여부, 데이터 SHA-256, 설정 SHA-256,
Python/CUDA/GPU, 패키지·모델 checkpoint 버전, seed, 시작·종료 시각을 기록한다.
각 그림은 제목, 축 단위, 표본 수, 조건, seed, 95% CI 여부를 표시하고 색상 매핑을
전 단계에서 동일하게 유지한다. 결과가 실패하거나 가설과 반대여도 보고서에서 제외하지
않는다.

공격 표는 공격 방법, 예산과 condition을 명시하고 실패 공격도 보존한다. 공격별 결론은
각 조건의 원자료와 제약 검증이 있는 경우에만 작성한다.

참고: [TabPFN 공식 저장소](https://github.com/PriorLabs/TabPFN),
[scikit-learn 데이터 누수 가이드](https://scikit-learn.org/stable/common_pitfalls.html).
