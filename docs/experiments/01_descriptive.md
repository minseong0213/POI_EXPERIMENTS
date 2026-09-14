# 01. 기술통계와 Table One

목적은 모델링 전에 34,000개 POI와 공격 변형의 구조·결측·분포 변화를 확인하는 것이다.
원본과 공격을 독립 표본처럼 세지 않고 POI_ID로 짝지어 분석한다.

## 산출물

- 전체 및 지역별 Table One: normal/adversarial의 N, 결측, 평균±표준편차,
  중앙값[IQR], 범주 빈도(%), standardized mean difference
- 지역별 양성/음성 건수 및 train/validation/test 분포
- 피처별 결측률, 고유값 수, 범주 빈도, 숫자 범위와 이상값
- 공격 전후 값이 변경된 POI 수와 변경률
- `report.md`, `tables/table_one_*.csv`, `figures/missingness.*`,
  `figures/class_balance.*`

Table One의 p-value를 효과 크기처럼 해석하지 않는다. 공격 전후는 짝지어진 자료이므로
추론 통계는 04단계의 paired 검정에서 수행한다. 데이터 타입과 코드값의 선행 0을
보존하며, 지역/조건별 표본 수 합계가 manifest와 일치해야 완료다.
