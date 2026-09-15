# 01. 기술통계와 Table One

Table One 형식의 지역·조건별 기술통계와 범주 빈도를 생성했다. 분석 단위는 POI_ID이며 clean/adversarial은 짝지어진 관측이다. 좌표 변경률은 X 100.0%, Y 100.0%였다.

## 생성 파일

- `figures/class_balance_train.png`
- `figures/paired_change_rate.png`
- `tables/class_balance.csv`
- `tables/missing_counts.csv`
- `tables/paired_change_rates.csv`
- `tables/table_one_categorical_by_region.csv`
- `tables/table_one_numeric_by_region.csv`

## 실행 범위

- 분석 데이터: train split의 paired POI 23,800개
- Seed: 42
- 모든 그림을 PNG 300 dpi로 생성함
- 상세 수치는 CSV와 metrics.json에 저장함

## 공격 범위

시각화는 사전 지정 대표 조건 `caa_high`를 사용했다. 전체 32개 공격 조건의 수치 결과는 CSV에 기록했다.
