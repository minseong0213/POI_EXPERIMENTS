# 02. 분포 시각화

지역별 box plot, 공통 bin histogram, 좌표 밀도와 paired 이동 그림을 생성했다. 그래프는 같은 축과 조건 색상을 사용하며 산점도 이동선은 지역당 30개 고정 표본이다.

## 생성 파일

- `figures/boxplot_region_condition.png`
- `figures/coordinate_hexbin.png`
- `figures/histograms_and_deltas.png`
- `figures/paired_coordinate_scatter.png`
- `tables/distribution_summary.csv`
- `tables/paired_numeric_deltas.csv`

## 실행 범위

- 분석 데이터: train split의 paired POI 23,800개
- Seed: 42
- 모든 그림을 PNG 300 dpi로 생성함
- 상세 수치는 CSV와 metrics.json에 저장함

## 공격 범위

시각화는 사전 지정 대표 조건 `caa_high`를 사용했다. 전체 32개 공격 조건의 수치 결과는 CSV에 기록했다.
