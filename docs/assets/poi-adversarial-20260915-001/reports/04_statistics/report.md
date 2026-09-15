# 04. CCA, correlation, paired 통계 검정

좌표의 paired t/Wilcoxon, 범주 코드의 Bowker symmetry, 지역 연관 chi-square와 Cramér's V를 계산하고 전체 검정에 BH-FDR을 적용했다. CCA는 joint one-hot/표준화 공간을 공통 PCA로 축소한 뒤 paired clean/adversarial 블록에 적용했다.

## 생성 파일

- `figures/cca_correlations.png`
- `figures/correlation_heatmap.png`
- `figures/paired_t_effects.png`
- `tables/cca_correlations.csv`
- `tables/correlations.csv`
- `tables/statistical_tests.csv`

## 실행 범위

- 분석 데이터: train split의 paired POI 23,800개
- Seed: 42
- 모든 그림을 PNG 300 dpi로 생성함
- 상세 수치는 CSV와 metrics.json에 저장함

## 공격 범위

시각화는 사전 지정 대표 조건 `caa_high`를 사용했다. 전체 32개 공격 조건의 수치 결과는 CSV에 기록했다.
