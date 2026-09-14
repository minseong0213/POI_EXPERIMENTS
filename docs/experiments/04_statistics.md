# 04. 연관성과 통계 검정

## CCA와 correlation

POI_ID로 짝지어진 normal 피처 블록과 adversarial 피처 블록 사이의 CCA를 수행한다.
범주형 피처는 train에서 정한 인코딩을 사용하고 고차원 희소 행렬에는 regularized CCA
또는 사전 차원 축소를 사용한다. canonical correlation, loading, permutation p-value,
bootstrap 95% CI를 전체와 지역별로 저장한다. Pearson/Spearman/Cramér's V 등 피처
유형에 맞는 상관·연관 지표와 heatmap을 함께 만든다.

## 검정

- 연속형 공격 전후: paired t-test와 Wilcoxon signed-rank를 함께 기록
- 범주형 공격 전후: 이진은 McNemar, 다범주는 Bowker/Stuart–Maxwell 계열
- 지역과 범주형 피처 연관: chi-square와 Cramér's V
- 분포 가정, 기대 빈도, 효과 크기, 95% CI를 기록
- 모든 다중 검정은 Benjamini–Hochberg FDR 보정

독립표본 t-test/chi-square만으로 paired 공격 전후 차이를 주장하지 않는다. 요청한
chi/t 결과는 반드시 효과 크기와 q-value를 함께 표와 forest plot으로 남긴다.
