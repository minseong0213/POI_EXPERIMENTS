# 07. 설명 가능성과 odds ratio

최종 모델에 대해 17개 OvR 과제의 clean/adversarial 설명을 같은 background 표본과
같은 POI 평가 표본으로 생성한다.

- 17개 지역별 clean·공격 SHAP summary beeswarm/dot plot
- 주요 연속형 피처 SHAP dependence
- 대표 TP/FP/FN/TN의 SHAP waterfall/force HTML과 정적 대체 그림
- LIME 지역별 대표 사례와 반복 안정성
- clean↔adversarial SHAP 변화량과 ranking 변화
- 주요 피처 제거 후 성능 및 SHAP/LIME 재계산

SHAP 시각 증적은 bar chart로 만들지 않는다. 각 행은 피처, x축은 SHAP value이며 0의
왼쪽은 음의 기여, 오른쪽은 양의 기여다. 점 색상은 낮은 피처값을 파랑, 높은 값을
빨강으로 표시한다. 모든 그림은 PNG 300 dpi로만 저장하고, 점을 재현할 SHAP 값과
피처값을 CSV 또는 Parquet으로 함께 남긴다.

TabPFN v2.5는 train background 32개와 지역별 양성·음성 각 2개 validation POI를 사용한
model-agnostic permutation SHAP으로 해석한다. 같은 POI의 clean/adversarial 설명을
비교하며, 대표 지역은 Seoul이다. LIME은 train 분포에서 1,000개 perturbation을 만들고
범주형 코드는 관측 범주 안에서만 치환한다. 서로 다른 설명 알고리즘의 절댓값은 직접
동등 비교하지 않는다.

Odds ratio는 해석용 logistic regression을 별도로 적합한다. 연속형은 train 기준 1 SD
단위, 범주는 기준 범주를 명시한다. 지역분류 17개 각각에 대해 OR, 95% CI, p-value,
FDR q-value를 forest plot과 CSV로 남긴다. 예측 최고 모델의 feature importance를
odds ratio로 부르지 않는다.
