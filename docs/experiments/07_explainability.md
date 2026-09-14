# 07. 설명 가능성과 odds ratio

최종 모델에 대해 17개 OvR 과제의 clean/adversarial 설명을 같은 background 표본과
같은 POI 평가 표본으로 생성한다.

- SHAP summary beeswarm/bar
- 주요 연속형 피처 SHAP dependence
- 대표 TP/FP/FN/TN의 SHAP waterfall/force HTML과 정적 대체 그림
- LIME 지역별 대표 사례와 반복 안정성
- clean↔adversarial SHAP 변화량과 ranking 변화
- 주요 피처 제거 후 성능 및 SHAP/LIME 재계산

트리 계열은 TreeSHAP을 우선 사용하고 TabPFN은 공식 지원 방식 또는 model-agnostic
방법을 별도 표기한다. 서로 다른 설명 알고리즘의 절댓값을 직접 동등 비교하지 않는다.

Odds ratio는 해석용 logistic regression을 별도로 적합한다. 연속형은 train 기준 1 SD
단위, 범주는 기준 범주를 명시한다. 지역분류 17개 각각에 대해 OR, 95% CI, p-value,
FDR q-value를 forest plot과 CSV로 남긴다. 예측 최고 모델의 feature importance를
odds ratio로 부르지 않는다.
