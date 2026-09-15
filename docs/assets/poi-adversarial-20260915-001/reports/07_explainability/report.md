# 07. 선택 모델 SHAP·LIME·odds ratio

clean validation 1위 `tabpfn_v3`의 17개 OvR 모델을 clean과 `caa_high`에서 설명했다. SHAP은 5개 원본 피처의 음·양 방향, 점 분포와 피처값 색상을 함께 표시하는 beeswarm dot plot이며 17지역×2조건 34개를 300dpi PNG로 저장했다.

전체 조건 평균 |SHAP| 1위 `Y_COORD`를 제거한 설명도 재계산했다. LIME, 좌표 dependence, waterfall/force와 별도 다변량 Binomial GLM의 odds ratio·95% CI·p-value·FDR q-value는 tables, figures, html에 기록했다. test는 사용하지 않았다.
