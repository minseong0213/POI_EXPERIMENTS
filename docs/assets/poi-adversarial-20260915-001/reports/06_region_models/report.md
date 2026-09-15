# 06. 17개 지역 OvR 8개 모델 clean 비교

동일한 clean train/validation split에서 트리 모델 5개와 TabPFN 3개를 17개 지역 OvR,
3개 seed로 비교했다. 총 408회 학습과 408개 clean 평가 조합이다.

clean validation Macro F1 1위는 `tabpfn_v3`이며 평균은 0.997267,
seed·지역 표준편차는 0.003197이다. test split은 사용하지 않았다.

`tables/`에는 전체 지표·예측·순위·시간을, `figures/`에는 normalized confusion
matrix와 전체 및 17개 지역별 PR/ROC curve를 PNG 300 dpi로 저장했다.
TabPFN 세 버전의 clean 평균·표준편차는
`tables/tabpfn_validation_performance.csv`에 별도 성능표로 기록했다.
