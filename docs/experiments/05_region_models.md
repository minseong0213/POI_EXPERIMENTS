# 05. 17개 지역 OvR 모델 비교

8개 모델 × 17개 지역 × 3 seeds를 동일 분할에서 실행한다. 모든 모델은 지역별
`region=1`, `other_region=0`의 확률을 출력해야 한다. 전처리와 class weight는 train에서만
fit한다. 튜닝 예산과 탐색 횟수를 모델별로 동일하게 제한하고 validation 외 데이터는
사용하지 않는다.

## 시각화와 표

- condition/region/model별 precision, recall, F1, accuracy, ROC-AUC, AP
- 17개 지역 macro 평균과 seed 표준편차·95% CI
- 지역별 confusion matrix와 행 정규화 버전
- 지역별 PR/ROC curve, 모델별 전체 summary curve
- clean↔adversarial 성능 변화 slope plot
- 성능 heatmap, 학습·추론 시간, peak CPU/GPU memory

ROC-AUC는 양성 확률로 계산하고 한 클래스만 있는 평가셋은 즉시 실패시킨다. threshold는
validation에서 정한 뒤 테스트에 고정한다. 모델 선택은 00단계의 robust F1 규칙을 따른다.
`all_predictions.parquet`에는 POI_ID, target_region, y_true, y_score, y_pred, condition,
model, seed, split을 저장해 모든 그래프와 검정을 재생성할 수 있게 한다.
