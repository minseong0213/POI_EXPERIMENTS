# 모델 실험 보고서: <EXP_ID>

## 실행 정보

| 항목 | 값 |
| --- | --- |
| 단계 / 모델 / checkpoint | |
| 대상 지역 / task | |
| 데이터 버전 / SHA-256 | |
| Git commit / dirty | |
| 설정 / seed | |
| Python / CUDA / GPU | |
| 시작 / 종료 / 소요시간 | |

## 실험 질문과 선택 규칙

비교 가설, 1차 지표, threshold 결정법, 튜닝 예산, 사전 정의한 동률 규칙을 기록한다.

## 데이터와 전처리

train/validation/test의 양성·음성 수, class weight, 포함·제외 피처, train에서 fit한
전처리와 결측 처리를 기록한다. 같은 POI의 normal/adversarial pairing을 검증한다.

## 결과

precision, recall, F1, accuracy, ROC-AUC, AP와 95% CI를 clean/adversarial 조건별로
기록한다. confusion matrix, PR/ROC curve, 지역×모델 heatmap, 시간·메모리 그림을 링크한다.

## 비교와 오류 분석

paired 차이와 CI, TP/FP/FN/TN 대표 사례, 지역별 실패 양상, seed 변동을 기록한다.
test 결과는 최종 잠금 실행에서만 채운다.

## 해석과 제한

모델 선택 근거와 반대 결과도 남긴다. Accuracy 단독 해석, 불균형, 공격 종류 부재,
라이선스와 checkpoint 접근 조건을 명시한다.

## 완료 확인

- [ ] 모든 지역·모델·seed 실행 또는 실패 사유 기록
- [ ] 예측 원자료에 POI_ID·score·label·condition 포함
- [ ] 지표를 예측 원자료에서 재계산해 일치 확인
- [ ] checkpoint 및 데이터·설정 해시 기록
- [ ] 모든 그림의 PNG 300 dpi와 표 CSV/Parquet 생성
- [ ] 출력 경로 덮어쓰기 방지 확인
