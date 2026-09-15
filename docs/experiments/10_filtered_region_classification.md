# 10. 탐지 후 지역분류 파이프라인

방어 파이프라인은 `공격 탐지 → 정상으로 통과한 표본 → 지역 OvR 분류` 순서다.
통과 표본 정확도만 보고하면 거부가 많은 방법이 유리해지므로 아래를 모두 기록한다.

- clean acceptance와 정상 오탐률
- attack rejection과 공격 누락률
- coverage(전체 입력 중 분류 결과를 낸 비율)
- accepted-only 지역분류 성능
- 거부를 실패로 센 end-to-end precision/recall/F1/accuracy
- 공격 없는 clean baseline 대비 효용 변화

성능 표, selective risk–coverage curve, 파이프라인 confusion/Sankey, 지역별 heatmap,
모델 시간·메모리를 남긴다. 앙상블 실험은 12단계 문서에서 별도로 정의한다.
