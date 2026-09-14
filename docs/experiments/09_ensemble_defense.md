# 09. 앙상블과 방어 파이프라인

단일 모델 결과가 고정된 뒤 validation 순위 상위 3개인 TabPFN v2.5, TabPFN v3,
LightGBM의 확률을 동일 가중치로 평균한 soft-voting 앙상블을 만든다. test 예측으로
구성이나 가중치를 선택하지 않는다. 단일 최고 모델과 동일한 POI·지표·seed에서 비교한다.

방어 파이프라인은 `공격 탐지 → 정상으로 통과한 표본 → 지역 OvR 분류` 순서다.
통과 표본 정확도만 보고하면 거부가 많은 방법이 유리해지므로 아래를 모두 기록한다.

- clean acceptance와 정상 오탐률
- attack rejection과 공격 누락률
- coverage(전체 입력 중 분류 결과를 낸 비율)
- accepted-only 지역분류 성능
- 거부를 실패로 센 end-to-end precision/recall/F1/accuracy
- 공격 없는 clean baseline 대비 효용 변화

성능 표, selective risk–coverage curve, 파이프라인 confusion/Sankey, 지역별 heatmap,
모델 시간·메모리를 남긴다. 새 아이디어가 여러 개면 각각 별도 config와 EXP_ID를 쓴다.
