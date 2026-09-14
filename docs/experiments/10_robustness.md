# 10. Sensitivity, stability, ranking, 최종 ablation

## Sensitivity

공격 비율 0/10/25/50/75/100%, 지역 판별 threshold 0.3~0.7, 공격 탐지 threshold
0.1~0.9를 한 번에 하나씩 바꾼다. 공격 종류와 강도 정보가 현재 데이터에 없으므로
PGD/CW/FGSM별 분석은 수행하지 않는다. seed별 원자료, 평균과 표준편차를 저장한다.

## Stability

3개 seed와 17개 지역에 따른 robust F1과 모델 순위의 변동을 본다. 평균·표준편차·
최솟값·최댓값과 seed 쌍별 Kendall 순위 상관을 기록한다. 피처 제거 안정성은 06단계의
지역별 ablation heatmap과 bootstrap CI로 분리해 제시한다.

## Ranking

각 seed와 지역에서 clean/adversarial 평균 F1로 모델 순위를 매긴 뒤 평균 순위와
지역별 순위 heatmap, 성능 radar를 만든다. radar는 절대 수치 표를 대체하지 않는다. 17개 지역을 독립 반복
실험으로 과장하지 않고 seed 및 paired POI 불확실성을 함께 제시한다.

## 최종 ablation과 보고서

앙상블 구성요소, 탐지기, threshold, 주요 피처군을 제거해 end-to-end 성능 기여를
검증한다. 최종 보고서는 단계별 보고서를 링크하고 데이터 버전, 선택 규칙, 실패 실행,
한계, 재현 명령을 포함한다. 공격 종류 부재와 지역 균등 표본이라는 제한을 결론에
반드시 명시한다.
