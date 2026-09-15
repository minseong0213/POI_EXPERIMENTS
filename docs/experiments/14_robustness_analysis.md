# 14. Sensitivity, stability, ranking, 최종 ablation

## Sensitivity

공격 비율 0/10/25/50/75/100%, 공격 방법·예산, 지역 판별 threshold 0.3~0.7, 공격 탐지
threshold와 clean retention 목표를 한 번에 하나씩 바꾼다. 재생성 공격 표의 8개 공격
방법과 32개 조건을 구분해 방법별·강도별 결과를 제시한다. seed별 원자료, 평균과
표준편차를 저장한다.

## Stability

3개 seed와 17개 지역에 따른 robust F1과 모델 순위의 변동을 본다. 평균·표준편차·
최솟값·최댓값과 seed 쌍별 Kendall 순위 상관을 기록한다. 피처 제거 안정성은 07단계의
지역별 ablation heatmap과 bootstrap CI로 분리해 제시한다.

## Ranking

각 seed와 지역에서 clean/adversarial 평균 F1로 모델 순위를 매긴 뒤 평균 순위와
지역별 순위 heatmap, 성능 radar를 만든다. radar는 절대 수치 표를 대체하지 않는다. 17개 지역을 독립 반복
실험으로 과장하지 않고 seed 및 paired POI 불확실성을 함께 제시한다.

## 최종 ablation과 보고서

앙상블 구성요소, 탐지기, threshold, 주요 피처군을 제거해 end-to-end 성능 기여를
검증한다. 최종 보고서는 단계별 보고서를 링크하고 데이터 버전, 선택 규칙, 실패 실행,
한계, 재현 명령을 포함한다. leave-one-attack-out, leave-one-source-model-out transfer와
감지기를 함께 회피하는 adaptive attack을 별도 조건으로 평가한다. 지역 균등 표본과
01단계 공격 유효성 검수 결과가 일반화에 주는 제한을 결론에 반드시 명시한다.
