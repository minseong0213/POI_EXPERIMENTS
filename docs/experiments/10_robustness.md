# 10. Sensitivity, stability, ranking, 최종 ablation

## Sensitivity

공격 비율 0/10/25/50/75/100%, class weight, threshold, 음성 표본 비율, 주요 전처리
파라미터를 한 번에 하나씩 바꾼다. 공격 종류 정보가 확보되면 PGD/CW/FGSM 강도별
분석을 추가한다. 성능 곡선과 95% CI를 저장한다.

## Stability

seed, bootstrap 표본, 피처 제거, 임베딩 파라미터에 따른 성능과 feature ranking의
변동을 본다. 평균±표준편차, CI, 순위 상관(Kendall/Spearman), SHAP top-k Jaccard를
기록한다.

## Ranking

각 지역과 조건에서 모델 순위를 매긴 뒤 평균 순위, critical-difference 형태의 그림,
성능 radar를 만든다. radar는 절대 수치 표를 대체하지 않는다. 17개 지역을 독립 반복
실험으로 과장하지 않고 seed 및 paired POI 불확실성을 함께 제시한다.

## 최종 ablation과 보고서

앙상블 구성요소, 탐지기, threshold, 주요 피처군을 제거해 end-to-end 성능 기여를
검증한다. 최종 보고서는 단계별 보고서를 링크하고 데이터 버전, 선택 규칙, 실패 실행,
한계, 재현 명령을 포함한다. 공격 종류 부재와 지역 균등 표본이라는 제한을 결론에
반드시 명시한다.
