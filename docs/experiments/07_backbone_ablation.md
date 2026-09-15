# 07. backbone ablation

06단계 validation 결과와 사전 정의한 규칙으로 최종 모델 계열과 hyperparameter를
고정한다. 선정 근거, 동률 처리, 제외된 실행과 실패를 `model_selection.md`에 기록한다.

## Ablation 순서

1. 전체 기본 피처
2. 좌표 제거
3. 좌표만 사용
4. X 좌표와 Y 좌표를 각각 제거
5. 업종 대/중/소분류를 각각 제거

각 조건은 같은 validation split과 seed를 사용한다. 17개 지역×3 seeds의 clean/adversarial
F1 평균을 robust F1으로 정의하고, 2,000회 paired bootstrap 95% CI와 full 대비 변화량을
표·막대 그림·지역 heatmap으로 남긴다. 06단계 8모델 비교에서 선택된 모델만
대상으로 하며 test split은 사용하지 않는다.
