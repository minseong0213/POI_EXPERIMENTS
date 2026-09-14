# 06. 모델 고정과 ablation

05단계 validation 결과와 사전 정의한 규칙으로 최종 모델 계열과 hyperparameter를
고정한다. 선정 근거, 동률 처리, 제외된 실행과 실패를 `model_selection.md`에 기록한다.

## Ablation 순서

1. 전체 기본 피처
2. 좌표 제거
3. 업종 대/중/소분류를 각각 제거
4. 범주형 전체 제거
5. 좌표 전체 제거
6. 공격에서 변경률이 높은 피처군 제거

각 조건은 같은 split과 seed를 사용하고 one-factor-at-a-time 결과 및 누적 제거 결과를
구분한다. validation robust F1 차이, paired bootstrap CI, clean/adversarial 변화량을
표·forest plot·heatmap으로 남긴다. 이 단계가 끝나면 최종 설명 대상 모델을 잠근다.
