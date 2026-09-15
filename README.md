# POI 실험 프로젝트

지역을 직접 나타내는 컬럼을 제외한 5개 피처로 17개 지역 one-vs-rest 이진분류와
적대적 공격 강건성 실험을 재현 가능한 구조로 실행한다. 기존 결과 산출물은
2026-09-15에 초기화했으며, 공격 데이터 생성부터 EDA·통계, 8개 지역분류 모델 비교,
구조 변경 방어 모델, SHAP/LIME, 공격 탐지와 잠금 test 평가까지 새로 수행한다.

```text
src/poi/          데이터 검증·전처리, 모델, 학습, 평가
scripts/train.sh  공통 학습 진입점
configs/         모델과 학습 설정 (baseline / smoke)
experiments/     gpu-orchestrator의 코드·데이터·명령 연결
tests/           데이터 무결성, 분할, 학습과 저장 검증
docs/            데이터 버전과 실험 기준
```

기존 R2 연결·34,000건 추출·업로드 도구는 `scripts/`에 보존했다.

## 로컬 설치와 검증

Python 3.9에서 검증했다. 패키지 범위는 3.9–3.12이며 다른 Python/OS에서는
잠금 파일 설치와 smoke를 먼저 검증한다.

```bash
python -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/pytest -q
EXP_ID=smoke-001 bash scripts/train.sh configs/smoke.yaml
```

기본 데이터 경로는 `data/poi_34k_seed42/`이다. 다른 위치는 `DATA_DIR`로 지정한다.
smoke는 지역당 학습 20건·검증 5건만 사용한다. 학습 기준 실험은 다음과 같다.

```bash
EXP_ID=baseline-001 bash scripts/train.sh configs/baseline.yaml
```

기본 baseline은 학습 23,800건으로 학습하고 검증 3,400건의 원본·공격 성능을 측정한다.
테스트셋은 설정의 `evaluation.splits`에 `test`를 명시했을 때만 평가한다.
같은 출력 경로를 재사용하면 덮어쓰지 않고 실패한다. 새 실행에는 새 `EXP_ID`를 쓴다.

## 산출물

기본 위치는 `artifacts/<EXP_ID>/{checkpoints,results,logs}/`이다.
Orchestrator가 제공하는 `CHECKPOINT_DIR`, `RESULT_DIR`, `LOG_DIR`가 있으면 그 경로를 쓴다.

- `checkpoints/model.joblib`: 17개 OvR 전처리·이진분류기 매핑
- `results/metrics.json`: OvR precision/recall/F1/accuracy/ROC-AUC/AP와 혼동행렬
- `results/*_predictions.csv`: 원본/공격 조건별 POI_ID와 예측
- `results/used_manifest.csv`: 실제 사용한 ID와 split
- `results/run.json`, `config.yaml`: 데이터 해시, Git 상태, 패키지 버전, 실행 설정
- `logs/summary.json`: 성공 상태와 소요 시간

## 의존성 관리

의존성 정의는 `pyproject.toml`, 설치용 잠금은 `requirements.lock`이다.
잠금 파일은 pip-tools로 직접·간접 의존성과 배포 파일 해시를 고정한다.
`requirements.txt`는 기존 명령 호환용으로 잠금 파일을 참조한다.

```bash
.venv/bin/pip-compile --extra dev --allow-unsafe --generate-hashes \
  --strip-extras --no-emit-index-url --output-file requirements.lock pyproject.toml
```

## 원격 실행

`experiments/smoke.env`와 `baseline.env`는 기존 `gpu-orchestrator`와 연결된다.
서버에서는 프로젝트 전용 `.venv`를 생성하고 잠금 파일로 설치한 뒤 동일한
`train.sh`를 실행한다. 서버 접속 정보는 `GPU_CONNECTION_FILE`로 공급한다.

Orchestrator는 **커밋된 코드만** 배포한다. 이번 작업 내용을 검토·커밋한 뒤
아래의 연결 파일 경로와 실행 ID를 지정해 사용한다.

```bash
GPU_CONNECTION_FILE=/path/to/instance.env \
EXPERIMENT_CONFIG=/home/mlops/orca/projects/poi/experiments/smoke.env \
  /home/mlops/gpu-orchestrator/scripts/launch_experiment.sh poi-smoke-001
```

이 프로젝트 설정은 GPU 인스턴스를 생성하거나 삭제하지 않는다.

[데이터 구조·추출 이력](docs/dataset.md) · [단계별 실험 인덱스](docs/experiments.md) ·
[전체 모델 행렬](configs/experiment_matrix.yaml)

## 현재 실험 상태

기존 로컬 `artifacts/`와 R2 `results/poi/` 산출물은 삭제했다. 이전 성능 수치와 그림은
새 결론에 사용하지 않는다. [2026-09-15 전체 실험 계획](docs/experiments/2026-09-15_full_robustness_experiment_plan.md)에 따라 새 experiment ID로 단계별 결과를 생성한다.

모든 그림은 PNG 300 dpi로만 저장한다. SHAP 시각화는 17개 지역별 clean·공격
beeswarm/dot plot으로 만들며, CCA와 LIME도 필수 분석으로 수행한다. 새 잠금 test가
끝나기 전에는 최종 모델이나 최종 성능을 확정하지 않는다.
