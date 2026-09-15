# 저장소 파일 분리와 이름 규칙

이 저장소는 파일 형식과 책임을 함께 분리한다.

| 형식/영역 | 위치 | 책임 |
| --- | --- | --- |
| TOML | `pyproject.toml`, `.codex/config.toml`, `.codex/agents/` | 패키지 설정과 에이전트별 모델·추론 강도·역할 설정 |
| 에이전트 운영 규칙 | `AGENTS.md`, `docs/experiments/independent_review_protocol.md` | 역할 호출 순서, 증적 요구사항, 완료 게이트 |
| YAML | `configs/protocol.yaml`, `configs/runs/`, `configs/stages/`, `configs/legacy/` | 선언형 실험 파라미터와 입력·출력 경로 |
| ENV | `experiments/*.env` | 실행기 자원, 컨테이너, 정확한 실행 명령 |
| Python | `src/poi/` | 재사용 가능한 실험 로직 |
| Shell/Python 실행 도구 | `scripts/` | 얇은 실행·검증·번들 진입점 |
| Markdown | `docs/`, `docs/experiments/` | 설계, 단계별 프로토콜, 상태 인덱스 |
| GitHub용 결과 | `docs/assets/<EXP_ID>/` | 결과 허브에 연결할 요약 표·대표 PNG와 해시 manifest |
| 원천/파생 데이터 | `data/<dataset_id>/` | 고정 입력과 manifest; 실험 결과를 저장하지 않음 |
| 실행 증적 | `artifacts/<EXP_ID>/` | 표, Parquet, PNG, 메타데이터, 로그, 보고서 |

YAML에는 비밀값이나 측정 결과를, ENV에는 분석 규칙이나 성능 수치를 넣지 않는다.
Python 모듈의 기본 config 경로는 반드시 존재하는 canonical YAML을 가리킨다. 새 단계
결과는 `reports/<NN_stage_name>/`에 저장하며 문서 번호와 같은 00–15 번호를 사용한다.

에이전트의 모델과 추론 강도처럼 실행 시 적용할 값은 Markdown에만 기록하지 않고
`.codex/agents/*.toml`에 둔다. Markdown은 TOML로 표현할 수 없는 검수 절차와 완료 조건을
설명한다.

2026-09-15 artifact는 이미 생성되고 해시가 고정된 증적이다. 따라서 내부의 옛 번호를
바꾸지 않는다. [결과 허브](results.md)가 옛 번호를 canonical 번호에 대응시킨다.
