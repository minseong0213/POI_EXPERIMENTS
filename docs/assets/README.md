# GitHub용 결과 자료

이 디렉터리는 Git에서 제외되는 `artifacts/` 중 결과 허브에 필요한 요약 CSV, 보고서와
대표 PNG만 담는다. 원본 데이터, 전체 예측, 모델 checkpoint와 대용량 Parquet은 R2 및
로컬 artifact에 보존한다.

`poi-adversarial-20260915-001/manifest.json`은 공개 파일의 원본 상대 경로, 크기,
원본 SHA-256과 공개본 SHA-256을 기록한다. Markdown과 JSON은 내용은 유지하면서 마지막
개행만 정규화한다. 이 묶음의 전체 실험 판정은 `FAIL`이며 파일 공개가 실험 완료를
뜻하지 않는다.

다시 생성하려면 저장소 루트에서 다음을 실행한다.

```bash
.venv/bin/python scripts/build_docs_results.py
```
