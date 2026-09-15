# POI 전체 강건성 실험 독립 검수

- 검수 시각: 2026-09-16T00:13:22+09:00
- 전체 판정: **FAIL**
- 기준 문서: `docs/experiments/2026-09-15_full_robustness_experiment_plan.md`
- 검수 대상: `artifacts/poi-adversarial-20260915-001`, `artifacts/poi-final-delivery-20260915`
- 검수 방식: 기존 성공 마커와 보고서 결론을 증명으로 간주하지 않고 CSV/Parquet, metadata, PNG와 생성 코드를 직접 대조했다.

공격 생성·기본 모델 비교·선택 모델 공격 평가는 상당 부분 검증됐지만, 핵심 방어 단계
08–15는 실행되지 않았다. 두 artifact의 `complete` 표시는 전체 계획이 아니라 제한된
공격 평가 9개 디렉터리 범위에만 해당한다.

| requirement | expected | observed | verdict | evidence / limitation |
|---|---:|---:|---|---|
| 데이터 분할 | 34,000; train 23,800 / validation 3,400 / test 6,800; 17지역 | 정확히 일치, POI_ID 34,000개 고유, split 간 교집합 0 | PASS | `data/poi_34k_seed42/sample_manifest.csv` |
| 모델 입력·누수 제외 | 좌표 2 + 업종코드 3만 입력 | `FEATURES`가 정확히 5개이며 region, POI_ID, 주소·행정코드는 모델 행렬에 미포함 | PASS | `src/poi/data.py:9`, `src/poi/benchmark.py:150`, `src/poi/tabpfn_benchmark.py:25` |
| 공격 cardinality | 32조건 × 27,200 POI = 870,400행; train/validation만 | 870,400행, train 761,600·validation 108,800, 각 POI 32조건, 중복 0, 8개 공격 방법 | PASS | `reports/01_attack_generation/tables/attacks.parquet` |
| 공격 seed 범위 | 기본 생성 및 stage 14에서 42/202/340 안정성 | 공격 생성은 seed 42 한 개뿐. 3 seeds는 동일 공격을 평가한 모델 seed | PARTIAL | `attacks.parquet.seed`: `{42: 870400}` |
| 거리·L0·조합 제약 | 모든 행 독립 재검증 | 거리 오차 최대 `2.27e-13m`; 거리/L0/97개 허용 조합 위반 각 0; 변경 flag 불일치 0 | PASS | `src/poi/adversarial.py:454` 및 `attacks.parquet` 독립 재계산 |
| 수치 범위·label 보존·provenance | 실제 범위와 실제 label 검증, victim 기록 | `valid_range`는 범위가 아니라 거리 예산 검사. 관측 train+validation 범위 밖 X 12행·Y 15행. `label_preserved=True`가 상수이며 공간 경계 검증 없음. `victim_model` 870,400행 전부 빈 문자열 | FAIL | `src/poi/adversarial.py:484`, `:490`, `:496` |
| 공격 실패 보존 | 성공 행만 선별 금지 | 실패 703,741행도 유지; source clean-correct 0.988272, 전체 ASR 0.191474, eligible ASR 0.193746 | PASS | `attack_success_source`, `clean_source_correct` |
| test 격리 | 고정 전 test를 열지 않음 | 공격·예측 원자료에는 test 행 0. 다만 `load_dataset()`이 전체 CSV의 test 행도 메모리에 읽은 뒤 필터링함 | PARTIAL | `src/poi/data.py:30`, `src/poi/benchmark.py:150` |
| 기술통계·Table One | 전체 조건·지역·수치·범주 | 수치형은 32조건×17지역×2피처 1,088행. 범주형 지역표는 clean/`caa_high` 2조건뿐이며 XLSX 없음 | PARTIAL | `reports/01_descriptive/tables/` |
| 분포·임베딩 그림 | 공격별 시각화 | PNG는 `caa_high` 대표 조건만 그림으로 생성; 전 조건은 일부 CSV에만 존재 | PARTIAL | `reports/02_distribution/`, `reports/03_embedding/` |
| CCA·통계 | CCA, Pearson/Spearman, paired/Bowker, effect/CI/p/q | 32조건×5 CCA=160행; paired/Bowker 224행과 q-value 존재. 상관표는 대표 조건의 Spearman 8행뿐이며 Pearson·변화량·전 조건 상관 없음. CCA 표에는 q-value 없음 | PARTIAL | `reports/04_statistics/tables/cca_all_attacks.csv`, `statistical_tests_all_attacks.csv`, `correlations.csv` |
| clean 8모델 | 8×17×3=408, 6지표·혼동행렬·PR/ROC | 408조합, 1,387,200예측, 누락·중복 0. 전 지표 독립 재계산 최대차 `1.11e-16` | PASS | `reports/06_region_models/tables/metrics_by_region.csv`, `all_predictions.parquet` |
| 공통 paired bootstrap CI | 모든 지역분류 단계 | clean·공격 성능표에 paired bootstrap CI 없음 | FAIL | `reports/06_region_models/tables/`, `reports/06_attack_evaluation/tables/` |
| 공격 후 5개 성능지표 | 33×17×3=1,683 | 정확히 1,683조합·5,722,200예측. F1/precision/recall/accuracy/ROC-AUC 및 AP 모두 재계산 일치 | PASS | `reports/06_attack_evaluation/tables/metrics_by_region.csv` |
| SHAP 17지역×2조건 dot | 34 beeswarm/dot + 원자료 | PNG 34개, 각 지역 clean/adversarial 존재, 음·양 SHAP와 파랑–빨강 피처값을 직접 확인. 원자료 680행이지만 그룹당 POI는 4개뿐 | PASS | `reports/07_explainability/tables/shap_values.csv`, `src/poi/selected_explain.py:44` |
| 계획상 SHAP 모델 범위 | 제안 모델 + 고정 TabPFN 2.5 | 실제는 `tabpfn_v3` 한 모델과 대표 공격 `caa_high`만 | PARTIAL | `src/poi/selected_explain.py:167` |
| LIME 지역·조건 | 17지역별 대표 사례, clean/공격 범위 | Seoul clean 1개, 5개 rule만 존재; CSV에 region·condition·POI_ID 열도 없음 | FAIL | `reports/07_explainability/tables/lime_representative.csv`, `src/poi/selected_explain.py:247` |
| 제안 자체 방어모델 | encoder, corruption head, reconstruction, gate, attack/region heads | 관련 구현과 artifact 모두 없음 | NOT_RUN | 예상 `reports/08_proposed_model/` 부재 |
| URE-RF 감지기 | 실제 URE embedding + RF 결과 | 구현·설정·결과 모두 없음 | NOT_RUN | 소스에서 URE는 문서에만 등장 |
| 공격탐지 후 지역분류 | 95% clean retention, coverage/AURC, 동일 TabPFN 2.5 | 결과 없음 | NOT_RUN | `reports/09_attack_detection`, `10_filtered_region_classification` 부재 |
| full-coverage 방어 | 제안 복원, augmentation, URE-region, GROOT 비교 | 없음 | NOT_RUN | `reports/11_full_coverage_robustness` 부재 |
| ensemble·sensitivity·stability | 고정 모델 후 실행 결과 | 과거용 러너 코드는 있으나 대상 artifact 결과 0개이며 입력 경로도 존재하지 않는 이전 artifact를 가리킴 | NOT_RUN | `src/poi/defense.py`, `src/poi/robustness.py` |
| ablation | 피처 제거 및 제안 구조 모듈 제거 | 러너/config만 있고 현재 artifact 결과 없음. 구조 모듈 ablation 구현도 없음 | NOT_RUN | `src/poi/tabpfn_ablation.py`, 예상 `reports/07_backbone_ablation/` 부재 |
| locked test | 최종 6,800개 1회 평가 | test 평가 artifact 없음 | NOT_RUN | `reports/15_locked_test` 부재 |
| PNG/SVG | PNG 300dpi, SVG 0 | 원본 artifact PNG 137, 전달본 98; 모두 실제 PNG, 최저 299.9994dpi, SVG 0 | PASS | 두 artifact 전체 |
| 전달본 무결성 | manifest와 복사본 일치 | manifest 179개 항목 모두 hash/bytes 일치, 원본과 공통 177파일 byte-identical | PASS | `artifacts/poi-final-delivery-20260915/manifest.json` |
| 최종 전달 구성 | 00–15, 종합 XLSX, FINAL_REPORT, run manifest, 로그 | 방어·test 단계 없음; XLSX, `FINAL_REPORT.md`, `run_manifest.json`, 실행 로그 없음 | FAIL | `artifacts/poi-final-delivery-20260915/` |

## 중요 발견사항

- **치명적:** `reports/99_audit/report.md`의 자동 감사 `REQUIRED` 목록은 9개 초기 단계로
  제한된다. 원 계획의 07–15를 검사하지 않으므로 기존 `_SUCCESS.json`과 전달본의
  `status=complete`는 전체 계획 완료 근거가 아니다.
- **치명적:** 제안 모델, URE-RF, 감지 후 분류, full coverage, 앙상블,
  sensitivity/stability, 최종 ablation, locked test가 모두 `NOT_RUN`이다.
- **높음:** 공격 제약의 거리·L0·경험적 조합은 맞지만 실제 수치 범위와 label 보존은
  검증하지 않았고 victim provenance가 비어 있다. “모든 공격이 모든 제약을 통과했다”는
  표현은 실제 증거 범위를 넘는다.
- **높음:** LIME은 계획 범위의 1/17 지역이며 공격 조건 사례는 0건이다.
- **높음:** clean/공격 기본 성능 수치는 재계산과 일치하지만 계획이 요구한 POI paired
  bootstrap CI가 없다.
- **중간:** SHAP은 실제 dot 형식이나 지역·조건당 4 POI뿐이고, 계획 대상인 TabPFN 2.5와
  제안 모델이 아니라 clean 1위 TabPFN v3만 설명한다.
- **중간:** 공격 생성 checkpoint는 artifact에 없고 metadata의 `/workspace/checkpoints/...`
  경로만 남았다. 일부 단계는 git commit·dirty 상태·GPU instance·실행 로그가 빠져 있다.

확인된 핵심 수치는 clean 1위 `tabpfn_v3` Macro F1 `0.9972673138`, 최악 공격
`pgd_m1000` F1 `0.9884092716`이다. 두 수치는 원예측에서 재산출되어 맞지만,
“제안 방어가 URE-RF보다 낫다” 또는 “전체 강건성 실험이 완료됐다”는 결론에는 사용할 수 없다.
