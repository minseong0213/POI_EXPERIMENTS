# POI 34,000건 실험 데이터

R2 원본 `ml-experiments/datasets/poi/`의 CSV 두 개에서 지역별 균형 표본을 추출했다.
원본은 수정하지 않는다. 인증은 `gpu-orchestrator/secrets/rclone.conf`를 직접 읽으며
이 저장소에 자격 증명을 복사하지 않는다.

## 결과 위치

- 로컬: `data/poi_34k_seed42/`
- R2: `r2:ml-experiments/datasets/poi/subsets/poi_34k_seed42_v1/`
- 원본 로컬 캐시: `data/source/`

| 파일 | 설명 |
| --- | --- |
| `poi_data_region.csv` | 원본에서 선택한 34,000행, 기존 15개 컬럼 유지 |
| `poi_adversarial_data_final.csv` | 같은 POI_ID, 같은 순서의 공격 데이터 34,000행 |
| `sample_manifest.csv` | POI_ID, region, 원본 0-based 데이터 행 인덱스, split |
| `region_counts.csv` | 지역별 원본·선택·분할 건수와 비율 |
| `report.json` | 설정, 원본 해시, 검증 결과, 공격 파일 변경 컬럼, 제한사항 |
| `_SUCCESS.json` | R2 전체 파일 업로드 및 다운로드 SHA-256 검증 완료 표시 |

원본과 공격 파일은 각각 **913,684행**이고 POI_ID가 유일하며 두 파일의 ID와
지역 라벨은 같은 순서로 일치한다. 축소본은 원본 34,000개 POI와 그 공격 변형이다.
두 파일을 합친 68,000행을 독립 표본 68,000개로 해석하지 않는다.

## 추출 및 분할

NumPy `default_rng(42)`를 사용해 지역별로 비복원 무작위 추출한다.
세종은 1,831행 전부를 사용한다. 남은 지역은 각각 2,010~2,011행으로 채우며,
잔여 건수는 지역명 오름차순으로 배정한다. 전체는 정확히 34,000행이다.

각 지역 안에서 70/10/20으로 나누고 소수점은 최대 나머지 방식으로 배정한다.
원본 표본 기준 학습 **23,800**, 검증 **3,400**, 테스트 **6,800**이다.
공격 파일도 반드시 동일한 manifest의 split을 사용한다.
분할 정보는 CSV의 학습 특성에 추가하지 않고 별도 파일에 둔다.

```python
import pandas as pd

root = 'data/poi_34k_seed42/'
manifest = pd.read_csv(root + 'sample_manifest.csv', dtype=str, keep_default_na=False)
clean = pd.read_csv(root + 'poi_data_region.csv', dtype=str, keep_default_na=False)
attack = pd.read_csv(root + 'poi_adversarial_data_final.csv', dtype=str, keep_default_na=False)
clean = clean.merge(manifest[['POI_ID', 'split']], on='POI_ID', validate='one_to_one')
attack = attack.merge(manifest[['POI_ID', 'split']], on='POI_ID', validate='one_to_one')
train_clean = clean.loc[clean['split'].eq('train')]
test_attack = attack.loc[attack['split'].eq('test')]
# POI_ID, split은 모델 입력에서 제외하고, region을 타깃으로 별도 분리한다.
```

지역 균등 표본이므로 전체 정확도는 원본 지역 분포에서의 정확도와 다르다.
Macro-F1과 지역별 성능을 함께 비교한다. 공격 파일에는 공격 종류 컬럼이 없어
PGD/CW/FGSM 별 실험으로 구분할 수 없다. 기존 공격 생성 모델 및 학습 분할 이력도
없으므로, 이번 분할만으로 공격 생성 단계의 데이터 독립성을 보장하지 않는다.
POI_ID 간 누수는 막았지만 서로 다른 POI의 공간적 의존성은 별도로 다뤄야 한다.

## 재현

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sample_poi.py --output data/poi_34k_seed42_rerun
```

원본 CSV 두 개가 `data/source/`에 있어야 한다. 사용 버전과 원본 SHA-256을 고정해
같은 결과를 재현할 수 있다. 기존 출력 폴더는 덮어쓰지 않는다.
`scripts/publish_subset.py`는 새 subset prefix에만 업로드하며, 원본 MD5와 크기를
R2와 비교하고 모든 업로드 파일을 다시 다운로드해 SHA-256을 검증한다.

GPU에서 기존 orchestrator로 가져올 때:

```bash
/home/mlops/gpu-orchestrator/scripts/prepare_data.sh \
  <HOST> <PORT> datasets/poi/subsets/poi_34k_seed42_v1 poi-34k
```
