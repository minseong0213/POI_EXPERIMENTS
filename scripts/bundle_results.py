"""Create an immutable unified validation-study bundle from completed stage outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


SOURCES = {
    '01_descriptive': 'artifacts/poi-eda-train-001/reports/01_descriptive',
    '02_distribution': 'artifacts/poi-eda-train-001/reports/02_distribution',
    '03_embedding': 'artifacts/poi-eda-train-001/reports/03_embedding',
    '04_statistics': 'artifacts/poi-eda-train-001/reports/04_statistics',
    '05_region_models': 'artifacts/poi-models-validation-001/reports/05_region_models',
    '06_model_selection_ablation': 'artifacts/poi-advanced-validation-001/reports/06_model_selection_ablation',
    '07_explainability': 'artifacts/poi-advanced-validation-002/reports/07_explainability',
    '08_attack_detection': 'artifacts/poi-advanced-validation-001/reports/08_attack_detection',
    '09_ensemble_defense': 'artifacts/poi-advanced-validation-001/reports/09_ensemble_defense',
    '10_robustness': 'artifacts/poi-advanced-validation-001/reports/10_robustness',
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/poi-validation-study-001');args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    reports=output/'reports';reports.mkdir(parents=True)
    for name,source in SOURCES.items():
        source=Path(source)
        if not (source/'report.md').is_file() or not (source/'metadata.json').is_file():
            raise FileNotFoundError(f'Incomplete stage: {source}')
        shutil.copytree(source,reports/name)
    files={str(path.relative_to(output)):{'bytes':path.stat().st_size,'sha256':digest(path)}
           for path in sorted(output.rglob('*')) if path.is_file()}
    manifest={'status':'validation_partial','completed_stages':list(SOURCES),
              'missing_required_model_families':['tabpfn_v2_5','tabpfn_v2_6','tabpfn_v3'],
              'test_split_used':False,'files':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('''# POI validation study 001

실제 train/validation 기반 01~10단계 결과 묶음이다. 각 단계의 `report.md`에서 표와
그림을 확인한다. 트리 모델 5개 결과만 완료됐고 TabPFN v2.5/v2.6/v3와 최종 test
잠금 평가는 아직 포함하지 않았으므로 최종 연구 결과가 아니다.

`manifest.json`은 모든 파일의 크기와 SHA-256을 기록한다.
''')
    print(json.dumps({'status':manifest['status'],'files':len(files),'bytes':sum(x['bytes'] for x in files.values())},indent=2))


if __name__=='__main__':main()
