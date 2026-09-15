"""Create the immutable complete POI study bundle, including the locked test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


SOURCES = {
    '01_descriptive': 'artifacts/poi-eda-train-001/reports/01_descriptive',
    '02_distribution': 'artifacts/poi-eda-train-001/reports/02_distribution',
    '03_embedding': 'artifacts/poi-eda-train-001/reports/03_embedding',
    '04_statistics': 'artifacts/poi-eda-train-001/reports/04_statistics',
    '05_region_models': 'artifacts/poi-models-validation-002/reports/05_region_models',
    '06_model_selection_ablation': 'artifacts/poi-tabpfn-ablation-001/reports/06_model_selection_ablation',
    '07_explainability': 'artifacts/poi-tabpfn-explain-001/reports/07_explainability',
    '08_attack_detection': 'artifacts/poi-advanced-validation-001/reports/08_attack_detection',
    '09_ensemble_defense': 'artifacts/poi-final-validation-001/reports/09_ensemble_defense',
    '10_robustness': 'artifacts/poi-final-validation-001/reports/10_robustness',
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def copy_locked_test(source, reports):
    destination = reports / '11_locked_test'
    shutil.copytree(source, destination)
    stage10 = reports / '10_robustness'
    for path in (source / 'tables').iterdir():
        shutil.copy2(path, stage10 / 'tables' / path.name)
    for path in (source / 'figures').iterdir():
        shutil.copy2(path, stage10 / 'figures' / path.name)
    shutil.copy2(source / 'report.md', stage10 / 'locked_test_report.md')
    shutil.copy2(source / 'metadata.json', stage10 / 'locked_test_metadata.json')
    shutil.copy2(source / 'metrics.json', stage10 / 'locked_test_metrics.json')


def add_tabpfn_performance_tables(reports, output):
    validation_dir = reports / '05_region_models'
    validation = pd.read_csv(validation_dir / 'tables' / 'metrics_summary.csv')
    validation = validation.loc[validation.model.str.startswith('tabpfn_')].copy()
    validation.insert(0, 'scope', 'validation')
    validation.to_csv(validation_dir / 'tables' / 'tabpfn_validation_performance.csv', index=False)

    test_dir = reports / '11_locked_test'
    test = pd.read_csv(test_dir / 'tables' / 'final_test_summary.csv')
    test.insert(0, 'split', 'test')
    test.insert(0, 'model', 'tabpfn_v2_5')
    test.to_csv(test_dir / 'tables' / 'tabpfn_test_performance.csv', index=False)

    def rows(frame, model_column=True):
        rendered = []
        for _, row in frame.iterrows():
            model = row['model'] if model_column else 'tabpfn_v2_5'
            rendered.append(
                f"| {model} | {row['condition']} | "
                f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f} | "
                f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
                f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} | "
                f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                f"{row['roc_auc_mean']:.6f} ± {row['roc_auc_std']:.6f} |")
        return '\n'.join(rendered)

    header = ('| Model | Condition | Precision | Recall | F1 | Accuracy | ROC-AUC |\n'
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |\n')
    document = f'''# TabPFN 성능표

값은 17개 지역과 3개 seed의 평균 ± 표준편차다.

## Validation 모델 비교

{header}{rows(validation)}

## 잠금 test: 선택 모델 TabPFN v2.5

{header}{rows(test, model_column=False)}
'''
    (output / 'TABPFN_PERFORMANCE.md').write_text(document)
    with (validation_dir / 'report.md').open('a') as stream:
        stream.write('\n## TabPFN validation 성능표\n\n' + header + rows(validation) + '\n')
    with (test_dir / 'report.md').open('a') as stream:
        stream.write('\n## TabPFN v2.5 잠금 test 성능표\n\n' + header + rows(test, model_column=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/poi-study-final-002')
    parser.add_argument('--final-test', default='artifacts/poi-final-test-003/final_test')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    reports = output / 'reports'
    reports.mkdir(parents=True)
    for name, source_value in SOURCES.items():
        source = Path(source_value)
        if not (source / 'report.md').is_file() or not (source / 'metadata.json').is_file():
            raise FileNotFoundError(f'Incomplete stage: {source}')
        shutil.copytree(source, reports / name)
    final_test = Path(args.final_test)
    if not (final_test / '_SUCCESS.json').is_file():
        raise FileNotFoundError(f'Incomplete locked test: {final_test}')
    copy_locked_test(final_test, reports)
    add_tabpfn_performance_tables(reports, output)
    readme = output / 'README.md'
    readme.write_text('''# POI region classification and adversarial robustness study

34,000개 paired POI 표본으로 수행한 전체 실험 결과다. `reports/01_*`부터
`reports/10_*`까지 validation 분석·모델 비교·설명·공격 탐지·방어·강건성 분석을,
`reports/11_locked_test`에는 8모델 validation 순위로 고정한 최종 모델의 미사용 test
평가를 기록했다. 각 단계는 별도 `report.md`, 표, PNG 300dpi 그림과 metadata를 갖는다.
`TABPFN_PERFORMANCE.md`에는 TabPFN 세 버전의 validation 성능과 선택 모델의 잠금
test 성능을 한 표로 모았다.

`manifest.json`은 번들 파일의 크기와 SHA-256, 선택 모델, 데이터 사용 범위를 기록한다.
''')
    success = {
        'status': 'complete',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'selected_model': 'tabpfn_v2_5',
        'validation_model_count': 8,
        'test_split_used_only_after_model_lock': True,
    }
    (output / '_SUCCESS.json').write_text(json.dumps(success, indent=2) + '\n')
    files = {
        str(path.relative_to(output)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
        for path in sorted(output.rglob('*')) if path.is_file()
    }
    manifest = {
        'status': 'complete', 'completed_stages': list(SOURCES) + ['11_locked_test'],
        'selected_model': 'tabpfn_v2_5', 'validation_model_count': 8,
        'test_split_used_only_after_model_lock': True,
        'files': files,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'files': len(files),
                      'bytes': sum(item['bytes'] for item in files.values())}, indent=2))


if __name__ == '__main__':
    main()
