"""Merge tree and TabPFN Stage 05 outputs into the complete eight-model report."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from .benchmark import METRICS, plot_results


EXPECTED_MODELS = {
    'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'catboost',
    'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3',
}
EXPECTED_CONDITIONS = {'clean', 'adversarial'}


def _load(tree_dir, tabpfn_dir):
    tree_metrics = pd.read_csv(tree_dir / 'tables' / 'metrics_by_region.csv')
    tree_predictions = pd.read_parquet(tree_dir / 'tables' / 'all_predictions.parquet')
    tab_metrics = pd.read_csv(tabpfn_dir / 'metrics_by_region.csv')
    tab_predictions = pd.read_parquet(tabpfn_dir / 'all_predictions.parquet')
    tab_metrics = tab_metrics.drop(columns=['checkpoint_version'], errors='ignore')
    return (pd.concat([tree_metrics, tab_metrics], ignore_index=True),
            pd.concat([tree_predictions, tab_predictions], ignore_index=True))


def _validate(metrics, predictions):
    models = set(metrics.model.unique())
    if models != EXPECTED_MODELS:
        raise ValueError(f'Expected models {sorted(EXPECTED_MODELS)}, got {sorted(models)}')
    if set(metrics.condition.unique()) != EXPECTED_CONDITIONS:
        raise ValueError('Both clean and adversarial conditions are required')
    if metrics.region.nunique() != 17 or metrics.seed.nunique() != 3:
        raise ValueError('Expected 17 regions and 3 seeds')
    key = ['model', 'seed', 'region', 'condition']
    if metrics.duplicated(key).any() or len(metrics) != 8 * 3 * 17 * 2:
        raise ValueError('Metric rows are incomplete or duplicated')
    prediction_key = key + ['POI_ID']
    if predictions.duplicated(prediction_key).any():
        raise ValueError('Prediction rows are duplicated')
    expected_rows = 8 * 3 * 17 * 2 * 3400
    if len(predictions) != expected_rows or predictions.POI_ID.nunique() != 3400:
        raise ValueError('Prediction rows or validation POI count are incomplete')
    if set(predictions.model.unique()) != EXPECTED_MODELS:
        raise ValueError('Prediction and metric model sets differ')
    for column in METRICS:
        if metrics[column].isna().any():
            raise ValueError(f'Missing metric values in {column}')


def run(tree_dir, tabpfn_dir, output):
    tree_dir, tabpfn_dir, output = map(Path, (tree_dir, tabpfn_dir, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not (tabpfn_dir / '_SUCCESS.json').is_file():
        raise FileNotFoundError('TabPFN output has no _SUCCESS.json')
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True)
    figures.mkdir()
    metrics, predictions = _load(tree_dir, tabpfn_dir)
    _validate(metrics, predictions)
    metrics.to_csv(tables / 'metrics_by_region.csv', index=False)
    predictions.to_parquet(tables / 'all_predictions.parquet', index=False)
    plot_results(metrics, predictions, figures)
    robust = metrics.pivot_table(index=['model', 'seed', 'region'], columns='condition',
                                 values='f1').reset_index()
    robust['robust_f1'] = (robust.clean + robust.adversarial) / 2
    ranking = (robust.groupby('model').robust_f1.agg(['mean', 'std'])
               .sort_values('mean', ascending=False).reset_index())
    ranking.to_csv(tables / 'model_ranking_validation.csv', index=False)
    shutil.copy2(tabpfn_dir / 'timing.csv', tables / 'tabpfn_timing.csv')
    tree_timing = tree_dir / 'tables' / 'timing.csv'
    if tree_timing.is_file():
        shutil.copy2(tree_timing, tables / 'tree_timing.csv')
    tab_meta = json.loads((tabpfn_dir / 'metadata.json').read_text())
    tree_meta = json.loads((tree_dir / 'metadata.json').read_text())
    metadata = {
        'status': 'complete', 'scope': 'validation only',
        'models': sorted(EXPECTED_MODELS), 'regions': 17, 'seeds': 3,
        'metric_rows': len(metrics), 'prediction_rows': len(predictions),
        'test_split_used': False, 'tree_run': tree_meta, 'tabpfn_run': tab_meta,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'metrics.json').write_text(
        json.dumps({'ranking': ranking.to_dict(orient='records')}, indent=2) + '\n')
    best = ranking.iloc[0]
    (output / 'report.md').write_text(f'''# 05. 17개 지역 OvR 8개 모델 비교

동일한 train/validation split에서 트리 모델 5개와 TabPFN 3개를 17개 지역 OvR,
3개 seed로 비교했다. 총 408회 학습과 clean/adversarial 816개 평가 조합이다.

validation robust F1 1위는 `{best['model']}`이며 평균은 {best['mean']:.6f},
seed·지역 표준편차는 {best['std']:.6f}이다. test split은 사용하지 않았다.

`tables/`에는 전체 지표·예측·순위·시간을, `figures/`에는 normalized confusion
matrix와 전체 및 17개 지역별 PR/ROC curve를 PNG 300 dpi와 SVG로 저장했다.
''')
    (output.parent / '_SUCCESS.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(ranking.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree-dir', default='artifacts/poi-models-validation-001/reports/05_region_models')
    parser.add_argument('--tabpfn-dir', default='artifacts/poi-tabpfn-validation-001')
    parser.add_argument('--output', default='artifacts/poi-models-validation-002/reports/05_region_models')
    args = parser.parse_args()
    run(args.tree_dir, args.tabpfn_dir, args.output)


if __name__ == '__main__':
    main()
