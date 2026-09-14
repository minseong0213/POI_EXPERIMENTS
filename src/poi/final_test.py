"""Locked clean/adversarial test evaluation for the eight-model validation winner."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
import yaml

from .data import CATEGORICAL, NUMERIC, load_dataset, sha256

METRICS = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'average_precision']
EXPECTED_MODELS = {
    'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'catboost',
    'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3',
}


def binary_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_score),
            'average_precision': average_precision_score(y_true, y_score),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def save_figure(fig, figures, name):
    fig.savefig(figures / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(figures / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_model(path):
    ranking = pd.read_csv(path)
    if set(ranking.model) != EXPECTED_MODELS or ranking.model.duplicated().any():
        raise ValueError('Refusing test access before the complete eight-model ranking exists')
    score = 'mean' if 'mean' in ranking.columns else 'mean_robust_f1'
    if score not in ranking or ranking[score].isna().any():
        raise ValueError('Selection ranking has no complete robust F1 column')
    return ranking.sort_values(score, ascending=False).iloc[0].model


def numeric_matrix(frame):
    matrix = frame[NUMERIC + CATEGORICAL].apply(
        pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('TabPFN input contains non-finite values')
    return matrix


def tree_evaluate(model_id, tree_config, development, evaluation, regions, seeds, threshold):
    from .benchmark import estimator, preprocessor

    params = yaml.safe_load(Path(tree_config).read_text())['models'][model_id]
    transform = preprocessor()
    x_train = transform.fit_transform(development)
    x_test = {name: transform.transform(frame) for name, frame in evaluation.items()}
    rows, predictions = [], []
    for seed in seeds:
        for region in regions:
            y_train = development.region.eq(region).astype(int).to_numpy()
            positive = y_train.sum()
            negative = len(y_train) - positive
            weights = np.where(y_train == 1, len(y_train) / (2 * positive),
                               len(y_train) / (2 * negative))
            model = estimator(model_id, params, seed)
            model.fit(x_train, y_train, sample_weight=weights)
            for condition, frame in evaluation.items():
                score = model.predict_proba(x_test[condition])[:, 1]
                truth = frame.region.eq(region).astype(int).to_numpy()
                rows.append({'model': model_id, 'seed': seed, 'region': region,
                             'condition': condition, 'split': 'test',
                             **binary_metrics(truth, score, threshold)})
                predictions.append(pd.DataFrame({
                    'POI_ID': frame.POI_ID, 'model': model_id, 'seed': seed,
                    'region': region, 'condition': condition, 'split': 'test',
                    'y_true': truth, 'y_score': score,
                    'y_pred': (score >= threshold).astype(int),
                }))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), {}


def tabpfn_evaluate(model_id, development, evaluation, regions, seeds, threshold, cfg):
    if not os.environ.get('TABPFN_TOKEN'):
        token_path = Path(os.environ.get('TABPFN_TOKEN_FILE', '/root/.config/poi/tabpfn_token'))
        if token_path.is_file():
            os.environ['TABPFN_TOKEN'] = token_path.read_text().strip()
    if not os.environ.get('TABPFN_TOKEN'):
        raise RuntimeError('TABPFN_TOKEN or TABPFN_TOKEN_FILE is required')
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    import tabpfn
    import torch
    versions = {
        'tabpfn_v2_5': ModelVersion.V2_5,
        'tabpfn_v2_6': ModelVersion.V2_6,
        'tabpfn_v3': ModelVersion.V3,
    }
    x_train = numeric_matrix(development)
    x_test = np.vstack([numeric_matrix(evaluation['clean']),
                        numeric_matrix(evaluation['adversarial'])])
    rows, predictions = [], []
    for seed in seeds:
        model = TabPFNClassifier.create_default_for_version(
            versions[model_id], device=cfg['device'], n_estimators=cfg['tabpfn_n_estimators'],
            random_state=seed, categorical_features_indices=[2, 3, 4],
            show_progress_bar=True, memory_saving_mode='auto')
        for start in range(0, len(regions), cfg['tabpfn_region_batch_size']):
            batch = regions[start:start + cfg['tabpfn_region_batch_size']]
            labels = [development.region.eq(region).astype(int).to_numpy() for region in batch]
            probabilities = model.predict_proba_batched(
                [x_train] * len(batch), labels, [x_test] * len(batch))
            for index, region in enumerate(batch):
                score = np.asarray(probabilities[index])[:, 1]
                offset = 0
                for condition in ['clean', 'adversarial']:
                    frame = evaluation[condition]
                    values = score[offset:offset + len(frame)]
                    offset += len(frame)
                    truth = frame.region.eq(region).astype(int).to_numpy()
                    rows.append({'model': model_id, 'seed': seed, 'region': region,
                                 'condition': condition, 'split': 'test',
                                 **binary_metrics(truth, values, threshold)})
                    predictions.append(pd.DataFrame({
                        'POI_ID': frame.POI_ID, 'model': model_id, 'seed': seed,
                        'region': region, 'condition': condition, 'split': 'test',
                        'y_true': truth, 'y_score': values,
                        'y_pred': (values >= threshold).astype(int),
                    }))
            torch.cuda.empty_cache()
    runtime = {'tabpfn': tabpfn.__version__, 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), runtime


def plot_test(metrics, predictions, figures):
    long = metrics.melt(id_vars=['condition', 'region', 'seed'], value_vars=METRICS,
                        var_name='metric', value_name='value')
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x='metric', y='value', hue='condition', errorbar='sd', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Locked test metrics across regions and seeds')
    save_figure(fig, figures, 'final_test_metrics')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, condition in zip(axes, ['clean', 'adversarial']):
        part = metrics.loc[metrics.condition.eq(condition)]
        matrix = np.array([[part.tn.sum(), part.fp.sum()], [part.fn.sum(), part.tp.sum()]])
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(matrix, annot=True, fmt='.4f', vmin=0, vmax=1, ax=ax)
        ax.set_title(condition)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    save_figure(fig, figures, 'final_test_confusion_normalized')

    selected = predictions.loc[predictions.seed.eq(predictions.seed.min())]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition, group in selected.groupby('condition'):
        false_positive, true_positive, _ = roc_curve(group.y_true, group.y_score)
        axes[0].plot(false_positive, true_positive,
                     label=f'{condition} AUC={roc_auc_score(group.y_true, group.y_score):.4f}')
        precision, recall, _ = precision_recall_curve(group.y_true, group.y_score)
        axes[1].plot(recall, precision,
                     label=f'{condition} AP={average_precision_score(group.y_true, group.y_score):.4f}')
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='Locked test ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Locked test PR')
    for ax in axes:
        ax.legend()
    save_figure(fig, figures, 'final_test_pr_roc')


def run(config_path, output_override=None):
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    output = Path(output_override or cfg['output'])
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    model_id = select_model(cfg['selection_ranking'])
    frames, manifest = load_dataset(cfg['data_dir'], verify_hashes=True)
    development = frames['clean'].loc[frames['clean'].split.isin(['train', 'validation'])].reset_index(drop=True)
    evaluation = {
        'clean': frames['clean'].loc[frames['clean'].split.eq('test')].reset_index(drop=True),
        'adversarial': frames['attack'].loc[frames['attack'].split.eq('test')].reset_index(drop=True),
    }
    regions = sorted(manifest.region.unique())
    if model_id.startswith('tabpfn_'):
        metrics, predictions, runtime = tabpfn_evaluate(
            model_id, development, evaluation, regions, cfg['seeds'], cfg['threshold'], cfg)
    else:
        metrics, predictions, runtime = tree_evaluate(
            model_id, cfg['tree_config'], development, evaluation, regions,
            cfg['seeds'], cfg['threshold'])
    expected_metrics = 3 * 17 * 2
    expected_predictions = 3 * 17 * 2 * 6800
    if len(metrics) != expected_metrics or len(predictions) != expected_predictions:
        raise RuntimeError('Final test output cardinality is incomplete')
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True)
    figures.mkdir()
    metrics.to_csv(tables / 'final_test_metrics.csv', index=False)
    predictions.to_parquet(tables / 'final_test_predictions.parquet', index=False)
    summary = metrics.groupby('condition')[METRICS].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(filter(None, column)) for column in summary.columns]
    summary.to_csv(tables / 'final_test_summary.csv', index=False)
    plot_test(metrics, predictions, figures)
    metadata = {
        'status': 'complete', 'scope': 'locked test', 'model': model_id,
        'selection_ranking': cfg['selection_ranking'],
        'selection_ranking_sha256': file_sha256(cfg['selection_ranking']),
        'development_rows': len(development), 'test_rows_per_condition': len(evaluation['clean']),
        'regions': len(regions), 'seeds': cfg['seeds'], 'threshold': cfg['threshold'],
        'dataset_sha256': {name: sha256(Path(cfg['data_dir']) / name) for name in
                           ['poi_data_region.csv', 'poi_adversarial_data_final.csv',
                            'sample_manifest.csv']},
        'python': platform.python_version(), 'scikit_learn': importlib.metadata.version('scikit-learn'),
        'runtime': runtime, 'seconds': time.time() - started,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'metrics.json').write_text(
        json.dumps({'summary': summary.to_dict(orient='records')}, indent=2) + '\n')
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'report.md').write_text(f'''# 잠금 test 평가

8개 모델 validation robust F1 순위로 `{model_id}`를 고정한 뒤, clean
train+validation {len(development):,}건으로 재학습했다. 이전에 사용하지 않은 test
{len(evaluation['clean']):,}건을 clean/adversarial 두 조건에서 한 번 평가했다.

모델 선택 근거 파일의 SHA-256은 `{metadata['selection_ranking_sha256']}`이다. 전체
지역·seed 지표와 예측, confusion matrix, PR/ROC는 `tables/`와 `figures/`에 있다.
''')
    (output / '_SUCCESS.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/final_test.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
