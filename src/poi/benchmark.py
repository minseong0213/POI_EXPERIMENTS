"""Stage 06: evaluate five tree models on clean 17-region OvR tasks."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import CATEGORICAL, NUMERIC, load_dataset, sha256

METRICS = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'average_precision']


def estimator(name, params, seed):
    common = dict(random_state=seed)
    if name == 'decision_tree':
        return DecisionTreeClassifier(**common, **params)
    if name == 'random_forest':
        return RandomForestClassifier(**common, **params)
    if name == 'xgboost':
        return XGBClassifier(**common, **params)
    if name == 'lightgbm':
        return LGBMClassifier(**common, verbosity=-1, **params)
    if name == 'catboost':
        return CatBoostClassifier(**common, verbose=False, allow_writing_files=False, **params)
    raise ValueError(f'Unsupported model {name}')


def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])


def binary_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_score),
            'average_precision': average_precision_score(y_true, y_score),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def save_figure(fig, figures, name):
    fig.savefig(figures / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_results(metrics, predictions, figures):
    summary = metrics.groupby(['model', 'condition'])[METRICS].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(filter(None, col)) for col in summary.columns]
    summary.to_csv(figures.parent / 'tables' / 'metrics_summary.csv', index=False)
    long = metrics.melt(id_vars=['model', 'condition', 'region', 'seed'], value_vars=METRICS,
                        var_name='metric', value_name='value')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for metric, ax in zip(METRICS, axes.flat):
        sns.barplot(data=long.loc[long.metric.eq(metric)], x='model', y='value', hue='condition',
                    errorbar='sd', ax=ax)
        ax.tick_params(axis='x', rotation=35); ax.set_title(metric); ax.set_ylim(0, 1)
    fig.tight_layout(); save_figure(fig, figures, 'metric_summary')

    score_table = metrics.loc[metrics.condition.eq('clean')].pivot_table(
        index='region', columns='model', values='f1', aggfunc='mean')
    score_table.to_csv(figures.parent / 'tables' / 'clean_f1_by_region.csv')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(score_table, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='viridis', ax=ax)
    ax.set_title('Clean validation F1 by region and model')
    save_figure(fig, figures, 'clean_f1_heatmap')

    selected = predictions.loc[predictions.seed.eq(predictions.seed.min())]
    conditions = sorted(selected.condition.unique())
    for curve_name in ['roc', 'pr']:
        fig, axes = plt.subplots(1, len(conditions), figsize=(6.5 * len(conditions), 5), squeeze=False)
        for ax, condition in zip(axes.flat, conditions):
            for model, group in selected.loc[selected.condition.eq(condition)].groupby('model'):
                if curve_name == 'roc':
                    x, y, _ = roc_curve(group.y_true, group.y_score); label = f'{model} AUC={roc_auc_score(group.y_true, group.y_score):.3f}'
                    ax.plot(x, y, label=label); ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=.7)
                    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
                else:
                    y, x, _ = precision_recall_curve(group.y_true, group.y_score); label = f'{model} AP={average_precision_score(group.y_true, group.y_score):.3f}'
                    ax.plot(x, y, label=label); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            ax.set_title(f'{curve_name.upper()} — {condition}'); ax.legend(fontsize=7)
        save_figure(fig, figures, f'{curve_name}_summary_seed_{int(selected.seed.min())}')

    for region in sorted(selected.region.unique()):
        region_data = selected.loc[selected.region.eq(region)]
        for curve_name in ['roc', 'pr']:
            fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.5), squeeze=False)
            for ax, condition in zip(axes.flat, conditions):
                for model, group in region_data.loc[region_data.condition.eq(condition)].groupby('model'):
                    if curve_name == 'roc':
                        x, y, _ = roc_curve(group.y_true, group.y_score); score = roc_auc_score(group.y_true, group.y_score)
                        ax.plot(x, y, label=f'{model} {score:.2f}'); ax.plot([0,1],[0,1],'--',color='grey',linewidth=.6)
                        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
                    else:
                        y, x, _ = precision_recall_curve(group.y_true, group.y_score); score = average_precision_score(group.y_true, group.y_score)
                        ax.plot(x, y, label=f'{model} {score:.2f}'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
                    ax.set_title(f'{region} — {condition}'); ax.legend(fontsize=6)
            save_figure(fig, figures, f'{curve_name}_{region.lower()}')

    models = sorted(metrics.model.unique())
    fig, axes = plt.subplots(len(conditions), len(models),
                             figsize=(4 * len(models), 3.5 * len(conditions)), squeeze=False)
    for row, condition in enumerate(conditions):
        for col, model in enumerate(models):
            part = metrics.loc[(metrics.condition.eq(condition)) & (metrics.model.eq(model))]
            matrix = np.array([[part.tn.sum(), part.fp.sum()], [part.fn.sum(), part.tp.sum()]])
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            sns.heatmap(matrix, annot=True, fmt='.3f', vmin=0, vmax=1, cbar=False, ax=axes[row, col])
            axes[row, col].set_title(f'{model}\n{condition}'); axes[row, col].set_xlabel('Predicted'); axes[row, col].set_ylabel('Actual')
    fig.tight_layout(); save_figure(fig, figures, 'confusion_matrices_normalized')


def run(config_path, output_override=None):
    started = time.time()
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    output = Path(output_override or config['output'])
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True); figures.mkdir()
    frames, manifest = load_dataset(config['data_dir'], verify_hashes=True)
    train = frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    if config.get('conditions', ['clean']) != ['clean']:
        raise ValueError('Clean model selection accepts only conditions: [clean]')
    validation = {
        'clean': frames['clean'].loc[frames['clean'].split.eq('validation')].reset_index(drop=True)
    }
    transform = preprocessor()
    x_train = transform.fit_transform(train)
    x_validation = {condition: transform.transform(frame) for condition, frame in validation.items()}
    regions = sorted(manifest.region.unique())
    all_metrics, all_predictions, timing = [], [], []
    for seed in config['seeds']:
        for model_name, params in config['models'].items():
            for region in regions:
                y_train = train.region.eq(region).astype(int).to_numpy()
                positive = y_train.sum(); negative = len(y_train) - positive
                weights = np.where(y_train == 1, len(y_train) / (2 * positive), len(y_train) / (2 * negative))
                model = estimator(model_name, params, seed)
                fit_started = time.time(); model.fit(x_train, y_train, sample_weight=weights)
                fit_seconds = time.time() - fit_started
                for condition, frame in validation.items():
                    predict_started = time.time(); score = model.predict_proba(x_validation[condition])[:, 1]
                    predict_seconds = time.time() - predict_started
                    y_true = frame.region.eq(region).astype(int).to_numpy()
                    result = binary_metrics(y_true, score, config['threshold'])
                    all_metrics.append({'model': model_name, 'seed': seed, 'region': region,
                                        'condition': condition, **result})
                    all_predictions.append(pd.DataFrame({
                        'POI_ID': frame.POI_ID, 'model': model_name, 'seed': seed,
                        'region': region, 'condition': condition, 'split': 'validation',
                        'y_true': y_true, 'y_score': score,
                        'y_pred': (score >= config['threshold']).astype(int)}))
                    timing.append({'model': model_name, 'seed': seed, 'region': region,
                                   'condition': condition, 'fit_seconds': fit_seconds,
                                   'predict_seconds': predict_seconds})
                print(f'completed {model_name} seed={seed} region={region}', flush=True)
    metrics = pd.DataFrame(all_metrics); predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(tables / 'metrics_by_region.csv', index=False)
    pd.DataFrame(timing).to_csv(tables / 'timing.csv', index=False)
    predictions.to_parquet(tables / 'all_predictions.parquet', index=False)
    plot_results(metrics, predictions, figures)
    ranking = (metrics.loc[metrics.condition.eq('clean')]
               .groupby('model').f1.agg(['mean', 'std'])
               .sort_values('mean', ascending=False).reset_index())
    ranking.to_csv(tables / 'model_ranking_validation.csv', index=False)
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    metadata = {'status':'complete', 'scope':'validation only', 'task':'17 binary one-vs-rest',
                'models':list(config['models']), 'seeds':config['seeds'], 'threshold':config['threshold'],
                'fits':len(config['models'])*len(config['seeds'])*len(regions),
                'train_rows':len(train), 'validation_rows':len(validation['clean']),
                'git_commit':commit, 'git_dirty':dirty, 'python':platform.python_version(),
                'dataset_sha256':{name:sha256(Path(config['data_dir'])/name) for name in
                                  ['poi_data_region.csv','poi_adversarial_data_final.csv','sample_manifest.csv']},
                'versions':{name:importlib.metadata.version(name) for name in
                            ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost']},
                'seconds':time.time()-started}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'metrics.json').write_text(json.dumps({'ranking':ranking.to_dict(orient='records')},indent=2)+'\n')
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    best = ranking.iloc[0]
    report = f'''# 06. 17개 지역 OvR 트리 모델 clean 비교

Train 23,800개 POI로 17개 지역별 이진분류기를 학습하고 validation 3,400개 POI의
clean 조건만 평가했다. 5개 모델 × 17개 지역 × 3 seeds = {metadata['fits']}회 학습이다.

현재 1위는 `{best['model']}`이며 clean validation Macro F1 평균은 {best['mean']:.4f}이다.
이 순위는 트리 모델만 포함한 중간 결과이고 TabPFN 3개 모델과 사전 정의한 최종 분석을
완료하기 전에는 최종 모델로 고정하지 않는다. 테스트 split은 사용하지 않았다.

## 산출물

- `tables/metrics_by_region.csv`, `metrics_summary.csv`, `model_ranking_validation.csv`
- `tables/all_predictions.parquet`, `timing.csv`
- 모델·조건별 지표 그림, normalized confusion matrix
- 전체 및 17개 지역별 PR/ROC curve (PNG 300 dpi)

Accuracy는 약 1:16 불균형으로 높게 보일 수 있으므로 모델 선택에는 clean Macro F1을 사용한다.
'''
    (output/'report.md').write_text(report)
    (output.parent/'_TREE_SUCCESS.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(ranking.to_string(index=False)); print(json.dumps({'status':'complete','seconds':metadata['seconds']},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/tree_benchmark.yaml')
    parser.add_argument('--output'); args=parser.parse_args(); run(args.config, args.output)


if __name__=='__main__': main()
