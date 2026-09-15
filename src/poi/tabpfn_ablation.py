"""Stage 06 GPU feature ablation for the selected TabPFN v2.5 model."""
import argparse
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
                             f1_score, precision_score, recall_score, roc_auc_score)
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import tabpfn
import torch
import yaml

from .data import CATEGORICAL, load_dataset, sha256


def numeric_matrix(frame, features):
    matrix = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('TabPFN ablation input contains non-finite values')
    return matrix


def metrics(y_true, score, threshold):
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        'precision': precision_score(y_true, predicted, zero_division=0),
        'recall': recall_score(y_true, predicted, zero_division=0),
        'f1': f1_score(y_true, predicted, zero_division=0),
        'accuracy': accuracy_score(y_true, predicted),
        'roc_auc': roc_auc_score(y_true, score),
        'average_precision': average_precision_score(y_true, score),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def atomic_frame(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run(config_path, output_override=None):
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_override or cfg['output'])
    if (out / '_SUCCESS.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed {out}')
    if out.exists() and not cfg.get('resume_partial', False):
        raise FileExistsError(f'Refusing to resume {out}')
    if not os.environ.get('TABPFN_TOKEN'):
        token_path = Path(os.environ.get('TABPFN_TOKEN_FILE', '/root/.config/poi/tabpfn_token'))
        if token_path.is_file():
            os.environ['TABPFN_TOKEN'] = token_path.read_text().strip()
    if not os.environ.get('TABPFN_TOKEN'):
        raise RuntimeError('TABPFN_TOKEN or TABPFN_TOKEN_FILE is required')

    tables, figures, work = out / 'tables', out / 'figures', out / 'work'
    shards = work / 'metrics'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    data_dir = os.environ.get('DATA_DIR', cfg['data_dir'])
    frames, manifest = load_dataset(data_dir)
    train = frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    clean = frames['clean'].loc[frames['clean'].split.eq('validation')].reset_index(drop=True)
    attack = frames['attack'].loc[frames['attack'].split.eq('validation')].reset_index(drop=True)
    regions = sorted(manifest.region.unique())
    rows = []

    for ablation, features in cfg['feature_sets'].items():
        x_train = numeric_matrix(train, features)
        x_eval = np.vstack([numeric_matrix(clean, features), numeric_matrix(attack, features)])
        categorical_indices = [index for index, feature in enumerate(features)
                               if feature in CATEGORICAL]
        for seed in cfg['seeds']:
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2_5, device=cfg['device'], n_estimators=cfg['n_estimators'],
                random_state=seed, categorical_features_indices=categorical_indices,
                show_progress_bar=True, memory_saving_mode='auto')
            for start in range(0, len(regions), cfg['region_batch_size']):
                batch = regions[start:start + cfg['region_batch_size']]
                stem = f'{ablation}-seed{seed}-regions{start:02d}'
                shard = shards / f'{stem}.csv'
                if shard.exists():
                    existing = pd.read_csv(shard)
                    if len(existing) != len(batch) * 2:
                        raise RuntimeError(f'Incomplete checkpoint shard: {stem}')
                    rows.extend(existing.to_dict(orient='records'))
                    print(f'resumed {ablation} seed={seed} regions={batch}', flush=True)
                    continue
                labels = [train.region.eq(region).astype(int).to_numpy() for region in batch]
                probabilities = None
                for attempt in range(1, cfg.get('batch_retries', 1) + 1):
                    try:
                        probabilities = model.predict_proba_batched(
                            [x_train] * len(batch), labels, [x_eval] * len(batch))
                        break
                    except Exception as error:
                        if attempt >= cfg.get('batch_retries', 1):
                            raise
                        print(f'retrying {stem} after {type(error).__name__} '
                              f'attempt={attempt}', flush=True)
                        torch.cuda.empty_cache()
                        time.sleep(min(10 * attempt, 30))
                batch_rows = []
                for index, region in enumerate(batch):
                    scores = np.asarray(probabilities[index])[:, 1]
                    for condition, offset, frame in [
                            ('clean', 0, clean),
                            ('adversarial', len(clean), attack)]:
                        values = scores[offset:offset + len(frame)]
                        truth = frame.region.eq(region).astype(int).to_numpy()
                        batch_rows.append({
                            'model': 'tabpfn_v2_5', 'ablation': ablation,
                            'features': '|'.join(features), 'seed': seed,
                            'region': region, 'condition': condition,
                            **metrics(truth, values, cfg['threshold']),
                        })
                atomic_frame(pd.DataFrame(batch_rows), shard)
                rows.extend(batch_rows)
                print(f'completed {ablation} seed={seed} regions={batch}', flush=True)
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    expected = len(cfg['feature_sets']) * len(cfg['seeds']) * len(regions) * 2
    if len(result) != expected or set(result.model) != {'tabpfn_v2_5'}:
        raise RuntimeError('Incomplete TabPFN ablation cardinality')
    result.to_csv(tables / 'ablation_metrics.csv', index=False)
    paired = result.pivot_table(index=['ablation', 'seed', 'region'],
                                columns='condition', values='f1').reset_index()
    paired['robust_f1'] = (paired.clean + paired.adversarial) / 2
    bootstrap = np.random.default_rng(cfg['bootstrap_seed'])
    summary_rows = []
    for ablation, group in paired.groupby('ablation'):
        values = group.robust_f1.to_numpy()
        draws = bootstrap.choice(values, size=(cfg['bootstrap_repeats'], len(values)),
                                 replace=True).mean(axis=1)
        summary_rows.append({
            'ablation': ablation, 'mean': values.mean(), 'std': values.std(ddof=1),
            'ci_low': np.quantile(draws, .025), 'ci_high': np.quantile(draws, .975),
        })
    summary = pd.DataFrame(summary_rows).sort_values('mean', ascending=False).reset_index(drop=True)
    baseline = float(summary.loc[summary.ablation.eq('full'), 'mean'].iloc[0])
    summary['delta_from_full'] = summary['mean'] - baseline
    summary.to_csv(tables / 'ablation_summary.csv', index=False)
    heat = paired.pivot_table(index='region', columns='ablation', values='robust_f1',
                              aggfunc='mean')
    heat.to_csv(tables / 'ablation_by_region.csv')

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(heat, annot=True, fmt='.2f', vmin=.5, vmax=1, cmap='viridis', ax=ax)
    ax.set_title('TabPFN v2.5 robust F1 feature ablation')
    save(fig, figures, 'ablation_heatmap')
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=summary, x='ablation', y='delta_from_full', color='#7A5195', ax=ax)
    ax.axhline(0, color='black', linewidth=.8)
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylabel('Robust F1 delta')
    save(fig, figures, 'ablation_delta')

    metadata = {
        'status': 'complete', 'model': 'tabpfn_v2_5', 'scope': 'validation',
        'seeds': cfg['seeds'], 'regions': len(regions),
        'feature_sets': list(cfg['feature_sets']), 'fits': expected // 2,
        'bootstrap_repeats': cfg['bootstrap_repeats'],
        'n_estimators': cfg['n_estimators'], 'resumable_shards': len(list(shards.glob('*.csv'))),
        'python': platform.python_version(), 'tabpfn': tabpfn.__version__,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'data_sha256': {name: sha256(Path(data_dir) / name) for name in
                        ['poi_data_region.csv', 'poi_adversarial_data_final.csv',
                         'sample_manifest.csv']},
        'seconds': time.time() - started,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'metrics.json').write_text(
        json.dumps({'summary': summary.to_dict(orient='records')}, indent=2) + '\n')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    best = summary.iloc[0]
    (out / 'report.md').write_text(f'''# 06. TabPFN v2.5 피처 ablation

8개 모델 validation 비교에서 선택된 `tabpfn_v2_5`에 대해 {len(cfg['feature_sets'])}개
피처 구성을 17개 OvR × {len(cfg['seeds'])} seeds로 평가했다. clean/adversarial F1의
평균을 robust F1으로 사용했으며 test split은 사용하지 않았다.

가장 높은 구성은 `{best.ablation}`이고 robust F1은 {best['mean']:.4f}이다. 전체
지역별 결과와 full 대비 변화량은 `tables/`에, PNG 300dpi 그림은 `figures/`에 있다.
''')
    (out / '_SUCCESS.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tabpfn_ablation.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
