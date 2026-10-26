"""Assemble the eight clean-validation models without fitting or selecting on test."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .data import load_clean_splits, sha256


def validate_predictions(frame, validation, models, seeds, threshold=.5):
    required = {'POI_ID', 'model', 'seed', 'region', 'y_true', 'y_score', 'y_pred'}
    if not required.issubset(frame):
        raise ValueError(f'Missing prediction columns: {required-set(frame)}')
    if validation.POI_ID.duplicated().any():
        raise ValueError('Duplicate validation identity')
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing prediction values')
    regions = sorted(validation.region.unique())
    expected_groups = {(model, seed, region) for model in models for seed in seeds for region in regions}
    keys = ['model', 'seed', 'region', 'POI_ID']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate prediction identity')
    actual_groups = set(frame[['model', 'seed', 'region']].itertuples(index=False, name=None))
    if actual_groups != expected_groups:
        raise ValueError('Model/seed/region Cartesian coverage mismatch')
    expected_ids = set(validation.POI_ID)
    for _, group in frame.groupby(['model', 'seed', 'region'], sort=False):
        if set(group.POI_ID) != expected_ids:
            raise ValueError('Incomplete validation POI coverage')
    for column in ('condition', 'attack_condition'):
        if column in frame and not frame[column].eq('clean').all():
            raise ValueError('Non-clean prediction included')
    if 'split' in frame and not frame.split.eq('validation').all():
        raise ValueError('Non-validation prediction included')
    source_region = frame.POI_ID.map(validation.set_index('POI_ID').region)
    if not np.array_equal(frame.y_true.to_numpy(), source_region.eq(frame.region).astype(int)):
        raise ValueError('Prediction truth differs from validation manifest')
    scores = pd.to_numeric(frame.y_score, errors='raise').to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid probability')
    if not np.array_equal(frame.y_pred.to_numpy(), (scores >= threshold).astype(int)):
        raise ValueError('Predictions differ from the frozen independent OvR threshold')
    return len(expected_groups)


def run(config_path):
    from .benchmark import binary_metrics, plot_results, save_figure
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cfg = yaml.safe_load(Path(config_path).read_text())
    expected_models = {'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'catboost',
                       'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3'}
    if len(cfg['models']) != 8 or set(cfg['models']) != expected_models:
        raise ValueError('This comparison requires all eight planned models exactly once')
    if sorted(cfg['seeds']) != [42, 202, 340] or cfg['threshold'] != .5:
        raise ValueError('Frozen seeds and independent OvR threshold differ')
    if cfg['expected_pois'] != 3400 or cfg['expected_regions'] != 17:
        raise ValueError('Frozen development validation cardinality differs')
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError(output)
    validation, _ = load_clean_splits(cfg['data_dir'], splits=('validation',))
    if len(validation) != cfg['expected_pois'] or validation.region.nunique() != cfg['expected_regions']:
        raise ValueError('Frozen validation cardinality mismatch')
    paths = [Path(cfg['tree_predictions'])]
    for model, directory in cfg['tabpfn_prediction_directories'].items():
        found = sorted(Path(directory).glob(f'{model}_full_*_clean.parquet'))
        if len(found) != len(cfg['seeds'])*cfg['expected_regions']:
            raise ValueError(f'Incomplete full-feature shards for {model}: {len(found)}')
        paths.extend(found)
    inputs = {str(path): sha256(path) for path in paths}
    parts = []
    for path in paths:
        part = pd.read_parquet(path)
        if 'ablation' in part and not part.ablation.eq('full').all():
            raise ValueError('Feature-removal predictions cannot enter full-feature comparison')
        # Older ablation shards omit split; exact POI membership is checked below.
        if 'split' not in part:
            part['split'] = 'validation'
        if 'condition' not in part:
            part['condition'] = part['attack_condition']
        parts.append(part[['POI_ID', 'model', 'seed', 'region', 'condition', 'split',
                           'y_true', 'y_score', 'y_pred']])
    predictions = pd.concat(parts, ignore_index=True)
    group_count = validate_predictions(predictions, validation, cfg['models'], cfg['seeds'], cfg['threshold'])
    tables, figures, provenance = (output/name for name in ('tables', 'figures', 'provenance'))
    for directory in (tables, figures, provenance):
        directory.mkdir(parents=True)
    source_files = [Path(__file__), Path(__file__).with_name('benchmark.py'),
                    Path(__file__).with_name('data.py'), Path(config_path)]
    for source in source_files:
        shutil.copyfile(source, provenance/source.name)
    (provenance/'source_hashes.json').write_text(json.dumps({str(p):sha256(p) for p in source_files}, indent=2)+'\n')
    rows = []
    for (model, seed, region), part in predictions.groupby(['model', 'seed', 'region'], sort=True):
        rows.append(dict(model=model, seed=int(seed), region=region, condition='clean',
                         **binary_metrics(part.y_true, part.y_score, cfg['threshold'])))
    metrics = pd.DataFrame(rows)
    metric_names = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'average_precision']
    seed_macro = metrics.groupby(['model', 'seed'])[metric_names].mean().reset_index()
    summary = seed_macro.groupby('model')[metric_names].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.reset_index().sort_values('f1_mean', ascending=False)
    metrics.to_csv(tables/'metrics_by_region.csv', index=False)
    seed_macro.to_csv(tables/'macro_by_seed.csv', index=False)
    summary.to_csv(tables/'macro_seed_summary.csv', index=False)
    predictions.to_parquet(tables/'all_predictions.parquet', index=False)
    plot_results(metrics, predictions, figures)
    radar_metrics = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axis = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    for _, row in summary.iterrows():
        values = np.asarray([row[f'{metric}_mean'] for metric in radar_metrics])
        axis.plot(closed_angles, np.r_[values, values[0]], marker='o', label=row['model'])
    axis.set_xticks(angles, radar_metrics); axis.set_ylim(0, 1)
    axis.set_title('Clean validation | 17-region macro | mean of three seeds')
    axis.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    save_figure(fig, figures, 'eight_model_radar')
    visual_seed = min(cfg['seeds'])
    for region in sorted(validation.region.unique()):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for axis, model in zip(axes.flat, cfg['models']):
            part = predictions.loc[predictions.model.eq(model) & predictions.seed.eq(visual_seed)
                                   & predictions.region.eq(region)]
            matrix = confusion_matrix(part.y_true, part.y_pred, labels=[0, 1])
            sns.heatmap(matrix, annot=True, fmt='d', cbar=False, ax=axis)
            axis.set_title(model); axis.set_xlabel('Predicted'); axis.set_ylabel('Actual')
        fig.suptitle(f'{region} vs other regions | validation | seed {visual_seed}')
        fig.tight_layout(); save_figure(fig, figures, f'confusion_{region.lower()}_seed_{visual_seed}')
    if any(sha256(path) != inputs[str(path)] for path in paths):
        raise RuntimeError('Prediction source changed during assembly')
    metadata = dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW', test_features_read=0,
        prediction_rows=len(predictions), model_region_seed_groups=group_count,
        models=cfg['models'], seeds=cfg['seeds'], threshold=cfg['threshold'],
        visualization_seed=visual_seed, input_sha256=inputs,
        source_data_sha256={name:sha256(Path(cfg['data_dir'])/name)
                           for name in ('poi_data_region.csv', 'sample_manifest.csv')},
        runtime={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','matplotlib')},
        limitations=['Empirical validation ranking, not a locked test result or selection-bias correction.',
                     'Seed standard deviations are not confidence intervals.',
                     'Source fits used their recorded execution environments; assembly does not refit models.',
                     'Prior test-coordinate exposure is not erased by this validation-only assembler.'])
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output/'report.md').write_text('# Clean validation: eight models\n\n'
        f'{group_count} model/seed/region groups and {len(predictions):,} paired predictions. '
        'All five input features; independent OvR threshold 0.5. '
        'Tables average 17 regions within each seed, then report mean and sample SD over three seeds. '
        'Use macro_seed_summary.csv for seed variability. The legacy metric_summary plot/table '
        'instead describes pooled region-seed variability; neither SD is a confidence interval. '
        f'PR/ROC and regional confusion plots show seed {visual_seed}; all seeds remain in the tables. '
        'The aggregate confusion summary pools binary OvR decisions, not a 17-class confusion matrix. '
        'This is an empirical validation comparison; independent review pending.\n')
    (output/'_GENERATED.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
