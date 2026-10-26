"""Clean-only paired POI bootstrap; never combines predictions from different runs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .clean_comparison import validate_predictions
from .data import load_clean_splits
from .uncertainty import (METRICS, KEY, PREDICTION_COLUMNS, sha256,
                          make_bootstrap_weights, _analyze_groups, _summaries,
                          _all_clean_model_pairs, _check_truth_consistency)


def validate_clean(frame, validation, cfg):
    if cfg.get('attack_predictions'):
        raise ValueError('Clean-only uncertainty does not accept attack predictions')
    if tuple(cfg['metrics']) != METRICS or cfg['bootstrap_repeats'] < 2000:
        raise ValueError('Requires five planned metrics and at least 2000 draws')
    if not 0 < cfg['confidence_level'] < 1:
        raise ValueError('Invalid confidence level')
    if len(validation) != cfg['expected_pois'] or validation.region.nunique() != cfg['expected_regions']:
        raise ValueError('Validation cardinality mismatch')
    validate_predictions(frame, validation, cfg['models'], cfg['expected_seeds'], cfg['threshold'])
    first_key = next(iter(frame.groupby(KEY, sort=False, observed=True).groups))
    order = frame.groupby(KEY, sort=False, observed=True).get_group(first_key).POI_ID.to_numpy(copy=True)
    _check_truth_consistency(frame, order)
    return order


def run(config_path):
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError(output)
    sources = [Path(__file__), Path(__file__).with_name('uncertainty.py'),
               Path(__file__).with_name('clean_comparison.py'), Path(__file__).with_name('data.py'),
               config_path]
    input_path = Path(cfg['clean_predictions'])
    hashes = {str(p): sha256(p) for p in [*sources, input_path]}
    frame = pd.read_parquet(input_path, columns=PREDICTION_COLUMNS)
    validation, _ = load_clean_splits(cfg['data_dir'], splits=('validation',))
    order = validate_clean(frame, validation, cfg)
    tables, figures, work, provenance = (output/n for n in ('tables', 'figures', 'work', 'provenance'))
    for directory in (tables, figures, work, provenance):
        directory.mkdir(parents=True)
    for path in sources:
        shutil.copyfile(path, provenance/path.name)
    (provenance/'input_source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    weights = make_bootstrap_weights(len(order), cfg['bootstrap_repeats'], cfg['bootstrap_seed'])
    np.save(work/'bootstrap_poi_weights.npy', weights, allow_pickle=False)
    pd.DataFrame({'POI_ID': order}).to_csv(work/'bootstrap_poi_order.csv', index=False)
    strata, replicates = _analyze_groups(frame, frame.iloc[:0].copy(), order, weights, 'clean')
    summary = _summaries(strata, replicates, cfg['confidence_level'], cfg['bootstrap_repeats'], len(order))
    pairs = _all_clean_model_pairs(replicates, strata, cfg['confidence_level'])
    # Uncentered bootstrap tail mass is not a null-calibrated frequentist p-value.
    pairs = pairs.drop(columns=['two_sided_bootstrap_p', 'bh_fdr_q'])
    strata.to_csv(tables/'seed_region_point_metrics.csv', index=False)
    replicates.to_parquet(tables/'bootstrap_metric_replicates.parquet', index=False)
    summary.to_csv(tables/'metric_confidence_intervals.csv', index=False)
    pairs.to_csv(tables/'all_clean_model_pairwise_differences.csv', index=False)
    pd.DataFrame([{'role': 'clean_model_comparison', 'path': str(input_path),
                   'bytes': input_path.stat().st_size, 'sha256': hashes[str(input_path)]}]).to_csv(
        tables/'source_provenance.csv', index=False)
    for metric in METRICS:
        subset = summary.loc[summary.metric.eq(metric)].sort_values('estimate')
        fig, ax = plt.subplots(figsize=(10, 6))
        # Draw intervals directly: percentile intervals need not bracket the point estimate.
        ax.hlines(subset.model, subset.ci_low, subset.ci_high, color='#2b6f92')
        ax.scatter(subset.estimate, subset.model, color='#2b6f92')
        ax.set(xlabel=f'Macro {metric}', ylabel='Model',
               title=f'Clean validation: paired POI bootstrap {cfg["confidence_level"]:.0%} CI')
        ax.grid(axis='x', alpha=.25)
        fig.tight_layout()
        fig.savefig(figures/f'clean_model_{metric}_ci.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    after = {path: sha256(path) for path in hashes}
    if after != hashes:
        raise RuntimeError('Input/config/source changed during uncertainty analysis')
    metadata = dict(status='executed_pending_independent_review', scope='clean validation only',
                    test_split_used=False, attack_predictions_used=False,
                    bootstrap_unit='POI_ID', bootstrap_repeats=cfg['bootstrap_repeats'],
                    bootstrap_seed=cfg['bootstrap_seed'], confidence_level=cfg['confidence_level'],
                    unique_pois=len(order), regions=cfg['expected_regions'],
                    seeds=cfg['expected_seeds'], clean_models=cfg['models'], metrics=list(METRICS),
                    bootstrap_weights_sha256=sha256(work/'bootstrap_poi_weights.npy'),
                    bootstrap_poi_order_sha256=sha256(work/'bootstrap_poi_order.csv'),
                    aggregation='equal mean of per-region/per-model-seed metrics within each POI resample',
                    intervals='marginal percentile, not simultaneous or multiplicity adjusted',
                    limitations=['Conditional on fitted models and observed validation sample.',
                                 'Spatial dependence and training or attack-generation variability are not resampled.',
                                 'Selection on the same validation data can be optimistic.',
                                 'Historical test-coordinate access remains disclosed; no new test access here.'],
                    input_source_hashes=hashes, seconds=time.time()-started,
                    versions={n: importlib.metadata.version(n) for n in
                              ['numpy', 'pandas', 'numba', 'matplotlib', 'pyarrow']})
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    shutil.copyfile(config_path, output/'config.yaml')
    (output/'report.md').write_text(
        '# Clean model uncertainty\n\n'
        f'{len(order):,} validation POIs, {len(cfg["models"])} models, '
        f'{cfg["expected_regions"]} OvR regions, {len(cfg["expected_seeds"])} model seeds; '
        f'{cfg["bootstrap_repeats"]:,} shared POI bootstrap draws.\n\n'
        'Five metrics are computed separately per seed and region and equally averaged inside each draw. '
        'All model comparisons use the same POI multiplicities. Tables contain marginal percentile '
        'confidence intervals and all unordered model-pair differences. Seeds/regions are not independent '
        'resampling units. No legacy attack predictions are used.\n\n'
        'These conditional validation intervals do not establish attack robustness, remove model-selection '
        'bias, account for spatial dependence, or restore pristine historical test isolation. '
        'Independent review is pending.\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
