"""Explicit paired clean-to-attack correlation changes from immutable EDA tables."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import sha256

KEY = ['region', 'attack_condition', 'generation_seed', 'method']


def paired_changes(frame):
    if frame.duplicated(KEY+['condition']).any():
        raise ValueError('Duplicate correlation identity')
    if set(frame.condition) != {'clean', 'adversarial'}:
        raise ValueError('Requires clean and adversarial correlations')
    wide = frame.pivot(index=KEY, columns='condition', values='correlation')
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy()).all():
        raise ValueError('Missing/nonfinite paired correlation')
    wide['correlation_change_attack_minus_clean'] = wide.adversarial-wide.clean
    return wide.reset_index().rename(columns={'clean':'clean_correlation',
                                             'adversarial':'adversarial_correlation'})


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs = {str(source/'tables/correlations.csv'): sha256(source/'tables/correlations.csv'),
              str(source/'metadata.json'): sha256(source/'metadata.json'),
              str(Path(__file__)): sha256(__file__)}
    frame = pd.read_csv(source/'tables/correlations.csv')
    table = paired_changes(frame)
    expected = {(r, c, s, m) for r in frame.region.unique()
                for c in frame.attack_condition.unique() for s in [42, 202, 340]
                for m in ['pearson', 'spearman']}
    if (len(frame.region.unique()) != 17 or len(frame.attack_condition.unique()) != 32
            or set(table[KEY].itertuples(index=False, name=None)) != expected):
        raise ValueError('Incomplete 17-region/32-condition/3-seed/2-method grid')
    (output/'tables').mkdir(parents=True); (output/'figures').mkdir(); (output/'provenance').mkdir()
    shutil.copyfile(__file__, output/'provenance/correlation_changes.py')
    table.to_csv(output/'tables/correlation_changes.csv', index=False)
    for method in ['pearson', 'spearman']:
        part = table.loc[table.method.eq(method)]
        limit = max(float(part.correlation_change_attack_minus_clean.abs().max()), 1e-12)
        for seed in [42, 202, 340]:
            grid = part.loc[part.generation_seed.eq(seed)].pivot(
                index='region', columns='attack_condition', values='correlation_change_attack_minus_clean')
            fig, ax = plt.subplots(figsize=(16, 8))
            plot = ax.imshow(grid.to_numpy(), vmin=-limit, vmax=limit, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=90)
            ax.set_yticks(range(len(grid.index)), grid.index)
            ax.set_title(f'{method}: correlation change (attack − clean), generation seed {seed}')
            fig.colorbar(plot, ax=ax, label='Correlation difference')
            fig.tight_layout(); fig.savefig(output/'figures'/f'{method}_change_seed{seed}.png', dpi=300)
            plt.close(fig)
    if any(sha256(path) != value for path,value in inputs.items()):
        raise RuntimeError('Source changed during derivation')
    meta = dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW', paired_change_rows=len(table),
                input_sha256=inputs, test_features_read=0, source_features='X_COORD versus Y_COORD',
                inference='descriptive paired point differences; no new significance test or CI',
                remaining_stage05_scope=['CCA loadings and bootstrap CI',
                                         'region/category chi-square and Cramers V'])
    (output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    (output/'report.md').write_text('# Clean-to-attack correlation changes\n\n'
        '3,264 changes pair Pearson/Spearman coordinate correlations by region, attack condition and '
        'generation seed. Values are adversarial minus clean on the same eligible POI pairs. '
        'The six PNG heatmaps use a common symmetric color range within each method.\n\n'
        'This is a descriptive contrast. Separate correlation p-values do not test their difference; '
        'no such inference is claimed. CCA loading/bootstrap intervals and region/category associations '
        'remain separate outstanding analyses. Independent review is pending.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.source, args.output)
