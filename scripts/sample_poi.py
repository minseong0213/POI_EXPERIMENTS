"""Build a reproducible, region-balanced, ID-paired POI subset locally.

CSV values are read as strings, preserving identifiers, leading zeros and blanks.
No source files are modified. R2 publishing is a separate operation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FILES = ('poi_data_region.csv', 'poi_adversarial_data_final.csv')
READ = dict(dtype=str, keep_default_na=False)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def balanced_quotas(counts, total):
    if total > sum(counts.values()):
        raise ValueError('Requested more rows than available')
    quotas = {region: 0 for region in sorted(counts)}
    # Water filling; ties resolved by region name, with no replacement.
    remaining = total
    while remaining:
        for region in quotas:
            if quotas[region] < counts[region]:
                quotas[region] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


def apportion(quotas, fraction, total):
    result = {r: int(n * fraction) for r, n in quotas.items()}
    order = sorted(quotas, key=lambda r: (-(quotas[r] * fraction - result[r]), r))
    for region in order[:total - sum(result.values())]:
        result[region] += 1
    assert sum(result.values()) == total
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/source'))
    parser.add_argument('--output', type=Path, default=Path('data/poi_34k_seed42'))
    parser.add_argument('--size', type=int, default=34000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError('size must be positive')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')

    metadata = [pd.read_csv(args.source / name, usecols=['POI_ID', 'region'], **READ) for name in FILES]
    clean, attack = metadata
    for frame in metadata:
        assert frame.POI_ID.is_unique and not frame.POI_ID.eq('').any()
        assert not frame.region.eq('').any()
    assert set(clean.POI_ID) == set(attack.POI_ID)
    attack_regions = attack.set_index('POI_ID').region.reindex(clean.POI_ID)
    assert np.array_equal(clean.region.to_numpy(), attack_regions.to_numpy())

    counts = clean.region.value_counts().to_dict()
    quotas = balanced_quotas(counts, args.size)
    validation = apportion(quotas, .1, round(args.size * .1))
    testing = apportion(quotas, .2, round(args.size * .2))
    rng = np.random.default_rng(args.seed)
    selected = []
    for region, quota in quotas.items():
        indices = clean.index[clean.region.eq(region)].to_numpy()
        chosen = rng.choice(indices, size=quota, replace=False)
        n_train = quota - validation[region] - testing[region]
        rows = clean.loc[chosen].copy()
        rows['source_row_index'] = chosen
        rows['split'] = (['train'] * n_train + ['validation'] * validation[region]
                         + ['test'] * testing[region])
        selected.append(rows)
    manifest = pd.concat(selected).sort_values('source_row_index').reset_index(drop=True)
    assert len(manifest) == args.size and manifest.POI_ID.is_unique
    assert manifest.split.value_counts().to_dict() == {
        'train': args.size - round(args.size * .1) - round(args.size * .2),
        'validation': round(args.size * .1), 'test': round(args.size * .2)}

    args.output.mkdir(parents=True)
    manifest.to_csv(args.output / 'sample_manifest.csv', index=False)
    ids = set(manifest.POI_ID)
    extracted = []
    source_stats = {}
    for name in FILES:
        parts = []
        n_rows = 0
        for chunk in pd.read_csv(args.source / name, chunksize=50000, **READ):
            n_rows += len(chunk)
            parts.append(chunk.loc[chunk.POI_ID.isin(ids)])
        subset = pd.concat(parts).set_index('POI_ID').loc[manifest.POI_ID].reset_index()
        assert len(subset) == args.size
        assert subset.POI_ID.tolist() == manifest.POI_ID.tolist()
        assert subset.region.tolist() == manifest.region.tolist()
        output = args.output / name
        subset.to_csv(output, index=False)
        # Verify the actual saved values, not just in-memory selection.
        pd.testing.assert_frame_equal(subset, pd.read_csv(output, **READ))
        extracted.append(subset)
        source_stats[name] = {'rows': n_rows, 'bytes': (args.source / name).stat().st_size,
                              'sha256': digest(args.source / name),
                              'md5': digest(args.source / name, 'md5')}
    assert extracted[0].columns.tolist() == extracted[1].columns.tolist()
    changed = extracted[0].ne(extracted[1])
    region_report = pd.DataFrame([
        {'region': r, 'source_rows': counts[r], 'selected_rows': quotas[r],
         'train': quotas[r] - validation[r] - testing[r],
         'validation': validation[r], 'test': testing[r],
         'source_share': counts[r] / len(clean), 'sample_share': quotas[r] / args.size}
        for r in quotas])
    region_report.to_csv(args.output / 'region_counts.csv', index=False)
    report = {
        'seed': args.seed, 'original_rows': len(clean), 'selected_original_rows': args.size,
        'selected_attack_rows': args.size, 'unique_original_poi_ids': args.size,
        'sampling': 'Capped region-balanced sampling without replacement; lexical tie breaks',
        'split_method': 'Within-region random 70/10/20 allocation with largest-remainder rounding',
        'split_counts': manifest.split.value_counts().to_dict(),
        'pairing': 'POI_ID join; same order and same region labels in both outputs',
        'source_ids_unique': True, 'source_id_sets_equal': True,
        'source_order_equal': clean.POI_ID.equals(attack.POI_ID),
        'changed_rows_in_attack_subset': int(changed.any(axis=1).sum()),
        'changed_cells_by_column': {c: int(changed[c].sum()) for c in changed},
        'sources': source_stats,
        'regions': region_report.to_dict(orient='records'),
        'limitations': [
            'Balanced subset does not preserve original region prevalence.',
            'Sejong has only 1831 source rows; all are included.',
            'Existing attack file has no attack-type label; PGD/CW/FGSM cannot be separated.',
            'Attack-generation model, training split and parameters are unknown; this split does not validate attack-generation independence.',
            'Split isolates POI_ID only; spatial or other cross-POI dependencies are not excluded.',
            'Use sample_manifest.csv for both files; do not independently split clean and attack rows.',
            'Filtered defense evaluation must report coverage/rejection and end-to-end performance, not only accuracy among accepted rows.',
        ],
        'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                    for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['original_rows', 'selected_original_rows',
          'selected_attack_rows', 'split_counts', 'changed_cells_by_column']}, indent=2))


if __name__ == '__main__':
    main()
