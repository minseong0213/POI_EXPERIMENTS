"""Read paired data with strict manifest validation; never re-split the data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ['X_COORD', 'Y_COORD']
CATEGORICAL = ['ASORT_LCLASDC', 'ASORT_MLSFCDC', 'ASORT_SDASDC']
FEATURES = NUMERIC + CATEGORICAL
FILES = ['poi_data_region.csv', 'poi_adversarial_data_final.csv', 'sample_manifest.csv']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_dataset(directory, verify_hashes=True):
    directory = Path(directory)
    if verify_hashes:
        report = json.loads((directory / 'report.json').read_text())
        for name in FILES:
            if sha256(directory / name) != report['outputs'][name]['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {name}')
    clean, attack, manifest = [pd.read_csv(directory / f, dtype=str, keep_default_na=False) for f in FILES]
    for name, frame in zip(FILES, [clean, attack, manifest]):
        required = {'POI_ID', 'region'} | ({'split'} if name == FILES[2] else set(FEATURES))
        if not required.issubset(frame.columns):
            raise ValueError(f'Missing columns in {name}: {required - set(frame.columns)}')
        if not frame.POI_ID.is_unique or frame.POI_ID.eq('').any() or frame.region.eq('').any():
            raise ValueError(f'Duplicate/empty POI_ID or empty region: {name}')
        if set(frame.POI_ID) != set(manifest.POI_ID):
            raise ValueError(f'ID set differs from manifest: {name}')
    if set(manifest.split) != {'train', 'validation', 'test'}:
        raise ValueError('Manifest must contain exactly train, validation, test splits')
    frames = {}
    for label, frame in [('clean', clean), ('attack', attack)]:
        frame = frame.set_index('POI_ID').loc[manifest.POI_ID].reset_index()
        if frame.region.tolist() != manifest.region.tolist():
            raise ValueError(f'Region labels differ from manifest: {label}')
        frame['split'] = manifest.split.to_numpy()
        frames[label] = frame
    regions = set(manifest.region)
    for split in ['train', 'validation', 'test']:
        if set(manifest.loc[manifest.split.eq(split), 'region']) != regions:
            raise ValueError(f'Missing regions in {split}')
    return frames, manifest


def features(frame):
    result = frame[FEATURES].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col].replace('', np.nan), errors='raise')
        if np.isinf(result[col]).any():
            raise ValueError(f'Infinite numeric values in {col}')
    for col in CATEGORICAL:
        result[col] = result[col].replace('', '__MISSING__')
    return result


def select_split(frame, split, per_region=None, seed=42):
    subset = frame.loc[frame.split.eq(split)]
    if per_region is not None:
        if not isinstance(per_region, int) or per_region <= 0:
            raise ValueError('per_region must be a positive integer or null')
        subset = pd.concat([group.sample(n=min(per_region, len(group)), random_state=seed)
                            for _, group in subset.groupby('region', sort=True)])
    return subset.reset_index(drop=True)
