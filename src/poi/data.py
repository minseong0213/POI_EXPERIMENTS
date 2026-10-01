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
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _verify_files(directory, names):
    report = json.loads((directory / 'report.json').read_text())
    for name in names:
        if sha256(directory / name) != report['outputs'][name]['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {name}')


def _read_manifest(directory):
    manifest = pd.read_csv(
        directory / 'sample_manifest.csv', dtype=str, keep_default_na=False)
    required = {'POI_ID', 'region', 'split'}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f'Missing columns in sample_manifest.csv: {required - set(manifest.columns)}')
    if (not manifest.POI_ID.is_unique or manifest.POI_ID.eq('').any() or
            manifest.region.eq('').any()):
        raise ValueError('Duplicate/empty POI_ID or empty region: sample_manifest.csv')
    if set(manifest.split) != set(SPLITS):
        raise ValueError('Manifest must contain exactly train, validation, test splits')
    regions = set(manifest.region)
    for split in SPLITS:
        if set(manifest.loc[manifest.split.eq(split), 'region']) != regions:
            raise ValueError(f'Missing regions in {split}')
    return manifest


def _validate_feature_frame(frame, manifest, filename):
    required = {'POI_ID', 'region'} | set(FEATURES)
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns in {filename}: {required - set(frame.columns)}')
    if (not frame.POI_ID.is_unique or frame.POI_ID.eq('').any() or
            frame.region.eq('').any()):
        raise ValueError(f'Duplicate/empty POI_ID or empty region: {filename}')
    if frame.POI_ID.tolist() != manifest.POI_ID.tolist():
        raise ValueError(f'POI order differs from manifest: {filename}')
    if frame.region.tolist() != manifest.region.tolist():
        raise ValueError(f'Region labels differ from manifest: {filename}')


def load_clean_splits(directory, splits=('train',), verify_hashes=True):
    """Load only requested clean splits without materializing held-out features.

    The manifest contains identifiers, labels and split membership but no model features, so it is
    safe to read in full. ``skiprows`` prevents feature rows outside ``splits`` from entering the
    clean DataFrame. Both returned frames are restricted to the requested splits and retain the
    frozen manifest order.
    """
    directory = Path(directory)
    requested = (splits,) if isinstance(splits, str) else tuple(splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('splits must contain one or more unique split names')
    unknown = set(requested) - set(SPLITS)
    if unknown:
        raise ValueError(f'Unknown splits: {sorted(unknown)}')
    if verify_hashes:
        _verify_files(directory, ['poi_data_region.csv', 'sample_manifest.csv'])
    full_manifest = _read_manifest(directory)
    selected_mask = full_manifest.split.isin(requested).to_numpy()
    selected_positions = set(np.flatnonzero(selected_mask).tolist())
    clean = pd.read_csv(
        directory / 'poi_data_region.csv', dtype=str, keep_default_na=False,
        skiprows=lambda row: row != 0 and (row - 1) not in selected_positions)
    manifest = full_manifest.loc[selected_mask].reset_index(drop=True)
    _validate_feature_frame(clean, manifest, 'poi_data_region.csv')
    clean['split'] = manifest.split.to_numpy()
    return clean, manifest


def load_generated_attacks(path, split, generation_seed, valid_only=False):
    """Load one explicit attack-generation seed without collapsing duplicate POI intents."""
    if split not in {'train', 'validation'}:
        raise ValueError('Generated attacks may only be loaded from train or validation')
    if generation_seed is None:
        raise ValueError('generation_seed is required')
    frame = pd.read_parquet(
        path, filters=[('split', '==', split),
                       ('generation_seed', '==', int(generation_seed))])
    required = {
        'POI_ID', 'region', 'split', 'attack_condition', 'generation_seed',
        'constraints_valid', 'primary_evaluation_eligible',
        'attack_schema_version', 'generation_identity_sha256',
        'surrogate_checkpoint_sha256', 'query_count_scope',
    }
    if not required.issubset(frame.columns):
        raise ValueError(f'Generated attack schema is missing: {required - set(frame.columns)}')
    if frame.empty:
        raise ValueError(
            f'No generated attacks for split={split}, generation_seed={generation_seed}')
    if set(frame.split) != {split} or set(frame.generation_seed.astype(int)) != {
            int(generation_seed)}:
        raise ValueError('Parquet filtering returned a different split or generation seed')
    if frame.duplicated(['POI_ID', 'attack_condition']).any():
        raise ValueError('Duplicate POI/attack condition rows within generation seed')
    for column in ('attack_schema_version', 'generation_identity_sha256',
                   'surrogate_checkpoint_sha256', 'query_count_scope'):
        values = frame[column].fillna('').astype(str)
        if values.eq('').any() or values.nunique() != 1:
            raise ValueError(f'Generated attack provenance is empty or inconsistent: {column}')
    if valid_only:
        frame = frame.loc[frame.primary_evaluation_eligible.astype(bool)]
    return frame.reset_index(drop=True)


def load_dataset(directory, verify_hashes=True):
    directory = Path(directory)
    if verify_hashes:
        _verify_files(directory, FILES)
    manifest = _read_manifest(directory)
    clean, attack = [pd.read_csv(directory / f, dtype=str, keep_default_na=False)
                     for f in FILES[:2]]
    for name, frame in zip(FILES[:2], [clean, attack]):
        required = {'POI_ID', 'region'} | set(FEATURES)
        if not required.issubset(frame.columns):
            raise ValueError(f'Missing columns in {name}: {required - set(frame.columns)}')
        if not frame.POI_ID.is_unique or frame.POI_ID.eq('').any() or frame.region.eq('').any():
            raise ValueError(f'Duplicate/empty POI_ID or empty region: {name}')
        if set(frame.POI_ID) != set(manifest.POI_ID):
            raise ValueError(f'ID set differs from manifest: {name}')
    frames = {}
    for label, frame in [('clean', clean), ('attack', attack)]:
        frame = frame.set_index('POI_ID').loc[manifest.POI_ID].reset_index()
        if frame.region.tolist() != manifest.region.tolist():
            raise ValueError(f'Region labels differ from manifest: {label}')
        frame['split'] = manifest.split.to_numpy()
        frames[label] = frame
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
