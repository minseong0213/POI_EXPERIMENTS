import pandas as pd
import pytest

from poi.data import (FEATURES, features, load_clean_splits, load_dataset,
                      load_generated_attacks, select_split)


def test_pairing_and_split_isolation(dataset):
    frames, manifest = load_dataset(dataset, verify_hashes=False)
    assert frames['clean'].POI_ID.tolist() == frames['attack'].POI_ID.tolist()
    assert frames['clean'].ASORT_LCLASDC.iloc[0] == '001'
    ids = {s: set(select_split(frames['clean'], s).POI_ID) for s in ['train', 'validation', 'test']}
    assert not (ids['train'] & ids['test'] or ids['train'] & ids['validation'] or ids['validation'] & ids['test'])
    assert len(select_split(frames['clean'], 'train', per_region=2)) == 4
    assert list(features(frames['clean'])) == FEATURES
    assert not {'POI_ID', 'region', 'split', 'SGG_CD_PREFIX'} & set(FEATURES)


@pytest.mark.parametrize('fault', ['duplicate', 'label', 'missing_id', 'bad_split'])
def test_reject_corrupt_pair_or_manifest(dataset, fault):
    file = 'sample_manifest.csv' if fault == 'bad_split' else 'poi_adversarial_data_final.csv'
    path = dataset / file
    frame = pd.read_csv(path, dtype=str)
    if fault == 'duplicate':
        frame.loc[0, 'POI_ID'] = frame.loc[1, 'POI_ID']
    elif fault == 'label':
        frame.loc[0, 'region'] = 'wrong'
    elif fault == 'missing_id':
        frame = frame.iloc[1:]
    else:
        frame.loc[0, 'split'] = 'unknown'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(dataset, verify_hashes=False)


def test_hash_verification(dataset):
    import json
    from poi.data import FILES, sha256
    report = {'outputs': {f: {'sha256': sha256(dataset / f)} for f in FILES}}
    (dataset / 'report.json').write_text(json.dumps(report))
    load_dataset(dataset)
    with (dataset / FILES[0]).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        load_dataset(dataset)


def test_load_clean_splits_never_materializes_heldout_features(dataset):
    clean_path = dataset / 'poi_data_region.csv'
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(dataset / 'sample_manifest.csv', dtype=str, keep_default_na=False)
    test_ids = set(manifest.loc[manifest.split.eq('test'), 'POI_ID'])
    clean.loc[clean.POI_ID.isin(test_ids), 'X_COORD'] = 'HELDOUT_FEATURE_SENTINEL'
    clean.to_csv(clean_path, index=False)

    development, development_manifest = load_clean_splits(
        dataset, splits=('train', 'validation'), verify_hashes=False)

    assert set(development.split) == {'train', 'validation'}
    assert not set(development.POI_ID) & test_ids
    assert 'HELDOUT_FEATURE_SENTINEL' not in set(development.X_COORD)
    assert development.POI_ID.tolist() == development_manifest.POI_ID.tolist()


def test_load_clean_splits_rejects_test_typo(dataset):
    with pytest.raises(ValueError, match='Unknown splits'):
        load_clean_splits(dataset, splits=('train', 'testing'), verify_hashes=False)


def test_generated_attack_loader_requires_and_filters_generation_seed(tmp_path):
    rows = []
    for seed in [42, 202, 340]:
        for poi_id, valid in [('p1', True), ('p2', False)]:
            rows.append({
                'POI_ID': poi_id, 'region': 'A', 'split': 'validation',
                'attack_condition': 'pgd_m10', 'generation_seed': seed,
                'constraints_valid': valid, 'primary_evaluation_eligible': valid,
                'attack_schema_version': 'poi-attack-generation-v2',
                'generation_identity_sha256': 'identity',
                'surrogate_checkpoint_sha256': 'checkpoint',
                'query_count_scope': 'search only',
            })
    path = tmp_path / 'attacks.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)

    selected = load_generated_attacks(path, 'validation', 202, valid_only=True)

    assert selected.POI_ID.tolist() == ['p1']
    assert selected.generation_seed.tolist() == [202]
    with pytest.raises(ValueError, match='generation_seed is required'):
        load_generated_attacks(path, 'validation', None)
    with pytest.raises(ValueError, match='train or validation'):
        load_generated_attacks(path, 'test', 202)
