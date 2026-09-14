import pandas as pd
import pytest

from poi.data import FEATURES, features, load_dataset, select_split


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
