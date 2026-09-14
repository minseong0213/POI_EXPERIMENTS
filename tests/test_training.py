import json

import joblib
import pandas as pd
import pytest
import yaml

from poi.data import features, load_dataset, select_split
from poi.model import build_model
from poi.train import run


def test_end_to_end_outputs_and_no_overwrite(dataset, tmp_path, monkeypatch):
    cfg = {'name': 'test', 'seed': 42,
           'data': {'directory': str(dataset), 'verify_hashes': False},
           'model': {'name': 'random_forest', 'params': {'n_estimators': 5, 'n_jobs': 1}},
           'sampling': {'train_per_region': None, 'eval_per_region': None},
           'evaluation': {'splits': ['validation', 'test'], 'threshold': 0.5}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    for env, folder in [('CHECKPOINT_DIR', 'ckpt'), ('RESULT_DIR', 'results'), ('LOG_DIR', 'logs')]:
        monkeypatch.setenv(env, str(tmp_path / folder))
    monkeypatch.setenv('DATA_DIR', str(dataset))
    monkeypatch.setenv('EXP_ID', 'pytest')
    metrics = run(path)
    assert metrics['test_clean']['summary']['accuracy'] == 1.0
    assert metrics['test_clean']['summary']['regions'] == 2
    assert set(metrics) == {'validation_clean', 'validation_attack', 'test_clean', 'test_attack'}
    clean = pd.read_csv(tmp_path / 'results/test_clean_predictions.csv')
    attack = pd.read_csv(tmp_path / 'results/test_attack_predictions.csv')
    assert clean.POI_ID.tolist() == attack.POI_ID.tolist()
    assert json.loads((tmp_path / 'results/run.json').read_text())['train_rows'] == 16
    saved = joblib.load(tmp_path / 'ckpt/model.joblib')
    assert sorted(saved) == ['A', 'B']
    assert all(model.classes_.tolist() == [0, 1] for model in saved.values())
    before = (tmp_path / 'results/metrics.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(path)
    assert (tmp_path / 'results/metrics.json').read_bytes() == before


def test_preprocessor_only_learns_training_values(dataset):
    frames, _ = load_dataset(dataset, verify_hashes=False)
    train = select_split(frames['clean'], 'train')
    validation = select_split(frames['clean'], 'validation')
    validation['ASORT_LCLASDC'] = 'unseen-validation-category'
    model = build_model({'name': 'random_forest', 'params': {'n_estimators': 2, 'n_jobs': 1}}, 42)
    model.fit(features(train), train.region)
    encoder = model.named_steps['preprocess'].named_transformers_['categorical']
    assert 'unseen-validation-category' not in encoder.categories_[0]
    assert len(model.predict(features(validation))) == len(validation)
