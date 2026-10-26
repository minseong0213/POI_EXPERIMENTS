import numpy as np
import pandas as pd
import pytest

from poi.clean_comparison import validate_predictions


def sample():
    validation = pd.DataFrame({'POI_ID': ['p1', 'p2', 'p3', 'p4'],
                               'region': ['A', 'A', 'B', 'B']})
    rows = []
    for model in ['m1', 'm2']:
        for seed in [42, 202]:
            for region in ['A', 'B']:
                for poi, source in validation.itertuples(index=False, name=None):
                    truth = int(source == region)
                    score = .8 if truth else .2
                    rows.append(dict(POI_ID=poi, model=model, seed=seed, region=region,
                                     y_true=truth, y_score=score, y_pred=int(score >= .5),
                                     condition='clean', split='validation'))
    return pd.DataFrame(rows), validation


def test_complete_paired_prediction_grid_is_validated():
    frame, validation = sample()
    assert validate_predictions(frame, validation, ['m1', 'm2'], [42, 202]) == 8


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_poi', 'missing_model', 'wrong_truth',
                                     'wrong_threshold', 'bad_score', 'test_split'])
def test_corrupted_comparison_cannot_be_reported(mutation):
    frame, validation = sample()
    if mutation == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == 'missing_poi':
        frame = frame.iloc[1:].copy()
    elif mutation == 'missing_model':
        frame = frame.loc[frame.model.eq('m1')]
    elif mutation == 'wrong_truth':
        frame.loc[0, 'y_true'] = 1-frame.loc[0, 'y_true']
    elif mutation == 'wrong_threshold':
        frame.loc[0, 'y_pred'] = 1-frame.loc[0, 'y_pred']
    elif mutation == 'bad_score':
        frame.loc[0, 'y_score'] = np.inf
    elif mutation == 'test_split':
        frame.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError):
        validate_predictions(frame, validation, ['m1', 'm2'], [42, 202])
