import numpy as np
import pytest

from poi.clean_uncertainty import validate_clean
from poi.uncertainty import METRICS, make_bootstrap_weights, _analyze_groups, _all_clean_model_pairs
from test_clean_comparison import sample


def config():
    return dict(metrics=list(METRICS), bootstrap_repeats=2000, confidence_level=.95,
                expected_pois=4, expected_regions=2, models=['m1', 'm2'],
                expected_seeds=[42, 202], threshold=.5)


def test_shuffle_retains_poi_pairing_and_identical_models_have_zero_delta():
    frame, validation = sample()
    frame = frame.sample(frac=1, random_state=12).reset_index(drop=True)
    order = validate_clean(frame, validation, config())
    weights = make_bootstrap_weights(4, 20, 123)
    strata, replicates = _analyze_groups(frame, frame.iloc[:0], order, weights, 'clean')
    pairs = _all_clean_model_pairs(replicates, strata, .95)
    assert len(pairs) == 5
    np.testing.assert_allclose(pairs.estimate_difference_a_minus_b, 0)
    defined = pairs.metric.ne('roc_auc')
    np.testing.assert_allclose(pairs.loc[defined, 'ci_low'], 0)
    np.testing.assert_allclose(pairs.loc[defined, 'ci_high'], 0)
    # Tiny ordinary bootstrap samples can lack either class: AUC remains undefined.
    assert pairs.loc[~defined, 'ci_low'].isna().all()
    assert set(strata.analysis_family) == {'clean_model_comparison'}


def test_legacy_attack_dependency_is_rejected():
    frame, validation = sample()
    cfg = dict(config(), attack_predictions='legacy.parquet')
    with pytest.raises(ValueError, match='does not accept attack'):
        validate_clean(frame, validation, cfg)


def test_missing_whole_model_seed_region_is_rejected():
    frame, validation = sample()
    frame = frame.loc[~(frame.model.eq('m1') & frame.seed.eq(42) & frame.region.eq('A'))]
    with pytest.raises(ValueError, match='Cartesian'):
        validate_clean(frame, validation, config())
