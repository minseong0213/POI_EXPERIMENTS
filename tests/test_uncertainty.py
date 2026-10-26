import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from poi.uncertainty import (
    _all_clean_model_pairs,
    _aligned_arrays,
    _bh_adjust,
    make_bootstrap_weights,
    weighted_metric_replicates,
)


def test_bootstrap_weights_are_reproducible_and_resample_pois():
    first = make_bootstrap_weights(25, 50, 123)
    second = make_bootstrap_weights(25, 50, 123)
    assert first.dtype == np.uint16
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == 25)
    assert np.any(first != 1)


def test_weighted_metrics_match_expanded_bootstrap_sample_with_score_ties():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    y_score = np.array([0.1, 0.4, 0.4, 0.4, 0.7, 0.9])
    y_pred = (y_score >= 0.5).astype(np.int8)
    weights = np.array([[1, 2, 0, 3, 1, 2], [2, 0, 3, 1, 2, 1]], dtype=np.uint16)
    actual = weighted_metric_replicates(y_true, y_score, y_pred, weights)
    for repeat, row in enumerate(weights):
        indexes = np.repeat(np.arange(len(row)), row)
        expected = {
            "f1": f1_score(y_true[indexes], y_pred[indexes], zero_division=0),
            "precision": precision_score(y_true[indexes], y_pred[indexes], zero_division=0),
            "recall": recall_score(y_true[indexes], y_pred[indexes], zero_division=0),
            "accuracy": accuracy_score(y_true[indexes], y_pred[indexes]),
            "roc_auc": roc_auc_score(y_true[indexes], y_score[indexes]),
        }
        for metric, value in expected.items():
            assert actual[metric][repeat] == pytest.approx(value)


def test_shared_weights_preserve_exact_paired_zero_difference():
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_score = np.array([0.1, 0.3, 0.7, 0.9])
    y_pred = (y_score >= 0.5).astype(np.int8)
    weights = make_bootstrap_weights(4, 100, 8)
    left = weighted_metric_replicates(y_true, y_score, y_pred, weights)
    right = weighted_metric_replicates(y_true, y_score, y_pred, weights)
    for metric in left:
        assert np.array_equal(left[metric], right[metric], equal_nan=True)
        assert np.nanmax(np.abs(left[metric] - right[metric])) == 0


def test_alignment_rejects_a_stratum_without_common_pois():
    frame = pd.DataFrame({
        "POI_ID": ["a", "b"], "y_true": [0, 1],
        "y_score": [0.1, 0.9], "y_pred": [0, 1],
    })
    with pytest.raises(ValueError, match="common POI set"):
        _aligned_arrays(frame, np.array(["a", "c"]))


def test_bh_adjustment_is_monotone_in_ranked_p_values():
    values = np.array([0.04, 0.001, 0.02, 0.2])
    adjusted = _bh_adjust(values)
    ranked = adjusted[np.argsort(values)]
    assert np.all(np.diff(ranked) >= 0)
    assert np.all((adjusted >= values) & (adjusted <= 1))


def test_all_clean_model_pairs_exports_each_unordered_pair_and_metric():
    replicate_rows = []
    stratum_rows = []
    for model, offset in [("a", 0.0), ("b", 0.1), ("c", 0.2)]:
        stratum_rows.append({
            "analysis_family": "clean_model_comparison", "model": model,
            "condition": "clean", "seed": 42, "region": "r",
            **{metric: 0.5 + offset for metric in ["f1", "precision", "recall", "accuracy", "roc_auc"]},
        })
        for repeat in range(4):
            replicate_rows.append({
                "analysis_family": "clean_model_comparison", "model": model,
                "condition": "clean", "replicate": repeat,
                **{metric: 0.5 + offset + repeat / 100 for metric in
                   ["f1", "precision", "recall", "accuracy", "roc_auc"]},
            })
    result = _all_clean_model_pairs(
        pd.DataFrame(replicate_rows), pd.DataFrame(stratum_rows), 0.95
    )
    assert len(result) == 3 * 5
    assert set(zip(result.model_a, result.model_b)) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert (result.estimate_difference_a_minus_b < 0).all()
    assert not result.simultaneous_coverage.any()
