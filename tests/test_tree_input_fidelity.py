import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from poi.data import CATEGORICAL, FEATURES
from poi.inference_benchmark import (
    BenchmarkContractError,
    _raw_frame_sha256,
    measure_prediction_repeats,
    summarize_prediction_reproduction,
)
from poi.selected_explain import (
    ModelFactory,
    _category_contract,
    _dataframe_sha256,
    _matrix,
    _tree_prediction_frame,
)


def frame_with_integer_string_categories():
    return pd.DataFrame({
        "X_COORD": ["126.1234567891", "126.1234567899", "127.1234567891", "127.1234567899",
                    "128.1234567891", "128.1234567899"],
        "Y_COORD": ["35.0000000001", "35.0000000009", "36.0000000001", "36.0000000009",
                    "37.0000000001", "37.0000000009"],
        "ASORT_LCLASDC": ["1", "1", "2", "2", "3", "3"],
        "ASORT_MLSFCDC": ["10", "11", "10", "11", "10", "11"],
        "ASORT_SDASDC": ["20", "21", "22", "20", "21", "22"],
    })


def test_raw_tree_frame_keeps_full_precision_and_exact_category_spelling():
    train = frame_with_integer_string_categories()
    contract = _category_contract(train, FEATURES)
    raw = _tree_prediction_frame(train, FEATURES, contract)
    assert raw.X_COORD.tolist() == train.X_COORD.tolist()
    assert raw.ASORT_LCLASDC.tolist() == train.ASORT_LCLASDC.tolist()
    assert raw.dtypes.astype(str).tolist() == ["object"] * 5

    matrix64 = _matrix(train, dtype=np.float64)
    reconstructed = _tree_prediction_frame(matrix64, FEATURES, contract)
    assert reconstructed.ASORT_LCLASDC.tolist() == train.ASORT_LCLASDC.tolist()
    assert reconstructed.ASORT_MLSFCDC.tolist() == train.ASORT_MLSFCDC.tolist()
    assert reconstructed.X_COORD.dtype == np.float64

    with pytest.raises(ValueError, match="float64"):
        _tree_prediction_frame(_matrix(train, dtype=np.float32), FEATURES, contract)


def test_numeric_category_mapping_fails_when_original_vocabulary_is_ambiguous():
    train = frame_with_integer_string_categories()
    train.loc[0, "ASORT_LCLASDC"] = "1.0"
    contract = _category_contract(train, FEATURES)
    numeric = _matrix(train, dtype=np.float64)
    with pytest.raises(ValueError, match="ambiguous numeric category"):
        _tree_prediction_frame(numeric, FEATURES, contract)

    # An exact raw spelling remains unambiguous even when another spelling has the
    # same numeric value.
    exact = _tree_prediction_frame(train, FEATURES, contract)
    assert exact.loc[0, "ASORT_LCLASDC"] == "1.0"
    assert exact.loc[1, "ASORT_LCLASDC"] == "1"


def test_raw_frame_digest_sees_precision_that_float32_erases():
    left = frame_with_integer_string_categories()
    right = left.copy()
    right.loc[0, "X_COORD"] = "126.1234567892"
    assert np.array_equal(_matrix(left), _matrix(right))
    assert _dataframe_sha256(left) != _dataframe_sha256(right)
    assert _raw_frame_sha256(left) == _dataframe_sha256(left)


def test_tree_factory_predicts_raw_dataframe_and_attests_actual_fit(tmp_path):
    train = frame_with_integer_string_categories()
    truth = np.array([0, 0, 0, 1, 1, 1])
    tree_config = tmp_path / "trees.yaml"
    tree_config.write_text(yaml.safe_dump({
        "models": {"decision_tree": {"max_depth": 3, "min_samples_leaf": 1}}
    }))
    factory = ModelFactory(
        "decision_tree", {"tree_config": str(tree_config), "seed": 42}, train
    )
    predict = factory.fit(FEATURES, truth)
    raw_score = predict(train)
    array_score = predict(_matrix(train, dtype=np.float64))
    np.testing.assert_array_equal(raw_score, array_score)

    attestation = factory.last_fit_attestation
    assert attestation["format"] == "poi-tree-fit-attestation-v1"
    assert attestation["raw_train_frame_sha256"] == _dataframe_sha256(train)
    assert attestation["truth_sha256"] == hashlib.sha256(truth.astype(np.int8).tobytes()).hexdigest()
    assert attestation["transformed_train"]["sha256"]
    assert attestation["transformer_fitted_state_sha256"]
    assert attestation["estimator_fitted_state_sha256"]
    assert attestation["probe_prediction_sha256"] == attestation["probe_repeat_prediction_sha256"]
    assert attestation["probe_predictions_exactly_equal"] is True
    assert attestation["category_vocabulary"]["ASORT_LCLASDC"] == ["1", "2", "3"]


def test_timing_discloses_machine_epsilon_repeat_variation():
    calls = 0
    base = np.array([.1, .2])
    clocks = iter(float(value) for pair in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] for value in pair)

    def predict(_):
        nonlocal calls
        calls += 1
        return base + (1e-16 if calls % 2 == 0 else 0)

    _, summary, repeats = measure_prediction_repeats(
        predict, pd.DataFrame({"x": [1, 2]}), synchronize=lambda: None,
        clock=lambda: next(clocks), repeat_absolute_tolerance=1e-12,
    )
    assert summary["all_repeats_exactly_equal"] is False
    assert 0 < summary["maximum_repeat_absolute_difference"] < 1e-12
    assert any(not row["exactly_equal_to_warmup"] for row in repeats)


def test_reproduction_summary_uses_recorded_label_mismatch_counts():
    comparison = pd.DataFrame({
        "scores_exactly_equal": [True, False],
        "prediction_label_mismatches": [0, 0],
    })
    assert summarize_prediction_reproduction(comparison) == {
        "all_scores_exact": False,
        "all_labels_same": True,
    }
    comparison.loc[1, "prediction_label_mismatches"] = 1
    assert summarize_prediction_reproduction(comparison)["all_labels_same"] is False
    with pytest.raises(BenchmarkContractError, match="incomplete"):
        summarize_prediction_reproduction(pd.DataFrame())
