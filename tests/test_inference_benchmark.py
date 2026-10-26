import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from poi.inference_benchmark import (
    MODELS,
    BenchmarkContractError,
    aggregate_timing_rows,
    compare_fit_predictions,
    measure_prediction_repeats,
    select_by_protocol_rule,
    validate_checkpoint_contracts,
    validate_config,
    validate_tree_thread_contract,
)


def test_timing_excludes_warmup_uses_five_repeats_and_synchronizes():
    calls = {"predict": 0, "sync": 0}
    clock_values = iter([0., 1., 10., 12., 20., 23., 30., 34., 40., 45.])

    def predict(matrix):
        calls["predict"] += 1
        return np.linspace(.1, .9, len(matrix))

    def synchronize():
        calls["sync"] += 1

    prediction, summary, repeats = measure_prediction_repeats(
        predict, np.ones((4, 5)), synchronize=synchronize,
        clock=lambda: next(clock_values),
    )

    assert calls == {"predict": 6, "sync": 12}  # one untimed warmup plus five measurements
    assert prediction.tolist() == pytest.approx([.1, .3666666667, .6333333333, .9])
    assert [row["validation_batch_seconds"] for row in repeats] == [1., 2., 3., 4., 5.]
    assert summary["median_validation_batch_seconds"] == 3.
    assert summary["mean_validation_batch_seconds"] == 3.
    assert summary["timing_cv"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1) / 3)
    assert summary["warmup_excluded_from_primary_time"] is True


def test_timing_rejects_prediction_nondeterminism():
    calls = 0

    def changing_predict(matrix):
        nonlocal calls
        calls += 1
        return np.full(len(matrix), calls / 100)

    with pytest.raises(BenchmarkContractError, match="differs from the excluded warmup"):
        measure_prediction_repeats(
            changing_predict, np.ones((3, 5)), synchronize=lambda: None,
            clock=iter([0., 1.]).__next__,
        )


def test_prediction_comparison_aligns_ids_and_never_calls_different_scores_exact():
    reference = pd.DataFrame({
        "POI_ID": ["a", "b", "c"], "y_true": [1, 0, 1],
        "y_score": [.8, .2, .7], "y_pred": [1, 0, 1],
    })
    shuffled = reference.iloc[[2, 0, 1]].reset_index(drop=True)
    exact = compare_fit_predictions(shuffled, reference, absolute_tolerance=1e-6)
    assert exact["reproduction_status"] == "EXACT_SAME_SCORES_AND_LABELS"
    assert exact["scores_exactly_equal"] is True

    changed = shuffled.copy()
    changed.loc[changed.POI_ID.eq("a"), "y_score"] += 1e-7
    tolerant = compare_fit_predictions(changed, reference, absolute_tolerance=1e-6)
    assert tolerant["scores_exactly_equal"] is False
    assert tolerant["scores_within_configured_tolerance"] is True
    assert tolerant["reproduction_status"] == "DIFFERENT_SCORES_WITHIN_TOLERANCE_SAME_LABELS"
    assert tolerant["actual_score_sha256"] != tolerant["reference_score_sha256"]
    outside = changed.copy()
    outside.loc[outside.POI_ID.eq("a"), "y_score"] += .1
    different = compare_fit_predictions(outside, reference, absolute_tolerance=1e-6)
    assert different["reproduction_status"] == "DIFFERENT_OUTSIDE_TOLERANCE"
    assert different["scores_within_configured_tolerance"] is False


def test_prediction_comparison_rejects_truth_or_identity_misalignment():
    reference = pd.DataFrame({
        "POI_ID": ["a", "b"], "y_true": [1, 0], "y_score": [.8, .2], "y_pred": [1, 0],
    })
    wrong_truth = reference.copy()
    wrong_truth.loc[0, "y_true"] = 0
    with pytest.raises(BenchmarkContractError, match="truth differs"):
        compare_fit_predictions(wrong_truth, reference, absolute_tolerance=1e-9)
    wrong_ids = reference.assign(POI_ID=["a", "c"])
    with pytest.raises(BenchmarkContractError, match="POI sets differ"):
        compare_fit_predictions(wrong_ids, reference, absolute_tolerance=1e-9)


def test_protocol_selector_includes_exact_0p005_boundary_and_uses_shorter_time():
    scores = {
        "decision_tree": .900,
        "random_forest": .895,  # inclusive boundary candidate
        "xgboost": .894999,     # just outside
        "lightgbm": .88,
        "catboost": .87,
        "tabpfn_v2_5": .86,
        "tabpfn_v2_6": .85,
        "tabpfn_v3": .84,
    }
    times = {model: 10. + index for index, model in enumerate(MODELS)}
    times["decision_tree"] = 2.
    times["random_forest"] = 1.
    performance = pd.DataFrame({"model": list(scores), "macro_f1": list(scores.values())})
    timing = pd.DataFrame({
        "model": list(times), "primary_inference_seconds": list(times.values())
    })

    table, result = select_by_protocol_rule(performance, timing)

    assert result["candidate_models"] == ["random_forest", "decision_tree"]
    assert result["selected_model"] == "random_forest"
    assert table.loc[table.model.eq("random_forest"), "selected_by_protocol_rule"].item()
    assert not table.loc[
        table.model.eq("xgboost"), "within_inclusive_0p005_candidate_band"
    ].item()
    assert result["does_not_modify_explicit_common_downstream_choice"] is True


def test_timing_aggregation_equal_weights_fit_medians():
    rows = pd.DataFrame([
        {"model": model, "seed": seed, "region": "A",
         "median_validation_batch_seconds": seconds, "fit_seconds": seconds * 10,
         "timing_cv": cv, "validation_rows": 3400}
        for model, seed, seconds, cv in [
            ("a", 42, 1., .1), ("a", 202, 3., .3),
            ("b", 42, 2., .2), ("b", 202, 6., .4),
        ]
    ])

    result = aggregate_timing_rows(rows, models=["a", "b"], expected_fits_per_model=2)

    assert result.set_index("model").loc["a", "primary_inference_seconds"] == 2.
    assert result.set_index("model").loc["b", "primary_inference_seconds"] == 4.
    assert result.set_index("model").loc["a", "mean_per_fit_timing_cv"] == .2
    assert result.fit_time_excluded_from_selection.all()


def test_checkpoint_contract_rehashes_all_three_files_and_rejects_bad_digest(tmp_path):
    contracts = {}
    for model in ("tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"):
        path = tmp_path / f"{model}.ckpt"
        path.write_bytes(model.encode())
        contracts[model] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    checked = validate_checkpoint_contracts(contracts)
    assert set(checked) == set(contracts)
    contracts["tabpfn_v3"]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkContractError, match="mismatch"):
        validate_checkpoint_contracts(contracts)
    contracts["tabpfn_v3"] = {
        "path": str((tmp_path / "missing.ckpt").resolve()),
        "sha256": "0" * 64,
    }
    with pytest.raises(BenchmarkContractError, match="Missing/invalid"):
        validate_checkpoint_contracts(contracts)


def test_tree_thread_contract_requires_two_threads_for_parallel_estimators(tmp_path):
    models = {
        "decision_tree": {},
        "random_forest": {"n_jobs": 2},
        "xgboost": {"n_jobs": 2},
        "lightgbm": {"n_jobs": 2},
        "catboost": {"thread_count": 2},
    }
    path = tmp_path / "trees.yaml"
    path.write_text(yaml.safe_dump({"models": models}))
    assert validate_tree_thread_contract(path, 2)["sha256"]
    models["xgboost"]["n_jobs"] = 4
    path.write_text(yaml.safe_dump({"models": models}))
    with pytest.raises(BenchmarkContractError, match="xgboost.n_jobs"):
        validate_tree_thread_contract(path, 2)


def test_repository_config_encodes_frozen_design_without_running_models():
    path = Path("configs/stages/06_inference_benchmark.yaml")
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert tuple(cfg["models"]) == MODELS
    assert cfg["timing"] == {
        "warmup_repeats": 1, "measured_repeats": 5, "rows_per_repeat": 3400,
    }
    assert cfg["selection_rule"]["absolute_f1_tolerance"] == .005
    assert cfg["existing_common_downstream"] == "tabpfn_v2_5"
