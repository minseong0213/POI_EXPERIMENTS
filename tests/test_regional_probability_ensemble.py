import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from poi.regional_probability_ensemble import (
    ATTACK_MANIFEST_FORMAT,
    FROZEN_GENERATION_SEEDS,
    FROZEN_MODELS,
    FROZEN_SEEDS,
    RegionalEnsembleError,
    apply_frozen_ensembles,
    balanced_cohort_weights,
    fit_calibration_and_ensembles,
    fit_simplex_weights,
    fit_temperature,
    load_attack_shard,
    load_partition,
    metric_table,
    paired_bootstrap_table,
    preflight,
    rank_members_on_calibration,
    sha256,
    temperature_scale,
    validate_attack_manifest,
    validate_clean_predictions,
    validate_frozen_config,
    validate_partition_coverage,
    validate_train_oof_predictions,
)


MODELS = ["tabpfn_v2_5", "random_forest", "catboost"]
SEEDS = [42]
REGIONS = ["North", "South"]
POIS = ["n1", "n2", "s1", "s2"]


def clean_long() -> pd.DataFrame:
    rows = []
    source = {"n1": "North", "n2": "North", "s1": "South", "s2": "South"}
    # random_forest is perfect on calibration IDs n1/s1.  CatBoost is perfect only
    # on evaluation IDs n2/s2, making leakage into ranking observable.
    scores = {
        "random_forest": {"n1": .9, "s1": .1, "n2": .1, "s2": .9},
        "catboost": {"n1": .1, "s1": .9, "n2": .9, "s2": .1},
        "tabpfn_v2_5": {"n1": .4, "s1": .3, "n2": .7, "s2": .3},
    }
    for model in MODELS:
        for poi in POIS:
            north_score = scores[model][poi]
            for region in REGIONS:
                score = north_score if region == "North" else 1 - north_score
                rows.append({
                    "POI_ID": poi, "model": model, "seed": 42, "region": region,
                    "split": "validation", "condition": "clean",
                    "y_true": int(source[poi] == region), "y_score": score,
                    "y_pred": int(score >= .5),
                })
    return pd.DataFrame(rows)


def clean_wide() -> pd.DataFrame:
    return validate_clean_predictions(
        clean_long(), models=MODELS, seeds=SEEDS, regions=REGIONS, expected_pois=4,
    )


def add_cohort_columns(frame: pd.DataFrame, source: str, generation_seed: int, condition: str) -> pd.DataFrame:
    result = frame.copy()
    result["calibration_source"] = source
    result["generation_seed"] = generation_seed
    result["attack_condition"] = condition
    return result


def attack_wide() -> pd.DataFrame:
    result = clean_wide().copy()
    result["generation_seed"] = 42
    result["attack_condition"] = "pgd_m10"
    result["primary_evaluation_eligible"] = True
    result["quarantine_reason"] = ""
    result["attack_method"] = "pgd"
    result["numeric_budget_m"] = 10.0
    result["categorical_budget_l0"] = 0
    result["source_model"] = "surrogate_mlp"
    result["victim_model"] = "regional_classifier"
    result["constraints_valid"] = True
    result["label_preserved"] = True
    return result


def test_clean_validation_and_partition_are_exact(tmp_path):
    wide = clean_wide()
    assert len(wide) == 8
    assert set(wide.columns) >= {f"score__{model}" for model in MODELS}

    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "split": "validation", "overlap": 0,
        "calibration_ids": ["n1", "s1"], "evaluation_ids": ["n2", "s2"],
        "partition_seed": 340,
    }))
    calibration, evaluation, metadata = load_partition(partition)
    validate_partition_coverage(wide, calibration, evaluation, REGIONS)
    assert metadata["sha256"] == sha256(partition)

    payload = json.loads(partition.read_text())
    payload["evaluation_ids"] = ["n1", "s2"]
    partition.write_text(json.dumps(payload))
    with pytest.raises(RegionalEnsembleError, match="overlap"):
        load_partition(partition)


def test_clean_model_grid_and_truth_fail_closed():
    incomplete = clean_long().iloc[:-1]
    with pytest.raises(RegionalEnsembleError, match="cardinality|coverage"):
        validate_clean_predictions(
            incomplete, models=MODELS, seeds=SEEDS, regions=REGIONS, expected_pois=4,
        )
    changed = clean_long()
    changed.loc[0, "y_true"] = 1 - changed.loc[0, "y_true"]
    with pytest.raises(RegionalEnsembleError, match="truth"):
        validate_clean_predictions(
            changed, models=MODELS, seeds=SEEDS, regions=REGIONS, expected_pois=4,
        )


def test_ranking_reads_calibration_half_only():
    wide = clean_wide()
    calibration = wide.loc[wide.POI_ID.isin(["n1", "s1"])]
    ranking = rank_members_on_calibration(calibration, MODELS, threshold=.5, top_k=2)
    ordered = ranking.sort_values("calibration_rank").model.tolist()
    assert ordered[0] == "random_forest"
    assert ranking.fit_partition.eq("validation_calibration_half").all()

    # Scores on the disjoint evaluation IDs can change arbitrarily without changing
    # a fit that receives only calibration rows.
    poisoned_evaluation = wide.loc[wide.POI_ID.isin(["n2", "s2"])].copy()
    poisoned_evaluation[[f"score__{model}" for model in MODELS]] = .5
    repeated = rank_members_on_calibration(calibration, MODELS, threshold=.5, top_k=2)
    pd.testing.assert_frame_equal(ranking, repeated)


def test_temperature_simplex_and_balanced_cohorts():
    truth = np.array([0, 0, 1, 1])
    overconfident = np.array([.01, .8, .2, .99])
    temperature = fit_temperature(truth, overconfident)
    scaled = temperature_scale(overconfident, temperature)
    assert 0.05 <= temperature <= 20
    assert np.isfinite(scaled).all()

    member_scores = np.column_stack([np.array([.1, .2, .8, .9]), np.full(4, .5)])
    weights = fit_simplex_weights(truth, member_scores)
    assert np.isclose(weights.sum(), 1)
    assert weights[0] > weights[1]

    cohorts = pd.DataFrame({
        "calibration_source": ["clean", "attack", "attack", "attack"],
        "generation_seed": [-1, 42, 42, 42],
        "attack_condition": ["clean", "a", "a", "a"],
    })
    balanced = balanced_cohort_weights(cohorts)
    assert np.isclose(balanced[:1].sum(), balanced[1:].sum())

    with pytest.raises(RegionalEnsembleError):
        fit_temperature(truth, [np.nan, .2, .8, .9])
    with pytest.raises(RegionalEnsembleError):
        fit_simplex_weights(truth, np.ones((4, 1)))


def test_fit_then_apply_keeps_fixed_reference_separate():
    wide = clean_wide()
    clean_cal = add_cohort_columns(wide.loc[wide.POI_ID.isin(["n1", "s1"])], "clean", -1, "clean")
    attack_cal = add_cohort_columns(
        attack_wide().loc[lambda value: value.POI_ID.isin(["n1", "s1"])], "attack", 42, "pgd_m10"
    )
    ranking = rank_members_on_calibration(clean_cal, MODELS, threshold=.5, top_k=2)
    temperatures, weights = fit_calibration_and_ensembles(
        clean_cal, attack_cal, models=MODELS, regions=REGIONS,
        member_selection=ranking,
        calibration_variants=["uncalibrated", "validation_clean", "validation_clean_attack"],
        top_k=2,
    )
    assert set(temperatures.calibration_variant) == {
        "uncalibrated", "validation_clean", "validation_clean_attack"
    }
    assert temperatures.fit_partition.str.contains("evaluation").sum() == 0
    assert weights.fit_partition.eq("validation_calibration_half_clean_and_attack").all()

    evaluation = attack_wide().loc[lambda value: value.POI_ID.isin(["n2", "s2"])]
    prediction = apply_frozen_ensembles(
        evaluation, models=MODELS, member_selection=ranking, temperatures=temperatures,
        ensemble_weights=weights, calibration_variant="validation_clean_attack",
    )
    assert set(prediction.pipeline) >= {
        "fixed_tabpfn_v2_5_reference", "best_single_calibration",
        "all8_equal_soft_vote", "top_k_equal_soft_vote", "top_k_simplex_soft_vote",
    }
    reference = prediction.loc[prediction.pipeline.eq("fixed_tabpfn_v2_5_reference")]
    expected = evaluation.sort_values(["target_region", "POI_ID"])["score__tabpfn_v2_5"].to_numpy()
    assert np.array_equal(reference.y_score.to_numpy(), expected)
    assert prediction.y_score.between(0, 1).all()


def test_attack_manifest_hash_grid_and_shard_validity(tmp_path):
    shard = tmp_path / "attack.parquet"
    frame = attack_wide().rename(columns={"model_seed": "model_seed"})
    frame.to_parquet(shard, index=False)
    entry = {
        "path": shard.name, "sha256": sha256(shard), "model_seed": 42,
        "generation_seed": 42, "attack_condition": "pgd_m10",
        "rows": len(frame), "unique_pois": 4,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": ATTACK_MANIFEST_FORMAT, "models": MODELS, "model_seeds": [42],
        "generation_seeds": [42], "regions": REGIONS,
        "attack_conditions": ["pgd_m10"], "shards": [entry],
    }))
    entries, metadata = validate_attack_manifest(
        manifest, models=MODELS, seeds=[42], generation_seeds=[42], regions=REGIONS,
        attack_conditions=["pgd_m10"],
    )
    loaded = load_attack_shard(
        entries[0], models=MODELS, regions=REGIONS, validation_ids=set(POIS), expected_pois=4,
        expected_source_region={"n1": "North", "n2": "North", "s1": "South", "s2": "South"},
    )
    assert len(loaded) == 8
    assert metadata["shard_count"] == 1

    payload = json.loads(manifest.read_text())
    payload["shards"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RegionalEnsembleError, match="hash mismatch"):
        validate_attack_manifest(
            manifest, models=MODELS, seeds=[42], generation_seeds=[42], regions=REGIONS,
            attack_conditions=["pgd_m10"],
        )


def test_attack_quarantine_and_truth_are_fail_closed(tmp_path):
    entry = {
        "model_seed": 42, "generation_seed": 42, "attack_condition": "pgd_m10",
        "rows": 8, "unique_pois": 4,
    }
    invalid = attack_wide()
    invalid.loc[invalid.POI_ID.eq("n1"), "primary_evaluation_eligible"] = False
    invalid.loc[invalid.POI_ID.eq("n1"), "quarantine_reason"] = ""
    path = tmp_path / "invalid.parquet"
    invalid.to_parquet(path, index=False)
    entry["path"] = str(path)
    with pytest.raises(RegionalEnsembleError, match="quarantine_reason"):
        load_attack_shard(entry, models=MODELS, regions=REGIONS, validation_ids=set(POIS), expected_pois=4)

    invalid = attack_wide()
    invalid.loc[invalid.POI_ID.eq("n1"), "source_region"] = "South"
    invalid.to_parquet(path, index=False)
    with pytest.raises(RegionalEnsembleError, match="truth disagrees"):
        load_attack_shard(entry, models=MODELS, regions=REGIONS, validation_ids=set(POIS), expected_pois=4)

    invalid = attack_wide()
    invalid.loc[invalid.POI_ID.eq("n1"), "constraints_valid"] = False
    invalid.to_parquet(path, index=False)
    with pytest.raises(RegionalEnsembleError, match="cannot be primary"):
        load_attack_shard(entry, models=MODELS, regions=REGIONS, validation_ids=set(POIS), expected_pois=4)


def test_train_oof_alternative_requires_fold_scope_and_validation_disjointness():
    frame = clean_wide().rename(columns={"model_seed": "model_seed"})
    frame["POI_ID"] = frame.POI_ID.map(lambda value: f"train_{value}")
    frame["split"] = "train"
    frame["prediction_scope"] = "train_oof"
    frame["fold_id"] = frame.POI_ID.map({
        "train_n1": 0, "train_n2": 1, "train_s1": 0, "train_s2": 1,
    })
    checked = validate_train_oof_predictions(
        frame, models=MODELS, seeds=SEEDS, regions=REGIONS, validation_ids=set(POIS),
    )
    assert checked.prediction_scope.eq("train_oof").all()
    frame.loc[0, "prediction_scope"] = "train_in_sample"
    with pytest.raises(RegionalEnsembleError, match="train/train_oof"):
        validate_train_oof_predictions(
            frame, models=MODELS, seeds=SEEDS, regions=REGIONS, validation_ids=set(POIS),
        )


def test_five_metrics_scopes_and_paired_poi_bootstrap():
    rows = []
    truth = {"a": 0, "b": 0, "c": 1, "d": 1}
    scores = {
        "best_single_calibration": [.2, .6, .7, .8],
        "top_k_equal_soft_vote": [.1, .2, .8, .9],
    }
    for pipeline, values in scores.items():
        for poi, score in zip(truth, values):
            rows.append({
                "POI_ID": poi, "pipeline": pipeline, "calibration_variant": "validation_clean",
                "model_seed": 42, "generation_seed": 42, "attack_condition": "pgd_m10",
                "target_region": "North", "truth": truth[poi], "y_score": score,
                "primary_evaluation_eligible": poi != "b",
            })
    frame = pd.DataFrame(rows)
    metrics = metric_table(frame, .5)
    assert set(metrics.metric) == {"f1", "precision", "recall", "accuracy", "roc_auc"}
    assert set(metrics.evaluation_scope) == {"all_intent_to_attack", "primary_valid"}
    bootstrap = paired_bootstrap_table(
        frame, evaluation_ids=list(truth), threshold=.5, repeats=40, seed=9,
    )
    assert set(bootstrap.metric) == {"f1", "precision", "recall", "accuracy", "roc_auc"}
    assert bootstrap.pipeline.eq("top_k_equal_soft_vote").all()
    assert bootstrap.bootstrap_repeats.eq(40).all()


def frozen_config(tmp_path: Path) -> dict:
    return {
        "format": "poi-regional-probability-ensemble-v1",
        "clean_predictions": str(tmp_path / "missing-clean.parquet"),
        "validation_partition": str(tmp_path / "missing-partition.json"),
        "attack_prediction_manifest": str(tmp_path / "missing-attacks.json"),
        "output": str(tmp_path / "must-not-exist"),
        "models": list(FROZEN_MODELS), "model_seeds": list(FROZEN_SEEDS),
        "generation_seeds": list(FROZEN_GENERATION_SEEDS),
        "regions": [f"region_{index}" for index in range(17)],
        "attack_conditions": [f"condition_{index}" for index in range(32)],
        "expected_validation_pois": 3400, "expected_regions": 17,
        "selection_partition": "validation_calibration_half",
        "scoring_partition": "validation_evaluation_half",
        "classification_threshold": .5, "top_k": 3,
        "calibration_variants": ["uncalibrated", "validation_clean", "validation_clean_attack"],
        "bootstrap_repeats": 2000, "bootstrap_seed": 20261001,
    }


def test_preflight_missing_inputs_is_not_run_and_writes_nothing(tmp_path):
    config = frozen_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = preflight(config_path)
    assert result["status"] == "NOT_RUN"
    assert result["test_rows_read"] == 0
    assert set(result["missing_inputs"]) == {
        "clean_predictions", "validation_partition", "attack_prediction_manifest"
    }
    assert not Path(config["output"]).exists()


def test_frozen_config_rejects_weak_or_leaky_design(tmp_path):
    config = frozen_config(tmp_path)
    config["bootstrap_repeats"] = 1999
    with pytest.raises(RegionalEnsembleError, match="2000"):
        validate_frozen_config(config)
    config = frozen_config(tmp_path)
    config["scoring_partition"] = "whole_validation"
    with pytest.raises(RegionalEnsembleError, match="disjoint evaluation"):
        validate_frozen_config(config)
