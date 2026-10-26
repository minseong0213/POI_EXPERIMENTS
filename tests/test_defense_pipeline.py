import json
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poi.data import FEATURES
from poi.defense_pipeline import (
    DETECTOR_LABELS,
    PROPOSED_LABEL,
    URE_LABEL,
    RegionModelFactory,
    _cohort_metric_rows,
    cross_design_expected_counts,
    _classification_metric_rows,
    _detector_metric_rows,
    _prediction_rows,
    _plot_curves,
    _plot_risk_coverage,
    binary_ovr_risk_coverage,
    load_calibration_partition,
    macro_region_metrics,
    selective_operating_summary,
    split_validation_ids,
    validate_attack_rows,
)


def test_macro_metrics_average_regions_and_disclose_undefined_auc():
    common = {
        "seed": 42,
        "generation_seed": 202,
        "attack_condition": "pgd",
        "evaluation_cohort": "attack",
        "population_scope": "primary_valid",
        "detector": PROPOSED_LABEL,
        "clean_retention_target": .95,
        "detection_threshold": .4,
        "classification_mode": "fixed_unmodified_selective_accepted_only",
        "coverage": .5,
        "accepted": 2,
        "rejected": 2,
        "total": 4,
    }
    regional = pd.DataFrame([
        {**common, "target_region": "A", "f1": 1., "precision": 1., "recall": 1.,
         "accuracy": .75, "roc_auc": .8},
        {**common, "target_region": "B", "f1": 0., "precision": 0., "recall": 0.,
         "accuracy": .25, "roc_auc": np.nan},
    ])

    result = macro_region_metrics(regional, expected_regions=2).iloc[0]

    assert result.macro_f1 == .5
    assert result.macro_accuracy == .5
    assert result.macro_roc_auc == .8
    assert result.roc_auc_finite_regions == 1
    assert result.roc_auc_undefined_regions == 1
    assert result.denominator_pois == 4


def test_macro_metrics_reject_missing_or_disagreeing_region_rows():
    regional = pd.DataFrame([{
        "seed": 42, "generation_seed": 202, "attack_condition": "pgd",
        "evaluation_cohort": "attack", "population_scope": "primary_valid",
        "detector": PROPOSED_LABEL, "clean_retention_target": .95,
        "detection_threshold": .4, "classification_mode": "fixed",
        "target_region": "A", "coverage": .5, "accepted": 1, "rejected": 1,
        "total": 2, "f1": 1., "precision": 1., "recall": 1.,
        "accuracy": 1., "roc_auc": np.nan,
    }])
    with pytest.raises(ValueError, match="exactly one row"):
        macro_region_metrics(regional, expected_regions=2)


def test_selective_operating_summary_uses_strict_attack_rejection_and_denominators():
    clean = {label: np.array([.2, .5]) for label in DETECTOR_LABELS}
    attack = {label: np.array([.5, .6, .9]) for label in DETECTOR_LABELS}
    thresholds = {label: {.95: .5} for label in DETECTOR_LABELS}

    result = selective_operating_summary(
        clean, attack, np.array([True, False, True]), thresholds,
        seed=42, generation_seed=202, attack_condition="pgd",
    )
    primary = result.loc[
        result.detector.eq(PROPOSED_LABEL)
        & result.population_scope.eq("primary_valid")
    ].iloc[0]

    assert primary.clean_retention == 1.
    assert primary.attack_total_pois == 2
    assert primary.attack_accepted_pois == 1  # score == threshold is accepted
    assert primary.attack_rejected_pois == 1
    assert primary.attack_rejection == .5
    assert primary.combined_total_pois == 4
    assert primary.combined_coverage == .75


def test_risk_coverage_is_tie_aware_and_uses_binary_ovr_risk():
    truth = np.array([[1, 0], [1, 0], [0, 1]])
    scores = np.array([[.9, .1], [.1, .1], [.1, .9]])  # POI risks 0, .5, 0
    detector = np.array([.2, .2, .9])

    curve, summary = binary_ovr_risk_coverage(
        truth, scores, detector, ["a", "b", "c"], {.90: .2, .95: .1},
        classification_threshold=.5, seed=42, detector=PROPOSED_LABEL,
    )

    assert curve.accepted_pois.tolist() == [0, 2, 3]
    assert np.isnan(curve.iloc[0].selective_risk)
    assert curve.iloc[1].selective_risk == .25
    assert curve.iloc[1].calibrated_clean_retention_targets == "0.9"
    assert summary["curve_points_exact"] == 2
    assert summary["aurc"] == pytest.approx((2 / 3) * .25 + (1 / 3) * (1 / 6))


def test_risk_coverage_empty_population_emits_explicit_nan_row():
    curve, summary = binary_ovr_risk_coverage(
        np.empty((0, 2), dtype=int), np.empty((0, 2)), np.empty(0), [], {.95: .5},
        classification_threshold=.5, population_scope="primary_valid",
    )

    assert len(curve) == 1 and bool(curve.iloc[0].empty_population)
    assert curve.iloc[0].total_pois == 0
    assert np.isnan(curve.iloc[0].coverage)
    assert np.isnan(summary["aurc"])


def test_risk_coverage_plot_survives_no_primary_valid_attack_curve(tmp_path):
    curves = pd.DataFrame([{
        "evaluation_cohort": "attack", "population_scope": "primary_valid",
        "coverage": np.nan, "selective_risk": np.nan, "detector": PROPOSED_LABEL,
    }])

    _plot_risk_coverage(curves, tmp_path)

    assert (tmp_path / "risk_coverage.png").is_file()


def test_full_cross_design_has_exact_canonical_clean_and_metric_counts():
    counts = cross_design_expected_counts(
        model_seeds=3, generation_seeds=3, attack_conditions=32,
        regions=17, detectors=2, retentions=4,
    )
    assert counts == {
        "model_generation_seed_pairs": 9,
        "clean_prediction_shards": 3,
        "attack_prediction_shards": 288,
        "prediction_shards": 291,
        "threshold_rows": 24,
        "detector_metric_rows": 4608,
        "classification_metric_rows": 570894,
        "region_model_fits": 51,
    }


def clean_frame(rows_per_region=4):
    rows = []
    for region_index, region in enumerate(("A", "B")):
        for row in range(rows_per_region):
            rows.append(
                {
                    "POI_ID": f"{region}-{row}",
                    "region": region,
                    "split": "validation",
                    "X_COORD": 100 + region_index * 10 + row,
                    "Y_COORD": 200 + region_index * 10 + row,
                    "ASORT_LCLASDC": region_index + 1,
                    "ASORT_MLSFCDC": region_index + 10,
                    "ASORT_SDASDC": region_index + 100,
                }
            )
    return pd.DataFrame(rows)


def attack_frame(clean, conditions=("fgsm_m50", "pcaa_l1")):
    rows = []
    for condition in conditions:
        for index, source in clean.reset_index(drop=True).iterrows():
            row = {
                "POI_ID": source.POI_ID,
                "region": source.region,
                "split": "validation",
                "generation_seed": 42,
                "attack_condition": condition,
                "attack_method": condition.split("_")[0],
                "numeric_budget_m": 50.0 if "m50" in condition else 0.0,
                "numeric_budget_was_implicit_zero": "m50" not in condition,
                "categorical_budget_l0": 1 if "l1" in condition else 0,
                "distance_metric": "local_equirectangular_wgs84_approximation",
                "distance_tolerance_m": 0.05 if "m50" in condition else 1e-6,
                "source_model": "surrogate_mlp",
                "victim_model": "surrogate_mlp",
                "victim_evaluation_scope": "direct_white_box_source_model",
                "transfer_evaluated": False,
                "targeted": False,
                "constraints_valid": index != 0,
                "label_preserved": index != 0,
                "primary_evaluation_eligible": index != 0,
                "quarantine_reason": "polygon_disagreement" if index == 0 else "",
                "attack_attempted": True,
                "attack_changed_any": "m50" in condition,
                "valid_numeric_train_range": True,
                "valid_category_codebook": True,
                "valid_clean_category_combo": True,
                "valid_attack_category_combo": True,
                "valid_category_combo": True,
                "within_budget": True,
                "clean_boundary_status": "unclassified" if index == 0 else "match",
                "attack_boundary_status": "match",
            }
            for feature in FEATURES:
                row[f"{feature}_clean"] = source[feature]
                row[f"{feature}_adv"] = source[feature] + (
                    0.0001 if feature == "X_COORD" and "m50" in condition else 0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_calibration_partition_requires_identical_detector_files(tmp_path):
    validation = clean_frame()
    calibration, evaluation = split_validation_ids(
        validation, calibration_fraction=0.5, seed=340
    )
    payload = {
        "split": "validation",
        "calibration_ids": calibration,
        "evaluation_ids": evaluation,
        "overlap": 0,
    }
    proposed, ure = tmp_path / "proposed.json", tmp_path / "ure.json"
    proposed.write_text(json.dumps(payload))
    ure.write_text(json.dumps(payload))

    observed_calibration, observed_evaluation, metadata = load_calibration_partition(
        validation, proposed, ure, fraction=0.5, seed=340
    )
    assert observed_calibration == calibration
    assert observed_evaluation == evaluation
    assert metadata["source"] == "matching_stage08_stage09_calibration_files"
    assert not set(observed_calibration) & set(observed_evaluation)

    changed = dict(payload)
    changed["calibration_ids"], changed["evaluation_ids"] = evaluation, calibration
    ure.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differ"):
        load_calibration_partition(validation, proposed, ure)


def test_calibration_partition_fallback_is_deterministic_and_fail_closed(tmp_path):
    validation = clean_frame()
    missing_a, missing_b = tmp_path / "a.json", tmp_path / "b.json"
    first = load_calibration_partition(validation, missing_a, missing_b, seed=340)
    second = load_calibration_partition(validation, missing_a, missing_b, seed=340)
    assert first[:2] == second[:2]
    assert first[2]["source"] == "deterministic_validation_clean_fallback"
    missing_a.write_text("{}")
    with pytest.raises(FileNotFoundError, match="together"):
        load_calibration_partition(validation, missing_a, missing_b)


def test_region_factory_fits_once_per_region_and_reuses_predictions():
    train = clean_frame().assign(split="train")

    class FakeFactory:
        instances = 0
        calls = 0

        def __init__(self, model_id, config, clean_train):
            assert model_id == "tabpfn_v2_5"
            assert config["seed"] == 42
            assert clean_train is not None
            FakeFactory.instances += 1

        def fit(self, features, truth):
            assert features == FEATURES
            FakeFactory.calls += 1
            value = float(np.mean(truth))
            return lambda matrix: np.full(len(matrix), value)

    factory = RegionModelFactory(
        train,
        ("A", "B"),
        seed=42,
        config={"model_id": "tabpfn_v2_5"},
        factory_class=FakeFactory,
    ).fit()
    first = factory.predict(np.ones((3, 5), dtype=np.float32))
    second = factory.predict(np.zeros((2, 5), dtype=np.float32))
    assert first.shape == (3, 2) and second.shape == (2, 2)
    assert factory.fit_count == 2
    assert FakeFactory.instances == 1 and FakeFactory.calls == 2
    with pytest.raises(RuntimeError, match="only be fitted once"):
        factory.fit()


def test_attack_validation_preserves_quarantine_and_rejects_anchor_changes():
    clean = clean_frame(rows_per_region=2)
    attacks = attack_frame(clean)
    train = clean.copy().assign(split="train")
    train["X_COORD"] = train["X_COORD"].astype(float)
    train.loc[train["X_COORD"].idxmax(), "X_COORD"] += 0.001
    validated, conditions = validate_attack_rows(
        attacks, clean, expected_conditions=2, clean_train=train
    )
    assert conditions == ("fgsm_m50", "pcaa_l1")
    assert len(validated) == len(attacks)
    assert (~validated["primary_evaluation_eligible"].astype(bool)).sum() == 2

    changed = attacks.copy()
    changed["X_COORD_clean"] = changed["X_COORD_clean"].astype(float)
    changed.loc[0, "X_COORD_clean"] += 0.25
    with pytest.raises(ValueError, match="clean anchor"):
        validate_attack_rows(changed, clean, expected_conditions=2, clean_train=train)

    dropped = attacks.drop(attacks.index[-1])
    with pytest.raises(ValueError, match="paired to every validation POI"):
        validate_attack_rows(dropped, clean, expected_conditions=2, clean_train=train)


def test_attack_validation_requires_clean_and_attack_tuples_seen_in_train():
    clean = pd.DataFrame(
        [
            {
                "POI_ID": "A-0",
                "region": "A",
                "split": "validation",
                "X_COORD": 100.0,
                "Y_COORD": 200.0,
                "ASORT_LCLASDC": 1.0,
                "ASORT_MLSFCDC": 11.0,
                "ASORT_SDASDC": 100.0,
            }
        ]
    )
    train = pd.DataFrame(
        [
            {
                "POI_ID": "T-0", "region": "A", "split": "train",
                "X_COORD": 99.0, "Y_COORD": 199.0,
                "ASORT_LCLASDC": 1.0, "ASORT_MLSFCDC": 10.0,
                "ASORT_SDASDC": 100.0,
            },
            {
                "POI_ID": "T-1", "region": "B", "split": "train",
                "X_COORD": 101.0, "Y_COORD": 201.0,
                "ASORT_LCLASDC": 2.0, "ASORT_MLSFCDC": 11.0,
                "ASORT_SDASDC": 101.0,
            },
        ]
    )
    attacks = attack_frame(clean, conditions=("pcaa_l1",))
    attacks.loc[:, "clean_boundary_status"] = "match"
    attacks.loc[:, "attack_boundary_status"] = "match"
    attacks.loc[:, "label_preserved"] = True
    attacks.loc[:, "valid_clean_category_combo"] = False
    attacks.loc[:, "valid_attack_category_combo"] = False
    attacks.loc[:, "valid_category_combo"] = False
    attacks.loc[:, "constraints_valid"] = False
    attacks.loc[:, "primary_evaluation_eligible"] = False
    attacks.loc[:, "quarantine_reason"] = "clean_category_tuple_not_observed_in_train"
    validated, _ = validate_attack_rows(
        attacks, clean, expected_conditions=1, clean_train=train
    )
    assert not validated["valid_clean_category_combo"].iloc[0]
    assert not validated["valid_attack_category_combo"].iloc[0]

    bypass = attacks.copy()
    bypass.loc[:, "valid_clean_category_combo"] = True
    bypass.loc[:, "valid_attack_category_combo"] = True
    bypass.loc[:, "valid_category_combo"] = True
    bypass.loc[:, "constraints_valid"] = True
    bypass.loc[:, "primary_evaluation_eligible"] = True
    bypass.loc[:, "quarantine_reason"] = ""
    with pytest.raises(ValueError, match="valid_clean_category_combo"):
        validate_attack_rows(bypass, clean, expected_conditions=1, clean_train=train)


def test_attack_validation_uses_implicit_zero_and_exact_distance_tolerances():
    clean = clean_frame(rows_per_region=2)
    train = clean.copy().assign(split="train")
    train["X_COORD"] = train["X_COORD"].astype(float)
    train.loc[train["X_COORD"].idxmax(), "X_COORD"] += 0.001
    attacks = attack_frame(clean)

    categorical_row = attacks.index[
        attacks["attack_condition"].eq("pcaa_l1") & attacks["POI_ID"].eq("A-1")
    ][0]
    attacks.loc[categorical_row, "X_COORD_adv"] += 1e-9
    attacks.loc[categorical_row, "attack_changed_any"] = True
    attacks.loc[categorical_row, "within_budget"] = False
    attacks.loc[categorical_row, "constraints_valid"] = False
    attacks.loc[categorical_row, "primary_evaluation_eligible"] = False
    attacks.loc[categorical_row, "quarantine_reason"] = "outside_attack_budget"

    numeric_row = attacks.index[
        attacks["attack_condition"].eq("fgsm_m50") & attacks["POI_ID"].eq("A-1")
    ][0]
    latitude = np.deg2rad(float(attacks.loc[numeric_row, "Y_COORD_clean"]))
    attacks.loc[numeric_row, "X_COORD_adv"] = (
        float(attacks.loc[numeric_row, "X_COORD_clean"])
        + 50.04 / (111_320.0 * np.cos(latitude))
    )

    validated, _ = validate_attack_rows(
        attacks, clean, expected_conditions=2, clean_train=train
    )
    observed_categorical = validated.loc[
        validated["POI_ID"].eq("A-1")
        & validated["attack_condition"].eq("pcaa_l1")
    ].iloc[0]
    observed_numeric = validated.loc[
        validated["POI_ID"].eq("A-1")
        & validated["attack_condition"].eq("fgsm_m50")
    ].iloc[0]
    assert not observed_categorical.within_budget
    assert observed_numeric.within_budget

    nonfinite = attacks.copy()
    nonfinite.loc[categorical_row, "numeric_budget_m"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_attack_rows(nonfinite, clean, expected_conditions=2, clean_train=train)

    wrong_tolerance = attacks.copy()
    wrong_tolerance.loc[categorical_row, "distance_tolerance_m"] = 0.05
    with pytest.raises(ValueError, match="distance tolerance"):
        validate_attack_rows(
            wrong_tolerance, clean, expected_conditions=2, clean_train=train
        )


def test_predictions_and_metrics_keep_all_intent_and_do_not_invent_ure_repair():
    frame = clean_frame(rows_per_region=1)
    regions = ("A", "B")
    fixed = np.array([[0.8, 0.2], [0.3, 0.7]])
    shared = np.array([[0.7, 0.3], [0.4, 0.6]])
    native = np.array([[0.9, 0.1], [0.2, 0.8]])
    detector = {
        PROPOSED_LABEL: np.array([0.1, 0.9]),
        URE_LABEL: np.array([0.2, 0.8]),
    }
    thresholds = {
        PROPOSED_LABEL: {0.95: 0.5},
        URE_LABEL: {0.95: 0.5},
    }
    valid = np.array([True, False])
    quarantine = np.array(["", "polygon_disagreement"])
    provenance = {
        "attack_method": np.array(["fgsm", "fgsm"]),
        "numeric_budget_m": np.array([50.0, 50.0]),
        "categorical_budget_l0": np.array([0, 0]),
        "source_model": np.array(["s", "s"]),
        "victim_model": np.array(["s", "s"]),
        "victim_evaluation_scope": np.array(["direct", "direct"]),
        "constraints_valid": valid,
        "label_preserved": valid,
        "attack_attempted": np.array([True, True]),
        "attack_changed_any": np.array([True, False]),
    }
    predictions = _prediction_rows(
        frame,
        seed=42,
        generation_seed=42,
        condition="fgsm_m50",
        regions=regions,
        valid=valid,
        quarantine_reason=quarantine,
        detector_scores=detector,
        fixed_scores=fixed,
        shared_repair_scores=shared,
        proposed_native_scores=native,
        thresholds=thresholds,
        provenance=provenance,
    )
    assert len(predictions) == len(frame) * len(regions) * len(DETECTOR_LABELS)
    assert predictions.loc[predictions.detector.eq(URE_LABEL),
                           "proposed_native_full_coverage_score"].isna().all()
    assert predictions.loc[predictions.detector.eq(PROPOSED_LABEL),
                           "proposed_native_full_coverage_score"].notna().all()
    fixed_by_detector = predictions.groupby("detector")["fixed_unmodified_score"].apply(list)
    assert fixed_by_detector.iloc[0] == fixed_by_detector.iloc[1]
    assert (~predictions["primary_evaluation_eligible"]).sum() == 4
    assert set(predictions["numeric_budget_m"]) == {50.0}

    clean_detector = {
        PROPOSED_LABEL: np.array([0.05, 0.10]),
        URE_LABEL: np.array([0.10, 0.15]),
    }
    detector_rows = pd.DataFrame(
        _detector_metric_rows(
            clean_detector,
            detector,
            valid,
            thresholds=thresholds,
            seed=42,
            generation_seed=42,
            condition="fgsm_m50",
        )
    )
    assert set(detector_rows.loc[detector_rows.population_scope.eq("all_intent"), "n"]) == {4}
    assert set(detector_rows.loc[detector_rows.population_scope.eq("primary_valid"), "n"]) == {3}
    assert set(detector_rows.quarantined_attack_rows) == {1}

    classification_rows = pd.DataFrame(
        _classification_metric_rows(
            np.array([[1, 0], [0, 1]]),
            fixed,
            shared,
            native,
            detector,
            thresholds,
            seed=42,
            generation_seed=42,
            condition="fgsm_m50",
            cohort="attack",
            population_scope="all_intent",
            all_intent_rows=2,
            primary_valid_rows=1,
            regions=regions,
            classification_threshold=0.5,
        )
    )
    native_rows = classification_rows.loc[
        classification_rows.classification_mode.eq("proposed_native_full_coverage")
    ]
    assert set(native_rows.detector) == {PROPOSED_LABEL}
    assert set(classification_rows.classification_mode) == {
        "fixed_unfiltered", "detector_accepted_only",
        "shared_nearest_train_repair", "proposed_native_full_coverage",
    }
    assert set(classification_rows.quarantined_rows) == {1}


def test_projected_repair_uses_fixed_classifier_only_for_rejected_rows():
    from poi.defense_pipeline import _classification_metric_rows, PROPOSED_LABEL, URE_LABEL
    rows=_classification_metric_rows(
        np.array([[1],[0]]),np.array([[.1],[.1]]),np.array([[.3],[.6]]),
        np.array([[.1],[.9]]),
        {PROPOSED_LABEL:np.array([.9,.1]),URE_LABEL:np.array([.9,.1])},
        {PROPOSED_LABEL:{.95:.5},URE_LABEL:{.95:.5}},
        seed=42,generation_seed=42,condition='pgd',cohort='attack',population_scope='primary_valid',
        all_intent_rows=2,primary_valid_rows=2,regions=['Seoul'],classification_threshold=.5,
        proposed_projected=np.array([[.9],[.9]]))
    projected=[row for row in rows if row['classification_mode']=='proposed_projected_repair_fixed_tabpfn25']
    assert len(projected)==1
    assert projected[0]['accuracy']==1
    assert projected[0]['coverage']==.5
    assert projected[0]['n']==2
    native=[row for row in rows if row['classification_mode']=='proposed_native_full_coverage'][0]
    assert native['accuracy']==0


def test_detector_metrics_use_each_calibrated_threshold_with_strict_ties():
    clean = {label: np.array([.1, .2]) for label in DETECTOR_LABELS}
    attack = {label: np.array([.3, .4]) for label in DETECTOR_LABELS}
    thresholds = {label: {.90: .2, .95: .3} for label in DETECTOR_LABELS}
    rows = pd.DataFrame(_detector_metric_rows(
        clean, attack, np.array([True, True]), thresholds=thresholds,
        seed=42, generation_seed=202, condition="pgd",
    ))
    assert len(rows) == len(DETECTOR_LABELS) * 2 * 2
    primary = rows.loc[
        rows.population_scope.eq("primary_valid")
        & rows.clean_retention_target.eq(.90)
    ]
    assert set(primary.recall) == {1.0}
    assert set(primary.detector_decision_rule) == {
        "attack_probability_strictly_gt_threshold"
    }
    tied = rows.loc[rows.clean_retention_target.eq(.95)]
    assert set(tied.threshold_tie_rows) == {1}


def test_empty_primary_attack_cohort_emits_n0_rows():
    clean = {
        "truth": np.array([[1], [0]]),
        "fixed": np.array([[.9], [.1]]),
        "shared": np.array([[.8], [.2]]),
        "native": np.array([[.85], [.15]]),
        "projected": np.array([[.82], [.18]]),
        "detector": {label: np.array([.1, .1]) for label in DETECTOR_LABELS},
    }
    attack = {**clean, "detector": {label: np.array([.9, .9]) for label in DETECTOR_LABELS}}
    rows = pd.DataFrame(_cohort_metric_rows(
        clean, attack, np.array([False, False]), seed=42, generation_seed=202,
        condition="pgd", regions=["Seoul"],
        thresholds={label: {.95: .5} for label in DETECTOR_LABELS},
        classification_threshold=.5,
    ))
    empty = rows.loc[
        rows.population_scope.eq("primary_valid") & rows.evaluation_cohort.eq("attack")
    ]
    assert len(empty) > 0 and set(empty.n) == {0}
    assert set(empty.f1) == {0.0}
    assert empty.accuracy.isna().all()


def test_plots_survive_fully_rejected_primary_valid_attack_rows(tmp_path):
    paths = []
    for condition, attack_label, accepted in [("clean", 0, True), ("pgd", 1, False)]:
        rows = []
        for detector in DETECTOR_LABELS:
            for poi, source in [("p0", "A"), ("p1", "B")]:
                for target in ("A", "B"):
                    rows.append({
                        "POI_ID": poi, "seed": 42, "attack_condition": condition,
                        "attack_label": attack_label, "detector": detector,
                        "detector_score": .1 if attack_label == 0 else .9,
                        "truth": int(source == target),
                        "fixed_unmodified_score": .8 if source == target else .2,
                        "primary_evaluation_eligible": True,
                        "retention_95_accepted": accepted,
                    })
        path = tmp_path / f"{condition}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths.append(path)
    figures = tmp_path / "figures"

    _plot_curves(paths, figures, .95)

    assert {path.name for path in figures.glob("*.png")} == {
        "detector_confusion.png", "detector_pr_roc.png",
        "passed_region_confusion_pr_roc.png",
    }


def test_selected_model_factory_binds_actual_checkpoint_and_repeat_probe(
    tmp_path, monkeypatch
):
    from poi.selected_explain import ModelFactory

    checkpoint = tmp_path / "official.ckpt"
    checkpoint.write_bytes(b"official checkpoint")
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    captured = {}

    class FakeModel:
        def fit(self, matrix, truth):
            captured["train"] = np.asarray(matrix).copy()
            return self

        def predict_proba(self, matrix):
            positive = np.full(len(matrix), .25)
            return np.column_stack((1 - positive, positive))

        def save_fit_state(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"official fitted state")

        @classmethod
        def load_from_fit_state(cls, path, *, device):
            assert Path(path).read_bytes() == b"official fitted state"
            assert device == "cpu"
            return cls()

    class FakeClassifier:
        @staticmethod
        def create_default_for_version(version, **overrides):
            captured.update(overrides)
            return FakeModel()

    tabpfn = types.ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = FakeClassifier
    tabpfn.__version__ = "fixture"
    constants = types.ModuleType("tabpfn.constants")
    constants.ModelVersion = types.SimpleNamespace(V2_5="v2.5", V2_6="v2.6", V3="v3")
    torch = types.ModuleType("torch")
    torch.__version__ = "fixture"
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    monkeypatch.setitem(sys.modules, "torch", torch)
    train = clean_frame().assign(split="train")
    factory = ModelFactory("tabpfn_v2_5", {
        "official_checkpoint": str(checkpoint),
        "expected_checkpoint_sha256": expected_sha,
        "require_checkpoint_attestation": True,
        "device": "cpu", "n_estimators": 4, "seed": 42,
    }, train)

    predictor = factory.fit(FEATURES, train.region.eq("A").astype(int).to_numpy())
    persisted = factory.persist_last_fit_state(tmp_path / "region.tabpfn_fit")

    # TabPFN 8.1 serializes constructor parameters in save_fit_state, so the
    # checkpoint is deliberately passed as a string rather than a Path.
    assert captured["model_path"] == str(checkpoint)
    assert np.allclose(predictor(np.ones((2, 5))), .25)
    attestation = factory.last_fit_attestation
    assert attestation["official_checkpoint_sha256"] == expected_sha
    assert attestation["probe_prediction_sha256"] == \
        attestation["probe_repeat_prediction_sha256"]
    assert attestation["feature_order"] == FEATURES
    assert persisted["fitted_state_format"] == "tabpfn_official_fit_state_v1"
    assert persisted["restored_probe_prediction_sha256"] == \
        persisted["probe_prediction_sha256"]
