import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from poi.sensitivity_stability import (
    CANONICAL_CLEAN_GENERATION_SEED,
    _broadcast_canonical_clean,
    _ordered_identity_sha256,
    _validate_shard_identity,
    _validate_sources,
    _validate_clean_broadcast_invariant,
    _weighted_f1,
    assess_ablations,
    assess_extended_plan,
    bootstrap_ranking,
    evaluate_sensitivity,
    factorial_seed_stability,
    fit_detector_ensembles,
    nested_attack_mask,
    run,
    seed_comparison_type,
    stage14_expected_counts,
)


MEMBERS = ["proposed_tabpfn25_attack_head", "URE-RF attack detector"]


def _metadata(eligible=True):
    return {
        "primary_evaluation_eligible": eligible,
        "quarantine_reason": "" if eligible else "outside_boundary",
        "attack_method": "clean",
        "numeric_budget_m": np.nan,
        "categorical_budget_l0": 0,
        "source_model": "none",
        "victim_model": "none",
        "victim_evaluation_scope": "clean",
        "constraints_valid": eligible,
        "label_preserved": True,
        "attack_attempted": False,
        "attack_changed_any": False,
        "transfer_evaluated": False,
        "targeted": False,
        "valid_numeric_train_range": eligible,
        "valid_category_codebook": True,
        "valid_category_combo": True,
        "within_budget": True,
        "clean_boundary_status": "match",
        "attack_boundary_status": "match" if eligible else "outside",
    }


def _detector_rows(prefix, include_targets=False):
    rows = []
    for poi, region in [(f"{prefix}0", "A"), (f"{prefix}1", "B")]:
        for condition, label in [("clean", 0), ("pgd_m10", 1)]:
            for detector, offset in [(MEMBERS[0], 0.05), (MEMBERS[1], 0.15)]:
                base = {
                    "POI_ID": poi,
                    "source_region": region,
                    "seed": 42,
                    "generation_seed": 42,
                    "attack_condition": condition,
                    "attack_label": label,
                    "detector": detector,
                    "detector_score": (0.1 + offset) if label == 0 else (0.7 + offset),
                    **_metadata(True),
                }
                if include_targets:
                    for target in ["A", "B"]:
                        rows.append({**base, "target_region": target})
                else:
                    rows.append(base)
    return pd.DataFrame(rows)


def test_attack_mixture_masks_are_nested_and_identity_stable():
    ids = [f"p{value}" for value in range(100)]
    low = nested_attack_mask(ids, .1, 9)
    high = nested_attack_mask(ids, .75, 9)
    assert low.sum() == 10
    assert high.sum() == 75
    assert np.all(~low | high)
    assert np.array_equal(low, nested_attack_mask(ids, .1, 9))


def test_detector_ensemble_fits_calibration_and_freezes_for_evaluation():
    calibration = _detector_rows("cal")
    evaluation = _detector_rows("eval", include_targets=True)
    cal, eva, thresholds, fits = fit_detector_ensembles(
        calibration,
        evaluation,
        MEMBERS,
        [.9, .95],
        {"enabled": True, "class_weight": "balanced", "max_iter": 1000,
         "random_state": 3},
    )
    assert np.allclose(cal["predefined_mean_score"], cal[MEMBERS].mean(axis=1))
    assert np.allclose(eva["predefined_mean_score"], eva[MEMBERS].mean(axis=1))
    assert eva["calibration_logistic"].between(0, 1).all()
    assert set(thresholds.detector_variant) == {*MEMBERS, "predefined_mean_score",
                                                "calibration_logistic"}
    logistic = fits.loc[fits.detector_variant.eq("calibration_logistic")].iloc[0]
    assert logistic.fit_partition == "validation_calibration_half"
    assert logistic.evaluation_partition == "disjoint_validation_evaluation_half"
    assert logistic.fit_rows == len(calibration) // len(MEMBERS)
    assert set(thresholds.generation_seed) == {CANONICAL_CLEAN_GENERATION_SEED}


def test_logistic_ensemble_pools_generation_seeds_without_tripling_clean_rows():
    base_calibration = _detector_rows("cal")
    base_evaluation = _detector_rows("eval", include_targets=True)
    calibration = []
    evaluation = []
    for generation_seed in (42, 202, 340):
        for source, target in ((base_calibration, calibration), (base_evaluation, evaluation)):
            part = source.copy()
            part["generation_seed"] = generation_seed
            target.append(part)
    calibration = pd.concat(calibration, ignore_index=True)
    evaluation = pd.concat(evaluation, ignore_index=True)

    cal, eva, thresholds, fits = fit_detector_ensembles(
        calibration, evaluation, MEMBERS, [.95],
        {"enabled": True, "class_weight": "balanced", "max_iter": 1000,
         "random_state": 3},
    )

    logistic = fits.loc[fits.detector_variant.eq("calibration_logistic")]
    assert len(logistic) == 1
    assert logistic.iloc[0].fit_clean_rows == 2
    assert logistic.iloc[0].fit_attack_rows == 6
    assert set(thresholds.generation_seed) == {CANONICAL_CLEAN_GENERATION_SEED}
    assert len(thresholds) == 4
    clean = eva.loc[eva.attack_label.eq(0)]
    pivot = clean.pivot_table(index="POI_ID", columns="generation_seed",
                              values="calibration_logistic")
    assert np.allclose(pivot[42], pivot[202]) and np.allclose(pivot[42], pivot[340])


def test_canonical_clean_is_broadcast_in_memory_and_never_stored_three_times():
    stored = _detector_rows("p", include_targets=True)
    stored.loc[stored.attack_label.eq(0), "generation_seed"] = CANONICAL_CLEAN_GENERATION_SEED
    expanded = _broadcast_canonical_clean(stored, [42, 202, 340], "fixture")
    clean = expanded.loc[expanded.attack_label.eq(0)]
    attack = expanded.loc[expanded.attack_label.eq(1)]
    assert set(clean.generation_seed) == {42, 202, 340}
    assert len(clean) == 3 * len(stored.loc[stored.attack_label.eq(0)])
    assert set(attack.generation_seed) == {42}


def test_clean_generation_anchor_difference_is_rejected():
    detector = _detector_rows("p")
    detector = pd.concat([
        detector.assign(generation_seed=42), detector.assign(generation_seed=202)
    ], ignore_index=True)
    region_rows = []
    for row in detector.loc[detector.detector.eq(MEMBERS[0])].itertuples(index=False):
        for target in ("A", "B"):
            region_rows.append({
                **row._asdict(), "target_region": target,
                "truth": int(row.source_region == target),
                "fixed_unmodified_score": .8,
                "shared_nearest_train_repair_score": .8,
                "proposed_native_full_coverage_score": .8,
            })
    region = pd.DataFrame(region_rows)
    wide = detector.pivot(
        index=["POI_ID", "seed", "generation_seed", "attack_condition", "attack_label"],
        columns="detector", values="detector_score",
    ).reset_index()
    wide["predefined_mean_score"] = wide[MEMBERS].mean(axis=1)
    wide.loc[
        wide.generation_seed.eq(202) & wide.attack_condition.eq("clean"),
        "predefined_mean_score",
    ] += .01
    with pytest.raises(ValueError, match="Clean broadcast differs"):
        _validate_clean_broadcast_invariant(
            region, wide, [*MEMBERS, "predefined_mean_score"]
        )


def test_weighted_f1_uses_explicit_zero_division_policy():
    truth = np.array([0, 1])
    score = np.array([0.1, 0.9])
    weights = np.array([[1, 1], [1, 0]], dtype=np.uint16)
    values = _weighted_f1(truth, score, np.array([True, True]), weights, .5)
    assert values[0] == 1
    assert values[1] == 0


def test_sensitivity_keeps_all_intent_coverage_but_metrics_are_primary_valid_only():
    region_rows = []
    detector_rows = []
    for condition, label in [("clean", 0), ("pgd_m10", 1)]:
        for poi in range(4):
            eligible = not (condition == "pgd_m10" and poi == 3)
            base = {
                "POI_ID": f"p{poi}", "source_region": "A" if poi % 2 == 0 else "B",
                "seed": 42, "generation_seed": 42, "attack_condition": condition,
                "attack_label": label, **_metadata(eligible),
            }
            base["attack_method"] = "clean" if label == 0 else "pgd"
            base["numeric_budget_m"] = np.nan if label == 0 else 10.0
            detector_rows.append({**base, MEMBERS[0]: .1 + .7 * label,
                                  "predefined_mean_score": .1 + .7 * label})
            for target in ["A", "B"]:
                truth = int(base["source_region"] == target)
                region_rows.append({
                    **base, "target_region": target, "truth": truth,
                    "fixed_unmodified_score": .9 if truth else .1,
                    "shared_nearest_train_repair_score": .8 if truth else .2,
                    "proposed_native_full_coverage_score": .85 if truth else .15,
                    "proposed_projected_repair_fixed_score": .82 if truth else .18,
                })
    thresholds = pd.DataFrame([
        {"seed": 42, "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
         "detector_variant": variant,
         "clean_retention_target": .95, "threshold": .5}
        for variant in [MEMBERS[0], "predefined_mean_score"]
    ])
    cfg = {
        "primary_clean_retention": .95,
        "attack_mixture_ratios": [0, .5, 1],
        "clean_retention_targets": [.95],
        "classification_threshold": .5,
        "mixture_seed": 12,
    }
    details, coverage, payload = evaluate_sensitivity(
        pd.DataFrame(region_rows), pd.DataFrame(detector_rows), thresholds, cfg
    )
    assert set(details.classification_mode) == {
        "full_coverage", "repaired_all", "proposed_projected_repair_fixed_tabpfn25",
    }
    assert set(coverage.population_scope) == {"all_intent_diagnostic", "primary_valid"}
    full_attack = coverage.loc[
        coverage.attack_mixture_ratio.eq(1)
        & coverage.detector_variant.eq("predefined_mean_score")
    ]
    assert full_attack.loc[full_attack.population_scope.eq("all_intent_diagnostic"),
                           "denominator_pois"].iloc[0] == 4
    assert full_attack.loc[full_attack.population_scope.eq("primary_valid"),
                           "denominator_pois"].iloc[0] == 3
    assert payload and all(
        np.array_equal(item["ids"], np.array(["p0", "p1", "p2", "p3"]))
        for item in payload
    )
    assert set(details.loc[details.attack_mixture_ratio.eq(0), "generation_seed"]) == {
        CANONICAL_CLEAN_GENERATION_SEED
    }


def test_sensitivity_emits_nan_status_for_empty_primary_valid_attack_cohort():
    region_rows, detector_rows = [], []
    for condition, label in [("clean", 0), ("pgd_m10", 1)]:
        for poi, source in [("p0", "A"), ("p1", "B")]:
            eligible = label == 0
            base = {
                "POI_ID": poi, "source_region": source, "seed": 42,
                "generation_seed": 42, "attack_condition": condition,
                "attack_label": label, **_metadata(eligible),
            }
            base["attack_method"] = "clean" if label == 0 else "pgd"
            detector_rows.append({
                **base, MEMBERS[0]: .1 + .8 * label,
                "predefined_mean_score": .1 + .8 * label,
            })
            for target in ("A", "B"):
                truth = int(source == target)
                region_rows.append({
                    **base, "target_region": target, "truth": truth,
                    "fixed_unmodified_score": .8 if truth else .2,
                    "shared_nearest_train_repair_score": .75 if truth else .25,
                    "proposed_native_full_coverage_score": .7 if truth else .3,
                })
    thresholds = pd.DataFrame([
        {"seed": 42, "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
         "detector_variant": variant, "clean_retention_target": .95, "threshold": .5}
        for variant in (MEMBERS[0], "predefined_mean_score")
    ])
    details, coverage, payload = evaluate_sensitivity(
        pd.DataFrame(region_rows), pd.DataFrame(detector_rows), thresholds,
        {"primary_clean_retention": .95, "attack_mixture_ratios": [0, 1],
         "clean_retention_targets": [.95], "classification_threshold": .5,
         "mixture_seed": 3},
    )
    empty = details.loc[details.attack_mixture_ratio.eq(1)]
    assert len(empty) and set(empty.n) == {0}
    assert set(empty.f1) == {0.0}
    assert empty.accuracy.isna().all() and empty.roc_auc.isna().all()
    primary_coverage = coverage.loc[
        coverage.attack_mixture_ratio.eq(1)
        & coverage.population_scope.eq("primary_valid")
    ]
    assert set(primary_coverage.denominator_pois) == {0}
    assert primary_coverage.coverage.isna().all()
    assert any(item["cohort"] == "attack" and not item["eligible"].any() for item in payload)


def test_factorial_seed_effects_and_expected_counts_are_separated():
    assert seed_comparison_type((42, 42), (202, 42)) == (
        "model_seed_effect_same_generation_seed"
    )
    assert seed_comparison_type((42, 42), (42, 202)) == (
        "generation_seed_effect_same_model_seed"
    )
    assert seed_comparison_type((42, 42), (202, 202)) == (
        "both_seeds_changed_not_effect_isolating"
    )
    rows = []
    for model_seed_index, model_seed in enumerate((42, 202, 340)):
        for generation_seed_index, generation_seed in enumerate((42, 202, 340)):
            for region in ("A", "B"):
                rows.append({
                    "pipeline": "p", "seed": model_seed,
                    "generation_seed": generation_seed, "target_region": region,
                    "robust_f1": .5 + .01 * model_seed_index + .02 * generation_seed_index,
                })
    marginals, variance = factorial_seed_stability(pd.DataFrame(rows))
    assert set(marginals.factor) == {"model_seed", "generation_seed"}
    assert len(marginals) == 6
    assert variance.iloc[0].model_seed_fraction == pytest.approx(.2)
    assert variance.iloc[0].generation_seed_fraction == pytest.approx(.8)
    assert not variance.iloc[0].inferential_test_performed

    counts = stage14_expected_counts(
        model_seeds=3, generation_seeds=3, attack_conditions=32, regions=17,
        ratios=6, detector_variants=4, retentions=4, projected_repair=True,
    )
    assert counts == {
        "model_generation_seed_pairs": 9,
        "canonical_clean_mixtures": 3,
        "attack_or_mixed_mixtures": 1440,
        "sensitivity_detail_rows": 539682,
        "coverage_rows": 46176,
        "ranking_payload_strata": 34629,
        "rank_rows": 1071,
        "rank_pair_correlations": 36,
        "seed_marginal_rows": 42,
        "variance_decomposition_rows": 7,
        "canonical_threshold_rows": 48,
    }


def test_paired_bootstrap_reports_fixed_strata_and_zero_division_policy():
    ids = np.asarray(["p0", "p1", "p2", "p3"])
    truth = np.asarray([0, 1, 0, 1])
    score = np.asarray([.1, .9, .2, .8])
    payload = []
    for pipeline in ["reference", "identical"]:
        for cohort in ["clean", "attack"]:
            payload.append({
                "pipeline": pipeline,
                "cohort": cohort,
                "ids": ids,
                "truth": truth,
                "score": score,
                "eligible": np.ones(4, dtype=bool),
            })
    cfg = {
        "classification_threshold": .5,
        "ranking": {
            "bootstrap_repeats": 200,
            "bootstrap_seed": 4,
            "confidence_level": .95,
            "reference_pipeline": "reference",
        },
    }
    intervals, pairwise = bootstrap_ranking(payload, cfg)
    assert (intervals.finite_draws <= intervals.bootstrap_repeats).all()
    assert (intervals.undefined_draws == intervals.bootstrap_repeats - intervals.finite_draws).all()
    difference = pairwise.loc[pairwise.pipeline.eq("identical")].iloc[0]
    assert difference.bootstrap_mean_difference == 0
    assert difference.ci_low == 0 and difference.ci_high == 0


def test_bootstrap_point_and_draws_keep_zero_positive_strata():
    ids = np.asarray(["a", "b"])
    payload = []
    for pipeline in ["reference", "same"]:
        for cohort in ["clean", "attack"]:
            payload.extend([
                {"pipeline": pipeline, "cohort": cohort, "ids": ids,
                 "truth": np.array([0, 1]), "score": np.array([.1, .9]),
                 "eligible": np.ones(2, dtype=bool)},
                {"pipeline": pipeline, "cohort": cohort, "ids": ids,
                 "truth": np.array([0, 0]), "score": np.array([.1, .1]),
                 "eligible": np.ones(2, dtype=bool)},
            ])
    cfg = {"classification_threshold": .5, "ranking": {
        "bootstrap_repeats": 20, "bootstrap_seed": 9, "confidence_level": .95,
        "reference_pipeline": "reference",
    }}
    intervals, pairwise = bootstrap_ranking(payload, cfg)
    row = intervals.loc[intervals.pipeline.eq("reference")].iloc[0]
    assert row.point_estimate_robust_f1 == pytest.approx(.5)
    assert row.finite_draws == 20 and row.undefined_draws == 0
    assert row.clean_point_strata_finite == row.clean_strata_required_per_draw == 2
    assert row.attack_point_strata_finite == row.attack_strata_required_per_draw == 2
    assert row.f1_zero_division_policy == 0
    assert pairwise.iloc[0].point_estimate_difference == 0


def test_missing_stage10_inputs_leave_stage14_not_run_without_output(tmp_path):
    output = tmp_path / "stage14"
    config = {
        "stage10_output": str(tmp_path / "stage10"),
        "prediction_manifest": str(tmp_path / "stage10/tables/prediction_manifest.csv"),
        "calibration_detector_scores": str(tmp_path / "stage10/tables/calibration.parquet"),
        "ablation_matrix": str(tmp_path / "missing_matrix.yaml"),
        "output": str(output),
        "model_seeds": [42, 202, 340],
        "generation_seeds": [42, 202, 340],
        "seed_design": "full_model_by_generation_seed_cross",
        "canonical_clean_generation_seed": -1,
        "attack_mixture_ratios": [0, .1, .25, .5, .75, 1],
        "clean_retention_targets": [.9, .95, .975, .99],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(FileNotFoundError, match="NOT_RUN"):
        run(path)
    assert not output.exists()


def test_ablation_assessment_does_not_claim_missing_runs(tmp_path):
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({
        "base_config": "unused.yaml",
        "variants": [
            {"name": "no_gate", "kind": "structural", "train": True,
             "output": str(tmp_path / "no_gate"), "changes": ["remove gate"]},
            {"name": "no_consistency", "kind": "objective", "train": True,
             "output": str(tmp_path / "no_consistency"), "changes": ["zero loss"]},
        ],
    }))
    result = assess_ablations({
        "ablation_matrix": str(matrix),
        "ablation_required_files": ["metadata.json", "training_history.csv"],
    })
    assert set(result.status) == {"NOT_RUN"}
    assert set(result.kind) == {"structural", "objective"}


def test_extended_plan_requirements_remain_explicit_test_unlock_blockers():
    result = assess_extended_plan({
        "available_attack_source_models": ["surrogate_mlp"],
        "required_extended_experiments": [
            "leave_one_attack_out_generalization",
            "leave_one_source_model_out_transfer",
            "adaptive_detector_evasion_attack",
            "regional_probability_ensemble_calibration",
        ],
    })
    assert len(result) == 4
    assert set(result.status) == {"NOT_RUN"}
    assert result.blocking_for_test_unlock.all()
    loso = result.loc[result.requirement_id.eq("leave_one_source_model_out_transfer")].iloc[0]
    assert "surrogate_mlp only" in loso.limitation
    regional = result.loc[
        result.requirement_id.eq("regional_probability_ensemble_calibration")
    ].iloc[0]
    assert regional.observed_scope == "detector-score ensemble only"


def test_stage14_rejects_attack_truth_that_differs_from_source_region():
    evaluation = _detector_rows("eval", include_targets=True)
    evaluation["truth"] = evaluation.source_region.eq(evaluation.target_region).astype(int)
    evaluation["fixed_unmodified_score"] = np.where(evaluation.truth.eq(1), .8, .2)
    evaluation["shared_nearest_train_repair_score"] = np.where(
        evaluation.truth.eq(1), .75, .25
    )
    evaluation["proposed_native_full_coverage_score"] = np.where(
        evaluation.truth.eq(1), .7, .3
    )
    calibration = _detector_rows("cal")
    cfg = {
        "model_seeds": [42], "generation_seeds": [42],
        "expected_regions": 2, "expected_attack_conditions": 1,
        "detector_members": MEMBERS,
    }
    bad = evaluation.copy()
    attack = bad.attack_label.eq(1)
    bad.loc[attack, "truth"] = 1 - bad.loc[attack, "truth"]

    with pytest.raises(ValueError, match="paired truth"):
        _validate_sources(bad, calibration, cfg)


def test_stage14_shard_identity_binds_truth_and_region_rows():
    frame = _detector_rows("eval", include_targets=True)
    frame = frame.loc[frame.attack_condition.eq("clean")].reset_index(drop=True)
    frame["truth"] = frame.source_region.eq(frame.target_region).astype(int)
    poi = frame[["POI_ID", "source_region"]].drop_duplicates("POI_ID", keep="first")

    class Manifest:
        seed = 42
        generation_seed = 42
        condition = "clean"
        unique_pois = 2
        poi_identity_sha256 = _ordered_identity_sha256(poi, ["POI_ID", "source_region"])
        row_identity_sha256 = _ordered_identity_sha256(
            frame, ["POI_ID", "source_region", "target_region", "truth", "detector"]
        )

    _validate_shard_identity(frame, Manifest(), Path("fixture.parquet"))
    bad = frame.copy()
    bad.loc[0, "truth"] = 1 - int(bad.loc[0, "truth"])
    with pytest.raises(ValueError, match="row identity"):
        _validate_shard_identity(bad, Manifest(), Path("fixture.parquet"))
