import json
import numpy as np
import pandas as pd
import pytest

from poi import unified_clean_features as module


def _validation_design():
    regions = [f"R{index:02d}" for index in range(17)]
    validation = pd.DataFrame({
        "POI_ID": [f"V{index:02d}{copy}" for index in range(17) for copy in range(2)],
        "region": [region for region in regions for _ in range(2)],
    })
    rows = []
    for model in module.MODELS:
        for seed in module.SEEDS:
            for region in regions:
                truth = validation["region"].eq(region).astype(np.int8).to_numpy()
                score = np.where(truth == 1, 0.9, 0.1) + (seed % 7) * 1e-5
                rows.append(pd.DataFrame({
                    "POI_ID": validation["POI_ID"],
                    "model": model,
                    "seed": seed,
                    "region": region,
                    "condition": "clean",
                    "split": "validation",
                    "y_true": truth,
                    "y_score": score,
                    "y_pred": (score >= 0.5).astype(np.int8),
                }))
    return validation, pd.concat(rows, ignore_index=True)


def test_normalize_requires_exact_cartesian_poi_truth_and_binary_values():
    validation, raw = _validation_design()
    normalized = module.normalize_prediction_frame(
        raw,
        feature_set="full",
        validation=validation,
        expected_models=module.MODELS,
    )
    assert len(normalized) == 8 * 3 * 17 * 34
    assert normalized.groupby(["model", "seed", "region"]).size().eq(34).all()
    assert normalized["y_pred_fixed_0p5"].isin([0, 1]).all()

    fractional = raw.copy()
    fractional.loc[0, "y_true"] = 0.9
    with pytest.raises(module.UnifiedCleanError, match="exact finite binary"):
        module.normalize_prediction_frame(
            fractional,
            feature_set="full",
            validation=validation,
            expected_models=module.MODELS,
        )

    missing = raw.drop(raw.index[0])
    with pytest.raises(module.UnifiedCleanError, match="row coverage"):
        module.normalize_prediction_frame(
            missing,
            feature_set="full",
            validation=validation,
            expected_models=module.MODELS,
        )


def test_common_calibration_thresholds_and_shared_poi_bootstrap_use_eval_only():
    validation, raw = _validation_design()
    frame = module.normalize_prediction_frame(
        raw,
        feature_set="full",
        validation=validation,
        expected_models=module.MODELS,
    )
    calibration_ids = [f"V{index:02d}0" for index in range(17)]
    evaluation_ids = [f"V{index:02d}1" for index in range(17)]
    weights = module.make_bootstrap_weights(17, 25, 123)
    thresholds, metrics, replicates = module.analyze_feature_predictions(
        frame,
        calibration_ids=calibration_ids,
        evaluation_ids=evaluation_ids,
        weights=weights,
    )

    assert len(thresholds) == 8 * 3 * 17
    assert thresholds["rows"].eq(17).all()
    assert thresholds["threshold_source"].eq("common_validation_calibration_ids_only").all()
    assert len(metrics) == 8 * 3 * 17 * 3
    assert metrics.loc[metrics["partition"].eq("evaluation"), "rows"].eq(17).all()
    assert metrics.loc[metrics["partition"].eq("full_validation"), "rows"].eq(34).all()
    assert set(metrics["threshold_rule"]) == {"calibrated_f1", "fixed_0p5", "fixed_0p5_full_validation"}
    assert len(replicates) == 8 * 6
    assert all(values.shape == (25,) for values in replicates.values())
    for metric in module.METRICS:
        reference = replicates[(f"full|{module.MODELS[0]}", metric)]
        if metric in {"f1", "precision", "recall", "accuracy"}:
            assert np.isfinite(reference).all()
        for model in module.MODELS[1:]:
            # One shared POI multiplicity matrix must make identical pipelines
            # exactly paired, including undefined AUC/AP draws.
            assert np.allclose(
                reference,
                replicates[(f"full|{model}", metric)],
                equal_nan=True,
            )


def test_bootstrap_summary_contains_all_24_pipeline_pairs_and_within_model_feature_contrasts():
    rows = []
    replicates = {}
    for feature_index, feature_set in enumerate(module.FEATURE_SETS):
        for model_index, model in enumerate(module.MODELS):
            for seed in module.SEEDS:
                for region_index in range(17):
                    rows.append({
                        "feature_set": feature_set,
                        "model": model,
                        "seed": seed,
                        "region": f"R{region_index:02d}",
                        "threshold_rule": "calibrated_f1",
                        **{metric: 0.5 + feature_index * 0.01 + model_index * 0.001 for metric in module.METRICS},
                    })
            for metric in module.METRICS:
                replicates[(f"{feature_set}|{model}", metric)] = np.linspace(0.4, 0.6, 20) + feature_index * 0.01 + model_index * 0.001
    replicate_table, intervals, pairs = module.summarize_bootstrap(
        pd.DataFrame(rows), replicates, confidence_level=0.95,
    )
    assert len(replicate_table) == 24 * 6 * 20
    assert len(intervals) == 24 * 6
    assert len(pairs) == (24 * 23 // 2) * 6
    same_model = pairs.loc[pairs["model_a"].eq(pairs["model_b"]) & pairs["feature_set_a"].ne(pairs["feature_set_b"])]
    assert len(same_model) == 8 * 3 * 6
    assert pairs["p_value"].isna().all()


def test_replay_contract_binds_exact_51_state_grid_partition_and_official_api(tmp_path):
    regions = tuple(f"R{index:02d}" for index in range(17))
    validation = pd.DataFrame({
        "POI_ID": [f"V{index:02d}_{row:03d}" for index in range(17) for row in range(200)],
        "region": [region for region in regions for _ in range(200)],
    })
    calibration_ids = [f"V{index:02d}_{row:03d}" for index in range(17) for row in range(99)]
    calibration_ids.extend(f"V{index:02d}_099" for index in range(10))
    evaluation_ids = [value for value in validation["POI_ID"] if value not in set(calibration_ids)]
    calibration_set = set(calibration_ids)
    (tmp_path / "states").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "tables").mkdir()
    thresholds = tmp_path / "tables/calibration_thresholds.csv"
    scores = tmp_path / "tables/calibration_scores.parquet"
    fits = []
    threshold_rows = []
    score_parts = []
    attest_hash = "a" * 64
    for seed in module.SEEDS:
        for index, region in enumerate(regions):
            state_relative = f"states/modelseed{seed}__region{index:02d}.tabpfn_fit"
            prediction_relative = f"predictions/tabpfn_v3_full_{seed}_{region}_clean.parquet"
            state = tmp_path / state_relative
            prediction = tmp_path / prediction_relative
            state.write_bytes(f"state-{seed}-{region}".encode())
            truth = validation["region"].eq(region).astype(np.int8).to_numpy()
            score = np.where(truth == 1, 0.9, 0.1).astype(np.float64)
            prediction_frame = pd.DataFrame({
                "model": "tabpfn_v3",
                "feature_set": "full",
                "seed": seed,
                "region": region,
                "POI_ID": validation["POI_ID"],
                "partition": ["calibration" if value in calibration_set else "evaluation" for value in validation["POI_ID"]],
                "y_true": truth,
                "y_score": score,
            })
            prediction_frame.to_parquet(prediction, index=False)
            calibration = prediction_frame.loc[prediction_frame["partition"].eq("calibration")].copy()
            selected = module.select_f1_threshold(calibration["y_true"], calibration["y_score"])
            score_hash = module.hashlib.sha256(np.ascontiguousarray(calibration["y_score"].to_numpy(np.float64)).tobytes()).hexdigest()
            truth_hash = module.hashlib.sha256(np.ascontiguousarray(calibration["y_true"].to_numpy(np.int8)).tobytes()).hexdigest()
            fits.append({
                "model_id": "tabpfn_v3",
                "model_seed": seed,
                "region": region,
                "region_index": index,
                "feature_order": list(module.FEATURE_SETS["full"]),
                "state_relative_path": state_relative,
                "state_sha256": module.sha256(state),
                "state_size_bytes": state.stat().st_size,
                "state_format": "tabpfn_official_fit_state_v1",
                "state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
                "official_checkpoint_path": "/frozen/checkpoint.ckpt",
                "official_checkpoint_sha256": attest_hash,
                "train_matrix_sha256": attest_hash,
                "train_truth_sha256": attest_hash,
                "train_poi_order_sha256": attest_hash,
                "probe_indices": [0, 3, 8],
                "probe_matrix_sha256": attest_hash,
                "original_probe_score_sha256": attest_hash,
                "repeat_probe_score_sha256": attest_hash,
                "restored_probe_score_sha256": attest_hash,
                "prediction_relative_path": prediction_relative,
                "prediction_sha256": module.sha256(prediction),
                "calibration_threshold": selected["threshold"],
                "calibration_score_float64_sha256": score_hash,
                "calibration_truth_int8_sha256": truth_hash,
            })
            threshold_rows.append({"model": "tabpfn_v3", "feature_set": "full", "seed": seed, "region": region, "region_index": index, **selected})
            score_parts.append(calibration)
            provenance = {
                "train_poi_order_sha256": attest_hash,
                "prediction_sha256": module.sha256(prediction),
                "calibration_threshold": selected,
                "calibration_score_float64_sha256": score_hash,
                "calibration_truth_int8_sha256": truth_hash,
                "fit_attestation": {
                    "fitted_state_sha256": module.sha256(state),
                    "fitted_state_size_bytes": state.stat().st_size,
                    "fitted_state_format": "tabpfn_official_fit_state_v1",
                    "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
                    "official_checkpoint_path": "/frozen/checkpoint.ckpt",
                    "official_checkpoint_sha256": attest_hash,
                    "train_matrix_sha256": attest_hash,
                    "truth_sha256": attest_hash,
                    "probe_indices": [0, 3, 8],
                    "probe_matrix_sha256": attest_hash,
                    "probe_prediction_sha256": attest_hash,
                    "probe_repeat_prediction_sha256": attest_hash,
                    "restored_probe_prediction_sha256": attest_hash,
                },
            }
            provenance_path = tmp_path / "work" / f"tabpfn_v3_full_{seed}_{region}_fit_provenance.json"
            provenance_path.parent.mkdir(exist_ok=True)
            provenance_path.write_text(json.dumps(provenance))
    pd.DataFrame(threshold_rows).sort_values(["seed", "region_index"]).to_csv(thresholds, index=False)
    pd.concat(score_parts, ignore_index=True).sort_values(["seed", "region", "POI_ID"]).to_parquet(scores, index=False)
    partition_sha = "b" * 64
    calibration_order_sha = module._ordered_ids_sha(calibration_ids)
    evaluation_order_sha = module._ordered_ids_sha(evaluation_ids)
    aggregate_partition = {
        "sha256": {"first": partition_sha, "second": partition_sha},
        "calibration_rows": 1693,
        "evaluation_rows": 1707,
        "calibration_poi_order_sha256": calibration_order_sha,
        "evaluation_poi_order_sha256": evaluation_order_sha,
    }
    replay_partition = {
        "sha256": partition_sha,
        "calibration_rows": 1693,
        "evaluation_rows": 1707,
        "calibration_poi_order_sha256": calibration_order_sha,
        "evaluation_poi_order_sha256": evaluation_order_sha,
    }
    contract = {
        "format": "poi-unified-clean-tabpfn3-replay-contract-v1",
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "model_id": "tabpfn_v3",
        "feature_order": list(module.FEATURE_SETS["full"]),
        "seeds": list(module.SEEDS),
        "regions": list(regions),
        "fit_count": 51,
        "partition": replay_partition,
        "state_load_rule": "TabPFNClassifier.load_from_fit_state exactly once per seed/region and reuse for all clean/attack/defense inference",
        "calibration_thresholds_sha256": module.sha256(thresholds),
        "calibration_scores_sha256": module.sha256(scores),
        "fits": fits,
        "test_rows_read": 0,
        "test_features_read": 0,
    }
    contract_path = tmp_path / "replay_contract.json"
    contract_path.write_text(json.dumps(contract))
    paths = [contract_path, thresholds, scores]
    paths.extend(tmp_path / row[key] for row in fits for key in ("state_relative_path", "prediction_relative_path"))
    paths.extend(
        tmp_path / "work" / f"tabpfn_v3_full_{seed}_{region}_fit_provenance.json"
        for seed in module.SEEDS for region in regions
    )
    snapshot = module._snapshot_files(paths)
    observed = module._validate_replay_contract(
        tmp_path, regions, aggregate_partition, calibration_ids, evaluation_ids,
        validation, snapshot,
    )
    assert observed["fit_count"] == 51

    contract["fits"][0]["state_api"] = "unreviewed_loader"
    contract_path.write_text(json.dumps(contract))
    changed_snapshot = module._snapshot_files(paths)
    with pytest.raises(module.UnifiedCleanError, match="identity/attestation"):
        module._validate_replay_contract(
            tmp_path, regions, aggregate_partition, calibration_ids,
            evaluation_ids, validation, changed_snapshot,
        )

    contract["fits"][0]["state_api"] = "TabPFNClassifier.save_fit_state/load_from_fit_state"
    duplicate_scores = pd.read_parquet(scores)
    duplicate_scores["POI_ID"] = "DUPLICATE_NOT_IN_CALIBRATION"
    duplicate_scores.to_parquet(scores, index=False)
    contract["calibration_scores_sha256"] = module.sha256(scores)
    contract_path.write_text(json.dumps(contract))
    duplicate_snapshot = module._snapshot_files(paths)
    with pytest.raises(module.UnifiedCleanError, match="cardinality/content"):
        module._validate_replay_contract(
            tmp_path, regions, aggregate_partition, calibration_ids,
            evaluation_ids, validation, duplicate_snapshot,
        )


def test_fit_identity_recomputes_fingerprint_and_binds_current_input_bytes():
    input_snapshot = {
        "/remote/data.csv": {"sha256": "a" * 64, "size_bytes": 123},
    }
    source_gate = {
        "manifest_sha256": "b" * 64,
        "verdict_sha256": "c" * 64,
    }
    identity = {
        "format": module.FIT_FORMAT,
        "mode": "coordinate-trees",
        "source_gate": source_gate,
        "input_snapshot": input_snapshot,
    }
    identity["fingerprint_sha256"] = module._canonical_json_sha({
        "format": module.FIT_FORMAT,
        "mode": "coordinate-trees",
        "models": list(module.TREE_MODELS),
        "feature_set": "coordinates_only",
        "features": list(module.FEATURE_SETS["coordinates_only"]),
        "seeds": list(module.SEEDS),
        "source_gate": source_gate,
        "input_snapshot": input_snapshot,
    })
    metadata = {"input_snapshot": input_snapshot}
    current_inventory = [("a" * 64, 123)]

    observed = module._validate_fit_identity_contract(
        identity,
        metadata,
        module.TREE_MODELS,
        "coordinate-trees",
        "coordinates_only",
        module.FEATURE_SETS["coordinates_only"],
        current_inventory,
    )
    assert observed == identity["fingerprint_sha256"]

    stale_fingerprint = json.loads(json.dumps(identity))
    stale_fingerprint["input_snapshot"]["/remote/data.csv"]["sha256"] = "0" * 64
    stale_metadata = {"input_snapshot": stale_fingerprint["input_snapshot"]}
    with pytest.raises(module.UnifiedCleanError, match="fingerprint"):
        module._validate_fit_identity_contract(
            stale_fingerprint,
            stale_metadata,
            module.TREE_MODELS,
            "coordinate-trees",
            "coordinates_only",
            module.FEATURE_SETS["coordinates_only"],
            current_inventory,
        )

    rebound_wrong_bytes = json.loads(json.dumps(stale_fingerprint))
    rebound_wrong_bytes["fingerprint_sha256"] = module._canonical_json_sha({
        "format": module.FIT_FORMAT,
        "mode": "coordinate-trees",
        "models": list(module.TREE_MODELS),
        "feature_set": "coordinates_only",
        "features": list(module.FEATURE_SETS["coordinates_only"]),
        "seeds": list(module.SEEDS),
        "source_gate": source_gate,
        "input_snapshot": rebound_wrong_bytes["input_snapshot"],
    })
    rebound_metadata = {"input_snapshot": rebound_wrong_bytes["input_snapshot"]}
    with pytest.raises(module.UnifiedCleanError, match="current data/config/partition/checkpoint bytes"):
        module._validate_fit_identity_contract(
            rebound_wrong_bytes,
            rebound_metadata,
            module.TREE_MODELS,
            "coordinate-trees",
            "coordinates_only",
            module.FEATURE_SETS["coordinates_only"],
            current_inventory,
        )
