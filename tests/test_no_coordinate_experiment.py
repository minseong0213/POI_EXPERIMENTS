import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
import yaml

import poi.no_coordinate_experiment as experiment


def test_gpu_config_freezes_three_features_seeds_checkpoint_and_output():
    path = Path("configs/stages/07_no_coordinate_tabpfn26.yaml")
    cfg = experiment.load_config(path)
    assert tuple(cfg["features"]) == experiment.FEATURES
    assert tuple(cfg["seeds"]) == experiment.SEEDS
    assert cfg["tabpfn_v2_6"] == {
        "device": "cuda",
        "n_estimators": 8,
        "official_checkpoint": (
            "/home/mlops/.cache/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt"
        ),
        "expected_checkpoint_sha256": (
            "0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a"
        ),
        "require_checkpoint_attestation": True,
    }
    assert cfg["tabpfn26_output"].endswith("07_no_coordinate_tabpfn26")


def test_threshold_selection_keeps_valid_zero_all_positive_candidate():
    truth = np.array([0, 0, 1, 1], dtype=np.int8)
    score = np.zeros(4, dtype=np.float64)
    selected = experiment.select_f1_threshold(truth, score)
    assert selected["threshold"] == 0.0
    assert selected["predicted_positive_rate"] == 1.0
    assert selected["f1"] == pytest.approx(2 / 3)
    assert selected["tie_break"].startswith("max_f1")


def test_threshold_selection_uses_only_supplied_calibration_arrays():
    truth = np.array([0, 0, 1, 1], dtype=np.int8)
    selected = experiment.select_f1_threshold(truth, np.array([.1, .2, .8, .9]))
    assert selected["threshold"] == .8
    assert selected["accepted"] == 2
    # An unrelated evaluation score cannot affect a function that never receives it.
    assert "evaluation" not in selected


def _baseline_frames():
    train = pd.DataFrame({
        "POI_ID": ["a", "b", "c", "d"],
        "ASORT_LCLASDC": ["1.0", "1.0", "2.0", "2.0"],
        "ASORT_MLSFCDC": ["1.0", "1.0", "1.0", "1.0"],
        "ASORT_SDASDC": ["1.0", "1.0", "1.0", "2.0"],
        "region": ["A", "A", "A", "B"],
    })
    validation = pd.DataFrame({
        "POI_ID": ["v1", "v2"],
        "ASORT_LCLASDC": ["1.0", "9.0"],
        "ASORT_MLSFCDC": ["1.0", "9.0"],
        "ASORT_SDASDC": ["1.0", "9.0"],
        "region": ["A", "B"],
    })
    return train, validation


def test_train_joint_frequency_smoothing_and_unseen_fallback_are_explicit():
    train, validation = _baseline_frames()
    fitted = experiment.fit_joint_frequency_baseline(train, ["A", "B"], alpha=1.0)
    probability, fallback = experiment.predict_joint_frequency(fitted, validation)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # Seen tuple count [2, 0] -> Laplace [3/4, 1/4].
    np.testing.assert_allclose(probability[0], [0.75, 0.25])
    # Unseen tuple -> smoothed regional prior ([3,1] + 1) / 6.
    np.testing.assert_allclose(probability[1], [4 / 6, 2 / 6])
    np.testing.assert_array_equal(fallback, [False, True])
    assert fitted["train_rows"] == 4
    assert fitted["observed_tuple_count"] == 3


def test_tree_prediction_receives_exact_raw_frame_and_tabpfn_receives_float32():
    frame = pd.DataFrame({
        "ASORT_LCLASDC": pd.Series(["1.0", "2.0"], dtype=object),
        "ASORT_MLSFCDC": pd.Series(["3.0", "4.0"], dtype=object),
        "ASORT_SDASDC": pd.Series(["5.0", "6.0"], dtype=object),
    })
    seen = {}

    def tree_predict(values):
        seen["tree"] = values.copy()
        return np.array([.1, .2])

    score, contract = experiment._predict_fitted_model("decision_tree", tree_predict, frame)
    pd.testing.assert_frame_equal(seen["tree"], frame)
    np.testing.assert_array_equal(score, [.1, .2])
    assert contract.startswith("raw_validation_dataframe")

    def tabpfn_predict(values):
        seen["tabpfn"] = values.copy()
        return np.array([.3, .4])

    _, contract = experiment._predict_fitted_model("tabpfn_v2_6", tabpfn_predict, frame)
    assert seen["tabpfn"].dtype == np.float32
    np.testing.assert_array_equal(seen["tabpfn"], [[1, 3, 5], [2, 4, 6]])
    assert contract.startswith("exact_float32")


def test_weighted_average_precision_matches_expanded_sample_weight():
    truth = np.array([1, 0, 1, 0], dtype=np.int8)
    score = np.array([.8, .8, .4, .1], dtype=np.float64)
    weights = np.array([[1, 2, 3, 1], [2, 0, 1, 4]], dtype=np.uint16)
    observed = experiment.weighted_average_precision_replicates(truth, score, weights)
    expected = np.array([
        average_precision_score(truth, score, sample_weight=row) for row in weights
    ])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_common_partition_requires_matching_payload_and_exact_coverage(tmp_path):
    validation = pd.DataFrame({
        "POI_ID": ["a", "b", "c", "d"], "region": ["A", "B", "A", "B"],
    })
    payload = {
        "split": "validation", "calibration_ids": ["a", "b"],
        "evaluation_ids": ["c", "d"], "overlap": 0,
    }
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(payload)); second.write_text(json.dumps(payload))
    calibration, evaluation, metadata = experiment.load_common_partition(
        validation, [first, second], expected_calibration=2, expected_evaluation=2,
    )
    assert calibration == ["a", "b"] and evaluation == ["c", "d"]
    assert metadata["selection_rule"].startswith("thresholds use calibration")
    changed = dict(payload, evaluation_ids=["d", "c"])
    second.write_text(json.dumps(changed))
    with pytest.raises(experiment.NoCoordinateError, match="payloads differ"):
        experiment.load_common_partition(
            validation, [first, second], expected_calibration=2, expected_evaluation=2,
        )


def test_common_partition_rejects_bytes_changed_after_snapshot(tmp_path):
    validation = pd.DataFrame({
        "POI_ID": ["a", "b", "c", "d"], "region": ["A", "B", "A", "B"],
    })
    payload = {
        "split": "validation", "calibration_ids": ["a", "b"],
        "evaluation_ids": ["c", "d"], "overlap": 0,
    }
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(payload)); second.write_text(json.dumps(payload))
    snapshot = experiment._snapshot_files([first, second])
    first.write_text(json.dumps({
        **payload, "calibration_ids": ["c", "d"], "evaluation_ids": ["a", "b"],
    }))
    with pytest.raises(experiment.NoCoordinateError, match="changed"):
        experiment.load_common_partition(
            validation, [first, second], expected_calibration=2, expected_evaluation=2,
            snapshot=snapshot,
        )


def test_prediction_shard_validator_rejects_feature_or_identity_changes(tmp_path):
    validation = pd.DataFrame({
        "POI_ID": ["a", "b"], "region": ["A", "B"],
    })
    for seed in experiment.SEEDS:
        frame = pd.DataFrame({
            "model": "fixture", "ablation": "no_coordinates", "seed": seed,
            "region": "A", "attack_condition": "clean",
            "features": "|".join(experiment.FEATURES),
            "POI_ID": ["a", "b"], "y_true": [1, 0],
            "y_score": [.8, .2], "y_pred": [1, 0],
        })
        frame.to_parquet(tmp_path / f"fixture_no_coordinates_{seed}_A_clean.parquet", index=False)
    observed, paths = experiment._read_prediction_shards(
        tmp_path, ["fixture"], validation, ["A"],
    )
    assert len(paths) == 3 and len(observed) == 6
    path = paths[0]
    changed = pd.read_parquet(path)
    changed["features"] = "wrong"
    changed.to_parquet(path, index=False)
    with pytest.raises(experiment.NoCoordinateError, match="identity differs"):
        experiment._read_prediction_shards(tmp_path, ["fixture"], validation, ["A"])


@pytest.mark.parametrize("column", ["y_true", "y_pred"])
def test_prediction_shard_validator_rejects_fractional_binary_values(tmp_path, column):
    validation = pd.DataFrame({"POI_ID": ["a", "b"], "region": ["A", "B"]})
    for seed in experiment.SEEDS:
        frame = pd.DataFrame({
            "model": "fixture", "ablation": "no_coordinates", "seed": seed,
            "region": "A", "attack_condition": "clean",
            "features": "|".join(experiment.FEATURES),
            "POI_ID": ["a", "b"], "y_true": [1, 0],
            "y_score": [.8, .2], "y_pred": [1, 0],
        })
        frame[column] = [1.9, .9]
        frame.to_parquet(
            tmp_path / f"fixture_no_coordinates_{seed}_A_clean.parquet", index=False,
        )
    with pytest.raises(experiment.NoCoordinateError, match="values/truth differ"):
        experiment._read_prediction_shards(tmp_path, ["fixture"], validation, ["A"])


def test_input_snapshot_binds_exact_read_and_detects_later_mutation(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    snapshot = experiment._snapshot_files([path])
    assert experiment._bound_bytes(path, snapshot) == b"first"
    path.write_bytes(b"second")
    with pytest.raises(experiment.NoCoordinateError, match="changed"):
        experiment._bound_bytes(path, snapshot)
    with pytest.raises(experiment.NoCoordinateError, match="changed"):
        experiment._assert_snapshot_unchanged(snapshot)


def test_bound_clean_loader_materializes_only_requested_features_and_rejects_swap(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = pd.DataFrame({
        "POI_ID": ["tr-a", "tr-b", "va-a", "va-b", "te-a", "te-b"],
        "region": ["A", "B", "A", "B", "A", "B"],
        "split": ["train", "train", "validation", "validation", "test", "test"],
    })
    clean = manifest[["POI_ID", "region"]].copy()
    clean["X_COORD"] = [1, 2, 3, 4, 999999, 999998]
    clean["Y_COORD"] = [5, 6, 7, 8, 999997, 999996]
    clean["ASORT_LCLASDC"] = "1.0"
    clean["ASORT_MLSFCDC"] = "2.0"
    clean["ASORT_SDASDC"] = "3.0"
    manifest_path, clean_path = data_dir / "sample_manifest.csv", data_dir / "poi_data_region.csv"
    manifest.to_csv(manifest_path, index=False)
    clean.to_csv(clean_path, index=False)
    report_path = data_dir / "report.json"
    _write_json(report_path, {"outputs": {
        "poi_data_region.csv": {"sha256": _sha(clean_path)},
        "sample_manifest.csv": {"sha256": _sha(manifest_path)},
    }})
    paths = [clean_path, manifest_path, report_path]
    snapshot = experiment._snapshot_files(paths)
    observed, selected_manifest = experiment._load_clean_splits_bound(
        data_dir, ("train", "validation"), snapshot,
    )
    assert observed["POI_ID"].tolist() == ["tr-a", "tr-b", "va-a", "va-b"]
    assert selected_manifest["POI_ID"].tolist() == observed["POI_ID"].tolist()
    assert not observed.astype(str).apply(lambda column: column.str.contains("99999")).any().any()
    clean_path.write_bytes(clean_path.read_bytes().replace(b"tr-a", b"XX-a"))
    with pytest.raises(experiment.NoCoordinateError, match="changed"):
        experiment._load_clean_splits_bound(
            data_dir, ("train", "validation"), snapshot,
        )


def test_aggregate_config_parse_is_bound_to_snapshot(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(Path("configs/stages/07_no_coordinate_comparison.yaml").read_bytes())
    snapshot = experiment._snapshot_files([path])
    path.write_text(path.read_text().replace("bootstrap_repeats: 2000", "bootstrap_repeats: 1"))
    with pytest.raises(experiment.NoCoordinateError, match="changed"):
        experiment.load_config(path, snapshot)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fresh_fixture(tmp_path):
    root = tmp_path / "fresh"
    data_dir = tmp_path / "data"
    source_root = tmp_path / "source_snapshot"
    for directory in (root / "predictions", root / "work", data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("poi_data_region.csv", b"clean"),
        ("sample_manifest.csv", b"manifest"),
        ("report.json", b"{}"),
    ):
        (data_dir / name).write_bytes(payload)
    partitions = []
    for index in (1, 2):
        path = tmp_path / f"partition{index}" / "calibration_ids.json"
        path.parent.mkdir()
        path.write_bytes(b'{"same":true}\n')
        partitions.append(path)
    source_files = {
        "src/poi/no_coordinate_experiment.py": b"frozen source\n",
        "configs/stages/synthetic.yaml": b"frozen config\n",
    }
    source_hashes = {}
    source_catalog = {}
    for relative, payload in source_files.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = _sha(path)
        source_hashes[f"/remote/project/{relative}"] = digest
        source_catalog[relative] = digest
    _write_json(source_root / "verified_after_run_source_manifest.json", {
        "capture": "synthetic completed fit", "files": source_catalog,
    })
    (root / "config.yaml").write_bytes(source_files["configs/stages/synthetic.yaml"])
    input_hashes = {
        f"/remote/data/{name}": _sha(data_dir / name)
        for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
    }
    input_hashes.update({
        f"/remote/partition{index}/calibration_ids.json": _sha(path)
        for index, path in enumerate(partitions, 1)
    })
    fingerprint = experiment._canonical_sha({
        "format": "poi-no-coordinate-fit-identity-v1", "mode": "trees",
        "models": ("decision_tree",), "features": experiment.FEATURES,
        "seeds": experiment.SEEDS, "source_sha256": source_hashes,
        "input_sha256": input_hashes,
    })
    _write_json(root / "input_identity.json", {
        "format": "poi-no-coordinate-fit-identity-v1",
        "fingerprint_sha256": fingerprint, "source_sha256": source_hashes,
        "input_sha256": input_hashes,
    })
    validation = pd.DataFrame({"POI_ID": ["a", "b"], "region": ["A", "B"]})
    prediction_hashes, provenance_hashes = {}, {}
    for seed in experiment.SEEDS:
        stem = f"decision_tree_no_coordinates_{seed}_A"
        prediction = root / "predictions" / f"{stem}_clean.parquet"
        pd.DataFrame({
            "model": "decision_tree", "ablation": "no_coordinates", "seed": seed,
            "region": "A", "attack_condition": "clean",
            "features": "|".join(experiment.FEATURES), "POI_ID": ["a", "b"],
            "y_true": [1, 0], "y_score": [.8, .2], "y_pred": [1, 0],
        }).to_parquet(prediction, index=False)
        prediction_sha = _sha(prediction)
        provenance = root / "work" / f"{stem}_fit_provenance.json"
        _write_json(provenance, {
            "format": "poi-no-coordinate-actual-fit-provenance-v1",
            "model": "decision_tree", "ablation": "no_coordinates", "seed": seed,
            "region": "A", "attack_condition": "clean",
            "features": "|".join(experiment.FEATURES),
            "fit_attestation": {
                "format": "poi-tree-fit-attestation-v1", "model_id": "decision_tree",
                "seed": seed, "feature_order": list(experiment.FEATURES),
                "train_rows": 2, "truth_sha256": "a" * 64,
                "sample_weight_sha256": "b" * 64,
                "transformer_fitted_state_sha256": "c" * 64,
                "estimator_fitted_state_sha256": "d" * 64,
            },
            "prediction_sha256": prediction_sha,
            "source_fingerprint_sha256": fingerprint,
            "train_rows": 2, "validation_rows": 2,
            "validation_poi_order_sha256": experiment._ordered_ids_sha(["a", "b"]),
        })
        provenance_sha = _sha(provenance)
        marker = root / "work" / f"{stem}.json"
        _write_json(marker, {
            "fingerprint_sha256": fingerprint,
            "files_sha256": {
                str(prediction.relative_to(root)): prediction_sha,
                str(provenance.relative_to(root)): provenance_sha,
            },
        })
        prediction_hashes[str(prediction)] = prediction_sha
        provenance_hashes[str(provenance)] = provenance_sha
    _write_json(root / "metadata.json", {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "scientific_completion": False,
        "format": "poi-no-coordinate-model-shards-v1", "mode": "trees",
        "models": ["decision_tree"], "features": list(experiment.FEATURES),
        "seeds": list(experiment.SEEDS), "regions": ["A"], "fit_count": 3,
        "train_rows": 2, "validation_rows": 2, "test_rows_read": 0,
        "test_features_read": 0, "source_sha256": source_hashes,
        "input_sha256": input_hashes, "prediction_sha256": prediction_hashes,
        "provenance_sha256": provenance_hashes,
    })
    _write_json(root / "_GENERATED_PENDING_REVIEW.json", {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "fit_count": 3,
        "test_rows_read": 0, "metadata_sha256": _sha(root / "metadata.json"),
    })
    return root, data_dir, source_root, partitions, validation


def test_fresh_artifact_authenticates_markers_provenance_and_source_snapshot(tmp_path):
    root, data_dir, source_root, partitions, validation = _fresh_fixture(tmp_path)
    paths = experiment._fresh_artifact_paths(
        root, ["decision_tree"], ["A"], source_root,
    )
    paths.extend(data_dir / name for name in (
        "poi_data_region.csv", "sample_manifest.csv", "report.json",
    ))
    snapshot = experiment._snapshot_files(paths)
    result = experiment._validate_fresh_artifact(
        root, ["decision_tree"], validation, ["A"], data_dir, source_root, snapshot,
        expected_mode="trees", expected_train_rows=2,
        expected_partition_sha256=_sha(partitions[0]),
    )
    assert result["fit_count"] == 3
    marker = root / "work" / "decision_tree_no_coordinates_42_A.json"
    value = json.loads(marker.read_text())
    value["fingerprint_sha256"] = "0" * 64
    _write_json(marker, value)
    changed_snapshot = experiment._snapshot_files(paths)
    with pytest.raises(experiment.NoCoordinateError, match="provenance differs"):
        experiment._validate_fresh_artifact(
            root, ["decision_tree"], validation, ["A"], data_dir, source_root,
            changed_snapshot, expected_mode="trees", expected_train_rows=2,
            expected_partition_sha256=_sha(partitions[0]),
        )
    marker.unlink()
    with pytest.raises(experiment.NoCoordinateError, match="absent"):
        experiment._snapshot_files(paths)


def _tar_member(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_imported_artifact_binds_old_markers_identity_and_executed_source(tmp_path):
    from poi.clean_ablation_subset_report import marker_fingerprint

    root, data_dir = tmp_path / "imported", tmp_path / "data"
    (root / "predictions").mkdir(parents=True)
    (root / "work").mkdir()
    data_dir.mkdir()
    (data_dir / "poi_data_region.csv").write_bytes(b"clean")
    (data_dir / "sample_manifest.csv").write_bytes(b"manifest")
    source_code, factory_code = b"old ablation source\n", b"old factory source\n"
    archive_path = tmp_path / "executed_source.tar"
    with tarfile.open(archive_path, "w") as archive:
        _tar_member(archive, "src/poi/feature_ablation.py", source_code)
        _tar_member(archive, "src/poi/selected_explain.py", factory_code)
    cfg = {
        "model_ids": ["tabpfn_v2_5"], "seeds": list(experiment.SEEDS),
        "feature_sets": {"no_coordinates": list(experiment.FEATURES)},
    }
    clean_sha, manifest_sha = _sha(data_dir / "poi_data_region.csv"), _sha(
        data_dir / "sample_manifest.csv"
    )
    fingerprint = marker_fingerprint(
        cfg, clean_sha, manifest_sha, hashlib.sha256(source_code).hexdigest(),
        hashlib.sha256(factory_code).hexdigest(),
    )
    _write_json(root / "input_identity.json", {"sha256": fingerprint})
    _write_json(root / "metadata.json", {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "test_used": False,
        "attack_sha256": None, "config": cfg, "clean_sha256": clean_sha,
    })
    for seed in experiment.SEEDS:
        stem = f"tabpfn_v2_5_no_coordinates_{seed}_A"
        prediction = root / "predictions" / f"{stem}_clean.parquet"
        prediction.write_bytes(f"prediction {seed}\n".encode())
        _write_json(root / "work" / f"{stem}.json", {
            "fingerprint": fingerprint,
            "metrics": [{
                "model": "tabpfn_v2_5", "ablation": "no_coordinates",
                "seed": seed, "region": "A", "attack_condition": "clean",
                "features": "|".join(experiment.FEATURES),
            }],
            "files": {str(prediction.relative_to(root)): _sha(prediction)},
        })
    validation = pd.DataFrame({"POI_ID": ["a", "b"], "region": ["A", "B"]})
    paths = experiment._imported_artifact_paths(
        root, ["tabpfn_v2_5"], ["A"], archive_path,
    )
    paths.extend(data_dir / name for name in ("poi_data_region.csv", "sample_manifest.csv"))
    snapshot = experiment._snapshot_files(paths)
    result = experiment._validate_imported_artifact(
        root, ["tabpfn_v2_5"], validation, ["A"], data_dir, archive_path, snapshot,
    )
    assert result["fit_count"] == 3
    assert result["per_fit_attestation"].startswith("unavailable")
    marker = root / "work" / "tabpfn_v2_5_no_coordinates_42_A.json"
    value = json.loads(marker.read_text())
    value["fingerprint"] = "0" * 64
    _write_json(marker, value)
    changed_snapshot = experiment._snapshot_files(paths)
    with pytest.raises(experiment.NoCoordinateError, match="marker differs"):
        experiment._validate_imported_artifact(
            root, ["tabpfn_v2_5"], validation, ["A"], data_dir, archive_path,
            changed_snapshot,
        )


def test_comparison_config_declares_separate_outputs_and_no_test_input():
    cfg = yaml.safe_load(Path("configs/stages/07_no_coordinate_comparison.yaml").read_text())
    assert cfg["tree_output"] != cfg["tabpfn26_output"] != cfg["aggregate_output"]
    assert cfg["bootstrap_repeats"] == 2000
    assert cfg["tree_fit_source_snapshot"].endswith("tree_revision2_source")
    assert cfg["tabpfn26_fit_source_snapshot"].endswith("tabpfn26_source")
    assert "clean_ablation_subset_report.py" in experiment.AGGREGATE_SOURCE_MODULES
    serialized = json.dumps(cfg).lower()
    assert "test_file" not in serialized and "test_data" not in serialized
