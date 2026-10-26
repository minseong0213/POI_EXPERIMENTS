import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poi import unified_clean_fit as module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_threshold_rejects_fractional_truth_before_integer_cast():
    with pytest.raises(ValueError, match="exact-binary"):
        module.select_f1_threshold([0.0, 1.0, 0.9], [0.1, 0.8, 0.7])

    selected = module.select_f1_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert selected["threshold"] == pytest.approx(0.8)
    assert selected["f1"] == pytest.approx(1.0)
    assert selected["decision_rule"] == "score_greater_than_or_equal_to_threshold"


def test_prediction_input_preserves_tree_precision_and_tabpfn_float32_contract():
    frame = pd.DataFrame({
        "X_COORD": [127.1234567890123],
        "Y_COORD": [36.9876543210987],
    })
    tree, tree_contract = module._prediction_input("random_forest", frame, module.COORDINATE_FEATURES)
    tabpfn, tabpfn_contract = module._prediction_input("tabpfn_v2_6", frame, module.COORDINATE_FEATURES)

    assert isinstance(tree, pd.DataFrame)
    assert tree.dtypes.tolist() == frame.dtypes.tolist()
    assert tree.iloc[0, 0] == frame.iloc[0, 0]
    assert tree_contract == "raw_validation_dataframe_original_float_precision"
    assert tabpfn.dtype == np.float32
    assert tabpfn_contract == "exact_float32_matrix_for_tabpfn"


def test_tabpfn81_string_checkpoint_real_cpu_fit_state_roundtrip(tmp_path):
    """Regression for TabPFN 8.1's JSON serialization of constructor params."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    checkpoint = Path("/home/mlops/.cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt")
    assert checkpoint.is_file()
    matrix = np.zeros((34, 5), dtype=np.float32)
    matrix[:, 0] = np.linspace(126, 129, len(matrix))
    matrix[:, 1] = np.linspace(35, 38, len(matrix))
    matrix[:, 2] = np.arange(len(matrix)) % 8
    matrix[:, 3] = np.arange(len(matrix)) % 5
    matrix[:, 4] = np.arange(len(matrix)) % 16
    truth = (np.arange(len(matrix)) % 2).astype(np.int8)
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device="cpu",
        n_estimators=1,
        random_state=42,
        model_path=str(checkpoint),
        categorical_features_indices=[2, 3, 4],
        show_progress_bar=False,
        memory_saving_mode="auto",
    )
    model.fit(matrix, truth)
    probe = matrix[:8]
    original = np.asarray(model.predict_proba(probe)[:, 1], dtype=np.float64)
    state = tmp_path / "real_cpu.tabpfn_fit"
    model.save_fit_state(state)
    restored = TabPFNClassifier.load_from_fit_state(state, device="cpu")
    replay = np.asarray(restored.predict_proba(probe)[:, 1], dtype=np.float64)

    assert state.is_file() and state.stat().st_size > 0
    assert np.array_equal(original, replay)


def test_source_gate_binds_manifest_verdict_and_current_bytes(tmp_path):
    source = tmp_path / "producer.py"
    config = tmp_path / "fit.yaml"
    protocol = tmp_path / "protocol.json"
    source.write_text("VALUE = 1\n")
    config.write_text("frozen: true\n")
    protocol.write_text("{}\n")
    manifest = tmp_path / "freeze.json"
    verdict = tmp_path / "verdict.json"
    cfg = {
        "fit_source_files": [str(source)],
        "_config_path": str(config),
        "protocol_file": str(protocol),
        "fit_source_freeze_manifest": str(manifest),
        "fit_source_review_verdict": str(verdict),
    }
    module.freeze_source(cfg)
    _write_json(verdict, {
        "status": "PASS",
        "review_scope": "unified_clean_fit_source_before_fits",
        "reviewed_source_manifest_sha256": module.sha256(manifest),
    })
    assert module.verify_source_gate(cfg)["manifest_sha256"] == module.sha256(manifest)

    source.write_text("VALUE = 2\n")
    with pytest.raises(module.UnifiedFitError, match="changed"):
        module.verify_source_gate(cfg)


def test_full_v3_job_emits_51_persisted_states_and_calibration_contract(tmp_path, monkeypatch):
    regions = [f"R{index:02d}" for index in range(17)]
    train_rows, validation_rows = [], []
    for index, region in enumerate(regions):
        for copy in range(2):
            common = {
                "region": region,
                "X_COORD": f"{127 + index * 0.001 + copy * 1e-8:.8f}",
                "Y_COORD": f"{36 + index * 0.001 + copy * 1e-8:.8f}",
                "ASORT_LCLASDC": str(index % 8),
                "ASORT_MLSFCDC": str(index % 5),
                "ASORT_SDASDC": str(index % 16),
            }
            train_rows.append({"POI_ID": f"T{index:02d}{copy}", "split": "train", **common})
            validation_rows.append({"POI_ID": f"V{index:02d}{copy}", "split": "validation", **common})
    clean = pd.DataFrame([*train_rows, *validation_rows])
    calibration_ids = [f"V{index:02d}0" for index in range(17)]
    evaluation_ids = [f"V{index:02d}1" for index in range(17)]
    partition_payload = {
        "split": "validation",
        "calibration_ids": calibration_ids,
        "evaluation_ids": evaluation_ids,
        "overlap": 0,
    }
    partition_a, partition_b = tmp_path / "partition_a.json", tmp_path / "partition_b.json"
    _write_json(partition_a, partition_payload)
    _write_json(partition_b, partition_payload)
    partition_sha = module.sha256(partition_a)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json"):
        (data_dir / name).write_text(f"fixture-{name}\n")
    tree_config = tmp_path / "trees.yaml"
    tree_config.write_text("models: {}\n")
    checkpoint = tmp_path / "v3.ckpt"
    checkpoint.write_bytes(b"official-fixture")
    manifest, verdict, config = tmp_path / "freeze.json", tmp_path / "verdict.json", tmp_path / "config.yaml"
    manifest.write_text("{}\n")
    verdict.write_text("{}\n")
    config.write_text("fixture: true\n")
    gate = {
        "manifest_path": str(manifest),
        "manifest_sha256": module.sha256(manifest),
        "verdict_path": str(verdict),
        "verdict_sha256": module.sha256(verdict),
    }

    class FakeFactory:
        def __init__(self, model, cfg, train):
            self.model = model
            self.cfg = cfg
            self.train = train
            self.last_fit_attestation = {}

        def fit(self, features, truth):
            train_matrix = np.asarray(module._matrix(self.train, features), dtype=np.float32)
            labels = np.asarray(truth, dtype=np.int8)
            self.last_fit_attestation = {
                "model_id": self.model,
                "seed": self.cfg["seed"],
                "feature_order": list(features),
                "official_checkpoint_path": str(checkpoint),
                "official_checkpoint_sha256": module.sha256(checkpoint),
                "train_matrix_sha256": hashlib.sha256(train_matrix.tobytes()).hexdigest(),
                "truth_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                "probe_indices": [0, 1],
                "probe_matrix_sha256": "a" * 64,
                "probe_prediction_sha256": "b" * 64,
                "probe_repeat_prediction_sha256": "b" * 64,
            }
            region_index = int(np.flatnonzero(labels)[0] // 2)
            def predict(values):
                size = len(values)
                score = np.full(size, 0.05, dtype=np.float64)
                score[2 * region_index:2 * region_index + 2] = 0.95
                return score
            return predict

        def persist_last_fit_state(self, path):
            path.write_bytes(f"state-{self.cfg['seed']}".encode())
            self.last_fit_attestation.update({
                "fitted_state_path": str(path.resolve()),
                "fitted_state_sha256": module.sha256(path),
                "fitted_state_size_bytes": path.stat().st_size,
                "fitted_state_format": "tabpfn_official_fit_state_v1",
                "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
                "restored_probe_prediction_sha256": "b" * 64,
            })
            return dict(self.last_fit_attestation)

    monkeypatch.setattr(module, "verify_source_gate", lambda cfg: gate)
    monkeypatch.setattr(module, "load_clean_splits", lambda *args, **kwargs: (clean.copy(), pd.DataFrame()))
    monkeypatch.setattr(module, "ModelFactory", FakeFactory)
    cfg = {
        "run_root": str(tmp_path / "run"),
        "outputs": {
            "coordinate_trees": "trees",
            "coordinate_tabpfn26": "tabpfn26",
            "full_tabpfn3_states": "fullv3",
        },
        "data_dir": str(data_dir),
        "tree_config": str(tree_config),
        "partition_files": [str(partition_a), str(partition_b)],
        "expected_partition_sha256": partition_sha,
        "expected_train_rows": 34,
        "expected_validation_rows": 34,
        "expected_calibration_rows": 17,
        "expected_evaluation_rows": 17,
        "expected_regions": 17,
        "repeat_prediction_absolute_tolerance": 1e-12,
        "tabpfn_v3": {
            "device": "cpu",
            "n_estimators": 8,
            "official_checkpoint": str(checkpoint),
            "expected_checkpoint_sha256": module.sha256(checkpoint),
            "require_checkpoint_attestation": True,
        },
        "_config_path": str(config),
    }
    module.run_fit(cfg, "full-tabpfn3-states")
    output = tmp_path / "run" / "fullv3"
    contract = json.loads((output / "replay_contract.json").read_text())
    assert contract["fit_count"] == 51
    assert len(contract["fits"]) == 51
    assert len(list((output / "states").glob("*.tabpfn_fit"))) == 51
    assert pd.read_csv(output / "tables" / "calibration_thresholds.csv").shape[0] == 51
    assert pd.read_parquet(output / "tables" / "calibration_scores.parquet").shape[0] == 51 * 17
    assert contract["partition"]["sha256"] == partition_sha
    assert all(row["original_probe_score_sha256"] == row["restored_probe_score_sha256"] for row in contract["fits"])
