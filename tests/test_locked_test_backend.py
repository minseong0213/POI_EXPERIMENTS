import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import poi.locked_test_backend as backend
from poi import adversarial as adv
from poi import defense_pipeline as defense
from poi.data import CATEGORICAL, FEATURES, NUMERIC, sha256


def _clean_frames():
    rows = [
        ("tr-a1", "A", "train", 0.0, 0.0, 1, 10, 100),
        ("tr-a2", "A", "train", 0.2, 0.2, 1, 10, 100),
        ("tr-b1", "B", "train", 0.8, 0.8, 2, 20, 200),
        ("tr-b2", "B", "train", 1.0, 1.0, 2, 20, 200),
        ("te-a", "A", "test", 0.1, 0.1, 1, 10, 100),
        ("te-b", "B", "test", 0.9, 0.9, 2, 20, 200),
    ]
    columns = ["POI_ID", "region", "split", *FEATURES]
    frame = pd.DataFrame(rows, columns=columns)
    return (
        frame.loc[frame.split.eq("train")].reset_index(drop=True),
        frame.loc[frame.split.eq("test")].reset_index(drop=True),
    )


class _Boundary:
    def validate(self, coordinates, expected_regions):
        expected = np.asarray(expected_regions, dtype=object)
        return {
            "matches": np.ones(len(expected), dtype=bool),
            "status": np.full(len(expected), "consensus_match", dtype=object),
            "consensus_region": expected.copy(),
            "source_labels": {"official2020": expected.copy(), "official2023": expected.copy()},
        }


def _attack_config():
    return {
        "surrogate_seed": 42,
        "generation_seeds": [11],
        "attack_batch_size": 8,
        "numeric_budgets_m": [10],
        "categorical_budgets_l0": [1],
        "mixed_budgets": [{"name": "low", "numeric_m": 10, "categorical_l0": 1}],
        "distance_validation": {
            "metric": adv.DISTANCE_METRIC,
            "numeric_budget_tolerance_m": 0.05,
            "categorical_only_tolerance_m": 0.000001,
        },
        "attacks": {
            "fgsm": {"steps": 1},
            "pgd": {"steps": 2},
            "cw_l2": {"steps": 2},
            "capgd": {"steps": 2},
            "pcaa": {"steps": 2, "candidate_draws": 2},
            "moeva": {"population": 2, "generations": 2},
        },
    }


def test_attack_materializer_executes_every_algorithm_branch_without_retraining(
    tmp_path, monkeypatch
):
    train, test = _clean_frames()
    cfg = _attack_config()
    specs = adv.attack_condition_specs(cfg)
    conditions = tuple(spec["condition"] for spec in specs)
    dimensions = backend.Dimensions(("A", "B"), (7,), (11,), conditions, len(test))
    surrogate = tmp_path / "surrogate"
    surrogate.mkdir()
    inputs = backend.FrozenInputs(
        attack_config=tmp_path / "attack.yaml",
        boundary_config=tmp_path / "boundary.yaml",
        generation_request=tmp_path / "request.json",
        generation_identity=tmp_path / "identity.json",
        surrogate_directory=surrogate,
        defense_config=tmp_path / "defense.yaml",
        detection_thresholds=tmp_path / "thresholds.csv",
        fixed_replay_contract=tmp_path / "replay.json",
        proposed_checkpoints={},
        ure_checkpoints={},
        fixed_checkpoint=tmp_path / "fixed.ckpt",
        device="cpu",
    )
    identity = {
        "generation_request_sha256": "a" * 64,
        "generation_identity_sha256": "b" * 64,
        "surrogate_checkpoint_sha256": "c" * 64,
    }
    encoder = {
        "regions": ["A", "B"],
        "category_values": {
            name: sorted(train[name].astype(float).unique().tolist()) for name in CATEGORICAL
        },
        "numeric_mean": train[NUMERIC].mean().to_numpy(np.float32),
        "numeric_std": train[NUMERIC].std(ddof=0).to_numpy(np.float32),
    }
    monkeypatch.setattr(
        adv, "load_surrogate", lambda *args, **kwargs: (object(), encoder, {}, surrogate / "surrogate.pt")
    )
    monkeypatch.setattr(
        adv, "train_surrogate", lambda *args, **kwargs: pytest.fail("surrogate retrained")
    )
    monkeypatch.setattr(
        backend.AdministrativeBoundaryValidator,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: _Boundary()),
    )

    def predict(model, coords, categories, encoder, device, batch_size):
        return np.zeros(len(coords), dtype=np.int64), np.zeros((len(coords), 2), dtype=float)

    def numeric(method, coords, categories, labels, model, encoder, budget, cfg, device):
        queries = {
            "fgsm": 1,
            "pgd": cfg["attacks"]["pgd"]["steps"] + 1,
            "cw_l2": cfg["attacks"]["cw_l2"]["steps"],
            "capgd": cfg["attacks"]["capgd"]["steps"] + 1,
        }[method]
        return coords.copy(), np.full(len(coords), queries, dtype=np.int64)

    def category_exact(coords, categories, labels, model, encoder, tuples, vectors, budget, cfg, device):
        return categories.copy(), np.full(len(coords), len(tuples), dtype=np.int64)

    def pcaa(coords, categories, labels, model, encoder, tuples, vectors, budget, cfg, device):
        queries = cfg["attacks"]["pcaa"]["steps"] + min(
            cfg["attacks"]["pcaa"]["candidate_draws"], len(tuples)
        )
        return categories.copy(), np.full(len(coords), queries, dtype=np.int64)

    def evolution(coords, categories, labels, model, encoder, tuples, vectors,
                  numeric_budget, categorical_budget, cfg, device, rng_seed):
        queries = cfg["attacks"]["moeva"]["population"] * cfg["attacks"]["moeva"]["generations"]
        return coords.copy(), categories.copy(), np.full(len(coords), queries, dtype=np.int64)

    monkeypatch.setattr(adv, "predict", predict)
    monkeypatch.setattr(adv, "numeric_attack", numeric)
    monkeypatch.setattr(adv, "categorical_exact", category_exact)
    monkeypatch.setattr(adv, "categorical_pcaa", pcaa)
    monkeypatch.setattr(adv, "mixed_evolution", evolution)

    output = tmp_path / "output"
    output.mkdir()
    paths, manifest = backend._materialize_attacks(
        train=train,
        test=test,
        cfg=cfg,
        specs=specs,
        identity=identity,
        inputs=inputs,
        dimensions=dimensions,
        output=output,
    )

    assert len(paths) == len(conditions)
    assert len(manifest["shards"]) == len(conditions)
    observed = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert set(observed.attack_condition) == set(conditions)
    assert observed.groupby("attack_condition").POI_ID.apply(list).to_dict() == {
        condition: ["te-a", "te-b"] for condition in sorted(conditions)
    }
    assert observed.primary_evaluation_eligible.all()
    categorical = observed.loc[observed.attack_method.isin(["category_exact", "pcaa"])]
    assert categorical.numeric_budget_m.eq(0).all()
    assert categorical.numeric_budget_was_implicit_zero.all()


def test_frozen_primary_thresholds_are_loaded_without_selection(tmp_path, monkeypatch):
    dimensions = backend.Dimensions(("A",), (7,), (11,), ("attack",), 2)
    path = tmp_path / "thresholds.csv"
    pd.DataFrame([
        {"seed": 7, "detector": backend.PROPOSED, "clean_retention_target": .95, "threshold": .4},
        {"seed": 7, "detector": backend.URE, "clean_retention_target": .95, "threshold": .6},
        {"seed": 7, "detector": backend.PROPOSED, "clean_retention_target": .90, "threshold": .3},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(
        defense,
        "select_detection_threshold",
        lambda *args, **kwargs: pytest.fail("test-time threshold selection"),
    )
    monkeypatch.setattr(
        defense,
        "RegionModelFactory",
        lambda *args, **kwargs: pytest.fail("test-time TabPFN refit"),
    )

    assert backend._load_thresholds(path, dimensions) == {
        7: {backend.PROPOSED: .4, backend.URE: .6}
    }

    broken = pd.read_csv(path).loc[lambda value: value.detector.ne(backend.URE)]
    broken.to_csv(path, index=False)
    with pytest.raises(backend.LockedTestBackendError, match="coverage is incomplete"):
        backend._load_thresholds(path, dimensions)


def _runtime(device="cpu"):
    import tabpfn
    import torch

    return {
        "tabpfn_version": tabpfn.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": device,
        "restored_on_device": device,
    }


class _StateModel:
    def __init__(self, region):
        self.region = region
        self.calls = 0

    def predict_proba(self, matrix):
        self.calls += 1
        x = np.clip(np.asarray(matrix, dtype=float)[:, 0], 0, 1)
        positive = 1 - x if self.region == "A" else x
        return np.column_stack([1 - positive, positive])


def _fit_attestation(seed, region, checkpoint, train, state_path, model):
    matrix = np.ascontiguousarray(defense._matrix(train), dtype=np.float32)
    probe_indices = list(range(len(train)))
    probe = np.ascontiguousarray(matrix[probe_indices], dtype=np.float32)
    prediction = np.ascontiguousarray(model.predict_proba(probe)[:, 1], dtype=np.float64)
    prediction_sha = hashlib.sha256(prediction.tobytes()).hexdigest()
    truth = np.ascontiguousarray(
        train.region.eq(region).astype(int).to_numpy(), dtype=np.int8
    )
    return {
        "format": "poi-tabpfn25-fit-replay-v1",
        "model_id": "tabpfn_v2_5",
        "seed": seed,
        "model_seed": seed,
        "region": region,
        "official_checkpoint_path": str(checkpoint),
        "official_checkpoint_sha256": sha256(checkpoint),
        "feature_order": list(FEATURES),
        "categorical_feature_indices": [2, 3, 4],
        "n_estimators": 2,
        "classification_threshold": .5,
        "train_rows": len(train),
        "train_matrix_dtype": "float32",
        "train_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "train_poi_order_sha256": defense._ordered_identity_sha256(train, ["POI_ID"]),
        "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": "auto",
        "probe_indices": probe_indices,
        "probe_matrix_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
        "probe_prediction_sha256": prediction_sha,
        "probe_repeat_prediction_sha256": prediction_sha,
        "restored_probe_prediction_sha256": prediction_sha,
        "fitted_state_format": "tabpfn_official_fit_state_v1",
        "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
        "fitted_state_path": str(state_path),
        "fitted_state_sha256": sha256(state_path),
        "fitted_state_size_bytes": state_path.stat().st_size,
        **_runtime(),
    }


def test_frozen_state_loader_rejects_changed_restored_probe_prediction(tmp_path):
    train, _ = _clean_frames()
    state_path = tmp_path / "a.tabpfn_fit"
    state_path.write_bytes(b"frozen state")
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model = _StateModel("A")
    fit = _fit_attestation(7, "A", checkpoint, train, state_path, model)
    fit["restored_probe_prediction_sha256"] = "b" * 64

    with pytest.raises(backend.LockedTestBackendError, match="changes probe prediction"):
        backend._load_frozen_region_models(
            train=train,
            frozen_fits={(7, "A"): fit},
            seed=7,
            regions=("A",),
            device="cpu",
            state_loader=lambda path, device: model,
        )


class _Proposed:
    regions = ("A", "B")

    def __init__(self):
        self.calls = 0

    def predict(self, matrix):
        self.calls += 1
        matrix = np.asarray(matrix, dtype=float)
        p_b = np.clip(matrix[:, 0], 0, 1)
        return {
            "attack_probability": np.clip(np.abs(matrix[:, 0] - .5), 0, 1),
            "region_probability": np.column_stack([1 - p_b, p_b]),
            "repaired_matrix": matrix.copy(),
            "preprojection_numeric_out_of_bounds": np.zeros((len(matrix), 2), dtype=bool),
            "preprojection_invalid_category_tuple": np.zeros(len(matrix), dtype=bool),
        }


class _URE:
    def __init__(self):
        self.calls = 0

    def predict_proba(self, matrix):
        self.calls += 1
        attack = np.clip(np.asarray(matrix)[:, 1], 0, 1)
        return np.column_stack([1 - attack, attack])


def test_synthetic_end_to_end_evaluator_loads_frozen_states_once_and_preserves_contract(
    tmp_path, monkeypatch
):
    train, test = _clean_frames()
    dimensions = backend.Dimensions(("A", "B"), (7,), (11,), ("fgsm_m10",), len(test))
    fixed_checkpoint = tmp_path / "tabpfn.ckpt"
    fixed_checkpoint.write_bytes(b"frozen-tabpfn")
    checkpoint_sha = sha256(fixed_checkpoint)
    stage10 = tmp_path / "stage10"
    state_dir = stage10 / "work" / "fixed_downstream_fit_states"
    state_dir.mkdir(parents=True)
    state_models = {region: _StateModel(region) for region in dimensions.regions}
    state_paths = {}
    fits = []
    for index, region in enumerate(dimensions.regions):
        state_path = state_dir / f"modelseed7__region{index:02d}.tabpfn_fit"
        state_path.write_bytes(f"state-{region}".encode())
        state_paths[state_path] = state_models[region]
        fits.append(
            _fit_attestation(
                7, region, fixed_checkpoint, train, state_path, state_models[region]
            )
        )

    replay = {
        "format": "poi-fixed-downstream-replay-contract-v1",
        "model_id": "tabpfn_v2_5",
        "official_checkpoint_path": str(fixed_checkpoint),
        "official_checkpoint_sha256": checkpoint_sha,
        "fit_count": 2,
        "expected_fit_count": 2,
        "test_rows_read": 0,
        "model_seeds": [7],
        "regions": ["A", "B"],
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL),
        "n_estimators": 2,
        "classification_threshold": .5,
        "train_rows": len(train),
        "train_poi_order_sha256": defense._ordered_identity_sha256(train, ["POI_ID"]),
        "train_feature_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(defense._matrix(train), dtype=np.float32).tobytes()
        ).hexdigest(),
        "thresholds_sha256": "pending",
        "stage_config_sha256": "pending",
        "runtime_environment_sha256": "pending",
        "runtime": {
            "python": backend.platform.python_version(),
            "python_executable": backend.sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **_runtime(),
        },
        "fits": fits,
    }
    defense_path = tmp_path / "defense.yaml"
    defense_path.write_text(
        "seeds: [7]\ngeneration_seeds: [11]\nprimary_clean_retention: 0.95\n"
        "classification_threshold: 0.5\nfixed_region_model:\n  model_id: tabpfn_v2_5\n"
        "  n_estimators: 2\n  require_checkpoint_attestation: true\n"
    )
    thresholds_path = tmp_path / "thresholds.csv"
    pd.DataFrame([
        {"seed": 7, "detector": backend.PROPOSED, "clean_retention_target": .95, "threshold": .45},
        {"seed": 7, "detector": backend.URE, "clean_retention_target": .95, "threshold": .55},
    ]).to_csv(thresholds_path, index=False)
    replay["thresholds_sha256"] = sha256(thresholds_path)
    replay["stage_config_sha256"] = sha256(defense_path)
    runtime_path = stage10 / "runtime_environment.json"
    runtime_path.write_text(json.dumps({
        "format": "poi-stage10-runtime-environment-v1",
        "execution_scope": "stage10_gpu_model_fitting_and_validation_evaluation",
        **backend._current_runtime_environment("cpu"),
        "test_rows_read": 0,
    }))
    replay["runtime_environment_sha256"] = sha256(runtime_path)
    replay_path = stage10 / "fixed_downstream_replay_contract.json"
    replay_path.write_text(json.dumps(replay))
    proposed_path = tmp_path / "proposed.pt"
    ure_path = tmp_path / "ure.pt"
    rf_path = tmp_path / "ure.joblib"
    for path in (proposed_path, ure_path, rf_path):
        path.write_bytes(b"checkpoint")
    inputs = backend.FrozenInputs(
        attack_config=tmp_path / "attack.yaml",
        boundary_config=tmp_path / "boundary.yaml",
        generation_request=tmp_path / "request.json",
        generation_identity=tmp_path / "identity.json",
        surrogate_directory=tmp_path / "surrogate",
        defense_config=defense_path,
        detection_thresholds=thresholds_path,
        fixed_replay_contract=replay_path,
        proposed_checkpoints={7: proposed_path},
        ure_checkpoints={7: (ure_path, rf_path)},
        fixed_checkpoint=fixed_checkpoint,
        device="cpu",
        runtime_environment=runtime_path,
    )
    attack = test[["POI_ID", "region"]].copy()
    for feature in FEATURES:
        attack[f"{feature}_adv"] = test[feature]
    attack["generation_seed"] = 11
    attack["attack_condition"] = "fgsm_m10"
    attack["attack_method"] = "fgsm"
    attack["numeric_budget_m"] = 10.0
    attack["categorical_budget_l0"] = 0
    attack["source_model"] = adv.SURROGATE_ID
    attack["victim_model"] = adv.SURROGATE_ID
    attack["victim_evaluation_scope"] = "direct_white_box_source_model"
    attack["constraints_valid"] = [True, False]
    attack["label_preserved"] = [True, False]
    attack["attack_attempted"] = True
    attack["attack_changed_any"] = False
    attack["transfer_evaluated"] = False
    attack["targeted"] = False
    attack["valid_numeric_train_range"] = True
    attack["valid_category_codebook"] = True
    attack["valid_category_combo"] = True
    attack["within_budget"] = True
    attack["clean_boundary_status"] = "consensus_match"
    attack["attack_boundary_status"] = ["consensus_match", "label_mismatch"]
    attack["generation_identity_sha256"] = "a" * 64
    attack["surrogate_checkpoint_sha256"] = "b" * 64
    attack["query_count"] = 1
    attack["primary_evaluation_eligible"] = [True, False]
    attack["quarantine_reason"] = ["", "label_mismatch"]
    attack_path = tmp_path / "attack.parquet"
    attack.to_parquet(attack_path, index=False)
    proposed, ure = _Proposed(), _URE()
    monkeypatch.setattr(
        defense,
        "select_detection_threshold",
        lambda *args, **kwargs: pytest.fail("test-time threshold selection"),
    )
    monkeypatch.setattr(
        defense,
        "RegionModelFactory",
        lambda *args, **kwargs: pytest.fail("test-time TabPFN refit"),
    )

    output = tmp_path / "output"
    output.mkdir()
    loaded = []

    def load_state(path, device):
        loaded.append((path, device))
        return state_paths[path]

    predictions, detector_metrics, classification_metrics, state_count = backend._evaluate_defenses(
        train=train,
        test=test,
        attack_paths=[attack_path],
        inputs=inputs,
        dimensions=dimensions,
        output=output,
        model_loader=lambda inputs, cfg, seed: (proposed, ure),
        state_loader=load_state,
    )

    assert state_count == 2
    assert loaded == [(Path(fit["fitted_state_path"]), "cpu") for fit in fits]
    assert proposed.calls == 2 and ure.calls == 2
    assert len(predictions) == 2
    assert sum(row["rows"] for row in predictions) == 16
    assert len(detector_metrics) == 20
    assert len(classification_metrics) == 320
    modes = classification_metrics.groupby("detector").classification_mode.unique().to_dict()
    assert set(modes[backend.PROPOSED]) == set(backend.CLASSIFICATION_MODES)
    assert set(modes[backend.URE]) == set(backend.CLASSIFICATION_MODES[:3])
    assert not any("native" in value for value in modes[backend.URE])
    primary = classification_metrics.loc[
        classification_metrics.evaluation_scope.eq("primary_valid")
        & classification_metrics.attack_condition.eq("fgsm_m10")
    ]
    assert set(primary.total_rows) == {1}
    assert set(classification_metrics.loc[
        classification_metrics.classification_mode.eq("fixed_unmodified"), "detector_coverage"
    ]) == {1.0}


def test_metric_rows_mark_empty_accepted_population_undefined():
    observations = {
        "truth": np.array([[1, 0]], dtype=bool),
        "fixed": np.array([[.8, .2]]),
        "shared": np.array([[.7, .3]]),
        "native": np.array([[.6, .4]]),
        "projected": np.array([[.75, .25]]),
        "detector": {backend.PROPOSED: np.array([.9]), backend.URE: np.array([.9])},
    }
    rows = pd.DataFrame(backend._classification_rows(
        observations,
        model_seed=7,
        generation_seed=11,
        condition="attack",
        scope="all_intent_to_attack",
        mask=np.array([True]),
        regions=("A", "B"),
        thresholds={backend.PROPOSED: .1, backend.URE: .1},
    ))
    accepted = rows.loc[rows.classification_mode.eq("detector_accepted_only")]
    assert set(accepted.metric_defined) == {False}
    assert set(accepted.undefined_reason) == {"empty_population"}
    assert set(accepted.metric_population_rows) == {0}


def test_public_backend_validates_every_static_input_before_clean_data_access(
    tmp_path, monkeypatch
):
    dimensions = backend.Dimensions(("A",), (7,), (11,), ("attack",), 1)
    inputs = backend.FrozenInputs(
        attack_config=tmp_path / "attack.yaml",
        boundary_config=tmp_path / "boundary.yaml",
        generation_request=tmp_path / "request.json",
        generation_identity=tmp_path / "identity.json",
        surrogate_directory=tmp_path / "surrogate",
        defense_config=tmp_path / "defense.yaml",
        detection_thresholds=tmp_path / "thresholds.csv",
        fixed_replay_contract=tmp_path / "replay.json",
        proposed_checkpoints={},
        ure_checkpoints={},
        fixed_checkpoint=tmp_path / "fixed.ckpt",
        device="cpu",
    )
    monkeypatch.setattr(
        backend, "_resolve_inputs", lambda context: (tmp_path, dimensions, inputs)
    )
    monkeypatch.setattr(
        backend,
        "_prevalidate_static_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            backend.LockedTestBackendError("stale frozen bytes")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_numeric_clean_data",
        lambda *args, **kwargs: pytest.fail("clean/test data accessed before static validation"),
    )

    with pytest.raises(backend.LockedTestBackendError, match="stale frozen bytes"):
        backend.run_locked_test_backend({
            "project_root": str(tmp_path), "clean_dataset_dir": "data/sentinel"
        })


def test_unseen_locked_test_category_fails_closed_without_remapping():
    train, test = _clean_frames()
    test.loc[0, CATEGORICAL[1]] = 999
    original = test.loc[0, CATEGORICAL[1]]
    dimensions = backend.Dimensions(("A", "B"), (7,), (11,), ("attack",), len(test))

    with pytest.raises(
        backend.LockedTestBackendError, match="outside the frozen clean-train vocabulary"
    ):
        backend._validate_clean_contract(train, test, dimensions)
    assert test.loc[0, CATEGORICAL[1]] == original


def test_full_locked_design_cardinalities_include_every_poi_condition_and_seed():
    dimensions = backend.Dimensions(
        backend.REGIONS,
        backend.MODEL_SEEDS,
        backend.GENERATION_SEEDS,
        tuple(f"condition_{index:02d}" for index in range(32)),
        6800,
    )

    assert backend.expected_cardinalities(dimensions) == {
        "attack_cohorts": 96,
        "attack_rows": 652_800,
        "attack_shards": 96,
        "prediction_cohorts_per_model": 97,
        "prediction_shards": 291,
        "prediction_rows": 67_279_200,
        "detector_metric_rows": 5_760,
        "classification_metric_rows": 395_760,
        "region_model_states": 51,
    }


def test_runtime_environment_mismatch_fails_closed(tmp_path):
    runtime = {
        "format": "poi-stage10-runtime-environment-v1",
        "execution_scope": "stage10_gpu_model_fitting_and_validation_evaluation",
        **backend._current_runtime_environment("cpu"),
        "test_rows_read": 0,
    }
    runtime["numpy"] = "0.0-tampered"
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    inputs = backend.FrozenInputs(
        attack_config=tmp_path / "attack.yaml",
        boundary_config=tmp_path / "boundary.yaml",
        generation_request=tmp_path / "request.json",
        generation_identity=tmp_path / "identity.json",
        surrogate_directory=tmp_path / "surrogate",
        defense_config=tmp_path / "defense.yaml",
        detection_thresholds=tmp_path / "thresholds.csv",
        fixed_replay_contract=tmp_path / "replay.json",
        proposed_checkpoints={},
        ure_checkpoints={},
        fixed_checkpoint=tmp_path / "fixed.ckpt",
        device="cpu",
        runtime_environment=runtime_path,
    )

    with pytest.raises(backend.LockedTestBackendError, match="runtime differs.*numpy"):
        backend._validate_runtime_environment(inputs)
