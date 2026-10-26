import json
from pathlib import Path

import pytest
import yaml

import poi.locked_test as locked


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> dict:
    runner = tmp_path / "src/poi/locked_test.py"
    backend = tmp_path / "src/poi/reviewed_backend.py"
    _write(runner, "# reviewed locked runner\n")
    _write(backend, "# reviewed backend\n")
    monkeypatch.setattr(locked, "__file__", str(runner))

    critical_paths = {
        "artifacts/stage08/proposed42.pt": "proposed",
        "artifacts/stage09/ure42.pt": "ure",
        "artifacts/stage10/thresholds.csv": "thresholds",
        "artifacts/stage08/calibration.json": "calibration08",
        "artifacts/stage09/calibration.json": "calibration09",
        "artifacts/stage10/calibration.parquet": "calibration10",
        "artifacts/stage10/metadata.json": "metadata",
        "artifacts/stage10/replay.json": "replay",
    }
    for relative, contents in critical_paths.items():
        _write(tmp_path / relative, contents + "\n")
    files = [
        {"path": relative, "sha256": locked._sha256(tmp_path / relative)}
        for relative in critical_paths
    ]
    blockers = sorted(locked.REQUIRED_STAGE14_ITEMS)
    manifest = {
        "format": "poi-locked-test-protocol-freeze-v1",
        "status": "PROSPECTIVE_LOCK_FROZEN_PENDING_LOCKED_TEST_EXECUTION",
        "locked_test_descriptor": locked.HISTORICAL_DESCRIPTOR,
        "prior_test_coordinate_access": {
            "occurred": True,
            "historically_pristine_test_claim_permitted": False,
            "statement": "All 6,800 test coordinates were materialized previously; only prospective isolation applies.",
        },
        "stages": {
            "14": {"independent_review": {
                "required_items": blockers,
                "validated_items": blockers,
                "prospective_lock_readiness_verdict": "PASS",
            }}
        },
        "critical_freeze_items": {
            "selected_checkpoints": [
                {"label": "proposed42", "path": "artifacts/stage08/proposed42.pt"},
                {"label": "ure42", "path": "artifacts/stage09/ure42.pt"},
            ],
            "threshold_files": [
                {"label": "thresholds", "path": "artifacts/stage10/thresholds.csv"},
            ],
            "calibration_partitions": [
                {"label": "proposed", "path": "artifacts/stage08/calibration.json"},
                {"label": "ure", "path": "artifacts/stage09/calibration.json"},
                {"label": "ensemble", "path": "artifacts/stage10/calibration.parquet"},
            ],
            "fixed_downstream": {
                "actual_checkpoint_path": "/frozen/tabpfn-v2.5.ckpt",
                "expected_checkpoint_sha256": "a" * 64,
                "evidence_files": ["artifacts/stage10/metadata.json"],
                "replay_contract_file": "artifacts/stage10/replay.json",
            },
        },
        "files": files,
    }
    manifest_path = tmp_path / "artifacts/freeze.json"
    _write(manifest_path, json.dumps(manifest))

    selection = {
        "format": locked.SELECTION_FORMAT,
        "status": "FROZEN",
        "selection_scope": "validation_only",
        "test_rows_used_for_selection": 0,
        "test_evaluation_performed": False,
        "historical_test_descriptor": locked.HISTORICAL_DESCRIPTOR,
        "primary_clean_retention": 0.95,
        "regions": list(locked.REGIONS),
        "model_seeds": list(locked.MODEL_SEEDS),
        "generation_seeds": list(locked.GENERATION_SEEDS),
        "detectors": list(locked.DETECTORS),
        "metrics": list(locked.FIVE_METRICS),
        "evaluation_scopes": list(locked.EVALUATION_SCOPES),
        "classification_modes": list(locked.CLASSIFICATION_MODES),
        "attack_conditions": [f"condition_{index:02d}" for index in range(32)],
        "detector_labels": {"normal": 0, "attack": 1},
        "region_classifier": {
            "model_id": "tabpfn_v2_5",
            "fit_scope": "clean_train_only",
            "fit_count": 51,
            "shared_across_detectors_and_conditions": True,
            "refit_on_test": False,
            "ovr_regions": 17,
        },
        "repair_contract": {
            "shared_nearest_train_for_both_detectors": True,
            "proposed_native_full_population": True,
            "ure_native_reconstruction": False,
            "uses_dataset_label_for_repair": False,
        },
        "frozen_files": {entry["path"]: entry["sha256"] for entry in files},
        "fixed_downstream_checkpoint": {
            "path": "/frozen/tabpfn-v2.5.ckpt",
            "sha256": "a" * 64,
        },
    }
    selection_path = tmp_path / "artifacts/selection.json"
    _write(selection_path, json.dumps(selection))

    review_path = tmp_path / "reviews/execution.json"
    authorization_path = tmp_path / "authorization.json"
    config_path = tmp_path / "configs/execution.yaml"
    config = {
        "format": locked.CONFIG_FORMAT,
        "execution_id": "locked-test-fixture-001",
        "freeze_manifest": "artifacts/freeze.json",
        "selection_contract": "artifacts/selection.json",
        "execution_review": "reviews/execution.json",
        "authorization": "authorization.json",
        "data": {
            "clean_dataset_dir": "data/locked-sentinel",
            "split": "test",
            "expected_test_rows": 6800,
        },
        "execution_backend": {
            "module": "poi.reviewed_backend",
            "callable": "run_locked_test_backend",
            "source_path": "src/poi/reviewed_backend.py",
        },
        "output": "artifacts/test-output/locked-test-fixture-001",
    }
    _write(config_path, yaml.safe_dump(config, sort_keys=False))

    reviewed = {
        "artifacts/freeze.json": locked._sha256(manifest_path),
        "artifacts/selection.json": locked._sha256(selection_path),
        "configs/execution.yaml": locked._sha256(config_path),
        "src/poi/locked_test.py": locked._sha256(runner),
        "src/poi/reviewed_backend.py": locked._sha256(backend),
    }
    review = {
        "format": locked.REVIEW_FORMAT,
        "role": "independent_experiment_audit",
        "reviewer_role": "independent_reviewer",
        "requested_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "fork_context": "none",
        "overall_verdict": "PASS",
        "locked_test_execution_readiness_verdict": "PASS",
        "test_data_accessed": False,
        "test_evaluation_performed": False,
        "requirements": [
            {"requirement_id": item, "verdict": "PASS"}
            for item in sorted(locked.REQUIRED_REVIEW_ITEMS)
        ],
        "reviewed_sha256": reviewed,
    }
    _write(review_path, json.dumps(review))
    bindings = {
        "freeze_manifest": {"path": "artifacts/freeze.json", "sha256": reviewed["artifacts/freeze.json"]},
        "selection_contract": {"path": "artifacts/selection.json", "sha256": reviewed["artifacts/selection.json"]},
        "execution_config": {"path": "configs/execution.yaml", "sha256": reviewed["configs/execution.yaml"]},
        "runner_source": {"path": "src/poi/locked_test.py", "sha256": reviewed["src/poi/locked_test.py"]},
        "backend_source": {"path": "src/poi/reviewed_backend.py", "sha256": reviewed["src/poi/reviewed_backend.py"]},
        "execution_review": {"path": "reviews/execution.json", "sha256": locked._sha256(review_path)},
    }
    authorization = {
        "format": locked.AUTHORIZATION_FORMAT,
        "status": locked.AUTHORIZED,
        "execution_id": config["execution_id"],
        "authorization_scope": locked.AUTHORIZATION_SCOPE,
        "test_split": "test",
        "maximum_test_executions": 1,
        "historical_test_descriptor": locked.HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "prior_coordinate_access_acknowledged": True,
        "prior_coordinate_access_statement": manifest["prior_test_coordinate_access"]["statement"],
        "test_data_accessed_when_authorized": False,
        "test_evaluation_performed_when_authorized": False,
        "authorized_by": "test fixture operator",
        "authorized_at_utc": "2026-10-01T00:00:00Z",
        "bindings": bindings,
    }
    _write(authorization_path, json.dumps(authorization))
    authorization_path.chmod(0o444)

    monkeypatch.setattr(locked, "verify_freeze_manifest", lambda *args, **kwargs: {
        "status": "PASS", "test_data_accessed": False, "test_evaluation_performed": False,
    })
    return {
        "config": config_path,
        "manifest": manifest_path,
        "selection": selection_path,
        "review": review_path,
        "authorization": authorization_path,
        "output": tmp_path / config["output"],
        "data": tmp_path / "data/locked-sentinel",
    }


def test_preflight_verifies_every_gate_without_opening_or_importing_test_data(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["data"].mkdir(parents=True)
    sentinel = fixture["data"] / "test-features.csv"
    sentinel.write_text("THIS MUST NOT BE READ")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == sentinel:
            pytest.fail("preflight opened locked test features")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(locked, "_load_backend", lambda gate: pytest.fail("backend imported"))

    result = locked.preflight(fixture["config"], project_root=tmp_path)

    assert result["status"] == locked.READY
    assert result["test_data_accessed"] is False
    assert result["test_evaluation_performed"] is False
    assert result["historically_pristine_test_claim_permitted"] is False
    assert not fixture["output"].exists()


def test_invalid_freeze_stops_before_selection_authorization_or_data(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        locked, "verify_freeze_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(locked.ProtocolFreezeError("tampered")),
    )
    fixture["selection"].unlink()

    with pytest.raises(locked.LockedTestNotReady, match="freeze verification failed"):
        locked.preflight(fixture["config"], project_root=tmp_path)
    assert not fixture["output"].exists()


def test_missing_stage14_experiments_are_not_bypassed(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture["manifest"].read_text())
    missing = "adaptive_detector_evasion_attack"
    manifest["stages"]["14"]["independent_review"]["required_items"].remove(missing)
    manifest["stages"]["14"]["independent_review"]["validated_items"].remove(missing)
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(locked.LockedTestNotReady, match="Stage 14 blockers"):
        locked.preflight(fixture["config"], project_root=tmp_path)


def test_incomplete_or_invented_defense_selection_fails_closed(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    selection = json.loads(fixture["selection"].read_text())
    selection["repair_contract"]["ure_native_reconstruction"] = True
    fixture["selection"].write_text(json.dumps(selection))

    with pytest.raises(locked.LockedTestError, match="URE native reconstruction"):
        locked.preflight(fixture["config"], project_root=tmp_path)


def test_byte_tampering_after_review_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    selection = json.loads(fixture["selection"].read_text())
    selection["post_review_note"] = "changed bytes"
    fixture["selection"].write_text(json.dumps(selection))

    with pytest.raises(locked.LockedTestError, match="byte-bound"):
        locked.preflight(fixture["config"], project_root=tmp_path)


def test_mutation_between_gate_and_reservation_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    gate = locked.pre_access_gate(fixture["config"], project_root=tmp_path)
    backend_source = tmp_path / "src/poi/reviewed_backend.py"
    backend_source.write_text("# changed after gate\n")

    with pytest.raises(locked.LockedTestError, match="changed after gate"):
        locked._reserve(gate)
    assert not fixture["output"].exists()


def _backend_result(gate, output: Path) -> dict:
    artifacts = {}
    for label, name in {
        "detector_metrics": "tables/detector_metrics.csv",
        "classification_metrics": "tables/classification_metrics.csv",
        "prediction_manifest": "tables/prediction_manifest.json",
        "attack_manifest": "tables/attack_manifest.json",
        "metadata": "metadata.json",
    }.items():
        path = output / name
        _write(path, f"{label}\n")
        artifacts[label] = {"path": name, "sha256": locked._sha256(path)}
    return {
        "format": "poi-locked-test-backend-result-v1",
        "execution_id": gate.execution_id,
        "split": "test",
        "test_rows": 6800,
        "test_executions": 1,
        "region_classifier": "tabpfn_v2_5",
        "region_model_fit_count": 51,
        "region_model_fit_scope": "clean_train_only",
        "same_region_classifier_shared_across_detectors": True,
        "region_classifier_refit_on_test": False,
        "detectors": list(locked.DETECTORS),
        "detector_labels": {"normal": 0, "attack": 1},
        "regions": list(locked.REGIONS),
        "model_seeds": list(locked.MODEL_SEEDS),
        "generation_seeds": list(locked.GENERATION_SEEDS),
        "attack_conditions": gate.selection["attack_conditions"],
        "metrics": list(locked.FIVE_METRICS),
        "evaluation_scopes": list(locked.EVALUATION_SCOPES),
        "classification_modes": list(locked.CLASSIFICATION_MODES),
        "primary_clean_retention": 0.95,
        "shared_nearest_train_repair_for_both_detectors": True,
        "proposed_native_full_population": True,
        "ure_native_reconstruction": False,
        "uses_dataset_label_for_repair": False,
        "preserved_rejected_and_quarantined_denominators": True,
        "historical_test_descriptor": locked.HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "artifacts": artifacts,
    }


def test_execution_reserves_before_backend_and_never_marks_success(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(locked, "_validate_backend_output_artifacts", lambda *args: None)

    def load_backend(gate):
        def backend(context):
            reservation = Path(context["reservation"])
            assert reservation.is_file()
            assert json.loads(reservation.read_text())["test_data_accessed_at_reservation"] is False
            return _backend_result(gate, Path(context["output"]))
        return backend

    monkeypatch.setattr(locked, "_load_backend", load_backend)
    result = locked.execute_locked_test(fixture["config"], project_root=tmp_path)

    assert result["status"] == locked.PENDING_REVIEW
    assert result["scientific_completion_claim_permitted"] is False
    assert (fixture["output"] / "_PENDING_INDEPENDENT_REVIEW.json").is_file()
    assert not (fixture["output"] / "_SUCCESS.json").exists()
    with pytest.raises(locked.LockedTestNotReady, match="output already exists"):
        locked.execute_locked_test(fixture["config"], project_root=tmp_path)


def test_failure_after_reservation_is_non_retryable_and_visible(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        locked, "_load_backend", lambda gate: lambda context: (_ for _ in ()).throw(
            RuntimeError("synthetic backend failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        locked.execute_locked_test(fixture["config"], project_root=tmp_path)

    failure = json.loads((fixture["output"] / "_EXECUTION_FAILED.json").read_text())
    assert failure["retry_permitted"] is False
    assert failure["test_access_may_have_occurred"] is True
    assert not (fixture["output"] / "_SUCCESS.json").exists()


def test_backend_self_attestation_cannot_replace_auditable_tables(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)

    def load_backend(gate):
        return lambda context: _backend_result(gate, Path(context["output"]))

    monkeypatch.setattr(locked, "_load_backend", load_backend)
    with pytest.raises(locked.LockedTestError, match="cannot read backend metadata"):
        locked.execute_locked_test(fixture["config"], project_root=tmp_path)

    assert (fixture["output"] / "_EXECUTION_FAILED.json").is_file()
    assert not (fixture["output"] / "_PENDING_INDEPENDENT_REVIEW.json").exists()
