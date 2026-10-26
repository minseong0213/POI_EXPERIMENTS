import json
import hashlib
import os
import stat
from pathlib import Path

import pytest
import yaml

from poi.protocol_freeze import (
    FORMAT,
    REQUIRED_STAGES,
    FreezeNotReady,
    ProtocolFreezeError,
    create_freeze_manifest,
    preflight,
    verify_freeze_manifest,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_verdict(
    stage: str, artifact_root: str, items: list[str], root: Path,
    reviewed_paths: list[str],
) -> dict:
    return {
        "role": "independent_experiment_audit",
        "reviewer_role": "independent_reviewer",
        "requested_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "fork_context": "none",
        "review_scope": f"requirements/stage_{stage}.md {artifact_root}",
        "overall_verdict": "PARTIAL",
        "prospective_lock_readiness_verdict": "PASS",
        "reviewed_sha256": {path: _digest(root / path) for path in reviewed_paths},
        "requirements": [
            {
                "requirement_id": item,
                "requirement": item.replace("_", " "),
                "expected": "fixture evidence is complete",
                "observed": "fixture evidence was inspected directly",
                "verdict": "PASS",
                "evidence": [f"{artifact_root}/artifact.txt"],
                "limitation": "fixture-only protocol unit test",
                "scope": f"stage_{stage}",
            }
            for item in items
        ] + [{
            "requirement_id": "historical_pristine_test_isolation",
            "requirement": "Historical pristine test isolation",
            "expected": "Prior coordinate access is disclosed and cannot be repaired",
            "observed": "Prior access is retained as a permanent limitation",
            "verdict": "PARTIAL",
            "evidence": [f"{artifact_root}/artifact.txt"],
            "limitation": "This is only a prospectively isolated locked evaluation.",
            "scope": f"stage_{stage}",
        }],
    }


def _fixture_contract(tmp_path: Path, *, complete: bool = True) -> tuple[Path, dict]:
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "reviews").mkdir()
    stages = {}
    checkpoint = tmp_path / "cache" / "tabpfn-v2.5.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture official checkpoint bytes")
    checkpoint_sha = _digest(checkpoint)
    for stage in REQUIRED_STAGES:
        artifact_root = f"artifacts/stage{stage}"
        artifact = tmp_path / artifact_root
        if complete:
            artifact.mkdir(parents=True)
            (artifact / "artifact.txt").write_text(f"stage {stage}\n")
        config_file = tmp_path / "configs" / f"stage{stage}.yaml"
        source_file = tmp_path / "src" / f"stage{stage}.py"
        config_file.write_text(f"stage: '{stage}'\n")
        source_file.write_text(f"STAGE = '{stage}'\n")
        items = [f"stage_{stage}_scope", f"stage_{stage}_test_isolation"]
        stages[stage] = {
            "artifact_roots": [artifact_root],
            "config_files": [f"configs/stage{stage}.yaml"],
            "code_files": [f"src/stage{stage}.py"],
            "required_artifact_files": [f"{artifact_root}/artifact.txt"],
            "independent_review": {
                "verdict_file": f"reviews/stage{stage}.json",
                "agent_role": "independent_experiment_audit",
                "reviewer_role": "independent_reviewer",
                "requested_model": "gpt-6-astra",
                "reasoning_effort": "medium",
                "fork_context": "none",
                "scope_tokens": [f"requirements/stage_{stage}.md", artifact_root],
                "required_items": items,
                "gate_verdict_field": "prospective_lock_readiness_verdict",
                "allowed_overall_verdicts": ["PARTIAL"],
                "acknowledged_nonpass_items": {
                    "historical_pristine_test_isolation": "PARTIAL",
                },
            },
        }
    dependency_files = [
        "src/poi/protocol_freeze.py", "src/poi/data.py", "src/poi/uncertainty.py",
        "src/poi/clean_uncertainty.py",
        "src/poi/final_test.py", "src/poi/locked_test.py", "src/poi/defense_evaluate.py",
        "src/poi/locked_test_backend.py",
        "src/poi/selected_explain.py", "src/poi/attack_evaluate.py",
        "src/poi/ablation.py", "src/poi/benchmark.py",
        "configs/stages/15_locked_test.yaml",
        "pyproject.toml", "requirements.txt", "requirements.lock",
    ]
    for relative in dependency_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"fixture dependency {relative}\n")
    replay_relative = "artifacts/stage10/fixed_downstream_replay_contract.json"
    runtime_relative = "artifacts/stage10/runtime_environment.json"
    replay_path = tmp_path / replay_relative
    if complete:
        digest = "1" * 64
        stage10_artifact_sha = _digest(tmp_path / "artifacts/stage10/artifact.txt")
        stage10_config_sha = _digest(tmp_path / "configs/stage10.yaml")
        runtime_path = tmp_path / runtime_relative
        runtime_path.write_text(json.dumps({
            "format": "poi-stage10-runtime-environment-v1",
            "execution_scope": "stage10_gpu_model_fitting_and_validation_evaluation",
            "python_version": "fixture", "python_executable": "/fixture/python",
            "platform": "fixture", "numpy": "fixture", "pandas": "fixture",
            "tabpfn_version": "fixture", "torch_version": "fixture",
            "cuda_version": None, "cuda_available": False, "gpu_device_name": None,
            "requested_device": "cpu", "restored_on_device": "cpu",
            "installed_distributions": {
                "numpy": "fixture", "pandas": "fixture",
                "tabpfn": "fixture", "torch": "fixture",
            },
            "test_rows_read": 0,
        }))
        source_snapshots = {}
        for name in (
            "defense_pipeline.py", "selected_explain.py", "defense_evaluate.py", "data.py",
            "attack_evaluate.py", "ablation.py", "benchmark.py",
        ):
            relative = f"artifacts/stage10/work/fixed_downstream_replay_sources/{name}"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"executed source snapshot: {name}\n")
            source_snapshots[relative] = _digest(path)
        replay_path.write_text(json.dumps({
            "format": "poi-fixed-downstream-replay-contract-v1",
            "model_id": "tabpfn_v2_5",
            "official_checkpoint_path": str(checkpoint),
            "official_checkpoint_sha256": checkpoint_sha,
            "fit_count": 1,
            "expected_fit_count": 1,
            "test_rows_read": 0,
            "features": ["num", "cat"], "categorical_features": ["cat"],
            "train_rows": 2,
            "model_seeds": [42], "regions": ["A"], "n_estimators": 4,
            "classification_threshold": .5,
                "runtime": {
                    "python": "fixture", "numpy": "fixture", "pandas": "fixture",
                    "python_executable": "/fixture/python", "tabpfn_version": "fixture",
                    "torch_version": "fixture", "cuda_version": None,
                    "cuda_available": False, "gpu_device_name": None,
                    "requested_device": "cpu", "restored_on_device": "cpu",
            },
            "raw_data_sha256": digest,
            "sample_manifest_sha256": digest,
            "train_poi_order_sha256": digest,
            "train_feature_matrix_sha256": digest,
            "calibration_poi_order_sha256": digest,
            "evaluation_poi_order_sha256": digest,
            "thresholds_sha256": stage10_artifact_sha,
            "calibration_scores_sha256": stage10_artifact_sha,
            "prediction_manifest_sha256": stage10_artifact_sha,
            "stage_config_sha256": stage10_config_sha,
            "runtime_environment_sha256": _digest(runtime_path),
            "source_snapshots_sha256": source_snapshots,
            "fits": [{
                "format": "poi-tabpfn25-fit-replay-v1",
                "model_id": "tabpfn_v2_5", "seed": 42, "model_seed": 42,
                "region": "A", "feature_order": ["num", "cat"],
                "n_estimators": 4,
                "official_checkpoint_path": str(checkpoint),
                "official_checkpoint_sha256": checkpoint_sha,
                "train_rows": 2, "train_matrix_dtype": "float32",
                "categorical_feature_indices": [1],
                "probe_indices": [0, 1], "fit_mode": "fit_preprocessors",
                "memory_saving_mode": "auto",
                "train_matrix_sha256": digest, "truth_sha256": digest,
                "probe_matrix_sha256": digest, "probe_prediction_sha256": digest,
                "probe_repeat_prediction_sha256": digest,
                "restored_probe_prediction_sha256": digest,
                    "tabpfn_version": "fixture", "torch_version": "fixture",
                    "cuda_version": None, "cuda_available": False,
                    "gpu_device_name": None, "requested_device": "cpu",
                    "restored_on_device": "cpu",
                "train_poi_order_sha256": digest, "classification_threshold": .5,
                "fitted_state_format": "tabpfn_official_fit_state_v1",
                "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
                    "fitted_state_path": str(
                        tmp_path / "artifacts/stage10/work/fixed_downstream_fit_states/"
                        "modelseed42__region00.tabpfn_fit"
                    ),
                "fitted_state_sha256": "REPLACED_BELOW",
                "fitted_state_size_bytes": 0,
            }],
        }))
        fitted_state = (
            tmp_path / "artifacts/stage10/work/fixed_downstream_fit_states/"
            "modelseed42__region00.tabpfn_fit"
        )
        fitted_state.parent.mkdir(parents=True, exist_ok=True)
        fitted_state.write_bytes(b"fixture persisted fit state")
        replay = json.loads(replay_path.read_text())
        replay["fits"][0]["fitted_state_sha256"] = _digest(fitted_state)
        replay["fits"][0]["fitted_state_size_bytes"] = fitted_state.stat().st_size
        replay_path.write_text(json.dumps(replay))
        stages["10"]["required_artifact_files"].append(replay_relative)
        for stage in REQUIRED_STAGES:
            artifact_root = stages[stage]["artifact_roots"][0]
            reviewed = sorted(
                [path.relative_to(tmp_path).as_posix()
                 for path in (tmp_path / artifact_root).rglob("*") if path.is_file()]
                + stages[stage]["config_files"] + stages[stage]["code_files"]
            )
            items = stages[stage]["independent_review"]["required_items"]
            verdict = _valid_verdict(stage, artifact_root, items, tmp_path, reviewed)
            (tmp_path / "reviews" / f"stage{stage}.json").write_text(json.dumps(verdict))
    spec = {
        "format": FORMAT,
        "manifest_output": "freeze/protocol_freeze_manifest.json",
        "required_stages": list(REQUIRED_STAGES),
        "protocol_dependencies": {
            "source_files": [value for value in dependency_files if value.startswith("src/")],
            "config_files": ["configs/stages/15_locked_test.yaml"],
            "environment_files": ["pyproject.toml", "requirements.txt", "requirements.lock"],
        },
        "common_independent_review": {
            "verdict_file": "reviews/common.json",
            "agent_role": "independent_experiment_audit",
            "reviewer_role": "independent_reviewer",
            "requested_model": "gpt-6-astra",
            "reasoning_effort": "medium",
            "fork_context": "none",
            "scope_tokens": ["protocol.yaml", "protocol_dependencies"],
            "required_items": [
                "common_dependency_byte_inventory", "freeze_verifier_and_config_review",
                "actual_runtime_environment_binding",
            ],
            "gate_verdict_field": "prospective_lock_readiness_verdict",
            "allowed_overall_verdicts": ["PARTIAL"],
            "acknowledged_nonpass_items": {
                "historical_pristine_test_isolation": "PARTIAL",
            },
        },
        "stages": stages,
        "critical_freeze_items": {
            "selection_evidence": [
                {"label": "clean_eight_model_selection", "stage": "06",
                 "path": "artifacts/stage06/artifact.txt"},
            ],
            "selected_checkpoints": [
                {"label": "proposed_seed42", "stage": "08",
                 "path": "artifacts/stage08/artifact.txt"},
                {"label": "ure_seed42", "stage": "09",
                 "path": "artifacts/stage09/artifact.txt"},
            ],
            "threshold_files": [
                {"label": "detector_thresholds", "stage": "10",
                 "path": "artifacts/stage10/artifact.txt"},
            ],
            "calibration_partitions": [
                {"label": "proposed_ids", "stage": "08",
                 "path": "artifacts/stage08/artifact.txt"},
                {"label": "ure_ids", "stage": "09",
                 "path": "artifacts/stage09/artifact.txt"},
                {"label": "ensemble_calibration_scores", "stage": "10",
                 "path": "artifacts/stage10/artifact.txt"},
            ],
            "fixed_downstream": {
                "model_id": "tabpfn_v2_5",
                "expected_checkpoint_sha256": checkpoint_sha,
                "actual_checkpoint_path": str(checkpoint),
                "replay_contract_file": replay_relative,
                "expected_fit_count": 1,
                "replay_hash_bindings": {
                    "thresholds_sha256": "artifacts/stage10/artifact.txt",
                    "calibration_scores_sha256": "artifacts/stage10/artifact.txt",
                    "prediction_manifest_sha256": "artifacts/stage10/artifact.txt",
                    "stage_config_sha256": "configs/stage10.yaml",
                    "runtime_environment_sha256": runtime_relative,
                },
                "evidence_files": ["artifacts/stage10/artifact.txt"],
            },
        },
        "prior_test_coordinate_access": {
            "occurred": True,
            "historically_pristine_test_claim_permitted": False,
            "statement": "Coordinates were previously materialized; only prospective isolation applies.",
            "evidence_files": ["artifacts/stage01/artifact.txt"],
        },
    }
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump(spec, sort_keys=False))
    if complete:
        reviewed = ["protocol.yaml", *dependency_files]
        common = _valid_verdict(
            "COMMON", "protocol_dependencies",
            spec["common_independent_review"]["required_items"], tmp_path, reviewed,
        )
        common["review_scope"] = "protocol.yaml protocol_dependencies"
        (tmp_path / "reviews/common.json").write_text(json.dumps(common))
    return config, spec


def test_missing_stage_artifacts_are_not_run_and_write_nothing(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=False)

    result = preflight(config, project_root=tmp_path)

    assert result["status"] == "NOT_RUN"
    assert not result["ready"]
    assert result["test_data_accessed"] is False
    assert not (tmp_path / "freeze").exists()
    with pytest.raises(FreezeNotReady, match="blocking checks"):
        create_freeze_manifest(config, project_root=tmp_path)
    assert not (tmp_path / "freeze").exists()


def test_stage_report_cannot_substitute_for_independent_verdict(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=True)
    verdict = tmp_path / "reviews" / "stage10.json"
    verdict.unlink()
    (tmp_path / "artifacts" / "stage10" / "report.md").write_text(
        "overall_verdict: PASS\nstatus: COMPLETE\n"
    )

    result = preflight(config, project_root=tmp_path)

    review_check = next(
        row for row in result["checks"]
        if row["stage"] == "10" and row["kind"] == "independent_review"
    )
    assert review_check["status"] == "NOT_RUN"
    assert "absent" in review_check["detail"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda verdict: verdict.update(role="poi_experiment_runner"), "role"),
        (lambda verdict: verdict.update(requested_model="gpt-5.6-sol"), "requested_model"),
        (lambda verdict: verdict.update(reasoning_effort="xhigh"), "reasoning_effort"),
        (lambda verdict: verdict.update(fork_context="all"), "fork_context"),
        (lambda verdict: verdict.update(review_scope="some unrelated stage"), "scope omits"),
        (lambda verdict: verdict["requirements"].pop(1), "omits required items"),
        (lambda verdict: verdict["requirements"][0].update(verdict="PARTIAL"), "PARTIAL"),
        (lambda verdict: verdict.update(overall_verdict="NOT_RUN"), "overall_verdict"),
        (lambda verdict: verdict.update(prospective_lock_readiness_verdict="PARTIAL"),
         "prospective_lock_readiness_verdict"),
        (lambda verdict: verdict["requirements"][-1].update(verdict="PASS"),
         "expected persistent PARTIAL"),
    ],
)
def test_independent_verdict_provenance_and_coverage_are_enforced(
    tmp_path, mutation, message
):
    config, _ = _fixture_contract(tmp_path, complete=True)
    path = tmp_path / "reviews" / "stage08.json"
    verdict = json.loads(path.read_text())
    mutation(verdict)
    path.write_text(json.dumps(verdict))

    result = preflight(config, project_root=tmp_path)

    assert result["status"] == "FAIL"
    check = next(
        row for row in result["checks"]
        if row["stage"] == "08" and row["kind"] == "independent_review"
    )
    assert check["status"] == "FAIL"
    assert message in check["detail"]


def test_manifest_is_exclusive_read_only_and_rehashes_every_file(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=True)

    manifest = create_freeze_manifest(config, project_root=tmp_path)
    manifest_path = tmp_path / "freeze" / "protocol_freeze_manifest.json"

    assert manifest["status"] == "PROSPECTIVE_LOCK_FROZEN_PENDING_LOCKED_TEST_EXECUTION"
    assert manifest["test_evaluation_performed"] is False
    assert manifest["historically_pristine_test_claim_permitted"] is False
    assert manifest["authorization_scope"] == "prospective_lock_readiness_only"
    assert manifest["scientific_completion_claim_permitted"] is False
    assert stat.S_IMODE(manifest_path.stat().st_mode) & 0o222 == 0
    verification = verify_freeze_manifest(manifest_path, project_root=tmp_path)
    assert verification["status"] == "PASS"
    assert verification["verified_files"] == len(manifest["files"])
    with pytest.raises(FileExistsError):
        create_freeze_manifest(config, project_root=tmp_path)

    frozen = tmp_path / "artifacts" / "stage07" / "artifact.txt"
    frozen.write_text("tampered\n")
    with pytest.raises(ProtocolFreezeError, match="mismatch"):
        verify_freeze_manifest(manifest_path, project_root=tmp_path)


def test_new_artifact_after_freeze_breaks_inventory(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=True)
    create_freeze_manifest(config, project_root=tmp_path)
    manifest_path = tmp_path / "freeze" / "protocol_freeze_manifest.json"
    (tmp_path / "artifacts" / "stage13" / "late_file.csv").write_text("late\n")

    with pytest.raises(ProtocolFreezeError, match="inventory changed"):
        verify_freeze_manifest(manifest_path, project_root=tmp_path)


def test_stale_independent_review_hash_blocks_preflight(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=True)
    (tmp_path / "artifacts/stage10/artifact.txt").write_text("changed after review\n")

    result = preflight(config, project_root=tmp_path)

    check = next(
        row for row in result["checks"]
        if row["stage"] == "10" and row["kind"] == "independent_review"
    )
    assert result["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert "stale" in check["detail"]


def test_arbitrary_checkpoint_digest_cannot_pass_with_text_evidence(tmp_path):
    config, spec = _fixture_contract(tmp_path, complete=True)
    spec["critical_freeze_items"]["fixed_downstream"][
        "expected_checkpoint_sha256"
    ] = "a" * 64
    config.write_text(yaml.safe_dump(spec, sort_keys=False))

    result = preflight(config, project_root=tmp_path)

    checkpoint = next(
        row for row in result["checks"] if row["kind"] == "fixed_downstream_checkpoint"
    )
    assert result["status"] == "FAIL"
    assert checkpoint["status"] == "FAIL"
    assert "mismatch" in checkpoint["detail"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda replay: replay["fits"][0].update(train_matrix_sha256="2" * 64),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].update(train_poi_order_sha256="2" * 64),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].pop("probe_indices"),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].pop("train_rows"),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].pop("train_matrix_dtype"),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].pop("categorical_feature_indices"),
         "train/preprocessor/probe"),
        (lambda replay: replay["fits"][0].update(fitted_state_sha256="3" * 64),
         "official fitted state"),
        (lambda replay: replay["runtime"].pop("torch_version"),
         "runtime contract"),
        (lambda replay: replay["fits"][0].update(torch_version="other"),
         "fit runtime differs"),
    ],
)
def test_replay_contract_rejects_inconsistent_fit_state_and_recipe(
    tmp_path, mutation, message
):
    config, _ = _fixture_contract(tmp_path, complete=True)
    path = tmp_path / "artifacts/stage10/fixed_downstream_replay_contract.json"
    replay = json.loads(path.read_text())
    mutation(replay)
    path.write_text(json.dumps(replay))

    result = preflight(config, project_root=tmp_path)

    check = next(
        row for row in result["checks"]
        if row["kind"] == "fixed_downstream_replay_contract"
    )
    assert check["status"] == "FAIL"
    assert message in check["detail"]


def test_external_checkpoint_is_rehashed_during_manifest_verification(tmp_path):
    config, spec = _fixture_contract(tmp_path, complete=True)
    create_freeze_manifest(config, project_root=tmp_path)
    checkpoint = Path(
        spec["critical_freeze_items"]["fixed_downstream"]["actual_checkpoint_path"]
    )
    checkpoint.write_bytes(b"tampered checkpoint")

    with pytest.raises(ProtocolFreezeError, match="checkpoint SHA-256"):
        verify_freeze_manifest(
            tmp_path / "freeze/protocol_freeze_manifest.json", project_root=tmp_path
        )


@pytest.mark.parametrize(
    "dependency",
    [
        "src/poi/final_test.py", "src/poi/locked_test_backend.py",
        "src/poi/clean_uncertainty.py",
        "src/poi/attack_evaluate.py",
        "src/poi/ablation.py", "src/poi/benchmark.py",
    ],
)
def test_protocol_behavior_dependency_coverage_is_mandatory(tmp_path, dependency):
    config, spec = _fixture_contract(tmp_path, complete=True)
    spec["protocol_dependencies"]["source_files"].remove(dependency)
    config.write_text(yaml.safe_dump(spec, sort_keys=False))

    with pytest.raises(ProtocolFreezeError, match=Path(dependency).stem):
        preflight(config, project_root=tmp_path)


def test_replay_runtime_environment_must_match_reported_and_installed_versions(tmp_path):
    config, _ = _fixture_contract(tmp_path, complete=True)
    runtime_path = tmp_path / "artifacts/stage10/runtime_environment.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["installed_distributions"]["torch"] = "different"
    runtime_path.write_text(json.dumps(runtime))
    replay_path = tmp_path / "artifacts/stage10/fixed_downstream_replay_contract.json"
    replay = json.loads(replay_path.read_text())
    replay["runtime_environment_sha256"] = _digest(runtime_path)
    replay_path.write_text(json.dumps(replay))

    result = preflight(config, project_root=tmp_path)

    check = next(
        row for row in result["checks"]
        if row["kind"] == "fixed_downstream_replay_contract"
    )
    assert check["status"] == "FAIL"
    assert "package inventory differs for torch" in check["detail"]


@pytest.mark.parametrize(
    "relative",
    ["src/poi/protocol_freeze.py", "protocol.yaml"],
)
def test_common_review_binds_shared_dependencies_and_freeze_config(tmp_path, relative):
    config, _ = _fixture_contract(tmp_path, complete=True)
    path = tmp_path / relative
    suffix = "# changed after common review\n" if path.suffix == ".yaml" else \
        "changed after common review\n"
    path.write_text(path.read_text() + suffix)

    result = preflight(config, project_root=tmp_path)

    check = next(
        row for row in result["checks"] if row["kind"] == "common_independent_review"
    )
    assert result["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert "stale" in check["detail"]
    with pytest.raises(FreezeNotReady):
        create_freeze_manifest(config, project_root=tmp_path)


@pytest.mark.parametrize(
    "disclosure",
    [
        {"occurred": False, "historically_pristine_test_claim_permitted": False},
        {"occurred": True, "historically_pristine_test_claim_permitted": True},
    ],
)
def test_prior_coordinate_access_must_forbid_pristine_claim(tmp_path, disclosure):
    config, spec = _fixture_contract(tmp_path, complete=True)
    spec["prior_test_coordinate_access"].update(disclosure)
    config.write_text(yaml.safe_dump(spec, sort_keys=False))

    with pytest.raises(ProtocolFreezeError, match="prior|pristine"):
        preflight(config, project_root=tmp_path)


def test_freeze_contract_rejects_any_dataset_path_without_opening_it(tmp_path, monkeypatch):
    config, spec = _fixture_contract(tmp_path, complete=True)
    data_file = tmp_path / "data" / "poi_34k_seed42" / "poi_data_region.csv"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("LOCKED SENTINEL THAT MUST NOT BE OPENED")
    spec["stages"]["01"]["required_artifact_files"].append(
        "data/poi_34k_seed42/poi_data_region.csv"
    )
    config.write_text(yaml.safe_dump(spec, sort_keys=False))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == data_file:
            pytest.fail("freeze preflight attempted to open a dataset")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ProtocolFreezeError, match="must not access datasets"):
        preflight(config, project_root=tmp_path)
