"""Fail-closed manifest creation for the locked-test protocol.

This module freezes development artifacts.  It deliberately contains no dataset loader and no
locked-test evaluation entry point.  The legacy test runner remains disabled until a manifest
created here passes an independent review gate and a separate, future unlock procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


FORMAT = "poi-locked-test-protocol-freeze-v1"
REQUIRED_STAGES = ("01", "06", "07", "08", "09", "10", "13", "14")
REQUIRED_PROTOCOL_SOURCE_DEPENDENCIES = frozenset({
    "src/poi/protocol_freeze.py",
    "src/poi/data.py",
    "src/poi/uncertainty.py",
    "src/poi/clean_uncertainty.py",
    "src/poi/final_test.py",
    "src/poi/locked_test.py",
    "src/poi/locked_test_backend.py",
    "src/poi/defense_evaluate.py",
    "src/poi/selected_explain.py",
    "src/poi/attack_evaluate.py",
    "src/poi/ablation.py",
    "src/poi/benchmark.py",
})
REQUIRED_PROTOCOL_CONFIG_DEPENDENCIES = frozenset({
    "configs/stages/15_locked_test.yaml",
})
REQUIRED_PROTOCOL_ENVIRONMENT_DEPENDENCIES = frozenset({
    "pyproject.toml", "requirements.txt", "requirements.lock",
})
PASS = "PASS"
NON_PASS = frozenset({"PARTIAL", "FAIL", "NOT_RUN"})
FORBIDDEN_SOURCE_TOP_LEVEL = frozenset({"data"})
_GLOB_CHARS = frozenset("*?[]{}")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolFreezeError(RuntimeError):
    """The freeze contract or its evidence is invalid."""


class FreezeNotReady(ProtocolFreezeError):
    """Development evidence is absent or has not passed its independent review."""


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolFreezeError(f"{field} must be a mapping")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ProtocolFreezeError(f"{field} must be a non-empty list")
    return value


def _lexical_repo_path(root: Path, raw: Any, field: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ProtocolFreezeError(f"{field} must be a non-empty path string")
    if any(character in raw for character in _GLOB_CHARS):
        raise ProtocolFreezeError(f"{field} must be explicit; globs are forbidden: {raw}")
    supplied = Path(raw)
    candidate = supplied if supplied.is_absolute() else root / supplied
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ProtocolFreezeError(f"{field} escapes project root: {raw}") from exc
    if not relative.parts:
        raise ProtocolFreezeError(f"{field} cannot be the project root")
    if relative.parts[0].lower() in FORBIDDEN_SOURCE_TOP_LEVEL:
        raise ProtocolFreezeError(
            f"{field} points into {relative.parts[0]}/; freeze code must not access datasets"
        )
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ProtocolFreezeError(f"{field} contains a symlink: {relative.as_posix()}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProtocolFreezeError(f"{field} resolves outside project root: {raw}") from exc
    return candidate, relative.as_posix()


def _config_path_list(
    root: Path, values: Any, field: str
) -> list[tuple[Path, str]]:
    paths = []
    for index, raw in enumerate(_require_list(values, field)):
        paths.append(_lexical_repo_path(root, raw, f"{field}[{index}]"))
    if len({relative for _, relative in paths}) != len(paths):
        raise ProtocolFreezeError(f"{field} contains duplicate paths")
    return paths


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolFreezeError(f"cannot read {field} as JSON: {path}: {exc}") from exc
    return _require_mapping(value, field)


def _scope_text(verdict: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in ("review_scope", "reviewed_paths", "artifact_paths", "config_paths"):
        value = verdict.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            values.extend(str(item) for item in value)
    return _normal_text(" ".join(values).replace("\\", "/"))


def _requirement_identifier(row: Mapping[str, Any]) -> str:
    value = row.get("requirement_id", row.get("requirement"))
    if not isinstance(value, str) or not value.strip():
        raise ProtocolFreezeError("independent review requirement lacks requirement_id/requirement")
    return _normal_text(value)


def validate_independent_verdict(
    verdict: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    stage_id: str,
) -> dict[str, Any]:
    """Validate review provenance and every required review item.

    A stage report, success marker, or a top-level PASS alone cannot satisfy this function.
    """

    expected_agent_role = str(contract.get("agent_role", "independent_experiment_audit"))
    expected_reviewer_role = str(contract.get("reviewer_role", "independent_reviewer"))
    expected_model = str(contract.get("requested_model", "gpt-6-astra"))
    expected_effort = str(contract.get("reasoning_effort", "medium"))
    expected_fork = str(contract.get("fork_context", "none"))
    observed_agent_role = verdict.get("role")
    observed_reviewer_role = verdict.get("reviewer_role", verdict.get("role"))
    checks = (
        ("role", observed_agent_role, expected_agent_role),
        ("reviewer_role", observed_reviewer_role, expected_reviewer_role),
        ("requested_model", verdict.get("requested_model"), expected_model),
        ("reasoning_effort", verdict.get("reasoning_effort"), expected_effort),
        ("fork_context", verdict.get("fork_context"), expected_fork),
    )
    for field, observed, expected in checks:
        if _normal_text(observed) != _normal_text(expected):
            raise ProtocolFreezeError(
                f"stage {stage_id} independent verdict {field}={observed!r}; expected {expected!r}"
            )

    gate_field = str(contract.get("gate_verdict_field", "prospective_lock_readiness_verdict"))
    if _normal_text(verdict.get(gate_field)) != "pass":
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict {gate_field} must be PASS"
        )
    allowed_overall = [str(value).strip().upper() for value in _require_list(
        contract.get("allowed_overall_verdicts", ["PARTIAL"]),
        f"stages.{stage_id}.review.allowed_overall_verdicts",
    )]
    observed_overall = str(verdict.get("overall_verdict", "")).strip().upper()
    if observed_overall not in allowed_overall:
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict overall_verdict={observed_overall!r}; "
            f"expected one of {allowed_overall}"
        )

    scope = _scope_text(verdict)
    scope_tokens = [str(item) for item in _require_list(
        contract.get("scope_tokens"), f"stages.{stage_id}.review.scope_tokens"
    )]
    missing_scope = [token for token in scope_tokens if _normal_text(token) not in scope]
    if missing_scope:
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict scope omits: {missing_scope}"
        )

    raw_acknowledged = _require_mapping(
        contract.get("acknowledged_nonpass_items", {
            "historical_pristine_test_isolation": "PARTIAL",
        }),
        f"stages.{stage_id}.review.acknowledged_nonpass_items",
    )
    acknowledged = {
        _normal_text(item): str(status).strip().upper()
        for item, status in raw_acknowledged.items()
    }
    if not acknowledged:
        raise ProtocolFreezeError(
            f"stage {stage_id} must retain an acknowledged historical-isolation limitation"
        )
    if any(status == PASS or status not in NON_PASS for status in acknowledged.values()):
        raise ProtocolFreezeError(
            f"stage {stage_id} acknowledged_nonpass_items must contain PARTIAL/FAIL/NOT_RUN"
        )

    required_items = [str(item) for item in _require_list(
        contract.get("required_items"), f"stages.{stage_id}.review.required_items"
    )]
    normalized_required = [_normal_text(item) for item in required_items]
    if len(set(normalized_required)) != len(normalized_required):
        raise ProtocolFreezeError(f"stage {stage_id} review required_items has duplicates")
    overlap = sorted(set(normalized_required) & set(acknowledged))
    if overlap:
        raise ProtocolFreezeError(
            f"stage {stage_id} items cannot be both PASS-gated and acknowledged non-PASS: {overlap}"
        )

    rows = _require_list(verdict.get("requirements"), "independent verdict requirements")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"requirements[{index}]")
        identifier = _requirement_identifier(row)
        if identifier in indexed:
            raise ProtocolFreezeError(
                f"stage {stage_id} independent verdict duplicates requirement {identifier!r}"
            )
        for required_field in ("expected", "observed", "evidence", "limitation", "scope"):
            if required_field not in row:
                raise ProtocolFreezeError(
                    f"stage {stage_id} review requirement {identifier!r} lacks {required_field}"
                )
        if not str(row["expected"]).strip() or not str(row["observed"]).strip():
            raise ProtocolFreezeError(
                f"stage {stage_id} review requirement {identifier!r} has empty expected/observed"
            )
        evidence = row["evidence"]
        if not evidence or (isinstance(evidence, Sequence) and not isinstance(evidence, str)
                            and len(evidence) == 0):
            raise ProtocolFreezeError(
                f"stage {stage_id} review requirement {identifier!r} has no evidence"
            )
        if not str(row["scope"]).strip():
            raise ProtocolFreezeError(
                f"stage {stage_id} review requirement {identifier!r} has empty scope"
            )
        row_verdict = str(row.get("verdict", "")).strip().upper()
        expected_nonpass = acknowledged.get(identifier)
        if expected_nonpass is not None:
            if row_verdict != expected_nonpass:
                raise ProtocolFreezeError(
                    f"stage {stage_id} historical limitation {identifier!r} is "
                    f"{row_verdict!r}; expected persistent {expected_nonpass}"
                )
        elif row_verdict != PASS:
            label = row_verdict if row_verdict in NON_PASS else repr(row.get("verdict"))
            raise ProtocolFreezeError(
                f"stage {stage_id} review requirement {identifier!r} is {label}, not PASS"
            )
        indexed[identifier] = row

    missing_items = [item for item, normalized in zip(required_items, normalized_required)
                     if normalized not in indexed]
    if missing_items:
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict omits required items: {missing_items}"
        )
    missing_acknowledgements = sorted(set(acknowledged) - set(indexed))
    if missing_acknowledgements:
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict omits historical limitations: "
            f"{missing_acknowledgements}"
        )
    return {
        "agent_role": expected_agent_role,
        "reviewer_role": expected_reviewer_role,
        "requested_model": expected_model,
        "reasoning_effort": expected_effort,
        "fork_context": expected_fork,
        "scope_tokens": scope_tokens,
        "required_items": required_items,
        "gate_verdict_field": gate_field,
        "allowed_overall_verdicts": allowed_overall,
        "acknowledged_nonpass_items": dict(raw_acknowledged),
        "validated_items": sorted(indexed),
        "prospective_lock_readiness_verdict": PASS,
        "overall_verdict": observed_overall,
    }


def _validate_reviewed_sha256(
    verdict: Mapping[str, Any], expected_paths: Iterable[str], root: Path, *, stage_id: str
) -> dict[str, str]:
    """Bind an independent verdict to all stage inputs and every additionally declared byte."""
    raw = _require_mapping(verdict.get("reviewed_sha256"), "independent verdict reviewed_sha256")
    reviewed = {str(path): str(digest).lower() for path, digest in raw.items()}
    expected = set(expected_paths)
    missing = sorted(expected - set(reviewed))
    if missing:
        raise ProtocolFreezeError(
            f"stage {stage_id} independent verdict hash inventory omits: {missing}"
        )
    normalized: dict[str, str] = {}
    for raw_path, expected_sha in reviewed.items():
        if not _SHA256_RE.fullmatch(expected_sha):
            raise ProtocolFreezeError(
                f"stage {stage_id} independent verdict has invalid sha256 for {raw_path}"
            )
        path, relative = _lexical_repo_path(
            root, raw_path, f"stage {stage_id} independent verdict reviewed_sha256"
        )
        if relative in normalized:
            raise ProtocolFreezeError(
                f"stage {stage_id} independent verdict duplicates reviewed path {relative}"
            )
        if not path.is_file():
            raise ProtocolFreezeError(
                f"stage {stage_id} independently reviewed file is absent: {relative}"
            )
        observed = _sha256(path)
        if observed != expected_sha:
            raise ProtocolFreezeError(
                f"stage {stage_id} independent verdict is stale for {relative}: "
                f"reviewed={expected_sha}, current={observed}"
            )
        normalized[relative] = expected_sha
    return dict(sorted(normalized.items()))


def _load_spec(config_path: str | Path, project_root: str | Path | None) -> tuple[dict[str, Any], Path, Path]:
    root = Path(project_root or Path.cwd()).resolve()
    config, _ = _lexical_repo_path(root, os.fspath(config_path), "config_path")
    try:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolFreezeError(f"cannot load freeze config {config}: {exc}") from exc
    spec = dict(_require_mapping(raw, "freeze config"))
    if spec.get("format") != FORMAT:
        raise ProtocolFreezeError(f"freeze config format must be {FORMAT!r}")
    configured_stages = tuple(str(item).zfill(2) for item in spec.get("required_stages", []))
    if configured_stages != REQUIRED_STAGES:
        raise ProtocolFreezeError(
            f"required_stages must be exactly {list(REQUIRED_STAGES)}, got {list(configured_stages)}"
        )
    stages = _require_mapping(spec.get("stages"), "stages")
    if set(stages) != set(REQUIRED_STAGES):
        raise ProtocolFreezeError(f"stages keys must be exactly {list(REQUIRED_STAGES)}")
    _lexical_repo_path(root, spec.get("manifest_output"), "manifest_output")
    return spec, config, root


def _validate_disclosure(spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    disclosure = _require_mapping(
        spec.get("prior_test_coordinate_access"), "prior_test_coordinate_access"
    )
    if disclosure.get("occurred") is not True:
        raise ProtocolFreezeError("prior test coordinate access must be disclosed as occurred=true")
    if disclosure.get("historically_pristine_test_claim_permitted") is not False:
        raise ProtocolFreezeError(
            "historically_pristine_test_claim_permitted must be false"
        )
    statement = disclosure.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ProtocolFreezeError("prior test coordinate access disclosure needs a statement")
    paths = _config_path_list(root, disclosure.get("evidence_files"),
                              "prior_test_coordinate_access.evidence_files")
    return {
        "occurred": True,
        "historically_pristine_test_claim_permitted": False,
        "statement": statement.strip(),
        "evidence_files": [relative for _, relative in paths],
    }


def _validate_critical(spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    critical = _require_mapping(spec.get("critical_freeze_items"), "critical_freeze_items")
    result: dict[str, Any] = {}
    required_stage_coverage = {
        "selection_evidence": {"06"},
        "selected_checkpoints": {"08", "09"},
        "threshold_files": {"10"},
        "calibration_partitions": {"08", "09", "10"},
    }
    for category, stage_coverage in required_stage_coverage.items():
        entries = _require_list(critical.get(category), f"critical_freeze_items.{category}")
        parsed = []
        observed_stages: set[str] = set()
        labels: set[str] = set()
        for index, raw in enumerate(entries):
            entry = _require_mapping(raw, f"critical_freeze_items.{category}[{index}]")
            label = str(entry.get("label", "")).strip()
            stage = str(entry.get("stage", "")).zfill(2)
            if not label or label in labels:
                raise ProtocolFreezeError(f"{category} needs unique non-empty labels")
            if stage not in REQUIRED_STAGES:
                raise ProtocolFreezeError(f"{category}.{label} has invalid stage {stage!r}")
            _, relative = _lexical_repo_path(root, entry.get("path"), f"{category}.{label}.path")
            labels.add(label)
            observed_stages.add(stage)
            parsed.append({"label": label, "stage": stage, "path": relative})
        missing = sorted(stage_coverage - observed_stages)
        if missing:
            raise ProtocolFreezeError(f"{category} omits required stages {missing}")
        result[category] = parsed

    downstream = _require_mapping(critical.get("fixed_downstream"),
                                  "critical_freeze_items.fixed_downstream")
    model_id = str(downstream.get("model_id", "")).strip()
    checkpoint_sha = str(downstream.get("expected_checkpoint_sha256", "")).strip().lower()
    if not model_id:
        raise ProtocolFreezeError("fixed_downstream.model_id is required")
    if not _SHA256_RE.fullmatch(checkpoint_sha):
        raise ProtocolFreezeError("fixed_downstream.expected_checkpoint_sha256 is invalid")
    evidence = _config_path_list(root, downstream.get("evidence_files"),
                                 "critical_freeze_items.fixed_downstream.evidence_files")
    checkpoint_raw = downstream.get("actual_checkpoint_path")
    if not isinstance(checkpoint_raw, str) or not Path(checkpoint_raw).is_absolute():
        raise ProtocolFreezeError("fixed_downstream.actual_checkpoint_path must be absolute")
    checkpoint_path = Path(checkpoint_raw)
    if checkpoint_path.is_symlink() or checkpoint_path.suffix.lower() not in {".ckpt", ".pt"}:
        raise ProtocolFreezeError("fixed_downstream actual checkpoint must be a direct .ckpt/.pt file")
    replay_path, replay_relative = _lexical_repo_path(
        root, downstream.get("replay_contract_file"),
        "critical_freeze_items.fixed_downstream.replay_contract_file",
    )
    expected_fit_count = int(downstream.get("expected_fit_count", 0))
    if expected_fit_count <= 0:
        raise ProtocolFreezeError("fixed_downstream.expected_fit_count must be positive")
    raw_bindings = _require_mapping(
        downstream.get("replay_hash_bindings"),
        "critical_freeze_items.fixed_downstream.replay_hash_bindings",
    )
    required_binding_fields = {
        "thresholds_sha256", "calibration_scores_sha256",
        "prediction_manifest_sha256", "stage_config_sha256",
        "runtime_environment_sha256",
    }
    if set(raw_bindings) != required_binding_fields:
        raise ProtocolFreezeError(
            "fixed_downstream replay_hash_bindings must bind thresholds, calibration scores, "
            "prediction manifest, stage config, and actual runtime environment"
        )
    replay_hash_bindings: dict[str, str] = {}
    for field, raw_path in raw_bindings.items():
        _, relative = _lexical_repo_path(
            root, raw_path, f"fixed_downstream.replay_hash_bindings.{field}"
        )
        replay_hash_bindings[str(field)] = relative
    result["fixed_downstream"] = {
        "model_id": model_id,
        "expected_checkpoint_sha256": checkpoint_sha,
        "actual_checkpoint_path": str(checkpoint_path),
        "replay_contract_file": replay_relative,
        "expected_fit_count": expected_fit_count,
        "replay_hash_bindings": replay_hash_bindings,
        "evidence_files": [relative for _, relative in evidence],
    }
    return result


def _validate_protocol_dependencies(spec: Mapping[str, Any], root: Path) -> dict[str, list[str]]:
    raw = _require_mapping(spec.get("protocol_dependencies"), "protocol_dependencies")
    requirements = {
        "source_files": REQUIRED_PROTOCOL_SOURCE_DEPENDENCIES,
        "config_files": REQUIRED_PROTOCOL_CONFIG_DEPENDENCIES,
        "environment_files": REQUIRED_PROTOCOL_ENVIRONMENT_DEPENDENCIES,
    }
    result: dict[str, list[str]] = {}
    for category, required in requirements.items():
        paths = _config_path_list(root, raw.get(category), f"protocol_dependencies.{category}")
        observed = {relative for _, relative in paths}
        missing = sorted(required - observed)
        if missing:
            raise ProtocolFreezeError(
                f"protocol_dependencies.{category} omits required behavior dependencies: {missing}"
            )
        result[category] = sorted(observed)
    return result


def _validate_replay_contract(
    downstream: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    checkpoint = Path(str(downstream["actual_checkpoint_path"]))
    expected_sha = str(downstream["expected_checkpoint_sha256"])
    if not checkpoint.is_file():
        raise FreezeNotReady(f"fixed downstream checkpoint absent: {checkpoint}")
    observed_checkpoint_sha = _sha256(checkpoint)
    if observed_checkpoint_sha != expected_sha:
        raise ProtocolFreezeError(
            "fixed downstream checkpoint SHA-256 differs from the frozen expected digest"
        )
    replay_path, replay_relative = _lexical_repo_path(
        root, downstream["replay_contract_file"], "fixed_downstream.replay_contract_file"
    )
    if not replay_path.is_file():
        raise FreezeNotReady(f"fixed downstream replay contract absent: {replay_relative}")
    replay = _read_json(replay_path, "fixed downstream replay contract")
    checks = {
        "format": "poi-fixed-downstream-replay-contract-v1",
        "model_id": downstream["model_id"],
        "official_checkpoint_path": str(checkpoint),
        "official_checkpoint_sha256": expected_sha,
        "fit_count": int(downstream["expected_fit_count"]),
        "expected_fit_count": int(downstream["expected_fit_count"]),
        "test_rows_read": 0,
    }
    for field, expected in checks.items():
        if replay.get(field) != expected:
            raise ProtocolFreezeError(
                f"fixed downstream replay contract {field}={replay.get(field)!r}; "
                f"expected {expected!r}"
            )
    features = replay.get("features")
    fits = replay.get("fits")
    if not isinstance(features, list) or not features or len(set(map(str, features))) != len(features):
        raise ProtocolFreezeError("fixed downstream replay contract feature order is invalid")
    if not isinstance(fits, list) or len(fits) != int(downstream["expected_fit_count"]):
        raise ProtocolFreezeError("fixed downstream replay contract fit attestations are incomplete")
    model_seeds = replay.get("model_seeds")
    regions = replay.get("regions")
    n_estimators = int(replay.get("n_estimators", 0))
    classification_threshold = replay.get("classification_threshold")
    runtime = replay.get("runtime")
    if (
        not isinstance(model_seeds, list) or not model_seeds
        or not isinstance(regions, list) or not regions
        or len(model_seeds) * len(regions) != int(downstream["expected_fit_count"])
        or n_estimators <= 0
        or not isinstance(classification_threshold, (int, float))
        or not 0 <= float(classification_threshold) <= 1
        or not isinstance(runtime, Mapping)
        or not {
            "python", "numpy", "pandas", "python_executable", "tabpfn_version",
            "torch_version", "cuda_version", "restored_on_device",
        }.issubset(runtime)
    ):
        raise ProtocolFreezeError("fixed downstream replay model/runtime contract is incomplete")
    required_hashes = (
        "raw_data_sha256", "sample_manifest_sha256",
        "train_poi_order_sha256", "train_feature_matrix_sha256",
        "calibration_poi_order_sha256", "evaluation_poi_order_sha256",
        "thresholds_sha256", "calibration_scores_sha256", "prediction_manifest_sha256",
        "stage_config_sha256", "runtime_environment_sha256",
    )
    for field in required_hashes:
        if not _SHA256_RE.fullmatch(str(replay.get(field, ""))):
            raise ProtocolFreezeError(f"fixed downstream replay contract lacks valid {field}")
    for field, relative in downstream["replay_hash_bindings"].items():
        path, _ = _lexical_repo_path(root, relative, f"fixed downstream replay binding {field}")
        if not path.is_file():
            raise FreezeNotReady(f"fixed downstream replay-bound file absent: {relative}")
        if _sha256(path) != replay[field]:
            raise ProtocolFreezeError(
                f"fixed downstream replay {field} differs from {relative}"
            )
    runtime_relative = downstream["replay_hash_bindings"]["runtime_environment_sha256"]
    runtime_path, _ = _lexical_repo_path(
        root, runtime_relative, "fixed downstream actual runtime environment"
    )
    runtime_environment = _read_json(runtime_path, "fixed downstream actual runtime environment")
    required_runtime_fields = {
        "format", "execution_scope", "python_version", "python_executable", "platform",
        "numpy", "pandas", "tabpfn_version", "torch_version", "cuda_version",
        "cuda_available", "gpu_device_name", "requested_device", "restored_on_device",
        "installed_distributions", "test_rows_read",
    }
    if (
        set(runtime_environment) != required_runtime_fields
        or runtime_environment.get("format") != "poi-stage10-runtime-environment-v1"
        or runtime_environment.get("execution_scope")
        != "stage10_gpu_model_fitting_and_validation_evaluation"
        or runtime_environment.get("test_rows_read") != 0
        or not isinstance(runtime_environment.get("installed_distributions"), Mapping)
        or not isinstance(runtime_environment.get("cuda_available"), bool)
        or not str(runtime_environment.get("requested_device", "")).strip()
        or not str(runtime_environment.get("restored_on_device", "")).strip()
        or (
            runtime_environment.get("cuda_available")
            and not str(runtime_environment.get("gpu_device_name", "")).strip()
        )
    ):
        raise ProtocolFreezeError("fixed downstream actual runtime environment is incomplete")
    runtime_environment_bindings = {
        "python": "python_version", "python_executable": "python_executable",
        "numpy": "numpy", "pandas": "pandas", "tabpfn_version": "tabpfn_version",
        "torch_version": "torch_version", "cuda_version": "cuda_version",
        "cuda_available": "cuda_available", "gpu_device_name": "gpu_device_name",
        "requested_device": "requested_device", "restored_on_device": "restored_on_device",
    }
    for replay_field, environment_field in runtime_environment_bindings.items():
        if runtime.get(replay_field) != runtime_environment.get(environment_field):
            raise ProtocolFreezeError(
                f"fixed downstream replay/runtime environment differs for {replay_field}"
            )
    distributions = runtime_environment["installed_distributions"]
    for package, runtime_field in (
        ("numpy", "numpy"), ("pandas", "pandas"),
        ("tabpfn", "tabpfn_version"), ("torch", "torch_version"),
    ):
        if distributions.get(package) != runtime_environment.get(runtime_field):
            raise ProtocolFreezeError(
                f"fixed downstream runtime package inventory differs for {package}"
            )
    requested_device = str(runtime_environment["requested_device"])
    if (
        requested_device != str(runtime_environment["restored_on_device"])
        or (requested_device.startswith("cuda") and not runtime_environment["cuda_available"])
    ):
        raise ProtocolFreezeError("fixed downstream runtime device/restore contract is inconsistent")
    source_snapshots = _require_mapping(
        replay.get("source_snapshots_sha256"),
        "fixed downstream replay source_snapshots_sha256",
    )
    expected_snapshot_names = {
        "defense_pipeline.py", "selected_explain.py", "defense_evaluate.py", "data.py",
        "attack_evaluate.py", "ablation.py", "benchmark.py",
    }
    observed_snapshot_names: set[str] = set()
    for raw_path, raw_sha in source_snapshots.items():
        path, relative = _lexical_repo_path(
            root, raw_path, "fixed downstream replay source snapshot"
        )
        expected_source_sha = str(raw_sha)
        if not path.is_file() or not _SHA256_RE.fullmatch(expected_source_sha):
            raise ProtocolFreezeError(
                f"fixed downstream replay source snapshot is absent/invalid: {relative}"
            )
        if _sha256(path) != expected_source_sha:
            raise ProtocolFreezeError(
                f"fixed downstream replay source snapshot hash differs: {relative}"
            )
        expected_snapshot_path = (
            replay_path.parent / "work" / "fixed_downstream_replay_sources" / path.name
        )
        if path != expected_snapshot_path:
            raise ProtocolFreezeError(
                f"fixed downstream replay source snapshot has unexpected location: {relative}"
            )
        observed_snapshot_names.add(path.name)
    if observed_snapshot_names != expected_snapshot_names:
        raise ProtocolFreezeError(
            "fixed downstream replay source snapshot coverage is incomplete"
        )
    fit_keys: set[tuple[int, str]] = set()
    common_probe_matrix_sha: str | None = None
    categorical_features = list(replay.get("categorical_features", []))
    expected_categorical_indices = [
        index for index, feature in enumerate(features) if feature in categorical_features
    ]
    train_rows = int(replay.get("train_rows", 0))
    if train_rows <= 0 or not categorical_features:
        raise ProtocolFreezeError("fixed downstream train/categorical recipe is incomplete")
    for index, raw_fit in enumerate(fits):
        fit = _require_mapping(raw_fit, f"fixed downstream fits[{index}]")
        if fit.get("format") != "poi-tabpfn25-fit-replay-v1":
            raise ProtocolFreezeError("fixed downstream fit attestation format is invalid")
        if fit.get("official_checkpoint_sha256") != expected_sha:
            raise ProtocolFreezeError("fixed downstream fit checkpoint hash is inconsistent")
        if fit.get("official_checkpoint_path") != str(checkpoint):
            raise ProtocolFreezeError("fixed downstream fit checkpoint path is inconsistent")
        if fit.get("model_id") != downstream["model_id"]:
            raise ProtocolFreezeError("fixed downstream fit model identity is inconsistent")
        if list(fit.get("feature_order", [])) != features:
            raise ProtocolFreezeError("fixed downstream fit feature order is inconsistent")
        key = (int(fit.get("model_seed", fit.get("seed"))), str(fit.get("region", "")))
        if not key[1] or key in fit_keys:
            raise ProtocolFreezeError("fixed downstream fit seed/region identity is invalid")
        fit_keys.add(key)
        for field in (
            "train_matrix_sha256", "truth_sha256", "probe_matrix_sha256",
            "probe_prediction_sha256", "probe_repeat_prediction_sha256",
            "train_poi_order_sha256",
        ):
            if not _SHA256_RE.fullmatch(str(fit.get(field, ""))):
                raise ProtocolFreezeError(
                    f"fixed downstream fit attestation lacks valid {field}"
                )
        fitted_state_path, fitted_state_relative = _lexical_repo_path(
            root, fit.get("fitted_state_path"),
            f"fixed downstream fits[{index}].fitted_state_path",
        )
        expected_fit_state_directory = (
            replay_path.parent / "work" / "fixed_downstream_fit_states"
        )
        expected_region_index = regions.index(key[1]) if key[1] in regions else -1
        expected_fit_state_name = (
            f"modelseed{key[0]}__region{expected_region_index:02d}.tabpfn_fit"
        )
        if (
            fit.get("fitted_state_format") != "tabpfn_official_fit_state_v1"
            or fit.get("fitted_state_api")
            != "TabPFNClassifier.save_fit_state/load_from_fit_state"
            or not fitted_state_path.is_file()
            or fitted_state_path.suffix != ".tabpfn_fit"
            or not _SHA256_RE.fullmatch(str(fit.get("fitted_state_sha256", "")))
            or _sha256(fitted_state_path) != fit.get("fitted_state_sha256")
            or fitted_state_path.stat().st_size != int(fit.get("fitted_state_size_bytes", -1))
            or fitted_state_path.parent != expected_fit_state_directory
            or fitted_state_path.name != expected_fit_state_name
        ):
            raise ProtocolFreezeError(
                f"fixed downstream official fitted state is invalid: {fitted_state_relative}"
            )
        probe_indices = fit.get("probe_indices")
        if (
            int(fit.get("train_rows", -1)) != train_rows
            or fit.get("train_matrix_dtype") != "float32"
            or fit.get("train_matrix_sha256") != replay["train_feature_matrix_sha256"]
            or fit.get("train_poi_order_sha256") != replay["train_poi_order_sha256"]
            or fit.get("categorical_feature_indices") != expected_categorical_indices
            or not isinstance(probe_indices, list) or not probe_indices
            or not all(isinstance(value, int) for value in probe_indices)
            or probe_indices != sorted(set(probe_indices))
            or min(probe_indices) < 0 or max(probe_indices) >= train_rows
            or len(probe_indices) > 64
            or fit.get("fit_mode") != "fit_preprocessors"
            or fit.get("memory_saving_mode") != "auto"
        ):
            raise ProtocolFreezeError(
                "fixed downstream per-fit train/preprocessor/probe recipe is inconsistent"
            )
        if common_probe_matrix_sha is None:
            common_probe_matrix_sha = str(fit["probe_matrix_sha256"])
        elif fit["probe_matrix_sha256"] != common_probe_matrix_sha:
            raise ProtocolFreezeError("fixed downstream probe matrix changes across fits")
        if "classification_threshold" not in fit:
            raise ProtocolFreezeError(
                "fixed downstream fit attestation lacks classification_threshold"
            )
        if (
            int(fit.get("n_estimators", -1)) != n_estimators
            or float(fit["classification_threshold"]) != float(classification_threshold)
        ):
            raise ProtocolFreezeError(
                "fixed downstream fit hyperparameters differ from the replay contract"
            )
        if fit["probe_prediction_sha256"] != fit["probe_repeat_prediction_sha256"]:
            raise ProtocolFreezeError(
                "fixed downstream same-input probe replay checksum differs"
            )
        if fit.get("restored_probe_prediction_sha256") != fit["probe_prediction_sha256"]:
            raise ProtocolFreezeError(
                "fixed downstream persisted-state reload changes probe prediction"
            )
        for runtime_field in (
            "tabpfn_version", "torch_version", "cuda_version", "cuda_available",
            "gpu_device_name", "requested_device", "restored_on_device",
        ):
            if fit.get(runtime_field) != runtime.get(runtime_field):
                raise ProtocolFreezeError(
                    f"fixed downstream fit runtime differs for {runtime_field}"
                )
        if int(fit.get("seed", -1)) != key[0] or int(fit.get("model_seed", -1)) != key[0]:
            raise ProtocolFreezeError("fixed downstream fit seed identity is inconsistent")
    expected_fit_keys = {
        (int(seed), str(region)) for seed in model_seeds for region in regions
    }
    if fit_keys != expected_fit_keys:
        raise ProtocolFreezeError("fixed downstream fit seed/region coverage is incomplete")
    return {
        "replay_contract_file": replay_relative,
        "replay_contract_sha256": _sha256(replay_path),
        "actual_checkpoint_path": str(checkpoint),
        "actual_checkpoint_sha256": observed_checkpoint_sha,
        "actual_checkpoint_size_bytes": checkpoint.stat().st_size,
        "validated_fit_attestations": len(fits),
        "validated_official_fitted_states": len(fits),
    }


def _stage_contracts(spec: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    stages = _require_mapping(spec["stages"], "stages")
    result: dict[str, dict[str, Any]] = {}
    for stage_id in REQUIRED_STAGES:
        raw = _require_mapping(stages[stage_id], f"stages.{stage_id}")
        artifact_roots = _config_path_list(root, raw.get("artifact_roots"),
                                           f"stages.{stage_id}.artifact_roots")
        config_files = _config_path_list(root, raw.get("config_files"),
                                         f"stages.{stage_id}.config_files")
        code_files = _config_path_list(root, raw.get("code_files"),
                                       f"stages.{stage_id}.code_files")
        required_artifacts = _config_path_list(
            root, raw.get("required_artifact_files"),
            f"stages.{stage_id}.required_artifact_files"
        )
        review = _require_mapping(raw.get("independent_review"),
                                  f"stages.{stage_id}.independent_review")
        review_path, review_relative = _lexical_repo_path(
            root, review.get("verdict_file"), f"stages.{stage_id}.independent_review.verdict_file"
        )
        review_contract = dict(review)
        review_contract.pop("verdict_file", None)
        result[stage_id] = {
            "artifact_roots": artifact_roots,
            "config_files": config_files,
            "code_files": code_files,
            "required_artifact_files": required_artifacts,
            "review_path": review_path,
            "review_relative": review_relative,
            "review_contract": review_contract,
        }
    return result


def _common_review_contract(spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    raw = _require_mapping(
        spec.get("common_independent_review"), "common_independent_review"
    )
    review_path, review_relative = _lexical_repo_path(
        root, raw.get("verdict_file"), "common_independent_review.verdict_file"
    )
    contract = dict(raw)
    contract.pop("verdict_file", None)
    return {
        "review_path": review_path,
        "review_relative": review_relative,
        "review_contract": contract,
    }


def _common_review_expected_paths(
    config_relative: str, dependencies: Mapping[str, list[str]]
) -> list[str]:
    return sorted({config_relative, *(path for paths in dependencies.values() for path in paths)})


def _stage_review_expected_paths(
    stage: Mapping[str, Any], root: Path
) -> list[str]:
    expected: list[str] = []
    for artifact_root, relative_root in stage["artifact_roots"]:
        if artifact_root.is_dir():
            expected.extend(_artifact_inventory(root, artifact_root, relative_root))
    expected.extend(relative for _, relative in stage["config_files"])
    expected.extend(relative for _, relative in stage["code_files"])
    return sorted(set(expected))


def _path_checks(
    stages: Mapping[str, Mapping[str, Any]],
    critical: Mapping[str, Any],
    disclosure: Mapping[str, Any],
    root: Path,
    config_relative: str,
    protocol_dependencies: Mapping[str, list[str]],
    common_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [{
        "kind": "protocol_config", "stage": None, "path": config_relative,
        "status": PASS, "detail": "freeze configuration exists",
    }]
    for stage_id, stage in stages.items():
        for path, relative in stage["artifact_roots"]:
            status = PASS if path.is_dir() else "NOT_RUN"
            checks.append({"kind": "artifact_root", "stage": stage_id, "path": relative,
                           "status": status,
                           "detail": "directory exists" if status == PASS else "directory absent"})
        for kind in ("config_files", "code_files", "required_artifact_files"):
            for path, relative in stage[kind]:
                status = PASS if path.is_file() else "NOT_RUN"
                checks.append({"kind": kind[:-1], "stage": stage_id, "path": relative,
                               "status": status,
                               "detail": "file exists" if status == PASS else "file absent"})
        review_path = stage["review_path"]
        if not review_path.is_file():
            checks.append({"kind": "independent_review", "stage": stage_id,
                           "path": stage["review_relative"], "status": "NOT_RUN",
                           "detail": "independent verdict absent"})
        else:
            try:
                verdict = _read_json(review_path, f"stage {stage_id} independent verdict")
                validate_independent_verdict(verdict, stage["review_contract"], stage_id=stage_id)
                _validate_reviewed_sha256(
                    verdict, _stage_review_expected_paths(stage, root), root, stage_id=stage_id
                )
            except ProtocolFreezeError as exc:
                checks.append({"kind": "independent_review", "stage": stage_id,
                               "path": stage["review_relative"], "status": "FAIL",
                               "detail": str(exc)})
            else:
                checks.append({"kind": "independent_review", "stage": stage_id,
                               "path": stage["review_relative"], "status": PASS,
                               "detail": "independent PASS gate validated"})

    for category in (
        "selection_evidence", "selected_checkpoints", "threshold_files",
        "calibration_partitions",
    ):
        for entry in critical[category]:
            path, _ = _lexical_repo_path(root, entry["path"], f"{category}.{entry['label']}")
            status = PASS if path.is_file() else "NOT_RUN"
            checks.append({"kind": category, "stage": entry["stage"], "path": entry["path"],
                           "status": status,
                           "detail": "file exists" if status == PASS else "file absent"})
    for relative in critical["fixed_downstream"]["evidence_files"]:
        path, _ = _lexical_repo_path(root, relative, "fixed_downstream.evidence_files")
        status = PASS if path.is_file() else "NOT_RUN"
        checks.append({"kind": "fixed_downstream", "stage": "10", "path": relative,
                       "status": status,
                       "detail": "file exists" if status == PASS else "file absent"})
    checkpoint = Path(critical["fixed_downstream"]["actual_checkpoint_path"])
    checkpoint_status = "NOT_RUN"
    checkpoint_detail = "actual checkpoint absent"
    if checkpoint.is_file():
        observed = _sha256(checkpoint)
        if observed == critical["fixed_downstream"]["expected_checkpoint_sha256"]:
            checkpoint_status, checkpoint_detail = PASS, "actual checkpoint hash matches"
        else:
            checkpoint_status, checkpoint_detail = "FAIL", "actual checkpoint hash mismatches"
    checks.append({
        "kind": "fixed_downstream_checkpoint", "stage": "10", "path": str(checkpoint),
        "status": checkpoint_status, "detail": checkpoint_detail,
    })
    replay_relative = critical["fixed_downstream"]["replay_contract_file"]
    replay_path, _ = _lexical_repo_path(root, replay_relative, "fixed_downstream.replay_contract_file")
    if not replay_path.is_file():
        checks.append({
            "kind": "fixed_downstream_replay_contract", "stage": "10",
            "path": replay_relative, "status": "NOT_RUN", "detail": "replay contract absent",
        })
    else:
        try:
            _validate_replay_contract(critical["fixed_downstream"], root)
        except ProtocolFreezeError as exc:
            checks.append({
                "kind": "fixed_downstream_replay_contract", "stage": "10",
                "path": replay_relative, "status": "FAIL", "detail": str(exc),
            })
        else:
            checks.append({
                "kind": "fixed_downstream_replay_contract", "stage": "10",
                "path": replay_relative, "status": PASS,
                "detail": "checkpoint, fitted-state replay, and context binding validated",
            })
    dependency_kind = {
        "source_files": "protocol_source_dependency",
        "config_files": "protocol_config_dependency",
        "environment_files": "protocol_environment_dependency",
    }
    for category, relatives in protocol_dependencies.items():
        for relative in relatives:
            path, _ = _lexical_repo_path(root, relative, f"protocol_dependencies.{category}")
            status = PASS if path.is_file() else "NOT_RUN"
            checks.append({
                "kind": dependency_kind[category], "stage": None, "path": relative,
                "status": status, "detail": "file exists" if status == PASS else "file absent",
            })
    common_path = common_review["review_path"]
    if not common_path.is_file():
        checks.append({
            "kind": "common_independent_review", "stage": None,
            "path": common_review["review_relative"], "status": "NOT_RUN",
            "detail": "common dependency/config verdict absent",
        })
    else:
        try:
            verdict = _read_json(common_path, "common independent verdict")
            validate_independent_verdict(
                verdict, common_review["review_contract"], stage_id="COMMON"
            )
            _validate_reviewed_sha256(
                verdict,
                _common_review_expected_paths(config_relative, protocol_dependencies),
                root,
                stage_id="COMMON",
            )
        except ProtocolFreezeError as exc:
            checks.append({
                "kind": "common_independent_review", "stage": None,
                "path": common_review["review_relative"], "status": "FAIL",
                "detail": str(exc),
            })
        else:
            checks.append({
                "kind": "common_independent_review", "stage": None,
                "path": common_review["review_relative"], "status": PASS,
                "detail": "fresh common dependency/config byte inventory validated",
            })
    for relative in disclosure["evidence_files"]:
        path, _ = _lexical_repo_path(root, relative, "prior_test_coordinate_access.evidence_files")
        status = PASS if path.is_file() else "NOT_RUN"
        checks.append({"kind": "test_access_disclosure", "stage": "01", "path": relative,
                       "status": status,
                       "detail": "file exists" if status == PASS else "file absent"})
    return checks


def preflight(config_path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    """Check readiness without creating an output and without opening any dataset."""

    spec, config, root = _load_spec(config_path, project_root)
    stages = _stage_contracts(spec, root)
    critical = _validate_critical(spec, root)
    disclosure = _validate_disclosure(spec, root)
    dependencies = _validate_protocol_dependencies(spec, root)
    common_review = _common_review_contract(spec, root)
    checks = _path_checks(
        stages, critical, disclosure, root, config.relative_to(root).as_posix(), dependencies,
        common_review,
    )
    statuses = {check["status"] for check in checks}
    status = "FAIL" if "FAIL" in statuses else ("NOT_RUN" if "NOT_RUN" in statuses else PASS)
    return {
        "format": FORMAT,
        "status": status,
        "ready": status == PASS,
        "test_data_accessed": False,
        "manifest_written": False,
        "checks": checks,
    }


def _artifact_inventory(root: Path, artifact_root: Path, relative_root: str) -> list[str]:
    files: list[str] = []
    for path in sorted(artifact_root.rglob("*")):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ProtocolFreezeError(f"artifact root contains symlink: {relative_path}")
        if path.is_file():
            files.append(relative_path)
        elif not path.is_dir():
            raise ProtocolFreezeError(f"artifact root contains unsupported entry: {relative_path}")
    if not files:
        raise ProtocolFreezeError(f"artifact root has no files: {relative_root}")
    return files


def _add_file(
    catalog: dict[str, dict[str, Any]], root: Path, relative: str, kind: str, stage: str | None
) -> None:
    path, normalized = _lexical_repo_path(root, relative, f"frozen file {relative}")
    if not path.is_file():
        raise FreezeNotReady(f"required frozen file absent: {normalized}")
    entry = catalog.setdefault(normalized, {
        "path": normalized,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "kinds": [],
        "stages": [],
    })
    if kind not in entry["kinds"]:
        entry["kinds"].append(kind)
    if stage is not None and stage not in entry["stages"]:
        entry["stages"].append(stage)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}
    return {"commit": commit, "worktree_dirty": bool(status_output.strip())}


def _exclusive_read_only_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def create_freeze_manifest(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Create an exclusive, read-only manifest after every gate passes."""

    readiness = preflight(config_path, project_root=project_root)
    if not readiness["ready"]:
        failures = [check for check in readiness["checks"] if check["status"] != PASS]
        raise FreezeNotReady(
            f"protocol freeze is {readiness['status']}; {len(failures)} blocking checks remain"
        )
    spec, config, root = _load_spec(config_path, project_root)
    stages = _stage_contracts(spec, root)
    critical = _validate_critical(spec, root)
    disclosure = _validate_disclosure(spec, root)
    dependencies = _validate_protocol_dependencies(spec, root)
    common_review = _common_review_contract(spec, root)
    replay_validation = _validate_replay_contract(critical["fixed_downstream"], root)
    _, output_relative = _lexical_repo_path(root, spec["manifest_output"], "manifest_output")
    output = root / output_relative

    catalog: dict[str, dict[str, Any]] = {}
    config_relative = config.relative_to(root).as_posix()
    _add_file(catalog, root, config_relative, "protocol_config", None)
    manifest_stages: dict[str, Any] = {}
    inventories: dict[str, list[str]] = {}
    for stage_id, stage in stages.items():
        artifact_files: list[str] = []
        for artifact_root, relative_root in stage["artifact_roots"]:
            inventory = _artifact_inventory(root, artifact_root, relative_root)
            inventories[relative_root] = inventory
            artifact_files.extend(inventory)
            for relative in inventory:
                _add_file(catalog, root, relative, "artifact", stage_id)
        required_artifacts = [relative for _, relative in stage["required_artifact_files"]]
        missing_from_inventory = sorted(set(required_artifacts) - set(artifact_files))
        if missing_from_inventory:
            raise ProtocolFreezeError(
                f"stage {stage_id} required files are outside its artifact inventories: "
                f"{missing_from_inventory}"
            )
        for _, relative in stage["config_files"]:
            _add_file(catalog, root, relative, "stage_config", stage_id)
        for _, relative in stage["code_files"]:
            _add_file(catalog, root, relative, "source_code", stage_id)
        verdict = _read_json(stage["review_path"], f"stage {stage_id} independent verdict")
        validated_review = validate_independent_verdict(
            verdict, stage["review_contract"], stage_id=stage_id
        )
        reviewed_sha256 = _validate_reviewed_sha256(
            verdict, sorted(set(artifact_files + [
                relative for _, relative in stage["config_files"] + stage["code_files"]
            ])), root, stage_id=stage_id,
        )
        _add_file(catalog, root, stage["review_relative"], "independent_review", stage_id)
        manifest_stages[stage_id] = {
            "artifact_roots": [relative for _, relative in stage["artifact_roots"]],
            "config_files": [relative for _, relative in stage["config_files"]],
            "code_files": [relative for _, relative in stage["code_files"]],
            "required_artifact_files": required_artifacts,
            "independent_review": {
                "verdict_file": stage["review_relative"],
                "reviewed_sha256": reviewed_sha256,
                **validated_review,
            },
        }

    for category in (
        "selection_evidence", "selected_checkpoints", "threshold_files",
        "calibration_partitions",
    ):
        for entry in critical[category]:
            _add_file(catalog, root, entry["path"], category, entry["stage"])
    for relative in critical["fixed_downstream"]["evidence_files"]:
        _add_file(catalog, root, relative, "fixed_downstream", "10")
    _add_file(
        catalog, root, critical["fixed_downstream"]["replay_contract_file"],
        "fixed_downstream_replay_contract", "10",
    )
    for relative in critical["fixed_downstream"]["replay_hash_bindings"].values():
        _add_file(catalog, root, relative, "fixed_downstream_replay_binding", "10")
    dependency_kinds = {
        "source_files": "protocol_source_dependency",
        "config_files": "protocol_config_dependency",
        "environment_files": "protocol_environment_dependency",
    }
    for category, relatives in dependencies.items():
        for relative in relatives:
            _add_file(catalog, root, relative, dependency_kinds[category], None)
    common_verdict = _read_json(common_review["review_path"], "common independent verdict")
    validated_common_review = validate_independent_verdict(
        common_verdict, common_review["review_contract"], stage_id="COMMON"
    )
    common_reviewed_sha256 = _validate_reviewed_sha256(
        common_verdict,
        _common_review_expected_paths(config_relative, dependencies),
        root,
        stage_id="COMMON",
    )
    _add_file(
        catalog, root, common_review["review_relative"], "common_independent_review", None
    )
    for relative in disclosure["evidence_files"]:
        _add_file(catalog, root, relative, "test_access_disclosure", "01")

    for entry in catalog.values():
        entry["kinds"].sort()
        entry["stages"].sort()
    manifest = {
        "format": FORMAT,
        "status": "PROSPECTIVE_LOCK_FROZEN_PENDING_LOCKED_TEST_EXECUTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_stages": list(REQUIRED_STAGES),
        "authorization_scope": "prospective_lock_readiness_only",
        "scientific_completion_claim_permitted": False,
        "test_evaluation_performed": False,
        "test_data_accessed_during_freeze": False,
        "historically_pristine_test_claim_permitted": False,
        "locked_test_descriptor": "historically_exposed_coordinates_prospectively_locked",
        "project": _git_state(root),
        "protocol_config": config_relative,
        "source_binding_interpretation": {
            "artifact_execution_source": (
                "execution provenance snapshots and commits inside each artifact inventory are "
                "authoritative for what produced that artifact"
            ),
            "current_repository_source": (
                "separately hashed prospective replay, verification, and locked-test behavior; "
                "it is not presumed byte-identical to an earlier execution snapshot"
            ),
            "independent_review_requirement": (
                "each stage review must reconcile its execution snapshot/commit with produced "
                "bytes and separately inspect current replay dependencies"
            ),
        },
        "stages": manifest_stages,
        "artifact_inventories": inventories,
        "critical_freeze_items": critical,
        "fixed_downstream_external_attestation": replay_validation,
        "protocol_dependencies": dependencies,
        "common_independent_review": {
            "verdict_file": common_review["review_relative"],
            "reviewed_sha256": common_reviewed_sha256,
            **validated_common_review,
        },
        "prior_test_coordinate_access": disclosure,
        "files": sorted(catalog.values(), key=lambda item: item["path"]),
    }

    # Detect a changed input before making the one-shot manifest visible.
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise ProtocolFreezeError(f"input changed while freezing: {entry['path']}")
    checkpoint = Path(replay_validation["actual_checkpoint_path"])
    if (
        checkpoint.stat().st_size != replay_validation["actual_checkpoint_size_bytes"]
        or _sha256(checkpoint) != replay_validation["actual_checkpoint_sha256"]
    ):
        raise ProtocolFreezeError("fixed downstream checkpoint changed while freezing")
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _exclusive_read_only_write(output, payload)
    try:
        verify_freeze_manifest(output, project_root=root)
    except Exception:
        # Keep the immutable failed record for forensic inspection; never overwrite it.
        raise
    return manifest


def _verify_inventory(root: Path, relative_root: str, expected: Sequence[str]) -> None:
    artifact_root, normalized = _lexical_repo_path(root, relative_root, "artifact_inventory root")
    if not artifact_root.is_dir():
        raise ProtocolFreezeError(f"frozen artifact root absent: {normalized}")
    observed = _artifact_inventory(root, artifact_root, normalized)
    if list(expected) != observed:
        missing = sorted(set(expected) - set(observed))
        added = sorted(set(observed) - set(expected))
        raise ProtocolFreezeError(
            f"artifact inventory changed for {normalized}: missing={missing}, added={added}"
        )


def verify_freeze_manifest(
    manifest_path: str | Path, *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Revalidate manifest structure, review semantics, inventories, sizes, and hashes."""

    root = Path(project_root or Path.cwd()).resolve()
    manifest_file, manifest_relative = _lexical_repo_path(root, os.fspath(manifest_path),
                                                          "manifest_path")
    manifest = _read_json(manifest_file, "freeze manifest")
    if manifest.get("format") != FORMAT:
        raise ProtocolFreezeError("unsupported freeze manifest format")
    if manifest.get("status") != "PROSPECTIVE_LOCK_FROZEN_PENDING_LOCKED_TEST_EXECUTION":
        raise ProtocolFreezeError("freeze manifest status is not locked-test pending")
    if tuple(manifest.get("required_stages", [])) != REQUIRED_STAGES:
        raise ProtocolFreezeError("freeze manifest required stages are incomplete")
    if manifest.get("authorization_scope") != "prospective_lock_readiness_only":
        raise ProtocolFreezeError("freeze manifest overstates its authorization scope")
    if manifest.get("scientific_completion_claim_permitted") is not False:
        raise ProtocolFreezeError("freeze manifest improperly permits a completion claim")
    if manifest.get("test_evaluation_performed") is not False:
        raise ProtocolFreezeError("freeze manifest claims test evaluation occurred")
    if manifest.get("test_data_accessed_during_freeze") is not False:
        raise ProtocolFreezeError("freeze manifest claims test data access during freeze")
    if manifest.get("historically_pristine_test_claim_permitted") is not False:
        raise ProtocolFreezeError("freeze manifest improperly permits a pristine-test claim")
    if manifest.get("locked_test_descriptor") != \
            "historically_exposed_coordinates_prospectively_locked":
        raise ProtocolFreezeError("freeze manifest mislabels the historically exposed test")
    source_binding = _require_mapping(
        manifest.get("source_binding_interpretation"), "source_binding_interpretation"
    )
    if set(source_binding) != {
        "artifact_execution_source", "current_repository_source",
        "independent_review_requirement",
    }:
        raise ProtocolFreezeError("freeze manifest source-binding roles are incomplete")
    if stat.S_IMODE(manifest_file.stat().st_mode) & 0o222:
        raise ProtocolFreezeError(f"freeze manifest is writable: {manifest_relative}")

    disclosure = _require_mapping(manifest.get("prior_test_coordinate_access"),
                                  "prior_test_coordinate_access")
    if disclosure.get("occurred") is not True or \
            disclosure.get("historically_pristine_test_claim_permitted") is not False:
        raise ProtocolFreezeError("prior test coordinate access disclosure is invalid")
    if not str(disclosure.get("statement", "")).strip():
        raise ProtocolFreezeError("prior test coordinate access disclosure has no statement")

    files = _require_list(manifest.get("files"), "files")
    file_index: dict[str, Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(files):
        entry = _require_mapping(raw_entry, f"files[{index}]")
        path, relative = _lexical_repo_path(root, entry.get("path"), f"files[{index}].path")
        if relative in file_index:
            raise ProtocolFreezeError(f"duplicate frozen file: {relative}")
        expected_sha = str(entry.get("sha256", ""))
        if not _SHA256_RE.fullmatch(expected_sha):
            raise ProtocolFreezeError(f"invalid sha256 for {relative}")
        if not path.is_file():
            raise ProtocolFreezeError(f"frozen file absent: {relative}")
        observed_size = path.stat().st_size
        if observed_size != entry.get("size_bytes"):
            raise ProtocolFreezeError(f"size mismatch for frozen file: {relative}")
        if _sha256(path) != expected_sha:
            raise ProtocolFreezeError(f"sha256 mismatch for frozen file: {relative}")
        file_index[relative] = entry

    protocol_config = str(manifest.get("protocol_config", ""))
    if protocol_config not in file_index or "protocol_config" not in file_index[protocol_config].get(
        "kinds", []
    ):
        raise ProtocolFreezeError("protocol configuration is not explicitly hashed")

    inventories = _require_mapping(manifest.get("artifact_inventories"),
                                   "artifact_inventories")
    for relative_root, expected in inventories.items():
        if not isinstance(expected, list):
            raise ProtocolFreezeError(f"artifact inventory must be a list: {relative_root}")
        _verify_inventory(root, str(relative_root), expected)
        for relative in expected:
            if relative not in file_index:
                raise ProtocolFreezeError(f"inventory file is not hashed: {relative}")

    critical = _require_mapping(manifest.get("critical_freeze_items"),
                                "critical_freeze_items")
    # Reuse the structural validator on manifest-relative critical entries.
    normalized_critical = _validate_critical({"critical_freeze_items": critical}, root)
    replay_validation = _validate_replay_contract(normalized_critical["fixed_downstream"], root)
    for category in (
        "selection_evidence", "selected_checkpoints", "threshold_files",
        "calibration_partitions",
    ):
        for entry in normalized_critical[category]:
            if entry["path"] not in file_index:
                raise ProtocolFreezeError(f"critical file is not hashed: {entry['path']}")
            if category not in file_index[entry["path"]].get("kinds", []):
                raise ProtocolFreezeError(
                    f"critical file lacks {category} role: {entry['path']}"
                )
    for relative in normalized_critical["fixed_downstream"]["evidence_files"]:
        if relative not in file_index:
            raise ProtocolFreezeError(f"fixed downstream evidence is not hashed: {relative}")
        if "fixed_downstream" not in file_index[relative].get("kinds", []):
            raise ProtocolFreezeError(f"fixed downstream evidence lacks role: {relative}")
    replay_relative = normalized_critical["fixed_downstream"]["replay_contract_file"]
    if replay_relative not in file_index or "fixed_downstream_replay_contract" not in \
            file_index[replay_relative].get("kinds", []):
        raise ProtocolFreezeError("fixed downstream replay contract is not explicitly hashed")
    for relative in normalized_critical["fixed_downstream"]["replay_hash_bindings"].values():
        if relative not in file_index or "fixed_downstream_replay_binding" not in \
                file_index[relative].get("kinds", []):
            raise ProtocolFreezeError(
                f"fixed downstream replay binding is not explicitly hashed: {relative}"
            )
    external = _require_mapping(
        manifest.get("fixed_downstream_external_attestation"),
        "fixed_downstream_external_attestation",
    )
    for field in (
        "actual_checkpoint_path", "actual_checkpoint_sha256", "actual_checkpoint_size_bytes",
        "replay_contract_file", "replay_contract_sha256", "validated_fit_attestations",
        "validated_official_fitted_states",
    ):
        if external.get(field) != replay_validation.get(field):
            raise ProtocolFreezeError(
                f"fixed downstream external attestation changed for {field}"
            )
    for relative in disclosure.get("evidence_files", []):
        if relative not in file_index:
            raise ProtocolFreezeError(f"test-access disclosure evidence is not hashed: {relative}")
        if "test_access_disclosure" not in file_index[relative].get("kinds", []):
            raise ProtocolFreezeError(f"test-access disclosure evidence lacks role: {relative}")

    stages = _require_mapping(manifest.get("stages"), "stages")
    if set(stages) != set(REQUIRED_STAGES):
        raise ProtocolFreezeError("freeze manifest stage keys are incomplete")
    for stage_id in REQUIRED_STAGES:
        stage = _require_mapping(stages[stage_id], f"stages.{stage_id}")
        review = _require_mapping(stage.get("independent_review"),
                                  f"stages.{stage_id}.independent_review")
        verdict_relative = str(review.get("verdict_file", ""))
        if verdict_relative not in file_index:
            raise ProtocolFreezeError(f"stage {stage_id} review verdict is not hashed")
        if "independent_review" not in file_index[verdict_relative].get("kinds", []):
            raise ProtocolFreezeError(f"stage {stage_id} verdict lacks independent-review role")
        verdict_path, _ = _lexical_repo_path(root, verdict_relative,
                                             f"stages.{stage_id}.review verdict")
        contract = {
            key: review[key] for key in (
                "agent_role", "reviewer_role", "requested_model", "reasoning_effort",
                "fork_context", "scope_tokens", "required_items", "gate_verdict_field",
                "allowed_overall_verdicts", "acknowledged_nonpass_items"
            )
        }
        validate_independent_verdict(
            verdict := _read_json(verdict_path, f"stage {stage_id} independent verdict"),
            contract,
            stage_id=stage_id,
        )
        reviewed_expected: list[str] = []
        for relative_root in _require_list(
            stage.get("artifact_roots"), f"stages.{stage_id}.artifact_roots"
        ):
            reviewed_expected.extend(inventories[str(relative_root)])
        reviewed_expected.extend(_require_list(
            stage.get("config_files"), f"stages.{stage_id}.config_files"
        ))
        reviewed_expected.extend(_require_list(
            stage.get("code_files"), f"stages.{stage_id}.code_files"
        ))
        reviewed_sha = _validate_reviewed_sha256(
            verdict, reviewed_expected, root, stage_id=stage_id
        )
        if review.get("reviewed_sha256") != reviewed_sha:
            raise ProtocolFreezeError(
                f"stage {stage_id} frozen reviewed-byte inventory differs from verdict"
            )
        stage_roots = _require_list(stage.get("artifact_roots"),
                                    f"stages.{stage_id}.artifact_roots")
        for relative_root in stage_roots:
            if relative_root not in inventories:
                raise ProtocolFreezeError(
                    f"stage {stage_id} artifact root lacks frozen inventory: {relative_root}"
                )
        expected_kinds = {
            "config_files": "stage_config",
            "code_files": "source_code",
            "required_artifact_files": "artifact",
        }
        for category, expected_kind in expected_kinds.items():
            for relative in _require_list(stage.get(category), f"stages.{stage_id}.{category}"):
                if relative not in file_index:
                    raise ProtocolFreezeError(
                        f"stage {stage_id} {category} entry is not hashed: {relative}"
                    )
                entry = file_index[relative]
                if expected_kind not in entry.get("kinds", []) or stage_id not in entry.get(
                    "stages", []
                ):
                    raise ProtocolFreezeError(
                        f"stage {stage_id} {category} entry lacks its frozen role: {relative}"
                    )

    dependencies = _validate_protocol_dependencies(
        {"protocol_dependencies": manifest.get("protocol_dependencies")}, root
    )
    dependency_kinds = {
        "source_files": "protocol_source_dependency",
        "config_files": "protocol_config_dependency",
        "environment_files": "protocol_environment_dependency",
    }
    for category, relatives in dependencies.items():
        for relative in relatives:
            entry = file_index.get(relative)
            if entry is None or dependency_kinds[category] not in entry.get("kinds", []):
                raise ProtocolFreezeError(
                    f"protocol dependency lacks frozen {dependency_kinds[category]} role: {relative}"
                )

    common_review = _require_mapping(
        manifest.get("common_independent_review"), "common_independent_review"
    )
    common_verdict_relative = str(common_review.get("verdict_file", ""))
    common_entry = file_index.get(common_verdict_relative)
    if common_entry is None or "common_independent_review" not in common_entry.get("kinds", []):
        raise ProtocolFreezeError("common independent review verdict is not explicitly hashed")
    common_verdict_path, _ = _lexical_repo_path(
        root, common_verdict_relative, "common_independent_review.verdict_file"
    )
    common_contract = {
        key: common_review[key] for key in (
            "agent_role", "reviewer_role", "requested_model", "reasoning_effort",
            "fork_context", "scope_tokens", "required_items", "gate_verdict_field",
            "allowed_overall_verdicts", "acknowledged_nonpass_items",
        )
    }
    common_verdict = _read_json(common_verdict_path, "common independent verdict")
    validate_independent_verdict(common_verdict, common_contract, stage_id="COMMON")
    common_reviewed = _validate_reviewed_sha256(
        common_verdict,
        _common_review_expected_paths(protocol_config, dependencies),
        root,
        stage_id="COMMON",
    )
    if common_review.get("reviewed_sha256") != common_reviewed:
        raise ProtocolFreezeError(
            "frozen common dependency/config byte inventory differs from verdict"
        )

    return {
        "format": FORMAT,
        "status": PASS,
        "manifest": manifest_relative,
        "verified_files": len(file_index),
        "verified_artifact_roots": len(inventories),
        "required_stages": list(REQUIRED_STAGES),
        "test_data_accessed": False,
        "test_evaluation_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="freeze contract YAML")
    parser.add_argument("--project-root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", metavar="MANIFEST")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify:
            result = verify_freeze_manifest(args.verify, project_root=args.project_root)
        else:
            if not args.config:
                raise ProtocolFreezeError("--config is required for --preflight/--create")
            if args.preflight:
                result = preflight(args.config, project_root=args.project_root)
            else:
                result = create_freeze_manifest(args.config, project_root=args.project_root)
    except (ProtocolFreezeError, FileExistsError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.preflight and result["status"] != PASS:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
