"""One-shot, fail-closed entry point for the prospectively locked POI test.

This module is intentionally separate from :mod:`poi.final_test`.  The historical runner remains
disabled.  Nothing in ``pre_access_gate`` imports a model backend, stats a dataset path, or opens a
test row.  A future execution is allowed to reach a byte-reviewed backend only after all of these
events have succeeded, in order:

1. the development freeze is reverified;
2. the frozen selection and the known Stage-14 blockers are checked;
3. a fresh independent execution review is byte-bound;
4. a read-only, exact one-shot authorization is checked; and
5. an exclusive execution reservation is written.

The freeze establishes prospective isolation only.  All execution records permanently retain the
known prior exposure of all 6,800 test coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .protocol_freeze import ProtocolFreezeError, verify_freeze_manifest


CONFIG_FORMAT = "poi-locked-test-execution-config-v1"
SELECTION_FORMAT = "poi-locked-test-selection-v1"
AUTHORIZATION_FORMAT = "poi-locked-test-one-shot-authorization-v1"
REVIEW_FORMAT = "poi-locked-test-execution-review-v1"
READY = "READY_WITHOUT_TEST_ACCESS"
AUTHORIZED = "AUTHORIZED_FOR_EXACTLY_ONE_LOCKED_TEST_EXECUTION"
PENDING_REVIEW = "LOCKED_TEST_EXECUTED_PENDING_INDEPENDENT_REVIEW"
HISTORICAL_DESCRIPTOR = "historically_exposed_coordinates_prospectively_locked"
AUTHORIZATION_SCOPE = "exactly_one_locked_test_execution"

REGIONS = (
    "Busan", "Chungbuk", "Chungnam", "Daegu", "Daejeon", "Gangwon", "Gwangju",
    "Gyeongbuk", "Gyeonggi", "Gyeongnam", "Incheon", "Jeju", "Jeonbuk", "Jeonnam",
    "Sejong", "Seoul", "Ulsan",
)
MODEL_SEEDS = (42, 202, 340)
GENERATION_SEEDS = (42, 202, 340)
DETECTORS = ("proposed_tabpfn25_attack_head", "URE-RF attack detector")
FIVE_METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
EVALUATION_SCOPES = ("primary_valid", "all_intent_to_attack")
CLASSIFICATION_MODES = (
    "fixed_unmodified",
    "detector_accepted_only",
    "shared_nearest_train_repair",
    "proposed_projected_repair_fixed_tabpfn25",
    "proposed_native_full_coverage",
)
REQUIRED_STAGE14_ITEMS = frozenset({
    "leave_one_attack_out_generalization",
    "leave_one_source_model_out_transfer",
    "adaptive_detector_evasion_attack",
    "regional_probability_ensemble_calibration",
})
REQUIRED_REVIEW_ITEMS = frozenset({
    "freeze_manifest_reverification",
    "selection_completeness",
    "one_shot_authorization_contract",
    "no_test_access_before_gate",
    "same_frozen_downstream",
    "two_detector_outputs",
    "coverage_and_full_population_repairs",
    "five_metrics_17_ovr",
    "historical_coordinate_exposure_disclosure",
    "stage14_blockers_pass",
})
REQUIRED_BINDINGS = (
    "freeze_manifest", "selection_contract", "execution_config", "runner_source",
    "backend_source", "execution_review",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_GLOB_CHARS = frozenset("*?[]{}")


class LockedTestError(RuntimeError):
    """The locked-test execution contract is incomplete, stale, or unsafe."""


class LockedTestNotReady(LockedTestError):
    """A required development result, review, authorization, or backend is absent."""


@dataclass(frozen=True)
class LockedTestGate:
    """Validated non-data inputs.  Constructed before any dataset path is inspected."""

    project_root: Path
    config_path: Path
    output: Path
    data_dir_raw: str
    expected_test_rows: int
    execution_id: str
    manifest: Mapping[str, Any]
    selection: Mapping[str, Any]
    backend_module: str
    backend_callable: str
    backend_source: Path
    bindings: Mapping[str, Mapping[str, str]]
    authorization_path: Path
    authorization_sha256: str
    historical_statement: str


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedTestError(f"{field} must be a mapping")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise LockedTestError(f"{field} must be a non-empty list")
    return value


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), field)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockedTestError(f"cannot read {field}: {path}: {exc}") from exc


def _repo_file(root: Path, raw: Any, field: str) -> tuple[Path, str]:
    """Resolve a non-dataset contract file inside the repository."""
    if not isinstance(raw, str) or not raw.strip() or any(char in raw for char in _GLOB_CHARS):
        raise LockedTestError(f"{field} must be one explicit repository path")
    supplied = Path(raw)
    candidate = supplied if supplied.is_absolute() else root / supplied
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise LockedTestError(f"{field} escapes project root") from exc
    if not relative.parts or relative.parts[0].lower() == "data":
        raise LockedTestError(f"{field} must not point into a dataset")
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise LockedTestError(f"{field} contains a symlink: {relative.as_posix()}")
    if not candidate.is_file():
        raise LockedTestNotReady(f"{field} is absent: {relative.as_posix()}")
    return candidate, relative.as_posix()


def _output_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip() or any(char in raw for char in _GLOB_CHARS):
        raise LockedTestError("output must be one explicit repository path")
    supplied = Path(raw)
    candidate = Path(os.path.abspath(os.fspath(supplied if supplied.is_absolute() else root / supplied)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise LockedTestError("output escapes project root") from exc
    if not relative.parts or relative.parts[0] != "artifacts":
        raise LockedTestError("output must be below artifacts/")
    cursor = root
    for component in relative.parts[:-1]:
        cursor /= component
        if cursor.is_symlink():
            raise LockedTestError("output parent contains a symlink")
    return candidate


def _deferred_data_path(raw: Any) -> str:
    """Validate only lexical spelling; do not stat, resolve, or open the dataset."""
    if not isinstance(raw, str) or not raw.strip() or any(char in raw for char in _GLOB_CHARS):
        raise LockedTestError("data.clean_dataset_dir must be one explicit path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise LockedTestError("data.clean_dataset_dir must be a relative path below data/")
    return path.as_posix()


def _load_config(config_path: str | Path, root: Path) -> tuple[dict[str, Any], Path, str]:
    path, relative = _repo_file(root, os.fspath(config_path), "execution config")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LockedTestError(f"cannot load execution config: {exc}") from exc
    config = dict(_mapping(raw, "execution config"))
    if config.get("format") != CONFIG_FORMAT:
        raise LockedTestError(f"execution config format must be {CONFIG_FORMAT}")
    return config, path, relative


def _file_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _list(manifest.get("files"), "freeze manifest files"):
        entry = _mapping(raw, "freeze manifest file")
        path = str(entry.get("path", ""))
        digest = str(entry.get("sha256", ""))
        if not path or path in result or not _SHA256_RE.fullmatch(digest):
            raise LockedTestError("freeze manifest file catalog is malformed")
        result[path] = entry
    return result


def _validate_stage14(manifest: Mapping[str, Any]) -> None:
    stages = _mapping(manifest.get("stages"), "freeze manifest stages")
    stage = _mapping(stages.get("14"), "freeze manifest stage 14")
    review = _mapping(stage.get("independent_review"), "stage 14 independent review")
    required = set(map(str, _list(review.get("required_items"), "stage 14 required items")))
    validated = set(map(str, _list(review.get("validated_items"), "stage 14 validated items")))
    missing = sorted(REQUIRED_STAGE14_ITEMS - required)
    unvalidated = sorted(REQUIRED_STAGE14_ITEMS - validated)
    if missing or unvalidated or review.get("prospective_lock_readiness_verdict") != "PASS":
        raise LockedTestNotReady(
            f"Stage 14 blockers are not independently PASS: missing={missing}, "
            f"unvalidated={unvalidated}"
        )


def _validate_selection(selection: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if selection.get("format") != SELECTION_FORMAT or selection.get("status") != "FROZEN":
        raise LockedTestNotReady("selection contract is not frozen")
    scalar_checks = {
        "selection_scope": "validation_only",
        "test_rows_used_for_selection": 0,
        "test_evaluation_performed": False,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "primary_clean_retention": 0.95,
    }
    for field, expected in scalar_checks.items():
        if selection.get(field) != expected:
            raise LockedTestError(f"selection {field} must be {expected!r}")
    exact_sequences = {
        "regions": REGIONS,
        "model_seeds": MODEL_SEEDS,
        "generation_seeds": GENERATION_SEEDS,
        "detectors": DETECTORS,
        "metrics": FIVE_METRICS,
        "evaluation_scopes": EVALUATION_SCOPES,
        "classification_modes": CLASSIFICATION_MODES,
    }
    for field, expected in exact_sequences.items():
        observed = tuple(selection.get(field, ()))
        if observed != expected:
            raise LockedTestError(f"selection {field} differs from the locked contract")
    conditions = selection.get("attack_conditions")
    if not isinstance(conditions, list) or len(conditions) != 32 or len(set(conditions)) != 32:
        raise LockedTestNotReady("selection must freeze exactly 32 unique attack conditions")
    if not all(isinstance(value, str) and value.strip() for value in conditions):
        raise LockedTestError("attack condition names must be non-empty strings")

    classifier = _mapping(selection.get("region_classifier"), "selection region_classifier")
    expected_classifier = {
        "model_id": "tabpfn_v2_5",
        "fit_scope": "clean_train_only",
        "fit_count": 51,
        "shared_across_detectors_and_conditions": True,
        "refit_on_test": False,
        "ovr_regions": 17,
    }
    for field, expected in expected_classifier.items():
        if classifier.get(field) != expected:
            raise LockedTestError(f"region_classifier.{field} must be {expected!r}")
    detector_labels = _mapping(selection.get("detector_labels"), "selection detector_labels")
    if detector_labels != {"normal": 0, "attack": 1}:
        raise LockedTestError("detector labels must remain normal=0 and attack=1")
    repair = _mapping(selection.get("repair_contract"), "selection repair_contract")
    if repair.get("shared_nearest_train_for_both_detectors") is not True:
        raise LockedTestError("shared nearest-train repair must be retained for both detectors")
    if repair.get("proposed_native_full_population") is not True:
        raise LockedTestError("proposed native full-population output is required")
    if repair.get("ure_native_reconstruction") is not False:
        raise LockedTestError("URE native reconstruction must not be invented")
    if repair.get("uses_dataset_label_for_repair") is not False:
        raise LockedTestError("repair must not use dataset labels")

    _validate_stage14(manifest)
    catalog = _file_index(manifest)
    critical = _mapping(manifest.get("critical_freeze_items"), "critical freeze items")
    required_paths: set[str] = set()
    for category in ("selected_checkpoints", "threshold_files", "calibration_partitions"):
        for raw in _list(critical.get(category), f"critical {category}"):
            required_paths.add(str(_mapping(raw, f"critical {category} entry").get("path", "")))
    downstream = _mapping(critical.get("fixed_downstream"), "critical fixed_downstream")
    required_paths.update(map(str, _list(downstream.get("evidence_files"),
                                         "fixed downstream evidence")))
    required_paths.add(str(downstream.get("replay_contract_file", "")))
    frozen_files = _mapping(selection.get("frozen_files"), "selection frozen_files")
    missing = sorted(required_paths - set(map(str, frozen_files)))
    if missing:
        raise LockedTestNotReady(f"selection omits frozen critical files: {missing}")
    for path, digest in frozen_files.items():
        entry = catalog.get(str(path))
        if entry is None or digest != entry.get("sha256"):
            raise LockedTestError(f"selection frozen file is absent or stale: {path}")

    checkpoint = _mapping(selection.get("fixed_downstream_checkpoint"),
                          "selection fixed_downstream_checkpoint")
    if checkpoint.get("path") != downstream.get("actual_checkpoint_path") or \
            checkpoint.get("sha256") != downstream.get("expected_checkpoint_sha256"):
        raise LockedTestError("selection changes the fixed downstream checkpoint")


def _validate_execution_review(
    review: Mapping[str, Any], expected: Mapping[str, tuple[str, str]]
) -> None:
    exact = {
        "format": REVIEW_FORMAT,
        "role": "independent_experiment_audit",
        "reviewer_role": "independent_reviewer",
        "requested_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "fork_context": "none",
        "overall_verdict": "PASS",
        "locked_test_execution_readiness_verdict": "PASS",
        "test_data_accessed": False,
        "test_evaluation_performed": False,
    }
    for field, value in exact.items():
        if review.get(field) != value:
            raise LockedTestNotReady(f"execution review {field} must be {value!r}")
    rows = review.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise LockedTestError("execution review requirements must be non-empty")
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, "execution review requirement")
        identifier = str(row.get("requirement_id", ""))
        if not identifier or identifier in indexed:
            raise LockedTestError("execution review requirement IDs must be unique")
        indexed[identifier] = row
    missing = sorted(REQUIRED_REVIEW_ITEMS - set(indexed))
    nonpass = sorted(item for item in REQUIRED_REVIEW_ITEMS
                     if item in indexed and indexed[item].get("verdict") != "PASS")
    if missing or nonpass:
        raise LockedTestNotReady(
            f"execution review is incomplete: missing={missing}, nonpass={nonpass}"
        )
    reviewed = _mapping(review.get("reviewed_sha256"), "execution review reviewed_sha256")
    expected_paths = {relative: digest for relative, digest in expected.values()}
    if dict(reviewed) != expected_paths:
        raise LockedTestError("execution review is not byte-bound to every required input")


def _validate_authorization(
    authorization: Mapping[str, Any], *, execution_id: str,
    expected: Mapping[str, tuple[str, str]], historical_statement: str,
) -> Mapping[str, Mapping[str, str]]:
    exact = {
        "format": AUTHORIZATION_FORMAT,
        "status": AUTHORIZED,
        "execution_id": execution_id,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "test_split": "test",
        "maximum_test_executions": 1,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "prior_coordinate_access_acknowledged": True,
        "test_data_accessed_when_authorized": False,
        "test_evaluation_performed_when_authorized": False,
    }
    for field, value in exact.items():
        if authorization.get(field) != value:
            raise LockedTestNotReady(f"authorization {field} must be {value!r}")
    if authorization.get("prior_coordinate_access_statement") != historical_statement:
        raise LockedTestError("authorization changes the historical access disclosure")
    if not str(authorization.get("authorized_by", "")).strip() or not str(
            authorization.get("authorized_at_utc", "")).strip():
        raise LockedTestError("authorization must record its issuer and UTC time")
    bindings = _mapping(authorization.get("bindings"), "authorization bindings")
    if tuple(sorted(bindings)) != tuple(sorted(REQUIRED_BINDINGS)):
        raise LockedTestError("authorization bindings are incomplete or contain extra entries")
    normalized: dict[str, Mapping[str, str]] = {}
    for name in REQUIRED_BINDINGS:
        entry = _mapping(bindings[name], f"authorization binding {name}")
        path, digest = expected[name]
        if entry.get("path") != path or entry.get("sha256") != digest:
            raise LockedTestError(f"authorization binding is stale: {name}")
        normalized[name] = {"path": path, "sha256": digest}
    return normalized


def pre_access_gate(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> LockedTestGate:
    """Verify every non-data gate without inspecting the configured dataset path."""
    root = Path(project_root or Path.cwd()).resolve()
    config, config_file, config_relative = _load_config(config_path, root)
    output = _output_path(root, config.get("output"))
    execution_id = str(config.get("execution_id", ""))
    if not _EXECUTION_ID_RE.fullmatch(execution_id):
        raise LockedTestError("execution_id is absent or invalid")
    data = _mapping(config.get("data"), "data")
    if data.get("split") != "test" or data.get("expected_test_rows") != 6800:
        raise LockedTestError("locked data contract must be split=test with exactly 6800 rows")
    data_dir_raw = _deferred_data_path(data.get("clean_dataset_dir"))

    manifest_file, manifest_relative = _repo_file(
        root, config.get("freeze_manifest"), "freeze manifest"
    )
    try:
        verification = verify_freeze_manifest(manifest_file, project_root=root)
    except ProtocolFreezeError as exc:
        raise LockedTestNotReady(f"freeze verification failed: {exc}") from exc
    if verification.get("status") != "PASS" or verification.get("test_data_accessed") is not False \
            or verification.get("test_evaluation_performed") is not False:
        raise LockedTestNotReady("freeze verification did not return a no-test-access PASS")
    if output.exists():
        raise LockedTestNotReady(f"one-shot output already exists: {output.relative_to(root)}")
    manifest = _read_json(manifest_file, "freeze manifest")
    if manifest.get("locked_test_descriptor") != HISTORICAL_DESCRIPTOR:
        raise LockedTestError("freeze omits the historical test-coordinate exposure descriptor")
    disclosure = _mapping(manifest.get("prior_test_coordinate_access"),
                          "prior test coordinate access")
    historical_statement = str(disclosure.get("statement", "")).strip()
    if disclosure.get("occurred") is not True or not historical_statement:
        raise LockedTestError("freeze omits the known prior test-coordinate access")

    selection_file, selection_relative = _repo_file(
        root, config.get("selection_contract"), "selection contract"
    )
    selection = _read_json(selection_file, "selection contract")
    _validate_selection(selection, manifest)

    backend = _mapping(config.get("execution_backend"), "execution_backend")
    backend_source, backend_relative = _repo_file(
        root, backend.get("source_path"), "backend source"
    )
    module_name = str(backend.get("module", "")).strip()
    callable_name = str(backend.get("callable", "")).strip()
    if not module_name.startswith("poi.") or not callable_name or callable_name.startswith("_"):
        raise LockedTestError("execution backend must be one public callable in poi.*")
    runner_file = Path(__file__).resolve()
    try:
        runner_relative = runner_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise LockedTestError("locked-test runner is outside project root") from exc

    review_file, review_relative = _repo_file(
        root, config.get("execution_review"), "execution review"
    )
    authorization_file, _ = _repo_file(
        root, config.get("authorization"), "one-shot authorization"
    )
    if stat.S_IMODE(authorization_file.stat().st_mode) & 0o222:
        raise LockedTestError("one-shot authorization must be read-only")

    expected_without_review = {
        "freeze_manifest": (manifest_relative, _sha256(manifest_file)),
        "selection_contract": (selection_relative, _sha256(selection_file)),
        "execution_config": (config_relative, _sha256(config_file)),
        "runner_source": (runner_relative, _sha256(runner_file)),
        "backend_source": (backend_relative, _sha256(backend_source)),
    }
    review = _read_json(review_file, "execution review")
    _validate_execution_review(review, expected_without_review)
    expected = {
        **expected_without_review,
        "execution_review": (review_relative, _sha256(review_file)),
    }
    authorization = _read_json(authorization_file, "one-shot authorization")
    bindings = _validate_authorization(
        authorization, execution_id=execution_id, expected=expected,
        historical_statement=historical_statement,
    )
    return LockedTestGate(
        project_root=root,
        config_path=config_file,
        output=output,
        data_dir_raw=data_dir_raw,
        expected_test_rows=6800,
        execution_id=execution_id,
        manifest=manifest,
        selection=selection,
        backend_module=module_name,
        backend_callable=callable_name,
        backend_source=backend_source,
        bindings=bindings,
        authorization_path=authorization_file,
        authorization_sha256=_sha256(authorization_file),
        historical_statement=historical_statement,
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def _revalidate_gate_bindings(gate: LockedTestGate) -> None:
    """Close the gate-to-reservation/import mutation window without touching datasets."""
    manifest_binding = gate.bindings["freeze_manifest"]
    manifest_path = gate.project_root / manifest_binding["path"]
    try:
        verification = verify_freeze_manifest(manifest_path, project_root=gate.project_root)
    except ProtocolFreezeError as exc:
        raise LockedTestNotReady(f"freeze changed after gate validation: {exc}") from exc
    if verification.get("status") != "PASS":
        raise LockedTestNotReady("freeze revalidation stopped passing")
    for name, entry in gate.bindings.items():
        path = gate.project_root / entry["path"]
        if path.is_symlink() or not path.is_file() or _sha256(path) != entry["sha256"]:
            raise LockedTestError(f"authorized bytes changed after gate validation: {name}")
    if (gate.authorization_path.is_symlink() or not gate.authorization_path.is_file()
            or stat.S_IMODE(gate.authorization_path.stat().st_mode) & 0o222
            or _sha256(gate.authorization_path) != gate.authorization_sha256):
        raise LockedTestError("one-shot authorization changed after gate validation")


def _reserve(gate: LockedTestGate) -> Path:
    _revalidate_gate_bindings(gate)
    gate.output.parent.mkdir(parents=True, exist_ok=True)
    gate.output.mkdir(mode=0o700)
    reservation = gate.output / "_EXECUTION_RESERVED.json"
    _write_exclusive(reservation, {
        "format": "poi-locked-test-execution-reservation-v1",
        "status": "RESERVED_BEFORE_FIRST_TEST_ACCESS",
        "execution_id": gate.execution_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "maximum_test_executions": 1,
        "test_data_accessed_at_reservation": False,
        "test_evaluation_performed_at_reservation": False,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "prior_coordinate_access_statement": gate.historical_statement,
        "bindings": gate.bindings,
        "authorization_sha256": gate.authorization_sha256,
    })
    return reservation


def _load_backend(gate: LockedTestGate) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    _revalidate_gate_bindings(gate)
    module = importlib.import_module(gate.backend_module)
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    if module_file != gate.backend_source.resolve():
        raise LockedTestError("imported backend module does not match its authorized source path")
    backend = getattr(module, gate.backend_callable, None)
    if not callable(backend):
        raise LockedTestNotReady("authorized backend callable is absent")
    return backend


def _artifact_path(gate: LockedTestGate, entry: Mapping[str, Any], label: str) -> Path:
    raw = entry["path"]
    path = gate.output / str(raw)
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(gate.output)
    except ValueError as exc:
        raise LockedTestError(f"backend artifact escapes output: {label}") from exc
    return path


def _validate_json_artifact(path: Path, field: str) -> Mapping[str, Any]:
    return _read_json(path, f"backend {field}")


def _clean_and_attack_cohorts(selection: Mapping[str, Any]) -> set[tuple[int, str]]:
    cohorts = {(-1, "clean")}
    cohorts.update(
        (int(seed), str(condition))
        for seed in GENERATION_SEEDS
        for condition in selection["attack_conditions"]
    )
    return cohorts


def _validate_attack_manifest(
    gate: LockedTestGate, path: Path
) -> dict[tuple[int, str], dict[str, int]]:
    payload = _validate_json_artifact(path, "attack manifest")
    conditions = list(gate.selection["attack_conditions"])
    expected_rows = gate.expected_test_rows * len(GENERATION_SEEDS) * len(conditions)
    exact = {
        "format": "poi-locked-test-attack-manifest-v1",
        "split": "test",
        "test_rows": gate.expected_test_rows,
        "generation_seeds": list(GENERATION_SEEDS),
        "attack_conditions": conditions,
        "expected_rows": expected_rows,
        "observed_rows": expected_rows,
        "paired_to_every_test_poi": True,
        "all_source_pois_retained": True,
        "failed_attacks_retained": True,
        "quarantined_rows_retained": True,
        "primary_valid_and_all_intent_scopes_preserved": True,
        "test_labels_not_used_for_repair": True,
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise LockedTestError(f"attack manifest {field} must be {expected!r}")
    digest = str(payload.get("ordered_test_poi_ids_sha256", ""))
    if not _SHA256_RE.fullmatch(digest):
        raise LockedTestError("attack manifest lacks an ordered test-POI identity digest")
    raw_counts = _list(payload.get("cohort_counts"), "attack manifest cohort_counts")
    counts: dict[tuple[int, str], dict[str, int]] = {}
    for raw in raw_counts:
        row = _mapping(raw, "attack manifest cohort count")
        try:
            key = (int(row["generation_seed"]), str(row["attack_condition"]))
            all_intent = int(row["all_intent_rows"])
            primary = int(row["primary_valid_rows"])
            quarantined = int(row["quarantined_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedTestError("attack manifest cohort count is malformed") from exc
        if key in counts:
            raise LockedTestError(f"duplicate attack cohort count: {key}")
        if all_intent != gate.expected_test_rows or not 0 <= primary <= all_intent \
                or quarantined != all_intent - primary:
            raise LockedTestError(f"attack cohort denominator is inconsistent: {key}")
        counts[key] = {
            "all_intent_to_attack": all_intent,
            "primary_valid": primary,
        }
    expected_cohorts = {
        (seed, condition)
        for seed in GENERATION_SEEDS
        for condition in conditions
    }
    if set(counts) != expected_cohorts:
        raise LockedTestError("attack manifest cohort coverage is incomplete")
    return counts


def _validate_prediction_manifest(gate: LockedTestGate, path: Path) -> None:
    payload = _validate_json_artifact(path, "prediction manifest")
    conditions = list(gate.selection["attack_conditions"])
    cohorts = 1 + len(GENERATION_SEEDS) * len(conditions)
    expected_rows = (
        gate.expected_test_rows * len(REGIONS) * len(DETECTORS) * len(MODEL_SEEDS) * cohorts
    )
    exact = {
        "format": "poi-locked-test-prediction-manifest-v1",
        "split": "test",
        "test_rows": gate.expected_test_rows,
        "model_seeds": list(MODEL_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "attack_conditions": conditions,
        "detectors": list(DETECTORS),
        "regions": list(REGIONS),
        "expected_prediction_rows": expected_rows,
        "observed_prediction_rows": expected_rows,
        "paired_across_detectors": True,
        "paired_across_all_17_ovr_regions": True,
        "rejected_rows_retained": True,
        "quarantined_rows_retained": True,
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise LockedTestError(f"prediction manifest {field} must be {expected!r}")
    shards = _list(payload.get("shards"), "prediction manifest shards")
    observed_rows = 0
    observed_paths: set[str] = set()
    for index, raw in enumerate(shards):
        shard = _mapping(raw, f"prediction shard {index}")
        relative = str(shard.get("path", ""))
        expected_sha = str(shard.get("sha256", ""))
        rows = shard.get("rows")
        if (not relative or relative in observed_paths or not _SHA256_RE.fullmatch(expected_sha)
                or not isinstance(rows, int) or rows <= 0):
            raise LockedTestError("prediction shard binding is malformed")
        shard_path = gate.output / relative
        shard_path = Path(os.path.abspath(os.fspath(shard_path)))
        try:
            shard_path.relative_to(gate.output)
        except ValueError as exc:
            raise LockedTestError("prediction shard escapes output") from exc
        if shard_path.is_symlink() or not shard_path.is_file() or _sha256(shard_path) != expected_sha:
            raise LockedTestError(f"prediction shard is absent, linked, or stale: {relative}")
        observed_paths.add(relative)
        observed_rows += rows
    if observed_rows != expected_rows:
        raise LockedTestError(
            f"prediction shard row sum {observed_rows} differs from {expected_rows}"
        )


def _read_metric_rows(path: Path, required: set[str], field: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            missing = sorted(required - set(columns))
            if missing:
                raise LockedTestError(f"{field} columns are missing: {missing}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LockedTestError(f"cannot read {field}: {exc}") from exc
    return columns, rows


def _unit_interval(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise LockedTestError(f"{field} is not numeric") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise LockedTestError(f"{field} is outside [0,1]")
    return value


def _metric_value(row: Mapping[str, str], field: str) -> None:
    defined = str(row.get("metric_defined", "")).strip().lower()
    reason = str(row.get("undefined_reason", "")).strip()
    if defined == "true":
        _unit_interval(row.get("value", ""), field)
        if reason:
            raise LockedTestError(f"{field} is defined but has an undefined_reason")
    elif defined == "false":
        raw = str(row.get("value", "")).strip().lower()
        if raw not in {"", "nan", "na", "null"} or not reason:
            raise LockedTestError(f"{field} undefined encoding is incomplete")
    else:
        raise LockedTestError(f"{field} metric_defined must be true or false")


def _optional_unit_interval(raw: str, field: str, *, defined: bool) -> float | None:
    value = str(raw).strip().lower()
    if not defined:
        if value not in {"", "nan", "na", "null"}:
            raise LockedTestError(f"{field} must be undefined for an empty cohort")
        return None
    return _unit_interval(value, field)


def _validate_detector_metrics(gate: LockedTestGate, path: Path) -> None:
    required = {
        "detector", "model_seed", "generation_seed", "attack_condition",
        "evaluation_scope", "metric", "value", "metric_defined", "undefined_reason",
        "clean_retention",
    }
    _, rows = _read_metric_rows(path, required, "detector metrics")
    expected = {
        (detector, model_seed, generation_seed, condition, scope, metric)
        for detector in DETECTORS
        for model_seed in MODEL_SEEDS
        for generation_seed in GENERATION_SEEDS
        for condition in gate.selection["attack_conditions"]
        for scope in EVALUATION_SCOPES
        for metric in FIVE_METRICS
    }
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        try:
            key = (
                row["detector"], int(row["model_seed"]), int(row["generation_seed"]),
                row["attack_condition"], row["evaluation_scope"], row["metric"],
            )
        except (TypeError, ValueError) as exc:
            raise LockedTestError("detector metric key is malformed") from exc
        if key in observed:
            raise LockedTestError(f"duplicate detector metric key: {key}")
        observed.add(key)
        _metric_value(row, "detector metric value")
        if abs(_unit_interval(row["clean_retention"], "clean retention") - 0.95) > 1e-12:
            raise LockedTestError("detector metrics must use frozen clean retention 0.95")
    if observed != expected:
        raise LockedTestError(
            f"detector metric coverage differs: expected={len(expected)}, observed={len(observed)}"
        )


def _validate_classification_metrics(
    gate: LockedTestGate,
    path: Path,
    cohort_counts: Mapping[tuple[int, str], Mapping[str, int]],
) -> None:
    required = {
        "detector", "model_seed", "generation_seed", "attack_condition",
        "evaluation_scope", "classification_mode", "region", "metric", "value",
        "metric_defined", "undefined_reason", "detector_coverage", "accepted_rows",
        "total_rows", "metric_population_rows",
    }
    _, rows = _read_metric_rows(path, required, "classification metrics")
    allowed_modes = {
        DETECTORS[0]: CLASSIFICATION_MODES,
        DETECTORS[1]: CLASSIFICATION_MODES[:3],
    }
    cohorts = _clean_and_attack_cohorts(gate.selection)
    expected = {
        (detector, model_seed, generation_seed, condition, scope, mode, region, metric)
        for detector, modes in allowed_modes.items()
        for model_seed in MODEL_SEEDS
        for generation_seed, condition in cohorts
        for scope in EVALUATION_SCOPES
        for mode in modes
        for region in REGIONS
        for metric in FIVE_METRICS
    }
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        try:
            key = (
                row["detector"], int(row["model_seed"]), int(row["generation_seed"]),
                row["attack_condition"], row["evaluation_scope"],
                row["classification_mode"], row["region"], row["metric"],
            )
            total = int(row["total_rows"])
            accepted = int(row["accepted_rows"])
            metric_population = int(row["metric_population_rows"])
        except (TypeError, ValueError) as exc:
            raise LockedTestError("classification metric key or denominator is malformed") from exc
        if key in observed:
            raise LockedTestError(f"duplicate classification metric key: {key}")
        observed.add(key)
        _metric_value(row, "classification metric value")
        generation_seed, condition, scope, mode = key[2], key[3], key[4], key[5]
        expected_total = (gate.expected_test_rows if condition == "clean" else
                          cohort_counts[(generation_seed, condition)][scope])
        if total != expected_total or not 0 <= accepted <= total:
            raise LockedTestError("classification denominator differs from its frozen scope")
        coverage = _optional_unit_interval(
            row["detector_coverage"], "detector coverage", defined=total > 0
        )
        if total and abs(float(coverage) - accepted / total) > 1e-12:
            raise LockedTestError("detector coverage disagrees with accepted/total")
        if mode == "detector_accepted_only":
            if metric_population != accepted:
                raise LockedTestError("accepted-only metrics use the wrong population")
        elif metric_population != total:
            raise LockedTestError("all-population classification mode has incomplete coverage")
        if mode in {"fixed_unmodified", "proposed_native_full_coverage"} and accepted != total:
            raise LockedTestError("threshold-free full-population mode has incomplete coverage")
    if observed != expected:
        raise LockedTestError(
            f"classification metric coverage differs: expected={len(expected)}, "
            f"observed={len(observed)}"
        )


def _validate_backend_output_artifacts(
    gate: LockedTestGate, result: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    metadata_path = _artifact_path(gate, artifacts["metadata"], "metadata")
    metadata = _validate_json_artifact(metadata_path, "metadata")
    for field in (
        "execution_id", "split", "test_rows", "test_executions", "region_classifier",
        "region_model_fit_count", "region_model_fit_scope",
        "same_region_classifier_shared_across_detectors", "region_classifier_refit_on_test",
        "detectors", "detector_labels", "regions", "model_seeds", "generation_seeds",
        "attack_conditions", "metrics", "evaluation_scopes", "classification_modes",
        "primary_clean_retention", "shared_nearest_train_repair_for_both_detectors",
        "proposed_native_full_population", "ure_native_reconstruction",
        "uses_dataset_label_for_repair", "preserved_rejected_and_quarantined_denominators",
        "historical_test_descriptor", "historically_pristine_test_claim_permitted",
    ):
        if metadata.get(field) != result.get(field):
            raise LockedTestError(f"backend metadata disagrees with result for {field}")
    cohort_counts = _validate_attack_manifest(
        gate, _artifact_path(gate, artifacts["attack_manifest"], "attack_manifest")
    )
    _validate_prediction_manifest(
        gate, _artifact_path(gate, artifacts["prediction_manifest"], "prediction_manifest")
    )
    _validate_detector_metrics(
        gate, _artifact_path(gate, artifacts["detector_metrics"], "detector_metrics")
    )
    _validate_classification_metrics(
        gate, _artifact_path(gate, artifacts["classification_metrics"],
                             "classification_metrics"), cohort_counts
    )


def _validate_backend_result(gate: LockedTestGate, result: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_mapping(result, "backend result"))
    exact = {
        "format": "poi-locked-test-backend-result-v1",
        "execution_id": gate.execution_id,
        "split": "test",
        "test_rows": gate.expected_test_rows,
        "test_executions": 1,
        "region_classifier": "tabpfn_v2_5",
        "region_model_fit_count": 51,
        "region_model_fit_scope": "clean_train_only",
        "same_region_classifier_shared_across_detectors": True,
        "region_classifier_refit_on_test": False,
        "detectors": list(DETECTORS),
        "detector_labels": {"normal": 0, "attack": 1},
        "regions": list(REGIONS),
        "model_seeds": list(MODEL_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "metrics": list(FIVE_METRICS),
        "evaluation_scopes": list(EVALUATION_SCOPES),
        "classification_modes": list(CLASSIFICATION_MODES),
        "primary_clean_retention": 0.95,
        "shared_nearest_train_repair_for_both_detectors": True,
        "proposed_native_full_population": True,
        "ure_native_reconstruction": False,
        "uses_dataset_label_for_repair": False,
        "preserved_rejected_and_quarantined_denominators": True,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
    }
    for field, expected in exact.items():
        if result.get(field) != expected:
            raise LockedTestError(f"backend result {field} must be {expected!r}")
    if result.get("attack_conditions") != gate.selection.get("attack_conditions"):
        raise LockedTestError("backend result attack conditions differ from the frozen selection")
    artifacts = _mapping(result.get("artifacts"), "backend result artifacts")
    required_artifacts = {
        "detector_metrics", "classification_metrics", "prediction_manifest",
        "attack_manifest", "metadata",
    }
    if set(artifacts) != required_artifacts:
        raise LockedTestError("backend result artifact set is incomplete or contains extras")
    normalized: dict[str, dict[str, Any]] = {}
    for label, raw in artifacts.items():
        entry = _mapping(raw, f"backend artifact {label}")
        raw_path, expected_sha = entry.get("path"), str(entry.get("sha256", ""))
        if not isinstance(raw_path, str) or not _SHA256_RE.fullmatch(expected_sha):
            raise LockedTestError(f"backend artifact binding is malformed: {label}")
        path = Path(raw_path)
        path = path if path.is_absolute() else gate.output / path
        path = Path(os.path.abspath(os.fspath(path)))
        try:
            path.relative_to(gate.output)
        except ValueError as exc:
            raise LockedTestError(f"backend artifact escapes output: {label}") from exc
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha:
            raise LockedTestError(f"backend artifact is absent, linked, or stale: {label}")
        normalized[label] = {
            "path": path.relative_to(gate.output).as_posix(),
            "sha256": expected_sha,
            "size_bytes": path.stat().st_size,
        }
    _validate_backend_output_artifacts(gate, result, normalized)
    result["artifacts"] = normalized
    return result


def execute_locked_test(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Consume one authorization and invoke its reviewed backend exactly once.

    The backend is imported only after the exclusive reservation exists.  Thus even an import-time
    backend defect cannot silently consume test data without leaving the one-shot reservation.
    """
    gate = pre_access_gate(config_path, project_root=project_root)
    reservation = _reserve(gate)
    context = {
        "format": "poi-locked-test-backend-context-v1",
        "execution_id": gate.execution_id,
        "project_root": str(gate.project_root),
        "output": str(gate.output),
        "clean_dataset_dir": gate.data_dir_raw,
        "split": "test",
        "expected_test_rows": gate.expected_test_rows,
        "selection": gate.selection,
        "freeze_manifest": gate.manifest,
        "reservation": str(reservation),
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "prior_coordinate_access_statement": gate.historical_statement,
    }
    try:
        result = _validate_backend_result(gate, _load_backend(gate)(context))
        pending = {
            "format": "poi-locked-test-pending-independent-review-v1",
            "status": PENDING_REVIEW,
            "execution_id": gate.execution_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_completion_claim_permitted": False,
            "independent_artifact_review_required": True,
            "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
            "historically_pristine_test_claim_permitted": False,
            "prior_coordinate_access_statement": gate.historical_statement,
            "bindings": gate.bindings,
            "backend_result": result,
        }
        _write_exclusive(gate.output / "_PENDING_INDEPENDENT_REVIEW.json", pending)
        return pending
    except BaseException as exc:
        failure = gate.output / "_EXECUTION_FAILED.json"
        if not failure.exists():
            _write_exclusive(failure, {
                "format": "poi-locked-test-execution-failure-v1",
                "status": "FAILED_AFTER_ONE_SHOT_RESERVATION",
                "execution_id": gate.execution_id,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "test_access_may_have_occurred": True,
                "retry_permitted": False,
                "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
            })
        raise


def preflight(config_path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    gate = pre_access_gate(config_path, project_root=project_root)
    return {
        "format": CONFIG_FORMAT,
        "status": READY,
        "execution_id": gate.execution_id,
        "freeze_verified": True,
        "selection_complete": True,
        "authorization_verified": True,
        "stage14_blockers_pass": True,
        "test_data_accessed": False,
        "test_evaluation_performed": False,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
        "prior_coordinate_access_statement": gate.historical_statement,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (preflight(args.config, project_root=args.project_root) if args.preflight
                  else execute_locked_test(args.config, project_root=args.project_root))
    except (LockedTestError, ProtocolFreezeError, FileExistsError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
