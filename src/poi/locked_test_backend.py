"""Reviewed execution backend for the one-shot, prospectively locked POI test.

The public entry point is imported only after :mod:`poi.locked_test` has consumed the
one-shot reservation.  This module then performs the first permitted feature access:
it loads clean train and test, reloads the frozen attack surrogate, materializes every
frozen attack intent, and evaluates the two validation-trained detectors with the same
clean-train-only TabPFN 2.5 OvR classifiers.

There is deliberately no training or selection fallback here.  Every checkpoint,
threshold, configuration, and replay contract is named by the frozen selection and
checked against the freeze file catalog before it is used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from . import adversarial as adv
from . import defense_pipeline as defense
from .data import CATEGORICAL, FEATURES, NUMERIC, load_clean_splits, sha256
from .defense_evaluate import SharedNearestTrainRepair, binary_metrics
from .geospatial import AdministrativeBoundaryValidator
from .locked_test import (
    CLASSIFICATION_MODES,
    DETECTORS,
    EVALUATION_SCOPES,
    FIVE_METRICS,
    GENERATION_SEEDS,
    HISTORICAL_DESCRIPTOR,
    MODEL_SEEDS,
    REGIONS,
)


INPUT_FORMAT = "poi-locked-test-backend-inputs-v1"
CONTEXT_FORMAT = "poi-locked-test-backend-context-v1"
RESULT_FORMAT = "poi-locked-test-backend-result-v1"
PROPOSED = DETECTORS[0]
URE = DETECTORS[1]
PRIMARY_RETENTION = 0.95
CLASSIFICATION_THRESHOLD = 0.5
_SHA256 = frozenset("0123456789abcdef")


class LockedTestBackendError(RuntimeError):
    """A frozen backend input or execution invariant failed closed."""


@dataclass(frozen=True)
class Dimensions:
    regions: tuple[str, ...]
    model_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    test_rows: int


@dataclass(frozen=True)
class FrozenInputs:
    attack_config: Path
    boundary_config: Path
    generation_request: Path
    generation_identity: Path
    surrogate_directory: Path
    defense_config: Path
    detection_thresholds: Path
    fixed_replay_contract: Path
    proposed_checkpoints: Mapping[int, Path]
    ure_checkpoints: Mapping[int, tuple[Path, Path]]
    fixed_checkpoint: Path
    device: str
    project_root: Path | None = None
    runtime_environment: Path | None = None


def expected_cardinalities(dimensions: Dimensions) -> dict[str, int]:
    """Return the exact all-POI locked-test output sizes without opening data."""
    attack_cohorts = len(dimensions.generation_seeds) * len(dimensions.conditions)
    classification_modes = len(CLASSIFICATION_MODES) + len(CLASSIFICATION_MODES[:3])
    return {
        "attack_cohorts": attack_cohorts,
        "attack_rows": dimensions.test_rows * attack_cohorts,
        "attack_shards": attack_cohorts,
        "prediction_cohorts_per_model": 1 + attack_cohorts,
        "prediction_shards": len(dimensions.model_seeds) * (1 + attack_cohorts),
        "prediction_rows": (
            dimensions.test_rows * len(dimensions.regions) * len(DETECTORS)
            * len(dimensions.model_seeds) * (1 + attack_cohorts)
        ),
        "detector_metric_rows": (
            len(dimensions.model_seeds) * attack_cohorts * len(EVALUATION_SCOPES)
            * len(DETECTORS) * len(FIVE_METRICS)
        ),
        "classification_metric_rows": (
            len(dimensions.model_seeds) * (1 + attack_cohorts)
            * len(EVALUATION_SCOPES) * classification_modes
            * len(dimensions.regions) * len(FIVE_METRICS)
        ),
        "region_model_states": len(dimensions.model_seeds) * len(dimensions.regions),
    }


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= _SHA256


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_sha256(values: Iterable[Any]) -> str:
    return _canonical_sha256([str(value) for value in values])


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedTestBackendError(f"{field} must be a mapping")
    return value


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), field)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockedTestBackendError(f"cannot read {field}: {path}: {exc}") from exc


def _catalog(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise LockedTestBackendError("freeze manifest file catalog is absent")
    for raw in files:
        entry = _mapping(raw, "freeze manifest file")
        relative = str(entry.get("path", ""))
        digest = str(entry.get("sha256", ""))
        if not relative or relative in result or not _is_sha256(digest):
            raise LockedTestBackendError("freeze manifest file catalog is malformed")
        result[relative] = digest
    return result


def _frozen_file(
    root: Path,
    raw_path: Any,
    *,
    field: str,
    frozen_files: Mapping[str, Any],
    catalog: Mapping[str, str],
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise LockedTestBackendError(f"{field} must be one repository file")
    supplied = Path(raw_path)
    candidate = Path(os.path.abspath(os.fspath(supplied if supplied.is_absolute() else root / supplied)))
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise LockedTestBackendError(f"{field} escapes the project root") from exc
    cursor = root
    for component in Path(relative).parts:
        cursor /= component
        if cursor.is_symlink():
            raise LockedTestBackendError(f"{field} contains a symlink")
    expected = str(frozen_files.get(relative, ""))
    if not _is_sha256(expected) or catalog.get(relative) != expected:
        raise LockedTestBackendError(f"{field} is not byte-bound by the frozen selection")
    if not candidate.is_file() or sha256(candidate) != expected:
        raise LockedTestBackendError(f"{field} is absent or changed: {relative}")
    return candidate


def _seed_paths(
    root: Path,
    raw: Any,
    *,
    field: str,
    frozen_files: Mapping[str, Any],
    catalog: Mapping[str, str],
) -> dict[int, Path]:
    values = _mapping(raw, field)
    if set(map(str, values)) != set(map(str, MODEL_SEEDS)):
        raise LockedTestBackendError(f"{field} must name exactly model seeds {MODEL_SEEDS}")
    return {
        seed: _frozen_file(
            root, values[str(seed)] if str(seed) in values else values[seed], field=f"{field}.{seed}",
            frozen_files=frozen_files, catalog=catalog,
        )
        for seed in MODEL_SEEDS
    }


def _resolve_inputs(context: Mapping[str, Any]) -> tuple[Path, Dimensions, FrozenInputs]:
    if context.get("format") != CONTEXT_FORMAT or context.get("split") != "test":
        raise LockedTestBackendError("invalid locked-test backend context")
    root = Path(str(context.get("project_root", ""))).resolve()
    output = Path(str(context.get("output", ""))).resolve()
    if not output.is_dir() or not (output / "_EXECUTION_RESERVED.json").is_file():
        raise LockedTestBackendError("backend requires the exclusive one-shot reservation")
    try:
        output.relative_to(root / "artifacts")
    except ValueError as exc:
        raise LockedTestBackendError("backend output must remain below artifacts/") from exc
    selection = _mapping(context.get("selection"), "selection")
    manifest = _mapping(context.get("freeze_manifest"), "freeze manifest")
    spec = _mapping(selection.get("locked_test_backend"), "selection.locked_test_backend")
    if spec.get("format") != INPUT_FORMAT:
        raise LockedTestBackendError(f"locked_test_backend format must be {INPUT_FORMAT}")
    frozen_files = _mapping(selection.get("frozen_files"), "selection.frozen_files")
    catalog = _catalog(manifest)

    dimensions = Dimensions(
        regions=tuple(map(str, selection.get("regions", ()))),
        model_seeds=tuple(map(int, selection.get("model_seeds", ()))),
        generation_seeds=tuple(map(int, selection.get("generation_seeds", ()))),
        conditions=tuple(map(str, selection.get("attack_conditions", ()))),
        test_rows=int(context.get("expected_test_rows", -1)),
    )
    if dimensions != Dimensions(REGIONS, MODEL_SEEDS, GENERATION_SEEDS, dimensions.conditions, 6800):
        raise LockedTestBackendError("selection changes the locked regions, seeds, or test size")
    if len(dimensions.conditions) != 32 or len(set(dimensions.conditions)) != 32:
        raise LockedTestBackendError("selection must contain exactly 32 unique attack conditions")

    attack_config = _frozen_file(
        root, spec.get("attack_config"), field="attack_config",
        frozen_files=frozen_files, catalog=catalog,
    )
    boundary_config = _frozen_file(
        root, spec.get("boundary_config"), field="boundary_config",
        frozen_files=frozen_files, catalog=catalog,
    )
    try:
        boundary_cfg = yaml.safe_load(boundary_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LockedTestBackendError(f"cannot read frozen boundary config: {exc}") from exc
    boundary_sources = _mapping(boundary_cfg, "boundary config").get("sources")
    if not isinstance(boundary_sources, list) or len(boundary_sources) < 2:
        raise LockedTestBackendError("frozen boundary config requires two or more sources")
    for index, source in enumerate(boundary_sources):
        source_entry = _mapping(source, f"boundary source {index}")
        _frozen_file(
            root, source_entry.get("path"), field=f"boundary source {index}",
            frozen_files=frozen_files, catalog=catalog,
        )
    request = _frozen_file(
        root, spec.get("generation_request"), field="generation_request",
        frozen_files=frozen_files, catalog=catalog,
    )
    identity = _frozen_file(
        root, spec.get("generation_identity"), field="generation_identity",
        frozen_files=frozen_files, catalog=catalog,
    )
    surrogate_files = _mapping(spec.get("surrogate"), "locked_test_backend.surrogate")
    checkpoint = _frozen_file(
        root, surrogate_files.get("checkpoint"), field="surrogate.checkpoint",
        frozen_files=frozen_files, catalog=catalog,
    )
    encoder = _frozen_file(
        root, surrogate_files.get("encoder"), field="surrogate.encoder",
        frozen_files=frozen_files, catalog=catalog,
    )
    history = _frozen_file(
        root, surrogate_files.get("training_history"), field="surrogate.training_history",
        frozen_files=frozen_files, catalog=catalog,
    )
    if (
        checkpoint.name != "surrogate.pt"
        or encoder.name != "surrogate_encoder.json"
        or history.name != "surrogate_training.csv"
        or len({checkpoint.parent, encoder.parent, history.parent}) != 1
    ):
        raise LockedTestBackendError("surrogate files must be the exact persisted checkpoint bundle")

    defense_config = _frozen_file(
        root, spec.get("defense_config"), field="defense_config",
        frozen_files=frozen_files, catalog=catalog,
    )
    thresholds = _frozen_file(
        root, spec.get("detection_thresholds"), field="detection_thresholds",
        frozen_files=frozen_files, catalog=catalog,
    )
    replay = _frozen_file(
        root, spec.get("fixed_replay_contract"), field="fixed_replay_contract",
        frozen_files=frozen_files, catalog=catalog,
    )
    runtime_environment = _frozen_file(
        root, spec.get("runtime_environment"), field="runtime_environment",
        frozen_files=frozen_files, catalog=catalog,
    )
    if runtime_environment != replay.parent / "runtime_environment.json":
        raise LockedTestBackendError(
            "runtime_environment must be adjacent to the fixed replay contract"
        )
    replay_payload = _read_json(replay, "fixed replay contract")
    replay_fits = replay_payload.get("fits")
    if not isinstance(replay_fits, list) or len(replay_fits) != len(MODEL_SEEDS) * len(REGIONS):
        raise LockedTestBackendError("fixed replay must name all 51 fitted states")
    replay_fit_keys: set[tuple[int, str]] = set()
    for index, raw_fit in enumerate(replay_fits):
        fit = _mapping(raw_fit, f"fixed replay fit {index}")
        key = (int(fit.get("model_seed", fit.get("seed", -1))), str(fit.get("region", "")))
        if key in replay_fit_keys or key not in {
            (seed, region) for seed in MODEL_SEEDS for region in REGIONS
        }:
            raise LockedTestBackendError("fixed replay fitted-state grid is malformed")
        replay_fit_keys.add(key)
        state_path = _frozen_file(
            root, fit.get("fitted_state_path"), field=f"fitted state {index}",
            frozen_files=frozen_files, catalog=catalog,
        )
        expected_state = (
            replay.parent / "work" / "fixed_downstream_fit_states"
            / f"modelseed{key[0]}__region{REGIONS.index(key[1]):02d}.tabpfn_fit"
        )
        if state_path != expected_state:
            raise LockedTestBackendError(f"fitted state path/name differs for {key}")
    proposed = _seed_paths(
        root, spec.get("proposed_checkpoints"), field="proposed_checkpoints",
        frozen_files=frozen_files, catalog=catalog,
    )
    raw_ure = _mapping(spec.get("ure_checkpoints"), "ure_checkpoints")
    if set(map(str, raw_ure)) != set(map(str, MODEL_SEEDS)):
        raise LockedTestBackendError("ure_checkpoints must name all three model seeds")
    ure: dict[int, tuple[Path, Path]] = {}
    for seed in MODEL_SEEDS:
        raw = _mapping(raw_ure[str(seed)] if str(seed) in raw_ure else raw_ure[seed], f"ure.{seed}")
        model = _frozen_file(
            root, raw.get("detector"), field=f"ure.{seed}.detector",
            frozen_files=frozen_files, catalog=catalog,
        )
        forest = _frozen_file(
            root, raw.get("random_forest"), field=f"ure.{seed}.random_forest",
            frozen_files=frozen_files, catalog=catalog,
        )
        if forest != model.with_suffix(".joblib"):
            raise LockedTestBackendError(f"URE seed {seed} RF is not adjacent to its detector")
        ure[seed] = (model, forest)

    downstream = _mapping(selection.get("fixed_downstream_checkpoint"), "fixed checkpoint")
    fixed_checkpoint = Path(str(downstream.get("path", "")))
    expected_fixed_sha = str(downstream.get("sha256", ""))
    if (
        not fixed_checkpoint.is_absolute()
        or fixed_checkpoint.is_symlink()
        or not fixed_checkpoint.is_file()
        or not _is_sha256(expected_fixed_sha)
        or sha256(fixed_checkpoint) != expected_fixed_sha
    ):
        raise LockedTestBackendError("external fixed TabPFN checkpoint is absent or hash-mismatched")
    device = str(spec.get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise LockedTestBackendError("locked backend device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise LockedTestBackendError("frozen backend requires CUDA but CUDA is unavailable")
    return output, dimensions, FrozenInputs(
        attack_config=attack_config,
        boundary_config=boundary_config,
        generation_request=request,
        generation_identity=identity,
        surrogate_directory=checkpoint.parent,
        defense_config=defense_config,
        detection_thresholds=thresholds,
        fixed_replay_contract=replay,
        proposed_checkpoints=proposed,
        ure_checkpoints=ure,
        fixed_checkpoint=fixed_checkpoint,
        device=device,
        project_root=root,
        runtime_environment=runtime_environment,
    )


def _numeric_clean_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, _ = load_clean_splits(data_dir, splits=("train", "test"), verify_hashes=True)
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    test = clean.loc[clean["split"].eq("test")].reset_index(drop=True)
    return train, test


def _current_runtime_environment(device: str) -> dict[str, Any]:
    try:
        import tabpfn
    except ImportError as exc:
        raise LockedTestBackendError("TabPFN is required by the frozen runtime") from exc
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[str(name).lower()] = distribution.version
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tabpfn_version": tabpfn.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "requested_device": device,
        "restored_on_device": device,
        "installed_distributions": dict(sorted(installed.items())),
    }


def _validate_runtime_environment(inputs: FrozenInputs) -> Mapping[str, Any]:
    if inputs.runtime_environment is None:
        raise LockedTestBackendError("frozen Stage10 runtime_environment is required")
    frozen = _read_json(inputs.runtime_environment, "Stage10 runtime environment")
    exact_keys = {
        "format", "execution_scope", "python_version", "python_executable", "platform",
        "numpy", "pandas", "tabpfn_version", "torch_version", "cuda_version",
        "cuda_available", "gpu_device_name", "requested_device", "restored_on_device",
        "installed_distributions", "test_rows_read",
    }
    if (
        set(frozen) != exact_keys
        or frozen.get("format") != "poi-stage10-runtime-environment-v1"
        or frozen.get("execution_scope")
        != "stage10_gpu_model_fitting_and_validation_evaluation"
        or frozen.get("test_rows_read") != 0
    ):
        raise LockedTestBackendError("frozen Stage10 runtime environment header differs")
    requested = str(frozen.get("requested_device", ""))
    if (
        requested != str(frozen.get("restored_on_device", ""))
        or (requested.startswith("cuda") and (
            frozen.get("cuda_available") is not True
            or not str(frozen.get("gpu_device_name", "")).strip()
        ))
    ):
        raise LockedTestBackendError("frozen Stage10 runtime device contract is invalid")
    current = _current_runtime_environment(inputs.device)
    for field, value in current.items():
        if frozen.get(field) != value:
            raise LockedTestBackendError(f"locked-test runtime differs from Stage10: {field}")
    return frozen


def _prevalidate_static_contract(inputs: FrozenInputs, dimensions: Dimensions) -> None:
    """Reject non-data contract errors before the one permitted test feature read."""
    _validate_runtime_environment(inputs)
    try:
        attack_cfg = yaml.safe_load(inputs.attack_config.read_text(encoding="utf-8"))
        defense_cfg = yaml.safe_load(inputs.defense_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LockedTestBackendError(f"cannot read frozen backend configuration: {exc}") from exc
    if not isinstance(attack_cfg, dict) or not isinstance(defense_cfg, dict):
        raise LockedTestBackendError("frozen backend configurations must be mappings")
    specs = adv.attack_condition_specs(attack_cfg)
    if {spec["condition"] for spec in specs} != set(dimensions.conditions):
        raise LockedTestBackendError("attack configuration differs from frozen condition set")
    if tuple(map(int, attack_cfg.get("generation_seeds", ()))) != dimensions.generation_seeds:
        raise LockedTestBackendError("attack configuration changes generation seeds")
    if inputs.project_root is not None:
        configured_boundary = Path(str(attack_cfg.get("boundary_config", "")))
        if not configured_boundary.is_absolute():
            configured_boundary = inputs.project_root / configured_boundary
        if configured_boundary.resolve() != inputs.boundary_config.resolve():
            raise LockedTestBackendError("attack configuration changes the boundary source")
    request = _read_json(inputs.generation_request, "generation request")
    identity = _read_json(inputs.generation_identity, "generation identity")
    request_sha = _canonical_sha256(request)
    if (
        request.get("format") != "poi-generation-request-v2"
        or identity.get("format") != "poi-generation-identity-v2"
        or identity.get("generation_request_sha256") != request_sha
        or request.get("config_sha256") != sha256(inputs.attack_config)
        or request.get("boundary_config_sha256") != sha256(inputs.boundary_config)
        or request.get("conditions") != specs
        or tuple(map(int, request.get("generation_seeds", ()))) != dimensions.generation_seeds
    ):
        raise LockedTestBackendError("frozen attack request/configuration binding differs")
    expected_identity = {
        "format": "poi-generation-identity-v2",
        "generation_request_sha256": request_sha,
        "surrogate_checkpoint_sha256": sha256(inputs.surrogate_directory / "surrogate.pt"),
        "surrogate_encoder_sha256": sha256(inputs.surrogate_directory / "surrogate_encoder.json"),
        "surrogate_training_history_sha256": sha256(
            inputs.surrogate_directory / "surrogate_training.csv"
        ),
    }
    expected_identity["generation_identity_sha256"] = _canonical_sha256(expected_identity)
    if dict(identity) != expected_identity:
        raise LockedTestBackendError("surrogate bundle differs from its immutable identity")
    code_hashes = _mapping(request.get("code_sha256"), "generation request code hashes")
    requested_by_name = {Path(str(path)).name: str(digest) for path, digest in code_hashes.items()}
    current_sources = (Path(adv.__file__), Path(adv.__file__).with_name("data.py"),
                       Path(adv.__file__).with_name("geospatial.py"))
    if set(requested_by_name) != {path.name for path in current_sources} or any(
        requested_by_name[path.name] != sha256(path) for path in current_sources
    ):
        raise LockedTestBackendError("attack implementation differs from surrogate request")

    if tuple(map(int, defense_cfg.get("seeds", ()))) != dimensions.model_seeds or tuple(
        map(int, defense_cfg.get("generation_seeds", ()))
    ) != dimensions.generation_seeds:
        raise LockedTestBackendError("defense configuration changes the frozen seed cross")
    if (
        float(defense_cfg.get("primary_clean_retention", -1)) != PRIMARY_RETENTION
        or float(defense_cfg.get("classification_threshold", -1))
        != CLASSIFICATION_THRESHOLD
        or str(defense_cfg.get("device", inputs.device)) != inputs.device
    ):
        raise LockedTestBackendError("defense threshold/device contract differs")
    if inputs.project_root is not None:
        proposed_template = str(defense_cfg.get("proposed_checkpoint_template", ""))
        ure_template = str(defense_cfg.get("ure_checkpoint_template", ""))
        if not proposed_template or not ure_template:
            raise LockedTestBackendError("defense configuration omits checkpoint templates")
        for seed in dimensions.model_seeds:
            proposed_path = Path(proposed_template.format(seed=seed))
            ure_path = Path(ure_template.format(seed=seed))
            if not proposed_path.is_absolute():
                proposed_path = inputs.project_root / proposed_path
            if not ure_path.is_absolute():
                ure_path = inputs.project_root / ure_path
            if proposed_path.resolve() != inputs.proposed_checkpoints[seed].resolve():
                raise LockedTestBackendError(f"proposed checkpoint path differs for seed {seed}")
            if ure_path.resolve() != inputs.ure_checkpoints[seed][0].resolve():
                raise LockedTestBackendError(f"URE checkpoint path differs for seed {seed}")
    _load_thresholds(inputs.detection_thresholds, dimensions)
    replay = _read_json(inputs.fixed_replay_contract, "fixed replay contract")
    if (
        inputs.runtime_environment is None
        or replay.get("runtime_environment_sha256") != sha256(inputs.runtime_environment)
    ):
        raise LockedTestBackendError("fixed replay runtime-environment hash differs")
    static_replay = {
        "format": "poi-fixed-downstream-replay-contract-v1",
        "model_id": "tabpfn_v2_5",
        "official_checkpoint_path": str(inputs.fixed_checkpoint),
        "official_checkpoint_sha256": sha256(inputs.fixed_checkpoint),
        "fit_count": len(dimensions.model_seeds) * len(dimensions.regions),
        "expected_fit_count": len(dimensions.model_seeds) * len(dimensions.regions),
        "test_rows_read": 0,
        "model_seeds": list(dimensions.model_seeds),
        "regions": list(dimensions.regions),
        "features": list(FEATURES),
        "thresholds_sha256": sha256(inputs.detection_thresholds),
        "stage_config_sha256": sha256(inputs.defense_config),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    }
    for field, expected in static_replay.items():
        if replay.get(field) != expected:
            raise LockedTestBackendError(f"fixed replay static field differs: {field}")
    fixed_cfg = _mapping(defense_cfg.get("fixed_region_model"), "fixed region model")
    if int(replay.get("n_estimators", -1)) != int(fixed_cfg.get("n_estimators", -2)):
        raise LockedTestBackendError("fixed replay n_estimators differs")
    fits = replay.get("fits")
    if not isinstance(fits, list) or len(fits) != static_replay["fit_count"]:
        raise LockedTestBackendError("fixed replay fit attestations are incomplete")


def _validate_clean_contract(
    train: pd.DataFrame, test: pd.DataFrame, dimensions: Dimensions
) -> None:
    if len(test) != dimensions.test_rows or not test["POI_ID"].astype(str).is_unique:
        raise LockedTestBackendError("locked test row count or POI identity differs")
    if tuple(sorted(train["region"].astype(str).unique())) != dimensions.regions:
        raise LockedTestBackendError("clean train regions differ from the frozen contract")
    counts = test["region"].astype(str).value_counts()
    if dimensions.test_rows % len(dimensions.regions):
        raise LockedTestBackendError("locked test size is not region-balanced")
    expected_per_region = dimensions.test_rows // len(dimensions.regions)
    if set(counts.index) != set(dimensions.regions) or set(counts.tolist()) != {
        expected_per_region
    }:
        raise LockedTestBackendError("locked test region counts differ from the frozen balance")
    if train[FEATURES].isna().any().any() or test[FEATURES].isna().any().any():
        raise LockedTestBackendError("clean train/test features must be finite and complete")
    if not np.isfinite(train[FEATURES].to_numpy(float)).all() or not np.isfinite(
        test[FEATURES].to_numpy(float)
    ).all():
        raise LockedTestBackendError("clean train/test features contain non-finite values")
    unknown: dict[str, list[float]] = {}
    for feature in CATEGORICAL:
        train_values = set(train[feature].astype(float))
        unseen = sorted(set(test[feature].astype(float)) - train_values)
        if unseen:
            unknown[feature] = unseen
    if unknown:
        raise LockedTestBackendError(
            "locked test contains categorical codes outside the frozen clean-train vocabulary; "
            f"no clamp/remap/test-time vocabulary update is permitted: {unknown}"
        )


def _load_attack_contract(
    inputs: FrozenInputs,
    train: pd.DataFrame,
    dimensions: Dimensions,
    data_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    cfg = yaml.safe_load(inputs.attack_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise LockedTestBackendError("attack configuration is malformed")
    if tuple(map(int, cfg.get("generation_seeds", ()))) != dimensions.generation_seeds:
        raise LockedTestBackendError("attack configuration changes generation seeds")
    specs = adv.attack_condition_specs(cfg)
    if {spec["condition"] for spec in specs} != set(dimensions.conditions):
        raise LockedTestBackendError("attack configuration differs from frozen condition set")
    request = _read_json(inputs.generation_request, "generation request")
    identity = _read_json(inputs.generation_identity, "generation identity")
    request_sha = _canonical_sha256(request)
    if request.get("format") != "poi-generation-request-v2" or identity.get(
        "format"
    ) != "poi-generation-identity-v2":
        raise LockedTestBackendError("unsupported attack generation identity")
    if identity.get("generation_request_sha256") != request_sha:
        raise LockedTestBackendError("surrogate identity is not bound to the generation request")
    expected_identity = {
        "format": "poi-generation-identity-v2",
        "generation_request_sha256": request_sha,
        "surrogate_checkpoint_sha256": sha256(inputs.surrogate_directory / "surrogate.pt"),
        "surrogate_encoder_sha256": sha256(inputs.surrogate_directory / "surrogate_encoder.json"),
        "surrogate_training_history_sha256": sha256(
            inputs.surrogate_directory / "surrogate_training.csv"
        ),
    }
    expected_identity["generation_identity_sha256"] = _canonical_sha256(expected_identity)
    if dict(identity) != expected_identity:
        raise LockedTestBackendError("surrogate bundle differs from its immutable identity")
    if request.get("config_sha256") != sha256(inputs.attack_config):
        raise LockedTestBackendError("generation request attack-config hash differs")
    if request.get("boundary_config_sha256") != sha256(inputs.boundary_config):
        raise LockedTestBackendError("generation request boundary-config hash differs")
    request_conditions = request.get("conditions")
    if request_conditions != specs:
        raise LockedTestBackendError("generation request attack conditions differ")
    if tuple(map(int, request.get("generation_seeds", ()))) != dimensions.generation_seeds:
        raise LockedTestBackendError("generation request attack seeds differ")
    code_hashes = _mapping(request.get("code_sha256"), "generation request code hashes")
    requested_by_name = {Path(str(path)).name: str(digest) for path, digest in code_hashes.items()}
    current_sources = (Path(adv.__file__), Path(adv.__file__).with_name("data.py"),
                       Path(adv.__file__).with_name("geospatial.py"))
    if set(requested_by_name) != {path.name for path in current_sources}:
        raise LockedTestBackendError("generation request source coverage differs")
    for path in current_sources:
        if requested_by_name[path.name] != sha256(path):
            raise LockedTestBackendError(
                f"attack implementation differs from surrogate request: {path.name}"
            )
    dataset_hashes = _mapping(request.get("dataset_sha256"), "generation request dataset hashes")
    for name, digest in dataset_hashes.items():
        path = data_dir / str(name)
        if not path.is_file() or sha256(path) != digest:
            raise LockedTestBackendError(f"dataset differs from surrogate request: {name}")
    train_identity = _mapping(
        _mapping(request.get("split_identity"), "request split identity").get("train"),
        "request train identity",
    )
    if train_identity.get("rows") != len(train) or train_identity.get(
        "ordered_ids_sha256"
    ) != _ordered_sha256(train["POI_ID"]):
        raise LockedTestBackendError("clean train IDs differ from surrogate training identity")
    feature_digest = hashlib.sha256(
        train[["POI_ID", "region"] + FEATURES]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    if train_identity.get("ordered_id_region_feature_sha256") != feature_digest:
        raise LockedTestBackendError("clean train feature matrix differs from surrogate request")
    return cfg, specs, request, identity


def _attack_candidate(
    spec: Mapping[str, Any],
    *,
    coords: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    model: Any,
    encoder: Mapping[str, Any],
    tuples: np.ndarray,
    tuple_vectors: np.ndarray,
    cfg: Mapping[str, Any],
    device: torch.device,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = str(spec["method"])
    numeric_budget = spec["numeric_budget_m"]
    categorical_budget = int(spec["categorical_budget_l0"])
    if method in {"fgsm", "pgd", "cw_l2", "capgd"}:
        attacked_coords, queries = adv.numeric_attack(
            method, coords, categories, labels, model, encoder, numeric_budget, cfg, device
        )
        return (
            adv.enforce_numeric_budget(coords, attacked_coords, numeric_budget),
            categories.copy(),
            queries,
        )
    if method == "category_exact":
        attacked, queries = adv.categorical_exact(
            coords, categories, labels, model, encoder, tuples, tuple_vectors,
            categorical_budget, cfg, device,
        )
        return coords.copy(), attacked, queries
    if method == "pcaa":
        attacked, queries = adv.categorical_pcaa(
            coords, categories, labels, model, encoder, tuples, tuple_vectors,
            categorical_budget, cfg, device,
        )
        return coords.copy(), attacked, queries
    if method == "moeva":
        attacked_coords, attacked_categories, queries = adv.mixed_evolution(
            coords, categories, labels, model, encoder, tuples, tuple_vectors,
            numeric_budget, categorical_budget, cfg, device, rng_seed,
        )
        return (
            adv.enforce_numeric_budget(coords, attacked_coords, numeric_budget),
            attacked_categories,
            queries,
        )
    if method == "caa":
        attacked_coords, capgd_queries = adv.numeric_attack(
            "capgd", coords, categories, labels, model, encoder, numeric_budget, cfg, device
        )
        attacked_coords = adv.enforce_numeric_budget(coords, attacked_coords, numeric_budget)
        attacked_categories = categories.copy()
        capgd_pred, _ = adv.predict(
            model, attacked_coords, attacked_categories, encoder, device,
            int(cfg["attack_batch_size"]),
        )
        failed = capgd_pred == labels
        queries = np.asarray(capgd_queries, dtype=np.int64).copy() + 1
        if failed.any():
            evo_coords, evo_categories, evo_queries = adv.mixed_evolution(
                coords[failed], categories[failed], labels[failed], model, encoder,
                tuples, tuple_vectors, numeric_budget, categorical_budget, cfg, device,
                rng_seed + 10_000,
            )
            attacked_coords[failed] = evo_coords
            attacked_categories[failed] = evo_categories
            queries[failed] += evo_queries
        return (
            adv.enforce_numeric_budget(coords, attacked_coords, numeric_budget),
            attacked_categories,
            queries,
        )
    raise LockedTestBackendError(f"unknown frozen attack method: {method}")


def _materialize_attacks(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    inputs: FrozenInputs,
    dimensions: Dimensions,
    output: Path,
) -> tuple[list[Path], Mapping[str, Any]]:
    attack_dir = output / "tables" / "attacks"
    attack_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "work" / "attack_shard_manifest.json"
    manifest: dict[str, Any] = {
        "format": "poi-shard-manifest-v2",
        "generation_identity_sha256": identity["generation_identity_sha256"],
        "shards": {},
    }
    device = torch.device(inputs.device)
    adv.set_seed(int(cfg["surrogate_seed"]))
    model, encoder, _, _ = adv.load_surrogate(
        train, cfg, inputs.surrogate_directory, device, identity["generation_request_sha256"]
    )
    tuples, tuple_vectors = adv.valid_tuples(train, encoder)
    numeric_bounds = {
        "min": train[NUMERIC].min().to_numpy(np.float64),
        "max": train[NUMERIC].max().to_numpy(np.float64),
    }
    boundary = AdministrativeBoundaryValidator.from_config(
        inputs.boundary_config,
        project_root=inputs.project_root or inputs.boundary_config.parents[2],
    )
    coords = test[NUMERIC].to_numpy(np.float64)
    categories = test[CATEGORICAL].to_numpy(np.float32)
    lookup = {region: index for index, region in enumerate(encoder["regions"])}
    if tuple(encoder["regions"]) != dimensions.regions:
        raise LockedTestBackendError("surrogate region order differs from the frozen regions")
    labels = np.asarray([lookup[value] for value in test["region"]], dtype=np.int64)
    clean_boundary = boundary.validate(coords, test["region"].to_numpy())
    clean_pred, _ = adv.predict(
        model, coords, categories, encoder, device, int(cfg["attack_batch_size"])
    )
    paths: list[Path] = []
    for generation_seed in dimensions.generation_seeds:
        for spec in specs:
            condition = str(spec["condition"])
            rng_seed = adv.condition_rng_seed(generation_seed, "test", condition)
            adv.set_seed(rng_seed)
            attacked_coords, attacked_categories, queries = _attack_candidate(
                spec, coords=coords, categories=categories, labels=labels, model=model,
                encoder=encoder, tuples=tuples, tuple_vectors=tuple_vectors, cfg=cfg,
                device=device, rng_seed=rng_seed,
            )
            attack_pred, _ = adv.predict(
                model, attacked_coords, attacked_categories, encoder, device,
                int(cfg["attack_batch_size"]),
            )
            frame = adv.records(
                test, attacked_coords, attacked_categories, labels, clean_pred, attack_pred,
                spec["method"], condition, spec["numeric_budget_m"],
                spec["categorical_budget_l0"], queries, tuples, generation_seed,
                int(cfg["surrogate_seed"]), numeric_bounds, encoder, boundary,
                clean_boundary, cfg.get("distance_validation"),
                identity["generation_identity_sha256"],
                identity["surrogate_checkpoint_sha256"],
            )
            path = attack_dir / f"g{generation_seed}__{condition}.parquet"
            adv.persist_shard(
                path, frame, manifest, manifest_path, test, spec, generation_seed, cfg,
                tuples, numeric_bounds, encoder, boundary, clean_boundary, identity,
            )
            paths.append(path)
    if len(paths) != len(dimensions.generation_seeds) * len(dimensions.conditions):
        raise LockedTestBackendError("attack shard coverage is incomplete")
    return paths, manifest


def _load_thresholds(path: Path, dimensions: Dimensions) -> dict[int, dict[str, float]]:
    frame = pd.read_csv(path)
    required = {"seed", "detector", "clean_retention_target", "threshold"}
    if not required.issubset(frame.columns):
        raise LockedTestBackendError(f"frozen threshold file lacks {sorted(required - set(frame))}")
    frame = frame.loc[np.isclose(frame["clean_retention_target"].astype(float), PRIMARY_RETENTION)]
    expected = {(seed, detector) for seed in dimensions.model_seeds for detector in DETECTORS}
    observed: dict[tuple[int, str], float] = {}
    for row in frame.itertuples(index=False):
        key = (int(row.seed), str(row.detector))
        value = float(row.threshold)
        if key in observed or key not in expected or not np.isfinite(value) or not 0 <= value <= 1:
            raise LockedTestBackendError("primary frozen detector thresholds are malformed")
        observed[key] = value
    if set(observed) != expected:
        raise LockedTestBackendError("primary frozen detector threshold coverage is incomplete")
    return {
        seed: {detector: observed[(seed, detector)] for detector in DETECTORS}
        for seed in dimensions.model_seeds
    }


def _validate_replay_header(
    replay: Mapping[str, Any], inputs: FrozenInputs, dimensions: Dimensions, train: pd.DataFrame
) -> dict[tuple[int, str], Mapping[str, Any]]:
    expected_checkpoint_sha = sha256(inputs.fixed_checkpoint)
    exact = {
        "format": "poi-fixed-downstream-replay-contract-v1",
        "model_id": "tabpfn_v2_5",
        "official_checkpoint_path": str(inputs.fixed_checkpoint),
        "official_checkpoint_sha256": expected_checkpoint_sha,
        "fit_count": len(dimensions.model_seeds) * len(dimensions.regions),
        "expected_fit_count": len(dimensions.model_seeds) * len(dimensions.regions),
        "test_rows_read": 0,
        "model_seeds": list(dimensions.model_seeds),
        "regions": list(dimensions.regions),
        "features": list(FEATURES),
    }
    for field, value in exact.items():
        if replay.get(field) != value:
            raise LockedTestBackendError(f"fixed replay contract {field} differs")
    if replay.get("train_rows") != len(train):
        raise LockedTestBackendError("fixed replay train row count differs")
    if replay.get("train_poi_order_sha256") != defense._ordered_identity_sha256(
        train, ["POI_ID"]
    ):
        raise LockedTestBackendError("fixed replay clean-train identity differs")
    train_matrix_sha = hashlib.sha256(
        np.ascontiguousarray(defense._matrix(train), dtype=np.float32).tobytes()
    ).hexdigest()
    if replay.get("train_feature_matrix_sha256") != train_matrix_sha:
        raise LockedTestBackendError("fixed replay clean-train matrix differs")
    if replay.get("thresholds_sha256") != sha256(inputs.detection_thresholds):
        raise LockedTestBackendError("fixed replay threshold hash differs")
    if replay.get("stage_config_sha256") != sha256(inputs.defense_config):
        raise LockedTestBackendError("fixed replay defense-config hash differs")
    if (
        inputs.runtime_environment is None
        or replay.get("runtime_environment_sha256") != sha256(inputs.runtime_environment)
    ):
        raise LockedTestBackendError("fixed replay runtime-environment hash differs")
    runtime = _mapping(replay.get("runtime"), "fixed replay runtime")
    try:
        import tabpfn
    except ImportError as exc:
        raise LockedTestBackendError("TabPFN is required to restore frozen fit states") from exc
    current_runtime = {
        "tabpfn_version": tabpfn.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "requested_device": inputs.device,
        "restored_on_device": inputs.device,
    }
    expected_runtime_keys = {
        "python", "python_executable", "numpy", "pandas", *current_runtime.keys()
    }
    if set(runtime) != expected_runtime_keys:
        raise LockedTestBackendError("fixed replay runtime fields differ")
    current_global_runtime = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **current_runtime,
    }
    for field, value in current_global_runtime.items():
        if runtime.get(field) != value:
            raise LockedTestBackendError(f"fixed-state runtime differs for {field}")
    fits = replay.get("fits")
    if not isinstance(fits, list) or len(fits) != exact["fit_count"]:
        raise LockedTestBackendError("fixed replay fit attestations are incomplete")
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    common_probe_sha: str | None = None
    matrix = np.ascontiguousarray(defense._matrix(train), dtype=np.float32)
    categorical_indices = [
        index for index, feature in enumerate(FEATURES) if feature in CATEGORICAL
    ]
    for raw in fits:
        fit = _mapping(raw, "fixed replay fit")
        key = (int(fit.get("model_seed", fit.get("seed", -1))), str(fit.get("region", "")))
        if key in indexed:
            raise LockedTestBackendError("duplicate fixed replay fit identity")
        state_path = Path(str(fit.get("fitted_state_path", "")))
        try:
            region_index = dimensions.regions.index(key[1])
        except ValueError as exc:
            raise LockedTestBackendError(f"unknown fitted-state region: {key[1]}") from exc
        expected_state_path = (
            inputs.fixed_replay_contract.parent / "work" / "fixed_downstream_fit_states"
            / f"modelseed{key[0]}__region{region_index:02d}.tabpfn_fit"
        ).resolve()
        probe_indices = fit.get("probe_indices")
        if (
            fit.get("format") != "poi-tabpfn25-fit-replay-v1"
            or fit.get("model_id") != "tabpfn_v2_5"
            or fit.get("official_checkpoint_path") != str(inputs.fixed_checkpoint)
            or fit.get("official_checkpoint_sha256") != expected_checkpoint_sha
            or list(fit.get("feature_order", ())) != list(FEATURES)
            or fit.get("categorical_feature_indices") != categorical_indices
            or int(fit.get("train_rows", -1)) != len(train)
            or fit.get("train_matrix_dtype") != "float32"
            or fit.get("train_matrix_sha256") != train_matrix_sha
            or fit.get("train_poi_order_sha256") != replay.get("train_poi_order_sha256")
            or int(fit.get("n_estimators", -1)) != int(replay.get("n_estimators", -2))
            or float(fit.get("classification_threshold", -1))
            != float(replay.get("classification_threshold", -2))
            or fit.get("fit_mode") != "fit_preprocessors"
            or fit.get("memory_saving_mode") != "auto"
            or fit.get("fitted_state_format") != "tabpfn_official_fit_state_v1"
            or fit.get("fitted_state_api")
            != "TabPFNClassifier.save_fit_state/load_from_fit_state"
            or state_path.resolve() != expected_state_path
            or not state_path.is_file()
            or not _is_sha256(fit.get("fitted_state_sha256"))
            or sha256(state_path) != fit.get("fitted_state_sha256")
            or state_path.stat().st_size != int(fit.get("fitted_state_size_bytes", -1))
            or not isinstance(probe_indices, list)
            or not probe_indices
            or not all(isinstance(value, int) for value in probe_indices)
            or probe_indices != sorted(set(probe_indices))
            or min(probe_indices) < 0
            or max(probe_indices) >= len(train)
            or len(probe_indices) > 64
        ):
            raise LockedTestBackendError(f"frozen fitted-state recipe is invalid: {key}")
        truth = np.ascontiguousarray(
            train["region"].eq(key[1]).astype(int).to_numpy(), dtype=np.int8
        )
        probe = np.ascontiguousarray(matrix[probe_indices], dtype=np.float32)
        probe_sha = hashlib.sha256(probe.tobytes()).hexdigest()
        if (
            fit.get("truth_sha256") != hashlib.sha256(truth.tobytes()).hexdigest()
            or fit.get("probe_matrix_sha256") != probe_sha
            or fit.get("probe_prediction_sha256")
            != fit.get("probe_repeat_prediction_sha256")
            or fit.get("probe_prediction_sha256")
            != fit.get("restored_probe_prediction_sha256")
            or int(fit.get("seed", -1)) != key[0]
        ):
            raise LockedTestBackendError(f"frozen fitted-state train/probe hash differs: {key}")
        if common_probe_sha is None:
            common_probe_sha = probe_sha
        elif probe_sha != common_probe_sha:
            raise LockedTestBackendError("frozen fitted states use different probe matrices")
        for field, value in current_runtime.items():
            if fit.get(field) != value:
                raise LockedTestBackendError(
                    f"frozen fitted-state runtime differs for {key}: {field}"
                )
        indexed[key] = fit
    expected_keys = {
        (seed, region) for seed in dimensions.model_seeds for region in dimensions.regions
    }
    if set(indexed) != expected_keys:
        raise LockedTestBackendError("fixed replay seed/region coverage differs")
    return indexed


class FrozenRegionModelSet:
    """Inference-only wrapper around the 17 official persisted TabPFN fit states."""

    def __init__(self, regions: Sequence[str], models: Mapping[str, Any]) -> None:
        self.regions = tuple(regions)
        self._models = dict(models)
        if set(self._models) != set(self.regions):
            raise LockedTestBackendError("frozen region-state coverage is incomplete")
        self.state_count = len(self._models)

    def predict(self, matrix5: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix5, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all():
            raise LockedTestBackendError("frozen region classifier input must be finite N x 5")
        columns: list[np.ndarray] = []
        for region in self.regions:
            probability = np.asarray(
                self._models[region].predict_proba(matrix)[:, 1], dtype=float
            )
            if probability.shape != (len(matrix),) or not np.isfinite(probability).all() or (
                (probability < 0) | (probability > 1)
            ).any():
                raise LockedTestBackendError(f"invalid frozen fitted-state output: {region}")
            columns.append(probability)
        return np.column_stack(columns)


def _default_state_loader(path: Path, device: str) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier.load_from_fit_state(path, device=device)


def _load_frozen_region_models(
    *,
    train: pd.DataFrame,
    frozen_fits: Mapping[tuple[int, str], Mapping[str, Any]],
    seed: int,
    regions: Sequence[str],
    device: str,
    state_loader: Callable[[Path, str], Any] = _default_state_loader,
) -> FrozenRegionModelSet:
    matrix = np.ascontiguousarray(defense._matrix(train), dtype=np.float32)
    models: dict[str, Any] = {}
    for region in regions:
        fit = frozen_fits[(seed, region)]
        state_path = Path(str(fit["fitted_state_path"]))
        model = state_loader(state_path, device)
        probe = np.ascontiguousarray(matrix[fit["probe_indices"]], dtype=np.float32)
        prediction = np.ascontiguousarray(model.predict_proba(probe)[:, 1], dtype=np.float64)
        observed_sha = hashlib.sha256(prediction.tobytes()).hexdigest()
        if observed_sha != fit["restored_probe_prediction_sha256"]:
            raise LockedTestBackendError(
                f"loaded frozen fitted state changes probe prediction: seed={seed} region={region}"
            )
        models[region] = model
    return FrozenRegionModelSet(regions, models)


def _load_detectors(inputs: FrozenInputs, defense_cfg: Mapping[str, Any], seed: int) -> tuple[Any, Any]:
    from .proposed_defense import load_proposed_predictor
    from .ure import load_ure_attack_detector

    proposed = load_proposed_predictor(
        inputs.proposed_checkpoints[seed], device=inputs.device,
        official_checkpoint=inputs.fixed_checkpoint,
        batch_size=int(defense_cfg.get("proposed_batch_size", 128)),
    )
    proposed.assert_decision_contract(
        classification_threshold=CLASSIFICATION_THRESHOLD,
        regional_decision_rule="independent_ovr_probability_gte_threshold",
    )
    ure = load_ure_attack_detector(inputs.ure_checkpoints[seed][0], device=inputs.device)
    return proposed, ure


def _attack_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["POI_ID", "region"]].copy()
    for feature in FEATURES:
        result[feature] = pd.to_numeric(frame[f"{feature}_adv"], errors="raise")
    defense._matrix(result)
    return result


def _prediction_rows(
    frame: pd.DataFrame,
    observations: Mapping[str, Any],
    *,
    seed: int,
    generation_seed: int,
    condition: str,
    regions: Sequence[str],
    thresholds: Mapping[str, float],
    valid: np.ndarray,
    quarantine: np.ndarray,
    provenance: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    provenance = provenance or {}
    n = len(frame)
    region_count = len(regions)
    truth = np.asarray(observations["truth"], dtype=bool)
    parts: list[pd.DataFrame] = []
    for detector in DETECTORS:
        detector_score = np.asarray(observations["detector"][detector], dtype=float)
        accepted = detector_score <= thresholds[detector]
        parts.append(pd.DataFrame({
            "POI_ID": np.repeat(frame["POI_ID"].astype(str).to_numpy(), region_count),
            "source_region": np.repeat(frame["region"].astype(str).to_numpy(), region_count),
            "target_region": np.tile(np.asarray(regions, dtype=object), n),
            "truth": truth.reshape(-1).astype(np.int8),
            "model_seed": seed,
            "generation_seed": generation_seed,
            "attack_condition": condition,
            "attack_label": np.int8(condition != "clean"),
            "detector": detector,
            "detector_score": np.repeat(detector_score, region_count),
            "detector_threshold": thresholds[detector],
            "detector_accepted": np.repeat(accepted, region_count),
            "fixed_unmodified_score": np.asarray(observations["fixed"]).reshape(-1),
            "shared_nearest_train_repair_score": np.asarray(observations["shared"]).reshape(-1),
            "proposed_projected_repair_fixed_score": (
                np.asarray(observations["projected"]).reshape(-1)
                if detector == PROPOSED else np.full(n * region_count, np.nan)
            ),
            "proposed_native_full_coverage_score": (
                np.asarray(observations["native"]).reshape(-1)
                if detector == PROPOSED else np.full(n * region_count, np.nan)
            ),
            "primary_evaluation_eligible": np.repeat(valid.astype(bool), region_count),
            "attack_validity": np.repeat(
                np.where(valid.astype(bool), "primary_valid", "quarantined"), region_count
            ),
            "quarantine_reason": np.repeat(quarantine.astype(str), region_count),
            "attack_method": np.repeat(
                provenance.get("attack_method", np.full(n, "clean", object)), region_count
            ),
            "numeric_budget_m": np.repeat(
                provenance.get("numeric_budget_m", np.zeros(n)), region_count
            ),
            "categorical_budget_l0": np.repeat(
                provenance.get("categorical_budget_l0", np.zeros(n)), region_count
            ),
            "source_model": np.repeat(
                provenance.get("source_model", np.full(n, "none", object)), region_count
            ),
            "victim_model": np.repeat(
                provenance.get("victim_model", np.full(n, "none", object)), region_count
            ),
            "victim_evaluation_scope": np.repeat(
                provenance.get("victim_evaluation_scope", np.full(n, "clean", object)),
                region_count,
            ),
            "constraints_valid": np.repeat(
                provenance.get("constraints_valid", np.ones(n, bool)), region_count
            ),
            "label_preserved": np.repeat(
                provenance.get("label_preserved", np.ones(n, bool)), region_count
            ),
            "attack_attempted": np.repeat(
                provenance.get("attack_attempted", np.zeros(n, bool)), region_count
            ),
            "attack_changed_any": np.repeat(
                provenance.get("attack_changed_any", np.zeros(n, bool)), region_count
            ),
            "transfer_evaluated": np.repeat(
                provenance.get("transfer_evaluated", np.zeros(n, bool)), region_count
            ),
            "targeted": np.repeat(
                provenance.get("targeted", np.zeros(n, bool)), region_count
            ),
            "valid_numeric_train_range": np.repeat(
                provenance.get("valid_numeric_train_range", np.ones(n, bool)), region_count
            ),
            "valid_category_codebook": np.repeat(
                provenance.get("valid_category_codebook", np.ones(n, bool)), region_count
            ),
            "valid_category_combo": np.repeat(
                provenance.get("valid_category_combo", np.ones(n, bool)), region_count
            ),
            "within_budget": np.repeat(
                provenance.get("within_budget", np.ones(n, bool)), region_count
            ),
            "clean_boundary_status": np.repeat(
                provenance.get("clean_boundary_status", np.full(n, "clean", object)),
                region_count,
            ),
            "attack_boundary_status": np.repeat(
                provenance.get("attack_boundary_status", np.full(n, "clean", object)),
                region_count,
            ),
            "generation_identity_sha256": np.repeat(
                provenance.get("generation_identity_sha256", np.full(n, "clean", object)),
                region_count,
            ),
            "surrogate_checkpoint_sha256": np.repeat(
                provenance.get("surrogate_checkpoint_sha256", np.full(n, "clean", object)),
                region_count,
            ),
            "query_count": np.repeat(
                provenance.get("query_count", np.zeros(n, dtype=np.int64)), region_count
            ),
        }))
    return pd.concat(parts, ignore_index=True)


def _metric_encoding(metrics: Mapping[str, Any], metric: str) -> tuple[Any, bool, str]:
    if int(metrics.get("n", 0)) == 0:
        return np.nan, False, "empty_population"
    value = float(metrics[metric])
    if not np.isfinite(value):
        return np.nan, False, "single_class_population" if metric == "roc_auc" else "undefined"
    return value, True, ""


def _classification_rows(
    observations: Mapping[str, Any],
    *,
    model_seed: int,
    generation_seed: int,
    condition: str,
    scope: str,
    mask: np.ndarray,
    regions: Sequence[str],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    truth = np.asarray(observations["truth"], dtype=int)[mask]
    fixed = np.asarray(observations["fixed"], dtype=float)[mask]
    shared = np.asarray(observations["shared"], dtype=float)[mask]
    native = np.asarray(observations["native"], dtype=float)[mask]
    projected = np.asarray(observations["projected"], dtype=float)[mask]
    total = int(mask.sum())
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        scores = np.asarray(observations["detector"][detector], dtype=float)[mask]
        accepted_mask = scores <= thresholds[detector]
        accepted_count = int(accepted_mask.sum())
        threshold_coverage = accepted_count / total if total else np.nan
        accepted_fixed_correct = (
            ((fixed >= CLASSIFICATION_THRESHOLD) == truth) & accepted_mask[:, None]
        )
        shared_final = np.where(accepted_mask[:, None], fixed, shared)
        modes: list[tuple[str, np.ndarray, int, int]] = [
            ("fixed_unmodified", fixed, total, total),
            ("detector_accepted_only", fixed[accepted_mask], accepted_count, accepted_count),
            ("shared_nearest_train_repair", shared_final, accepted_count, total),
        ]
        if detector == PROPOSED:
            modes.extend([
                (
                    "proposed_projected_repair_fixed_tabpfn25",
                    np.where(accepted_mask[:, None], fixed, projected), accepted_count, total,
                ),
                ("proposed_native_full_coverage", native, total, total),
            ])
        for mode, values, accepted, population in modes:
            mode_truth = truth[accepted_mask] if mode == "detector_accepted_only" else truth
            coverage = accepted / total if total else np.nan
            for region_index, region in enumerate(regions):
                metrics = binary_metrics(
                    mode_truth[:, region_index], values[:, region_index],
                    CLASSIFICATION_THRESHOLD,
                )
                for metric in FIVE_METRICS:
                    value, defined, reason = _metric_encoding(metrics, metric)
                    rows.append({
                        "detector": detector,
                        "model_seed": model_seed,
                        "generation_seed": generation_seed,
                        "attack_condition": condition,
                        "evaluation_scope": scope,
                        "classification_mode": mode,
                        "region": region,
                        "metric": metric,
                        "value": value,
                        "metric_defined": defined,
                        "undefined_reason": reason,
                        "detector_coverage": coverage,
                        "accepted_rows": accepted,
                        "total_rows": total,
                        "metric_population_rows": population,
                        "threshold_detector_coverage": threshold_coverage,
                        "threshold_accepted_rows": accepted_count,
                        "threshold_rejected_rows": total - accepted_count,
                        "accepted_and_fixed_correct_fraction_all": (
                            float(accepted_fixed_correct[:, region_index].mean())
                            if total else np.nan
                        ),
                    })
    return rows


def _detector_rows(
    clean: Mapping[str, Any],
    attack: Mapping[str, Any],
    *,
    model_seed: int,
    generation_seed: int,
    condition: str,
    scope: str,
    attack_mask: np.ndarray,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_n = len(np.asarray(clean["fixed"]))
    for detector in DETECTORS:
        score = np.concatenate([
            np.asarray(clean["detector"][detector], dtype=float),
            np.asarray(attack["detector"][detector], dtype=float)[attack_mask],
        ])
        truth = np.concatenate([
            np.zeros(clean_n, dtype=np.int8), np.ones(int(attack_mask.sum()), dtype=np.int8)
        ])
        metrics = binary_metrics(
            truth, score, thresholds[detector], threshold_operator="gt"
        )
        for metric in FIVE_METRICS:
            value, defined, reason = _metric_encoding(metrics, metric)
            rows.append({
                "detector": detector,
                "model_seed": model_seed,
                "generation_seed": generation_seed,
                "attack_condition": condition,
                "evaluation_scope": scope,
                "metric": metric,
                "value": value,
                "metric_defined": defined,
                "undefined_reason": reason,
                "clean_retention": PRIMARY_RETENTION,
                "detection_threshold": thresholds[detector],
                "detector_decision_rule": "attack_probability_strictly_gt_threshold",
                "threshold_tie_rows": int(np.equal(score, thresholds[detector]).sum()),
                "clean_rows": clean_n,
                "attack_rows": int(attack_mask.sum()),
            })
    return rows


def _provenance(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    names = (
        "attack_method", "numeric_budget_m", "categorical_budget_l0", "source_model",
        "victim_model", "victim_evaluation_scope", "constraints_valid", "label_preserved",
        "attack_attempted", "attack_changed_any", "transfer_evaluated", "targeted",
        "valid_numeric_train_range", "valid_category_codebook", "valid_category_combo",
        "within_budget", "clean_boundary_status", "attack_boundary_status",
        "generation_identity_sha256", "surrogate_checkpoint_sha256", "query_count",
    )
    return {name: frame[name].to_numpy() for name in names}


def _prediction_shard_entry(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "rows": len(frame),
        "ordered_poi_ids_sha256": _ordered_sha256(
            frame.drop_duplicates("POI_ID", keep="first")["POI_ID"]
        ),
    }


def _evaluate_defenses(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    attack_paths: Sequence[Path],
    inputs: FrozenInputs,
    dimensions: Dimensions,
    output: Path,
    model_loader: Callable[[FrozenInputs, Mapping[str, Any], int], tuple[Any, Any]] = _load_detectors,
    state_loader: Callable[[Path, str], Any] = _default_state_loader,
    observation_fn: Callable[..., Mapping[str, Any]] = defense._observations,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, int]:
    defense_cfg = yaml.safe_load(inputs.defense_config.read_text(encoding="utf-8"))
    if not isinstance(defense_cfg, dict):
        raise LockedTestBackendError("defense configuration is malformed")
    if tuple(map(int, defense_cfg.get("seeds", ()))) != dimensions.model_seeds:
        raise LockedTestBackendError("defense configuration changes model seeds")
    if tuple(map(int, defense_cfg.get("generation_seeds", ()))) != dimensions.generation_seeds:
        raise LockedTestBackendError("defense configuration changes generation seeds")
    if float(defense_cfg.get("primary_clean_retention", -1)) != PRIMARY_RETENTION:
        raise LockedTestBackendError("defense configuration changes primary clean retention")
    if float(defense_cfg.get("classification_threshold", -1)) != CLASSIFICATION_THRESHOLD:
        raise LockedTestBackendError("defense configuration changes classification threshold")
    if str(defense_cfg.get("device", inputs.device)) != inputs.device:
        raise LockedTestBackendError("defense configuration changes the frozen device")
    root = inputs.project_root
    if root is not None:
        proposed_template = str(defense_cfg.get("proposed_checkpoint_template", ""))
        ure_template = str(defense_cfg.get("ure_checkpoint_template", ""))
        if not proposed_template or not ure_template:
            raise LockedTestBackendError("defense configuration omits checkpoint templates")
        for seed in dimensions.model_seeds:
            configured_proposed = Path(proposed_template.format(seed=seed))
            configured_ure = Path(ure_template.format(seed=seed))
            if not configured_proposed.is_absolute():
                configured_proposed = root / configured_proposed
            if not configured_ure.is_absolute():
                configured_ure = root / configured_ure
            if configured_proposed.resolve() != inputs.proposed_checkpoints[seed].resolve():
                raise LockedTestBackendError(f"proposed checkpoint path differs for seed {seed}")
            if configured_ure.resolve() != inputs.ure_checkpoints[seed][0].resolve():
                raise LockedTestBackendError(f"URE checkpoint path differs for seed {seed}")
    fixed_cfg = dict(_mapping(defense_cfg.get("fixed_region_model"), "fixed region model"))
    fixed_cfg["official_checkpoint"] = str(inputs.fixed_checkpoint)
    fixed_cfg["expected_checkpoint_sha256"] = sha256(inputs.fixed_checkpoint)
    fixed_cfg["classification_threshold"] = CLASSIFICATION_THRESHOLD
    thresholds = _load_thresholds(inputs.detection_thresholds, dimensions)
    replay = _read_json(inputs.fixed_replay_contract, "fixed replay contract")
    if int(replay.get("n_estimators", -1)) != int(fixed_cfg.get("n_estimators", -2)):
        raise LockedTestBackendError("fixed replay n_estimators differs from defense config")
    if float(replay.get("classification_threshold", -1)) != CLASSIFICATION_THRESHOLD:
        raise LockedTestBackendError("fixed replay classification threshold differs")
    frozen_fits = _validate_replay_header(replay, inputs, dimensions, train)

    by_key: dict[tuple[int, str], Path] = {}
    for path in attack_paths:
        frame = pd.read_parquet(path, columns=["generation_seed", "attack_condition"])
        if frame.empty:
            raise LockedTestBackendError(f"empty attack shard: {path}")
        key = (int(frame["generation_seed"].iloc[0]), str(frame["attack_condition"].iloc[0]))
        if key in by_key or set(frame["generation_seed"].astype(int)) != {key[0]} or set(
            frame["attack_condition"].astype(str)
        ) != {key[1]}:
            raise LockedTestBackendError("attack shard identity is duplicate or inconsistent")
        by_key[key] = path
    expected_keys = {
        (seed, condition)
        for seed in dimensions.generation_seeds
        for condition in dimensions.conditions
    }
    if set(by_key) != expected_keys:
        raise LockedTestBackendError("attack shard matrix is incomplete")

    prediction_dir = output / "tables" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    prediction_entries: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    state_count = 0
    repair = SharedNearestTrainRepair().fit(train[FEATURES])

    for model_seed in dimensions.model_seeds:
        proposed, ure = model_loader(inputs, defense_cfg, model_seed)
        fixed = _load_frozen_region_models(
            train=train, frozen_fits=frozen_fits, seed=model_seed,
            regions=dimensions.regions, device=inputs.device,
            state_loader=state_loader,
        )
        if fixed.state_count != len(dimensions.regions):
            raise LockedTestBackendError("frozen region-state load count differs")
        state_count += fixed.state_count
        clean_observations = observation_fn(test, proposed, ure, fixed, repair, dimensions.regions)
        clean_valid = np.ones(len(test), dtype=bool)
        clean_quarantine = np.full(len(test), "", dtype=object)
        clean_predictions = _prediction_rows(
            test, clean_observations, seed=model_seed, generation_seed=-1,
            condition="clean", regions=dimensions.regions,
            thresholds=thresholds[model_seed], valid=clean_valid,
            quarantine=clean_quarantine,
        )
        clean_path = prediction_dir / f"model{model_seed}__clean.parquet"
        _atomic_frame(clean_predictions, clean_path)
        clean_entry = _prediction_shard_entry(clean_path, clean_predictions)
        clean_entry["path"] = clean_path.relative_to(output).as_posix()
        prediction_entries.append(clean_entry)
        for scope in EVALUATION_SCOPES:
            classification_rows.extend(_classification_rows(
                clean_observations, model_seed=model_seed, generation_seed=-1,
                condition="clean", scope=scope, mask=clean_valid,
                regions=dimensions.regions, thresholds=thresholds[model_seed],
            ))

        for generation_seed in dimensions.generation_seeds:
            for condition in dimensions.conditions:
                attack = pd.read_parquet(by_key[(generation_seed, condition)])
                if len(attack) != len(test) or attack["POI_ID"].astype(str).tolist() != test[
                    "POI_ID"
                ].astype(str).tolist():
                    raise LockedTestBackendError("attack shard loses paired test POIs")
                attack_frame = _attack_feature_frame(attack)
                observations = observation_fn(
                    attack_frame, proposed, ure, fixed, repair, dimensions.regions
                )
                valid = attack["primary_evaluation_eligible"].astype(bool).to_numpy()
                quarantine = attack["quarantine_reason"].fillna("").astype(str).to_numpy()
                if (valid & (quarantine != "")).any() or ((~valid) & (quarantine == "")).any():
                    raise LockedTestBackendError("attack validity/quarantine contract differs")
                predictions = _prediction_rows(
                    attack_frame, observations, seed=model_seed,
                    generation_seed=generation_seed, condition=condition,
                    regions=dimensions.regions, thresholds=thresholds[model_seed],
                    valid=valid, quarantine=quarantine, provenance=_provenance(attack),
                )
                prediction_path = prediction_dir / (
                    f"model{model_seed}__g{generation_seed}__{condition}.parquet"
                )
                _atomic_frame(predictions, prediction_path)
                entry = _prediction_shard_entry(prediction_path, predictions)
                entry["path"] = prediction_path.relative_to(output).as_posix()
                prediction_entries.append(entry)
                masks = {
                    "primary_valid": valid,
                    "all_intent_to_attack": np.ones(len(attack), dtype=bool),
                }
                for scope, mask in masks.items():
                    detector_rows.extend(_detector_rows(
                        clean_observations, observations, model_seed=model_seed,
                        generation_seed=generation_seed, condition=condition,
                        scope=scope, attack_mask=mask,
                        thresholds=thresholds[model_seed],
                    ))
                    classification_rows.extend(_classification_rows(
                        observations, model_seed=model_seed,
                        generation_seed=generation_seed, condition=condition,
                        scope=scope, mask=mask, regions=dimensions.regions,
                        thresholds=thresholds[model_seed],
                    ))
    return (
        prediction_entries,
        pd.DataFrame(detector_rows),
        pd.DataFrame(classification_rows),
        state_count,
    )


def _attack_manifest(
    attack_paths: Sequence[Path], test: pd.DataFrame, dimensions: Dimensions, output: Path
) -> Mapping[str, Any]:
    counts: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    observed = 0
    for path in attack_paths:
        frame = pd.read_parquet(
            path, columns=[
                "POI_ID", "generation_seed", "attack_condition",
                "primary_evaluation_eligible", "quarantine_reason",
            ],
        )
        if frame["POI_ID"].astype(str).tolist() != test["POI_ID"].astype(str).tolist():
            raise LockedTestBackendError("attack manifest found an unpaired POI shard")
        seed = int(frame["generation_seed"].iloc[0])
        condition = str(frame["attack_condition"].iloc[0])
        primary = int(frame["primary_evaluation_eligible"].astype(bool).sum())
        counts.append({
            "generation_seed": seed,
            "attack_condition": condition,
            "all_intent_rows": len(frame),
            "primary_valid_rows": primary,
            "quarantined_rows": len(frame) - primary,
        })
        shards.append({
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256(path),
            "rows": len(frame),
        })
        observed += len(frame)
    expected = dimensions.test_rows * len(dimensions.generation_seeds) * len(dimensions.conditions)
    return {
        "format": "poi-locked-test-attack-manifest-v1",
        "split": "test",
        "test_rows": dimensions.test_rows,
        "generation_seeds": list(dimensions.generation_seeds),
        "attack_conditions": list(dimensions.conditions),
        "expected_rows": expected,
        "observed_rows": observed,
        "paired_to_every_test_poi": True,
        "all_source_pois_retained": True,
        "failed_attacks_retained": True,
        "quarantined_rows_retained": True,
        "primary_valid_and_all_intent_scopes_preserved": True,
        "test_labels_not_used_for_repair": True,
        "ordered_test_poi_ids_sha256": _ordered_sha256(test["POI_ID"]),
        "cohort_counts": sorted(
            counts, key=lambda row: (row["generation_seed"], row["attack_condition"])
        ),
        "shards": shards,
    }


def _base_result(
    context: Mapping[str, Any], dimensions: Dimensions, state_count: int
) -> dict[str, Any]:
    return {
        "format": RESULT_FORMAT,
        "execution_id": str(context["execution_id"]),
        "split": "test",
        "test_rows": dimensions.test_rows,
        "test_executions": 1,
        "region_classifier": "tabpfn_v2_5",
        # The outer v1 contract names the 51 frozen train-only fits.  This execution
        # restores their official fitted states and performs zero new fits.
        "region_model_fit_count": state_count,
        "region_model_fit_scope": "clean_train_only",
        "region_model_state_load_count": state_count,
        "region_model_runtime_fit_count": 0,
        "region_model_state_binding": "official_tabpfn_save_fit_state_v1",
        "same_region_classifier_shared_across_detectors": True,
        "region_classifier_refit_on_test": False,
        "detectors": list(DETECTORS),
        "detector_labels": {"normal": 0, "attack": 1},
        "regions": list(dimensions.regions),
        "model_seeds": list(dimensions.model_seeds),
        "generation_seeds": list(dimensions.generation_seeds),
        "attack_conditions": list(dimensions.conditions),
        "metrics": list(FIVE_METRICS),
        "evaluation_scopes": list(EVALUATION_SCOPES),
        "classification_modes": list(CLASSIFICATION_MODES),
        "primary_clean_retention": PRIMARY_RETENTION,
        "shared_nearest_train_repair_for_both_detectors": True,
        "proposed_native_full_population": True,
        "ure_native_reconstruction": False,
        "uses_dataset_label_for_repair": False,
        "preserved_rejected_and_quarantined_denominators": True,
        "historical_test_descriptor": HISTORICAL_DESCRIPTOR,
        "historically_pristine_test_claim_permitted": False,
    }


def run_locked_test_backend(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute the reviewed locked-test backend after the gate reservation.

    This callable has no dry-run or fallback branch.  Any missing frozen artifact,
    hash mismatch, incomplete condition, failed model replay, or malformed output
    raises before a result can be returned to the outer gate.
    """
    output, dimensions, inputs = _resolve_inputs(context)
    _prevalidate_static_contract(inputs, dimensions)
    data_dir = Path(str(context["project_root"])) / str(context["clean_dataset_dir"])
    train, test = _numeric_clean_data(data_dir)
    _validate_clean_contract(train, test, dimensions)
    cfg, specs, _, identity = _load_attack_contract(
        inputs, train, dimensions, data_dir
    )
    attack_paths, _ = _materialize_attacks(
        train=train, test=test, cfg=cfg, specs=specs, identity=identity,
        inputs=inputs, dimensions=dimensions, output=output,
    )
    attack_manifest = _attack_manifest(attack_paths, test, dimensions, output)
    attack_manifest_path = output / "tables" / "attack_manifest.json"
    _atomic_json(attack_manifest_path, attack_manifest)

    prediction_entries, detector_metrics, classification_metrics, state_count = _evaluate_defenses(
        train=train, test=test, attack_paths=attack_paths, inputs=inputs,
        dimensions=dimensions, output=output,
    )
    expected_state_count = len(dimensions.model_seeds) * len(dimensions.regions)
    if state_count != expected_state_count:
        raise LockedTestBackendError("fixed downstream fitted-state load count is incomplete")
    detector_path = output / "tables" / "detector_metrics.csv"
    classification_path = output / "tables" / "classification_metrics.csv"
    _atomic_frame(detector_metrics, detector_path)
    _atomic_frame(classification_metrics, classification_path)
    prediction_manifest = {
        "format": "poi-locked-test-prediction-manifest-v1",
        "split": "test",
        "test_rows": dimensions.test_rows,
        "model_seeds": list(dimensions.model_seeds),
        "generation_seeds": list(dimensions.generation_seeds),
        "attack_conditions": list(dimensions.conditions),
        "detectors": list(DETECTORS),
        "regions": list(dimensions.regions),
        "expected_prediction_rows": expected_cardinalities(dimensions)["prediction_rows"],
        "observed_prediction_rows": sum(entry["rows"] for entry in prediction_entries),
        "paired_across_detectors": True,
        "paired_across_all_17_ovr_regions": True,
        "rejected_rows_retained": True,
        "quarantined_rows_retained": True,
        "shards": prediction_entries,
    }
    prediction_manifest_path = output / "tables" / "prediction_manifest.json"
    _atomic_json(prediction_manifest_path, prediction_manifest)

    result = _base_result(context, dimensions, state_count)
    metadata_path = output / "metadata.json"
    _atomic_json(metadata_path, result)
    paths = {
        "detector_metrics": detector_path,
        "classification_metrics": classification_path,
        "prediction_manifest": prediction_manifest_path,
        "attack_manifest": attack_manifest_path,
        "metadata": metadata_path,
    }
    result["artifacts"] = {
        label: {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
        for label, path in paths.items()
    }
    return result
