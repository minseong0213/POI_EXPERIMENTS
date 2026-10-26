"""Fit the missing models for the prospective clean feature comparison.

The source contract for this module is frozen and independently reviewed before
any fit starts.  It supports only three jobs: coordinate-only trees,
coordinate-only TabPFN 2.6, and full-feature TabPFN v3 with official persisted
fit states for downstream attack/defense replay.  Test rows are never loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import yaml

from .data import load_clean_splits, sha256


# Kept injectable for focused synthetic tests; production resolves it lazily so
# source-freeze and config checks do not import optional explanation packages.
ModelFactory = None


FULL_FEATURES = ("X_COORD", "Y_COORD", "ASORT_LCLASDC", "ASORT_MLSFCDC", "ASORT_SDASDC")
COORDINATE_FEATURES = ("X_COORD", "Y_COORD")
TREE_MODELS = ("decision_tree", "random_forest", "xgboost", "lightgbm", "catboost")
SEEDS = (42, 202, 340)
SOURCE_FORMAT = "poi-unified-clean-fit-source-freeze-v1"
FIT_FORMAT = "poi-unified-clean-actual-fit-v1"


class UnifiedFitError(RuntimeError):
    """Raised when a frozen clean fit contract fails closed."""


def _matrix(frame: pd.DataFrame, features: Sequence[str], dtype=np.float32) -> np.ndarray:
    values = frame.loc[:, list(features)].copy()
    for name in features:
        values[name] = pd.to_numeric(values[name], errors="raise")
    matrix = values.to_numpy(dtype=dtype)
    if not np.isfinite(matrix).all():
        raise UnifiedFitError("Model matrix contains non-finite values")
    return matrix


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _ordered_ids_sha(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    return {"commit": commit, "dirty": dirty}


def _versions(packages: Sequence[str]) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved) or any(path.is_symlink() or not path.is_file() for path in resolved):
        raise UnifiedFitError("Fit input catalog requires unique regular non-symlink files")
    return {str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(resolved, key=str)}


def _source_catalog(cfg: Mapping[str, Any]) -> dict[str, Path]:
    project = Path(__file__).resolve().parents[2]
    catalog = {str(label): (project / str(label)).resolve() for label in cfg["fit_source_files"]}
    config = Path(str(cfg["_config_path"])).resolve()
    catalog[f"config::{config.name}"] = config
    protocol = Path(str(cfg["protocol_file"])).resolve()
    catalog[f"protocol::{protocol.name}"] = protocol
    if any(not path.is_file() for path in catalog.values()):
        raise FileNotFoundError("A fit source dependency is absent")
    return catalog


def freeze_source(cfg: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        label: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for label, path in sorted(_source_catalog(cfg).items())
    }
    payload = {
        "format": SOURCE_FORMAT,
        "status": "FROZEN_PENDING_INDEPENDENT_SOURCE_REVIEW",
        "scientific_completion": False,
        "files": files,
        "jobs": ["coordinate_trees", "coordinate_tabpfn_v2_6", "full_tabpfn_v3_replay_states"],
        "seeds": list(SEEDS),
        "test_rows_read": 0,
        "test_features_read": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(str(cfg["fit_source_freeze_manifest"]))
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != "created_utc"} != {k: v for k, v in payload.items() if k != "created_utc"}:
            raise UnifiedFitError("Existing fit source freeze differs; use a new run")
        return previous
    _atomic_json(payload, path)
    return payload


def verify_source_gate(cfg: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = Path(str(cfg["fit_source_freeze_manifest"]))
    verdict_path = Path(str(cfg["fit_source_review_verdict"]))
    if not manifest_path.is_file() or not verdict_path.is_file():
        raise UnifiedFitError("Fit source freeze and fresh independent verdict are required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != SOURCE_FORMAT or set(manifest.get("files", {})) != set(_source_catalog(cfg)):
        raise UnifiedFitError("Fit source manifest format/catalog differs")
    for label, path in _source_catalog(cfg).items():
        expected = manifest["files"][label]
        if expected.get("sha256") != sha256(path) or int(expected.get("size_bytes", -1)) != path.stat().st_size:
            raise UnifiedFitError(f"Frozen fit source changed: {label}")
    verdict = json.loads(verdict_path.read_text())
    status = verdict.get("status", verdict.get("overall_verdict", verdict.get("verdict")))
    if status != "PASS" or verdict.get("review_scope") != "unified_clean_fit_source_before_fits":
        raise UnifiedFitError("Independent fit-source review did not PASS the exact scope")
    if verdict.get("reviewed_source_manifest_sha256") != sha256(manifest_path):
        raise UnifiedFitError("Fit-source verdict is not bound to the current manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "verdict_path": str(verdict_path.resolve()),
        "verdict_sha256": sha256(verdict_path),
    }


def select_f1_threshold(y_true: Any, y_score: Any) -> dict[str, Any]:
    truth_raw = pd.to_numeric(pd.Series(y_true), errors="raise").to_numpy(np.float64)
    score = pd.to_numeric(pd.Series(y_score), errors="raise").to_numpy(np.float64)
    if truth_raw.ndim != 1 or score.shape != truth_raw.shape or not len(score) or not np.isfinite(truth_raw).all() or not np.isin(truth_raw, [0.0, 1.0]).all() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any() or len(np.unique(truth_raw)) != 2:
        raise ValueError("Threshold calibration requires finite exact-binary truth and probabilities")
    truth = truth_raw.astype(np.int8)
    candidates = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)]
    rows = []
    for threshold in candidates:
        prediction = (score >= threshold).astype(np.int8)
        tp = int(((truth == 1) & (prediction == 1)).sum())
        fp = int(((truth == 0) & (prediction == 1)).sum())
        fn = int(((truth == 1) & (prediction == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append({"threshold": float(threshold), "f1": f1, "precision": precision, "recall": recall, "accepted": int(prediction.sum()), "tp": tp, "fp": fp, "fn": fn})
    best = max(rows, key=lambda row: (row["f1"], row["precision"], -row["accepted"], row["threshold"]))
    return {**best, "rows": len(truth), "positive_truth": int(truth.sum()), "predicted_positive_rate": best["accepted"] / len(truth), "candidate_count": len(rows), "decision_rule": "score_greater_than_or_equal_to_threshold", "tie_break": "max_f1_then_precision_then_fewer_positives_then_higher_threshold"}


def _binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (score >= threshold).astype(np.int8)
    return {
        "f1": f1_score(truth, prediction, zero_division=0),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "accuracy": accuracy_score(truth, prediction),
        "roc_auc": roc_auc_score(truth, score),
        "average_precision": average_precision_score(truth, score),
        "predicted_positive_rate": float(prediction.mean()),
    }


def _load_partition(validation: pd.DataFrame, paths: Sequence[str | Path], cfg: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    if len(paths) != 2:
        raise UnifiedFitError("Exactly two independently emitted partition files are required")
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    if payloads[0] != payloads[1]:
        raise UnifiedFitError("Partition payloads differ")
    payload = payloads[0]
    calibration = list(map(str, payload.get("calibration_ids", [])))
    evaluation = list(map(str, payload.get("evaluation_ids", [])))
    validation_ids = set(validation["POI_ID"].astype(str))
    if len(calibration) != int(cfg["expected_calibration_rows"]) or len(evaluation) != int(cfg["expected_evaluation_rows"]) or len(set(calibration)) != len(calibration) or len(set(evaluation)) != len(evaluation) or set(calibration) & set(evaluation) or set(calibration) | set(evaluation) != validation_ids:
        raise UnifiedFitError("Common validation partition is invalid")
    digest = sha256(Path(paths[0]))
    if digest != str(cfg["expected_partition_sha256"]) or sha256(Path(paths[1])) != digest:
        raise UnifiedFitError("Common validation partition SHA differs")
    return calibration, evaluation, {"files": [str(Path(path).resolve()) for path in paths], "sha256": digest, "calibration_rows": len(calibration), "evaluation_rows": len(evaluation), "calibration_poi_order_sha256": _ordered_ids_sha(calibration), "evaluation_poi_order_sha256": _ordered_ids_sha(evaluation)}


def _job(cfg: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == "coordinate-trees":
        return {"models": TREE_MODELS, "feature_set": "coordinates_only", "features": COORDINATE_FEATURES, "output": cfg["outputs"]["coordinate_trees"], "persist_states": False}
    if mode == "coordinate-tabpfn26":
        return {"models": ("tabpfn_v2_6",), "feature_set": "coordinates_only", "features": COORDINATE_FEATURES, "output": cfg["outputs"]["coordinate_tabpfn26"], "persist_states": False}
    if mode == "full-tabpfn3-states":
        return {"models": ("tabpfn_v3",), "feature_set": "full", "features": FULL_FEATURES, "output": cfg["outputs"]["full_tabpfn3_states"], "persist_states": True}
    raise ValueError(f"Unknown fit mode: {mode}")


def _checkpoint_cfg(cfg: Mapping[str, Any], model: str) -> dict[str, Any]:
    model_cfg = dict(cfg[model])
    override = os.environ.get("TABPFN26_CHECKPOINT" if model == "tabpfn_v2_6" else "TABPFN3_CHECKPOINT")
    if override:
        model_cfg["official_checkpoint"] = override
    return model_cfg


def _prediction_input(model: str, frame: pd.DataFrame, features: Sequence[str]) -> tuple[Any, str]:
    """Preserve the fitted family's exact inference representation."""
    if model in TREE_MODELS:
        return frame.loc[:, list(features)].copy(), "raw_validation_dataframe_original_float_precision"
    return _matrix(frame, list(features), dtype=np.float32), "exact_float32_matrix_for_tabpfn"


def _output_path(cfg: Mapping[str, Any], relative: str) -> Path:
    root = Path(os.environ.get("UNIFIED_RUN_ROOT", str(cfg["run_root"])))
    return root / relative


def run_fit(cfg: Mapping[str, Any], mode: str) -> None:
    gate = verify_source_gate(cfg)
    job = _job(cfg, mode)
    output = _output_path(cfg, str(job["output"]))
    if (output / "_GENERATED_PENDING_REVIEW.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed fit: {output}")
    for subdir in ("predictions", "work", "states", "tables"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.environ.get("DATA_DIR", str(cfg["data_dir"]))).resolve()
    inputs = [data_dir / "poi_data_region.csv", data_dir / "sample_manifest.csv", data_dir / "report.json", *[Path(path).resolve() for path in cfg["partition_files"]], Path(str(cfg["tree_config"])).resolve(), Path(gate["manifest_path"]), Path(gate["verdict_path"])]
    global ModelFactory
    if ModelFactory is None:
        from .selected_explain import ModelFactory as factory_class
    else:
        factory_class = ModelFactory
    for model in job["models"]:
        if model.startswith("tabpfn_"):
            inputs.append(Path(str(_checkpoint_cfg(cfg, model)["official_checkpoint"])).resolve())
    initial = _snapshot(inputs)
    fingerprint = _canonical_sha({"format": FIT_FORMAT, "mode": mode, "models": job["models"], "feature_set": job["feature_set"], "features": job["features"], "seeds": SEEDS, "source_gate": gate, "input_snapshot": initial})
    identity = {"format": FIT_FORMAT, "fingerprint_sha256": fingerprint, "mode": mode, "source_gate": gate, "input_snapshot": initial}
    identity_path = output / "input_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise UnifiedFitError("Partial fit identity differs")
    if not identity_path.exists():
        _atomic_json(identity, identity_path)
    clean, _ = load_clean_splits(data_dir, splits=("train", "validation"), verify_hashes=True)
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    if len(train) != int(cfg["expected_train_rows"]) or len(validation) != int(cfg["expected_validation_rows"]):
        raise UnifiedFitError("Development split cardinality differs")
    regions = tuple(sorted(train["region"].astype(str).unique()))
    if len(regions) != int(cfg["expected_regions"]) or tuple(sorted(validation["region"].astype(str).unique())) != regions:
        raise UnifiedFitError("Development region contract differs")
    calibration_ids, evaluation_ids, partition = _load_partition(validation, cfg["partition_files"], cfg)
    partition_lookup = {value: "calibration" for value in calibration_ids}
    partition_lookup.update({value: "evaluation" for value in evaluation_ids})
    calibration_set = set(calibration_ids)
    started = time.time()
    threshold_rows, calibration_score_parts = [], []
    completed = 0
    for model in job["models"]:
        for seed in SEEDS:
            for region_index, region in enumerate(regions):
                stem = f"{model}_{job['feature_set']}_{seed}_{region}"
                prediction_path = output / "predictions" / f"{stem}_clean.parquet"
                provenance_path = output / "work" / f"{stem}_fit_provenance.json"
                marker_path = output / "work" / f"{stem}.json"
                state_path = output / "states" / f"modelseed{seed}__region{region_index:02d}.tabpfn_fit"
                expected_paths = [prediction_path, provenance_path, *( [state_path] if job["persist_states"] else [] )]
                if marker_path.exists():
                    marker = json.loads(marker_path.read_text())
                    files = {str(path.relative_to(output)): sha256(path) for path in expected_paths} if all(path.is_file() for path in expected_paths) else {}
                    if marker.get("fingerprint_sha256") != fingerprint or marker.get("files_sha256") != files:
                        raise UnifiedFitError(f"Invalid partial fit shard: {stem}")
                    provenance = json.loads(provenance_path.read_text())
                    if job["persist_states"]:
                        threshold_rows.append(provenance["calibration_threshold"])
                        calibration_score_parts.append(pd.read_parquet(prediction_path).loc[lambda x: x["partition"].eq("calibration")])
                    completed += 1
                    continue
                if any(path.exists() for path in expected_paths):
                    raise UnifiedFitError(f"Incomplete unmarked fit shard: {stem}")
                model_cfg = {"seed": seed, "tree_config": cfg["tree_config"], "repeat_prediction_absolute_tolerance": float(cfg["repeat_prediction_absolute_tolerance"])}
                if model.startswith("tabpfn_"):
                    model_cfg.update(_checkpoint_cfg(cfg, model))
                truth_train = train["region"].astype(str).eq(region).astype(np.int8).to_numpy()
                factory = factory_class(model, model_cfg, train)
                fit_started = time.perf_counter()
                predict = factory.fit(list(job["features"]), truth_train)
                fit_seconds = time.perf_counter() - fit_started
                values, input_contract = _prediction_input(model, validation, job["features"])
                predict_started = time.perf_counter()
                score = np.asarray(predict(values), dtype=np.float64)
                predict_seconds = time.perf_counter() - predict_started
                if score.shape != (len(validation),) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
                    raise UnifiedFitError(f"Invalid score output: {stem}")
                if job["persist_states"]:
                    attestation = factory.persist_last_fit_state(state_path)
                else:
                    attestation = dict(factory.last_fit_attestation)
                if attestation.get("model_id") != model or attestation.get("feature_order") != list(job["features"]):
                    raise UnifiedFitError(f"Fit attestation identity differs: {stem}")
                truth = validation["region"].astype(str).eq(region).astype(np.int8).to_numpy()
                prediction = pd.DataFrame({"model": model, "feature_set": job["feature_set"], "seed": seed, "region": region, "condition": "clean", "split": "validation", "features": "|".join(job["features"]), "POI_ID": validation["POI_ID"].astype(str).to_numpy(), "partition": validation["POI_ID"].astype(str).map(partition_lookup), "y_true": truth, "y_score": score, "y_pred_fixed_0p5": (score >= 0.5).astype(np.int8)})
                if prediction["partition"].isna().any():
                    raise UnifiedFitError("A validation POI is outside the common partition")
                _atomic_frame(prediction, prediction_path)
                calibration = prediction.loc[prediction["POI_ID"].isin(calibration_set)].copy()
                selected = select_f1_threshold(calibration["y_true"], calibration["y_score"])
                threshold_row = {"model": model, "feature_set": job["feature_set"], "seed": seed, "region": region, "region_index": region_index, "threshold_source": "common_validation_calibration_ids_only", **selected}
                training_policy = "balanced_OvR_sample_weight_n_over_2class_count" if model in TREE_MODELS else "TabPFN_default_unweighted_fit"
                provenance = {"format": "poi-unified-clean-actual-fit-provenance-v1", "model": model, "feature_set": job["feature_set"], "features": list(job["features"]), "seed": seed, "region": region, "region_index": region_index, "fit_attestation": attestation, "training_weight_policy": training_policy, "prediction_input_contract": input_contract, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "train_rows": len(train), "validation_rows": len(validation), "train_poi_order_sha256": _ordered_ids_sha(train["POI_ID"].astype(str).tolist()), "validation_poi_order_sha256": _ordered_ids_sha(validation["POI_ID"].astype(str).tolist()), "partition": partition, "calibration_threshold": threshold_row, "calibration_score_float64_sha256": hashlib.sha256(np.ascontiguousarray(calibration["y_score"].to_numpy(np.float64)).tobytes()).hexdigest(), "calibration_truth_int8_sha256": hashlib.sha256(np.ascontiguousarray(calibration["y_true"].to_numpy(np.int8)).tobytes()).hexdigest(), "source_fingerprint_sha256": fingerprint, "prediction_sha256": sha256(prediction_path), "test_rows_read": 0, "test_features_read": 0}
                _atomic_json(provenance, provenance_path)
                files = {str(path.relative_to(output)): sha256(path) for path in expected_paths}
                _atomic_json({"fingerprint_sha256": fingerprint, "files_sha256": files, "fixed_0p5_metrics": _binary_metrics(truth, score, 0.5), "calibration_threshold": threshold_row}, marker_path)
                if job["persist_states"]:
                    threshold_rows.append(threshold_row)
                    calibration_score_parts.append(calibration)
                completed += 1
                print(json.dumps({"mode": mode, "completed": completed, "expected": len(job["models"]) * len(SEEDS) * len(regions), "model": model, "seed": seed, "region": region}), flush=True)
    if _snapshot(inputs) != initial or verify_source_gate(cfg) != gate:
        raise UnifiedFitError("Fit source or input changed during execution")
    expected = len(job["models"]) * len(SEEDS) * len(regions)
    predictions = sorted((output / "predictions").glob("*.parquet"))
    provenances = sorted((output / "work").glob("*_fit_provenance.json"))
    markers = sorted(path for path in (output / "work").glob("*.json") if not path.name.endswith("_fit_provenance.json"))
    states = sorted((output / "states").glob("*.tabpfn_fit"))
    if completed != expected or len(predictions) != expected or len(provenances) != expected or len(markers) != expected or len(states) != (expected if job["persist_states"] else 0):
        raise UnifiedFitError("Fit output cardinality differs")
    replay_contract = None
    if job["persist_states"]:
        thresholds = pd.DataFrame(threshold_rows).sort_values(["seed", "region_index"]).reset_index(drop=True)
        calibration_scores = pd.concat(calibration_score_parts, ignore_index=True).sort_values(["seed", "region", "POI_ID"]).reset_index(drop=True)
        if len(thresholds) != 51 or len(calibration_scores) != 51 * len(calibration_ids):
            raise UnifiedFitError("Full-v3 calibration contract cardinality differs")
        _atomic_frame(thresholds, output / "tables" / "calibration_thresholds.csv")
        _atomic_frame(calibration_scores, output / "tables" / "calibration_scores.parquet")
        fits = []
        for provenance_path in provenances:
            provenance = json.loads(provenance_path.read_text())
            attestation = provenance["fit_attestation"]
            required = ("fitted_state_path", "fitted_state_sha256", "fitted_state_size_bytes", "official_checkpoint_sha256", "train_matrix_sha256", "truth_sha256", "probe_indices", "probe_matrix_sha256", "probe_prediction_sha256", "probe_repeat_prediction_sha256", "restored_probe_prediction_sha256")
            if any(key not in attestation for key in required):
                raise UnifiedFitError(f"Persisted replay attestation incomplete: {provenance_path}")
            state = Path(attestation["fitted_state_path"])
            prediction = output / "predictions" / f"{provenance['model']}_{provenance['feature_set']}_{provenance['seed']}_{provenance['region']}_clean.parquet"
            fits.append({"model_id": provenance["model"], "model_seed": provenance["seed"], "region": provenance["region"], "region_index": provenance["region_index"], "feature_order": provenance["features"], "state_path": str(state), "state_relative_path": str(state.relative_to(output.resolve())), "state_sha256": attestation["fitted_state_sha256"], "state_size_bytes": attestation["fitted_state_size_bytes"], "state_format": attestation["fitted_state_format"], "state_api": attestation["fitted_state_api"], "official_checkpoint_path": attestation["official_checkpoint_path"], "official_checkpoint_sha256": attestation["official_checkpoint_sha256"], "train_matrix_sha256": attestation["train_matrix_sha256"], "train_truth_sha256": attestation["truth_sha256"], "train_poi_order_sha256": provenance["train_poi_order_sha256"], "probe_indices": attestation["probe_indices"], "probe_matrix_sha256": attestation["probe_matrix_sha256"], "original_probe_score_sha256": attestation["probe_prediction_sha256"], "repeat_probe_score_sha256": attestation["probe_repeat_prediction_sha256"], "restored_probe_score_sha256": attestation["restored_probe_prediction_sha256"], "prediction_path": str(prediction.resolve()), "prediction_relative_path": str(prediction.relative_to(output)), "prediction_sha256": provenance["prediction_sha256"], "calibration_threshold": provenance["calibration_threshold"]["threshold"], "calibration_score_float64_sha256": provenance["calibration_score_float64_sha256"], "calibration_truth_int8_sha256": provenance["calibration_truth_int8_sha256"]})
        replay_contract = {"format": "poi-unified-clean-tabpfn3-replay-contract-v1", "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "model_id": "tabpfn_v3", "feature_order": list(FULL_FEATURES), "seeds": list(SEEDS), "regions": list(regions), "fit_count": 51, "partition": partition, "threshold_rule": "per seed/region inclusive F1 threshold selected only on 1693 clean calibration IDs", "state_resolution_rule": "verify state_relative_path under replay_contract parent; state_path records execution location only", "state_load_rule": "TabPFNClassifier.load_from_fit_state exactly once per seed/region and reuse for all clean/attack/defense inference", "calibration_thresholds_path": str((output / "tables" / "calibration_thresholds.csv").resolve()), "calibration_thresholds_relative_path": "tables/calibration_thresholds.csv", "calibration_thresholds_sha256": sha256(output / "tables" / "calibration_thresholds.csv"), "calibration_scores_path": str((output / "tables" / "calibration_scores.parquet").resolve()), "calibration_scores_relative_path": "tables/calibration_scores.parquet", "calibration_scores_sha256": sha256(output / "tables" / "calibration_scores.parquet"), "fits": sorted(fits, key=lambda row: (row["model_seed"], row["region_index"])), "test_rows_read": 0, "test_features_read": 0}
        _atomic_json(replay_contract, output / "replay_contract.json")
    metadata = {"format": FIT_FORMAT, "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "scientific_completion": False, "mode": mode, "models": list(job["models"]), "feature_set": job["feature_set"], "features": list(job["features"]), "seeds": list(SEEDS), "regions": list(regions), "fit_count": expected, "persisted_state_count": len(states), "train_rows": len(train), "validation_rows": len(validation), "partition": partition, "source_gate": gate, "input_snapshot": initial, "training_weight_policy": {model: ("balanced_OvR_sample_weight_n_over_2class_count" if model in TREE_MODELS else "TabPFN_default_unweighted_fit") for model in job["models"]}, "prediction_sha256": {str(path): sha256(path) for path in predictions}, "provenance_sha256": {str(path): sha256(path) for path in provenances}, "marker_sha256": {str(path): sha256(path) for path in markers}, "state_sha256": {str(path): sha256(path) for path in states}, "replay_contract_sha256": sha256(output / "replay_contract.json") if replay_contract else None, "execution_completed_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.time() - started, "git": _git_state(), "python": platform.python_version(), "versions": _versions(["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", *( ["tabpfn", "torch"] if any(model.startswith("tabpfn_") for model in job["models"]) else [] )]), "test_rows_read": 0, "test_features_read": 0, "historical_test_isolation": "PARTIAL: prior whole-dataset coordinate access is known; this fit reads train/validation only", "limitations": ["Development clean fit only; no locked test data were read.", "Independent result review remains pending."]}
    _atomic_json(metadata, output / "metadata.json")
    shutil.copy2(cfg["_config_path"], output / "config.yaml")
    _atomic_json({"status": metadata["status"], "metadata_sha256": sha256(output / "metadata.json"), "fit_count": expected, "persisted_state_count": len(states), "test_rows_read": 0}, output / "_GENERATED_PENDING_REVIEW.json")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    before = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    cfg = yaml.safe_load(path.read_bytes())
    after = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    if not isinstance(cfg, dict) or before != after:
        raise UnifiedFitError("Fit config invalid or changed during load")
    cfg["_config_path"] = str(path)
    if tuple(cfg["seeds"]) != SEEDS or float(cfg["fixed_threshold"]) != 0.5 or int(cfg["expected_regions"]) != 17:
        raise UnifiedFitError("Frozen seeds/threshold/region count differs")
    return cfg


def run(config_path: str | Path, mode: str) -> None:
    cfg = load_config(config_path)
    if mode == "freeze-source":
        freeze_source(cfg)
    else:
        run_fit(cfg, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("freeze-source", "coordinate-trees", "coordinate-tabpfn26", "full-tabpfn3-states"))
    args = parser.parse_args()
    run(args.config, args.mode)


if __name__ == "__main__":
    main()
