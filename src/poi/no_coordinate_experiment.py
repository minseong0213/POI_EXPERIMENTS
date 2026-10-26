"""No-coordinate clean validation experiment with calibration-only thresholds.

The three modes are deliberately separate:

``trees`` fits the five CPU tree families, ``tabpfn26`` prepares/runs only the
missing TabPFN 2.6 shards, and ``aggregate`` combines those fresh shards with
the immutable imported TabPFN 2.5/v3 shards.  None of the modes loads test rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import yaml

from .data import CATEGORICAL, SPLITS, _validate_feature_frame, load_clean_splits, sha256
from .selected_explain import ModelFactory, _matrix
from .uncertainty import make_bootstrap_weights, weighted_metric_replicates


FEATURES = tuple(CATEGORICAL)
TREE_MODELS = (
    "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
)
LEARNED_MODELS = (*TREE_MODELS, "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3")
BASELINES = ("constant_negative", "train_joint_category_frequency")
ALL_PIPELINES = (*LEARNED_MODELS, *BASELINES)
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc", "average_precision")
SEEDS = (42, 202, 340)
AGGREGATE_SOURCE_MODULES = (
    "data.py", "uncertainty.py", "clean_ablation_subset_report.py",
)


class NoCoordinateError(RuntimeError):
    """Raised when the no-coordinate experiment contract is violated."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    return {"commit": commit, "dirty": dirty}


def _ordered_ids_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def _runtime_versions(packages: Sequence[str]) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result


def _snapshot_files(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Hash a closed set of regular files before any experiment input is read."""
    requested = [Path(path) for path in paths]
    if any(path.is_symlink() for path in requested):
        raise NoCoordinateError("Aggregate input catalog may not contain symbolic links")
    resolved = [path.resolve() for path in requested]
    if len(set(resolved)) != len(resolved):
        raise NoCoordinateError("Aggregate input catalog contains duplicate paths")
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved, key=str):
        if not path.is_file():
            raise NoCoordinateError(f"Aggregate input is absent or not a regular file: {path}")
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size_bytes += len(block)
        snapshot[str(path)] = {
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        }
    return snapshot


def _assert_snapshot_unchanged(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail closed when any cataloged input changed after the initial snapshot."""
    current = _snapshot_files([Path(path) for path in snapshot])
    if current != dict(snapshot):
        raise NoCoordinateError("An aggregate input changed after its initial byte snapshot")


def _bound_bytes(
    path: str | Path, snapshot: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Read bytes and bind that exact read to the pre-computation snapshot."""
    resolved = Path(path).resolve()
    expected = snapshot.get(str(resolved))
    if expected is None:
        raise NoCoordinateError(f"Input was not in the initial aggregate catalog: {resolved}")
    payload = resolved.read_bytes()
    observed = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if observed != dict(expected):
        raise NoCoordinateError(f"Input changed before or while it was read: {resolved}")
    return payload


def _bound_json(
    path: str | Path, snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(path, snapshot))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NoCoordinateError(f"Invalid JSON aggregate input: {path}") from exc
    if not isinstance(value, dict):
        raise NoCoordinateError(f"Aggregate JSON input must be an object: {path}")
    return value


def _snapshot_digest_map(
    paths: Sequence[str | Path], snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(Path(path).resolve()): str(snapshot[str(Path(path).resolve())]["sha256"])
        for path in paths
    }


def _load_clean_splits_bound(
    data_dir: Path,
    splits: Sequence[str],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load development rows from bytes bound to the aggregate input snapshot.

    The full manifest has no model features and is safe to parse. The clean CSV is
    streamed once for its hash, while only requested physical rows are retained for
    parsing. A physical-line count mismatch fails closed rather than risking quoted
    multiline records that could invalidate the row filter.
    """
    requested = tuple(map(str, splits))
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(SPLITS):
        raise ValueError("Bound clean loader requires unique known split names")
    manifest_path = data_dir / "sample_manifest.csv"
    clean_path = data_dir / "poi_data_region.csv"
    report_path = data_dir / "report.json"
    manifest = pd.read_csv(
        io.BytesIO(_bound_bytes(manifest_path, snapshot)),
        dtype=str, keep_default_na=False,
    )
    required = {"POI_ID", "region", "split"}
    if not required.issubset(manifest.columns):
        raise NoCoordinateError("Bound sample manifest lacks required columns")
    if (
        not manifest["POI_ID"].is_unique or manifest["POI_ID"].eq("").any()
        or manifest["region"].eq("").any() or set(manifest["split"]) != set(SPLITS)
    ):
        raise NoCoordinateError("Bound sample manifest identity/splits differ")
    regions = set(manifest["region"])
    if any(
        set(manifest.loc[manifest["split"].eq(split), "region"]) != regions
        for split in SPLITS
    ):
        raise NoCoordinateError("Bound sample manifest omits a region from a split")
    report = _bound_json(report_path, snapshot)
    for name, path in (
        ("poi_data_region.csv", clean_path), ("sample_manifest.csv", manifest_path),
    ):
        expected = report.get("outputs", {}).get(name, {}).get("sha256")
        observed = snapshot[str(path.resolve())]["sha256"]
        if expected != observed:
            raise NoCoordinateError(f"Bound clean input checksum differs: {name}")

    selected_mask = manifest["split"].isin(requested).to_numpy()
    selected_positions = set(np.flatnonzero(selected_mask).tolist())
    selected_csv = io.BytesIO()
    digest = hashlib.sha256()
    physical_rows = -1
    with clean_path.open("rb") as stream:
        for physical_row, line in enumerate(stream):
            digest.update(line)
            if physical_row == 0 or physical_row - 1 in selected_positions:
                selected_csv.write(line)
            physical_rows = physical_row
    expected_entry = snapshot[str(clean_path.resolve())]
    if (
        digest.hexdigest() != expected_entry["sha256"]
        or clean_path.stat().st_size != expected_entry["size_bytes"]
        or physical_rows != len(manifest)
    ):
        raise NoCoordinateError(
            "Clean CSV changed during bound streaming or uses unsupported multiline rows"
        )
    selected_csv.seek(0)
    clean = pd.read_csv(selected_csv, dtype=str, keep_default_na=False)
    selected_manifest = manifest.loc[selected_mask].reset_index(drop=True)
    _validate_feature_frame(clean, selected_manifest, "poi_data_region.csv")
    clean["split"] = selected_manifest["split"].to_numpy()
    return clean, selected_manifest


def load_common_partition(
    validation: pd.DataFrame,
    paths: Sequence[str | Path],
    *,
    expected_calibration: int,
    expected_evaluation: int,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Load byte/accounting evidence for one common validation partition."""
    if len(paths) < 2:
        raise NoCoordinateError("Two independently emitted matching partition files are required")
    resolved = [Path(path) for path in paths]
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError("A required validation partition file is absent")
    payloads = [
        _bound_json(path, snapshot) if snapshot is not None else json.loads(path.read_text())
        for path in resolved
    ]
    canonical = [_canonical_sha(payload) for payload in payloads]
    if len(set(canonical)) != 1:
        raise NoCoordinateError("Detector validation partition payloads differ")
    payload = payloads[0]
    calibration = list(map(str, payload.get("calibration_ids", ())))
    evaluation = list(map(str, payload.get("evaluation_ids", ())))
    if (
        payload.get("split") != "validation"
        or int(payload.get("overlap", -1)) != 0
        or len(calibration) != expected_calibration
        or len(evaluation) != expected_evaluation
        or len(set(calibration)) != len(calibration)
        or len(set(evaluation)) != len(evaluation)
        or set(calibration) & set(evaluation)
    ):
        raise NoCoordinateError("Validation calibration/evaluation partition is invalid")
    observed = set(validation["POI_ID"].astype(str))
    if set(calibration) | set(evaluation) != observed:
        raise NoCoordinateError("Partition does not cover the exact clean validation IDs")
    lookup = validation.set_index("POI_ID")["region"]
    regions = set(validation["region"].astype(str))
    if set(lookup.loc[calibration].astype(str)) != regions or set(
        lookup.loc[evaluation].astype(str)
    ) != regions:
        raise NoCoordinateError("Both partition halves must cover every region")
    return calibration, evaluation, {
        "paths": [str(path) for path in resolved],
        "sha256": {
            str(path): (
                str(snapshot[str(path.resolve())]["sha256"])
                if snapshot is not None else sha256(path)
            )
            for path in resolved
        },
        "payload_sha256": canonical[0],
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "calibration_poi_order_sha256": _ordered_ids_sha(calibration),
        "evaluation_poi_order_sha256": _ordered_ids_sha(evaluation),
        "selection_rule": "thresholds use calibration IDs only; evaluation IDs are score-only",
    }


def select_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Select an inclusive threshold on calibration rows with deterministic ties.

    The candidates are every observed score plus the next float above the maximum
    (the all-negative decision).  Ties prefer precision, fewer accepted rows, then
    the higher threshold.  A score of zero is a valid candidate and can therefore
    yield an all-positive classifier; its predicted-positive rate is recorded.
    """
    truth = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float64)
    if (
        truth.ndim != 1 or score.shape != truth.shape or len(truth) == 0
        or not np.isin(truth, [0, 1]).all() or not np.isfinite(score).all()
        or ((score < 0) | (score > 1)).any() or len(np.unique(truth)) != 2
    ):
        raise ValueError("Threshold calibration requires finite binary paired scores")
    order = np.argsort(-score, kind="mergesort")
    sorted_score, sorted_truth = score[order], truth[order]
    positives = int(truth.sum())
    candidates = [{
        "threshold": float(np.nextafter(score.max(), np.inf)),
        "f1": 0.0, "precision": 0.0, "recall": 0.0,
        "accepted": 0, "tp": 0, "fp": 0, "fn": positives,
    }]
    tp = fp = position = 0
    while position < len(sorted_score):
        end = position + 1
        while end < len(sorted_score) and sorted_score[end] == sorted_score[position]:
            end += 1
        tp += int(sorted_truth[position:end].sum())
        fp += int(end - position - sorted_truth[position:end].sum())
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        candidates.append({
            "threshold": float(sorted_score[position]), "f1": float(f1),
            "precision": float(precision), "recall": float(recall),
            "accepted": int(end), "tp": tp, "fp": fp, "fn": fn,
        })
        position = end
    best = max(
        candidates,
        key=lambda row: (row["f1"], row["precision"], -row["accepted"], row["threshold"]),
    )
    return {
        **best,
        "rows": len(truth),
        "positive_truth": positives,
        "predicted_positive_rate": best["accepted"] / len(truth),
        "candidate_count": len(candidates),
        "decision_rule": "score_greater_than_or_equal_to_threshold",
        "tie_break": "max_f1_then_precision_then_fewer_positives_then_higher_threshold",
    }


def fit_joint_frequency_baseline(
    train: pd.DataFrame, regions: Sequence[str], *, alpha: float,
) -> dict[str, Any]:
    """Fit P(region | exact category tuple) with Laplace smoothing on train only."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Laplace alpha must be finite and positive")
    features = list(FEATURES)
    if train[features + ["region"]].isna().any().any():
        raise ValueError("Joint-frequency training values must be complete")
    region_order = tuple(map(str, regions))
    if len(region_order) < 2 or len(set(region_order)) != len(region_order):
        raise ValueError("Joint-frequency baseline needs unique regions")
    tuples = list(map(tuple, train[features].astype(str).itertuples(index=False, name=None)))
    labels = train["region"].astype(str).tolist()
    counts: dict[tuple[str, ...], np.ndarray] = {}
    region_index = {region: index for index, region in enumerate(region_order)}
    prior_counts = np.zeros(len(region_order), dtype=np.int64)
    for key, region in zip(tuples, labels):
        if region not in region_index:
            raise ValueError(f"Unknown train region: {region}")
        values = counts.setdefault(key, np.zeros(len(region_order), dtype=np.int64))
        values[region_index[region]] += 1
        prior_counts[region_index[region]] += 1
    return {
        "format": "poi-train-joint-category-frequency-v1",
        "feature_order": features,
        "region_order": list(region_order),
        "alpha": float(alpha),
        "train_rows": len(train),
        "observed_tuple_count": len(counts),
        "counts": counts,
        "prior_probability": (prior_counts + alpha) / (
            prior_counts.sum() + alpha * len(region_order)
        ),
        "train_poi_order_sha256": _ordered_ids_sha(train["POI_ID"].astype(str).tolist()),
        "fit_frame_sha256": hashlib.sha256(
            train[["POI_ID", *features, "region"]].to_csv(
                index=False, lineterminator="\n",
            ).encode()
        ).hexdigest(),
    }


def predict_joint_frequency(
    fitted: Mapping[str, Any], frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    features = list(map(str, fitted["feature_order"]))
    regions = list(map(str, fitted["region_order"]))
    alpha = float(fitted["alpha"])
    counts = fitted["counts"]
    prior = np.asarray(fitted["prior_probability"], dtype=np.float64)
    output = np.empty((len(frame), len(regions)), dtype=np.float64)
    fallback = np.zeros(len(frame), dtype=bool)
    for index, key in enumerate(map(
        tuple, frame[features].astype(str).itertuples(index=False, name=None),
    )):
        observed = counts.get(key)
        if observed is None:
            output[index] = prior
            fallback[index] = True
        else:
            output[index] = (observed + alpha) / (observed.sum() + alpha * len(regions))
    if not np.allclose(output.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise NoCoordinateError("Joint-frequency probabilities do not sum to one")
    return output, fallback


def _point_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float64)
    prediction = (score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_positive_rate": float(prediction.mean()),
    }


@njit(cache=True, parallel=True)
def weighted_average_precision_replicates(
    y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Exact weighted non-interpolated AP for shared bootstrap multiplicities."""
    order = np.argsort(y_score)[::-1]
    result = np.empty(weights.shape[0], dtype=np.float64)
    for repeat in prange(weights.shape[0]):
        positive_total = 0.0
        for original in range(y_true.size):
            if y_true[original] == 1:
                positive_total += weights[repeat, original]
        if positive_total <= 0:
            result[repeat] = np.nan
            continue
        tp = 0.0
        fp = 0.0
        previous_recall = 0.0
        average_precision = 0.0
        position = 0
        while position < order.size:
            score = y_score[order[position]]
            end = position
            while end < order.size and y_score[order[end]] == score:
                original = order[end]
                if y_true[original] == 1:
                    tp += weights[repeat, original]
                else:
                    fp += weights[repeat, original]
                end += 1
            recall = tp / positive_total
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
            position = end
        result[repeat] = average_precision
    return result


def _fit_inputs(cfg: Mapping[str, Any], mode: str) -> tuple[list[Path], list[Path]]:
    config_path = Path(cfg["_config_path"])
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("selected_explain.py").resolve(),
        Path(__file__).with_name("data.py").resolve(),
        Path(__file__).with_name("ablation.py").resolve(),
        Path(__file__).with_name("benchmark.py").resolve(),
        config_path,
        Path(cfg["tree_config"]).resolve(),
    ]
    data_dir = Path(cfg["data_dir"]).resolve()
    inputs = [
        data_dir / "poi_data_region.csv", data_dir / "sample_manifest.csv",
        data_dir / "report.json", *map(lambda value: Path(value).resolve(), cfg["partition_files"]),
    ]
    if mode == "tabpfn26":
        inputs.append(Path(cfg["tabpfn_v2_6"]["official_checkpoint"]).resolve())
    if any(not path.is_file() for path in [*sources, *inputs]):
        raise FileNotFoundError("No-coordinate fit source/input file is absent")
    return sources, inputs


def _model_output(cfg: Mapping[str, Any], mode: str) -> Path:
    return Path(cfg["tree_output"] if mode == "trees" else cfg["tabpfn26_output"])


def _model_list(mode: str) -> tuple[str, ...]:
    if mode == "trees":
        return TREE_MODELS
    if mode == "tabpfn26":
        return ("tabpfn_v2_6",)
    raise ValueError(f"Unsupported fit mode: {mode}")


def _predict_fitted_model(
    model_id: str, predict: Any, validation: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    """Use the exact input representation fitted by each model family."""
    if model_id in TREE_MODELS:
        # The fitted transformer saw the original object/string category tokens.
        # Supplying this exact raw frame avoids float conversion and token reconstruction.
        values = validation.loc[:, list(FEATURES)].copy()
        return (
            np.asarray(predict(values), dtype=np.float64),
            "raw_validation_dataframe_original_category_tokens",
        )
    return (
        np.asarray(
            predict(_matrix(validation, list(FEATURES), dtype=np.float32)), dtype=np.float64,
        ),
        "exact_float32_numeric_category_matrix_for_tabpfn",
    )


def _fit_models(cfg: Mapping[str, Any], mode: str) -> None:
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = _model_output(cfg, mode)
    if (output / "_GENERATED_PENDING_REVIEW.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed model run: {output}")
    for directory in (output / "predictions", output / "work", output / "provenance"):
        directory.mkdir(parents=True, exist_ok=True)
    sources, inputs = _fit_inputs(cfg, mode)
    source_hashes = {str(path): sha256(path) for path in sources}
    input_hashes = {str(path): sha256(path) for path in inputs}
    models = _model_list(mode)
    fingerprint = _canonical_sha({
        "format": "poi-no-coordinate-fit-identity-v1", "mode": mode,
        "models": models, "features": FEATURES, "seeds": SEEDS,
        "source_sha256": source_hashes, "input_sha256": input_hashes,
    })
    identity_path = output / "input_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()).get("fingerprint_sha256") != fingerprint:
            raise NoCoordinateError("Partial output belongs to different source/config/input bytes")
    else:
        _atomic_json({
            "format": "poi-no-coordinate-fit-identity-v1",
            "fingerprint_sha256": fingerprint,
            "source_sha256": source_hashes, "input_sha256": input_hashes,
        }, identity_path)

    clean, _ = load_clean_splits(
        cfg["data_dir"], splits=("train", "validation"), verify_hashes=True,
    )
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    if len(train) != int(cfg["expected_train_rows"]) or len(validation) != int(
        cfg["expected_validation_rows"]
    ):
        raise NoCoordinateError("Clean development split cardinality differs")
    regions = tuple(sorted(train["region"].astype(str).unique()))
    if len(regions) != int(cfg["expected_regions"]):
        raise NoCoordinateError("Train region cardinality differs")
    load_common_partition(
        validation, cfg["partition_files"],
        expected_calibration=int(cfg["expected_calibration_rows"]),
        expected_evaluation=int(cfg["expected_evaluation_rows"]),
    )
    raw_category_dtypes = {feature: str(train[feature].dtype) for feature in FEATURES}
    raw_vocabularies = {
        feature: list(dict.fromkeys(train[feature].astype(str).tolist())) for feature in FEATURES
    }
    if [len(raw_vocabularies[name]) for name in FEATURES] != list(
        cfg["expected_train_category_vocabulary_sizes"]
    ):
        raise NoCoordinateError("Train category vocabulary differs")

    completed = 0
    for model_id in models:
        for seed in SEEDS:
            for region_index, region in enumerate(regions):
                stem = f"{model_id}_no_coordinates_{seed}_{region}"
                prediction_path = output / "predictions" / f"{stem}_clean.parquet"
                provenance_path = output / "work" / f"{stem}_fit_provenance.json"
                marker_path = output / "work" / f"{stem}.json"
                if marker_path.exists():
                    marker = json.loads(marker_path.read_text())
                    expected = {
                        str(prediction_path.relative_to(output)): sha256(prediction_path),
                        str(provenance_path.relative_to(output)): sha256(provenance_path),
                    } if prediction_path.is_file() and provenance_path.is_file() else {}
                    if marker.get("fingerprint_sha256") != fingerprint or marker.get(
                        "files_sha256"
                    ) != expected:
                        raise NoCoordinateError(f"Invalid partial fit shard: {marker_path}")
                    completed += 1
                    continue
                if marker_path.exists() or prediction_path.exists() or provenance_path.exists():
                    raise NoCoordinateError(f"Incomplete partial fit shard: {stem}")
                model_cfg = {
                    "seed": seed, "tree_config": cfg["tree_config"],
                    "repeat_prediction_absolute_tolerance": float(
                        cfg.get("repeat_prediction_absolute_tolerance", 1e-12)
                    ),
                }
                if model_id == "tabpfn_v2_6":
                    model_cfg.update(cfg["tabpfn_v2_6"])
                truth = train["region"].astype(str).eq(region).astype(np.int8).to_numpy()
                factory = ModelFactory(model_id, model_cfg, train)
                fit_started = time.perf_counter()
                predict = factory.fit(list(FEATURES), truth)
                fit_seconds = time.perf_counter() - fit_started
                predict_started = time.perf_counter()
                score, prediction_input = _predict_fitted_model(
                    model_id, predict, validation,
                )
                predict_seconds = time.perf_counter() - predict_started
                y_true = validation["region"].astype(str).eq(region).astype(np.int8).to_numpy()
                if score.shape != (len(validation),) or not np.isfinite(score).all() or (
                    (score < 0) | (score > 1)
                ).any():
                    raise NoCoordinateError(f"Invalid probabilities from {stem}")
                identity = {
                    "model": model_id, "ablation": "no_coordinates", "seed": seed,
                    "region": region, "attack_condition": "clean",
                    "features": "|".join(FEATURES),
                }
                predictions = pd.DataFrame({
                    **{name: value for name, value in identity.items()},
                    "POI_ID": validation["POI_ID"].astype(str).to_numpy(),
                    "y_true": y_true, "y_score": score,
                    "y_pred": (score >= float(cfg["fixed_threshold"])).astype(np.int8),
                })
                _atomic_frame(predictions, prediction_path)
                attestation = dict(factory.last_fit_attestation)
                if attestation.get("model_id") != model_id or attestation.get(
                    "feature_order"
                ) != list(FEATURES):
                    raise NoCoordinateError(f"Actual fit attestation differs for {stem}")
                provenance = {
                    "format": "poi-no-coordinate-actual-fit-provenance-v1",
                    **identity, "region_index": region_index,
                    "fit_attestation": attestation,
                    "prediction_input_contract": prediction_input,
                    "raw_train_category_dtypes": raw_category_dtypes,
                    "raw_train_category_vocabulary": raw_vocabularies,
                    "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "prediction_sha256": sha256(prediction_path),
                    "train_rows": len(train), "validation_rows": len(validation),
                    "train_poi_order_sha256": _ordered_ids_sha(
                        train["POI_ID"].astype(str).tolist()
                    ),
                    "validation_poi_order_sha256": _ordered_ids_sha(
                        validation["POI_ID"].astype(str).tolist()
                    ),
                    "source_fingerprint_sha256": fingerprint,
                }
                _atomic_json(provenance, provenance_path)
                files = {
                    str(prediction_path.relative_to(output)): sha256(prediction_path),
                    str(provenance_path.relative_to(output)): sha256(provenance_path),
                }
                _atomic_json({
                    "fingerprint_sha256": fingerprint, "files_sha256": files,
                    "fixed_0p5_metrics": _point_metrics(
                        y_true, score, float(cfg["fixed_threshold"])
                    ),
                }, marker_path)
                completed += 1
                print(json.dumps({
                    "mode": mode, "completed": completed,
                    "expected": len(models) * len(SEEDS) * len(regions),
                    "model": model_id, "seed": seed, "region": region,
                }), flush=True)

    if {str(path): sha256(path) for path in sources} != source_hashes:
        raise NoCoordinateError("Fit source/config changed during execution")
    if {str(path): sha256(path) for path in inputs} != input_hashes:
        raise NoCoordinateError("Fit data/partition/checkpoint changed during execution")
    expected_fits = len(models) * len(SEEDS) * len(regions)
    prediction_paths = sorted((output / "predictions").glob("*.parquet"))
    provenance_paths = sorted((output / "work").glob("*_fit_provenance.json"))
    if completed != expected_fits or len(prediction_paths) != expected_fits or len(
        provenance_paths
    ) != expected_fits:
        raise NoCoordinateError("No-coordinate fit output is incomplete")
    metadata = {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scientific_completion": False,
        "format": "poi-no-coordinate-model-shards-v1",
        "mode": mode, "models": list(models), "features": list(FEATURES),
        "seeds": list(SEEDS), "regions": list(regions), "fit_count": expected_fits,
        "train_rows": len(train), "validation_rows": len(validation),
        "test_rows_read": 0, "test_features_read": 0,
        "execution_started_utc": started_utc,
        "execution_completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.time() - started,
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "prediction_sha256": {str(path): sha256(path) for path in prediction_paths},
        "provenance_sha256": {str(path): sha256(path) for path in provenance_paths},
        "git": _git_state(), "python": platform.python_version(),
        "versions": _runtime_versions([
            "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost",
            *( ["tabpfn", "torch"] if mode == "tabpfn26" else [] ),
        ]),
        "historical_test_access": (
            "PARTIAL: prior unrelated processing materialized all 34,000 coordinates; "
            "this run loads train/validation only and cannot restore pristine test isolation"
        ),
        "limitations": [
            "Validation-only no-coordinate experiment; it is not locked-test evidence.",
            "Independent review is pending.",
        ],
    }
    _atomic_json(metadata, output / "metadata.json")
    shutil.copy2(cfg["_config_path"], output / "config.yaml")
    _atomic_json({
        "status": metadata["status"], "metadata_sha256": sha256(output / "metadata.json"),
        "fit_count": expected_fits, "test_rows_read": 0,
    }, output / "_GENERATED_PENDING_REVIEW.json")


def _expected_stems(
    expected_models: Sequence[str], regions: Sequence[str],
) -> list[str]:
    return [
        f"{model}_no_coordinates_{seed}_{region}"
        for model in expected_models for seed in SEEDS for region in regions
    ]


def _fresh_artifact_paths(
    root: Path, expected_models: Sequence[str], regions: Sequence[str],
    source_snapshot_root: Path,
) -> list[Path]:
    stems = _expected_stems(expected_models, regions)
    paths = [
        root / "input_identity.json", root / "metadata.json", root / "config.yaml",
        root / "_GENERATED_PENDING_REVIEW.json",
    ]
    for stem in stems:
        paths.extend([
            root / "predictions" / f"{stem}_clean.parquet",
            root / "work" / f"{stem}.json",
            root / "work" / f"{stem}_fit_provenance.json",
        ])
    if not source_snapshot_root.is_dir():
        raise NoCoordinateError(
            f"Immutable completed-fit source snapshot is absent: {source_snapshot_root}"
        )
    source_paths = sorted(path for path in source_snapshot_root.rglob("*") if path.is_file())
    if not source_paths:
        raise NoCoordinateError("Immutable completed-fit source snapshot is empty")
    return [*paths, *source_paths]


def _imported_artifact_paths(
    root: Path, expected_models: Sequence[str], regions: Sequence[str], source_archive: Path,
) -> list[Path]:
    paths = [root / "input_identity.json", root / "metadata.json", source_archive]
    for stem in _expected_stems(expected_models, regions):
        paths.extend([
            root / "predictions" / f"{stem}_clean.parquet",
            root / "work" / f"{stem}.json",
        ])
    return paths


def _hashes_by_basename(values: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, digest in values.items():
        name = Path(path).name
        if name in result and result[name] != digest:
            raise NoCoordinateError(f"Conflicting hashes for repeated basename: {name}")
        result[name] = str(digest)
    return result


def _validate_source_snapshot(
    root: Path,
    source_hashes: Mapping[str, str],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = root / "verified_after_run_source_manifest.json"
    manifest = _bound_json(manifest_path, snapshot)
    catalog = manifest.get("files")
    if not isinstance(catalog, dict) or not catalog:
        raise NoCoordinateError("Completed-fit source manifest lacks its file catalog")
    actual_paths = {
        str(path.relative_to(root)): path
        for path in root.rglob("*") if path.is_file() and path != manifest_path
    }
    if set(actual_paths) != set(catalog):
        raise NoCoordinateError("Completed-fit source snapshot file set differs from manifest")
    for relative, expected_sha in catalog.items():
        path = actual_paths[relative]
        if snapshot[str(path.resolve())]["sha256"] != expected_sha:
            raise NoCoordinateError(f"Completed-fit source snapshot hash differs: {relative}")
        matches = [
            digest for original, digest in source_hashes.items()
            if Path(original).as_posix().endswith(relative)
        ]
        if matches != [expected_sha]:
            raise NoCoordinateError(
                f"Completed-fit identity is not bound to archived source: {relative}"
            )
    if len(catalog) != len(source_hashes):
        raise NoCoordinateError("Completed-fit source identity/catalog cardinality differs")
    return {
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": snapshot[str(manifest_path.resolve())]["sha256"],
        "source_files": len(catalog),
    }


def _validate_fresh_artifact(
    root: Path,
    expected_models: Sequence[str],
    validation: pd.DataFrame,
    regions: Sequence[str],
    data_dir: Path,
    source_snapshot_root: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
    *,
    expected_mode: str,
    expected_train_rows: int,
    expected_partition_sha256: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    identity = _bound_json(root / "input_identity.json", snapshot)
    metadata = _bound_json(root / "metadata.json", snapshot)
    completion = _bound_json(root / "_GENERATED_PENDING_REVIEW.json", snapshot)
    fingerprint = identity.get("fingerprint_sha256")
    source_hashes = identity.get("source_sha256")
    input_hashes = identity.get("input_sha256")
    if (
        identity.get("format") != "poi-no-coordinate-fit-identity-v1"
        or not isinstance(fingerprint, str) or len(fingerprint) != 64
        or not isinstance(source_hashes, dict) or not isinstance(input_hashes, dict)
    ):
        raise NoCoordinateError(f"Fresh fit identity is invalid: {root}")
    recomputed = _canonical_sha({
        "format": "poi-no-coordinate-fit-identity-v1", "mode": expected_mode,
        "models": tuple(expected_models), "features": FEATURES, "seeds": SEEDS,
        "source_sha256": source_hashes, "input_sha256": input_hashes,
    })
    expected_fits = len(expected_models) * len(SEEDS) * len(regions)
    if recomputed != fingerprint:
        raise NoCoordinateError(f"Fresh fit fingerprint cannot be reproduced: {root}")
    if (
        metadata.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or metadata.get("scientific_completion") is not False
        or metadata.get("format") != "poi-no-coordinate-model-shards-v1"
        or metadata.get("mode") != expected_mode
        or metadata.get("models") != list(expected_models)
        or metadata.get("features") != list(FEATURES)
        or metadata.get("seeds") != list(SEEDS)
        or metadata.get("regions") != list(regions)
        or metadata.get("fit_count") != expected_fits
        or metadata.get("train_rows") != expected_train_rows
        or metadata.get("validation_rows") != len(validation)
        or metadata.get("test_rows_read") != 0
        or metadata.get("test_features_read") != 0
        or metadata.get("source_sha256") != source_hashes
        or metadata.get("input_sha256") != input_hashes
    ):
        raise NoCoordinateError(f"Fresh fit metadata contract differs: {root}")
    metadata_sha = snapshot[str((root / "metadata.json").resolve())]["sha256"]
    if (
        completion.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or completion.get("fit_count") != expected_fits
        or completion.get("test_rows_read") != 0
        or completion.get("metadata_sha256") != metadata_sha
    ):
        raise NoCoordinateError(f"Fresh fit completion marker differs: {root}")
    source_binding = _validate_source_snapshot(
        source_snapshot_root, source_hashes, snapshot,
    )
    copied_config_sha = snapshot[str((root / "config.yaml").resolve())]["sha256"]
    if copied_config_sha not in source_hashes.values():
        raise NoCoordinateError("Fresh fit copied config is not in its source identity")

    actual_inputs = {
        name: str(snapshot[str((data_dir / name).resolve())]["sha256"])
        for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
    }
    identity_inputs = _hashes_by_basename(input_hashes)
    if any(identity_inputs.get(name) != digest for name, digest in actual_inputs.items()):
        raise NoCoordinateError("Fresh fit clean data hashes differ from aggregation inputs")
    partition_digests = [
        digest for original, digest in input_hashes.items()
        if Path(original).name == "calibration_ids.json"
    ]
    if (
        len(partition_digests) != 2 or len(set(partition_digests)) != 1
        or partition_digests[0] != expected_partition_sha256
    ):
        raise NoCoordinateError("Fresh fit partition identity is incomplete or inconsistent")
    checkpoint_entries = [
        digest for original, digest in input_hashes.items() if Path(original).suffix == ".ckpt"
    ]
    if expected_checkpoint_sha256 is None:
        if checkpoint_entries:
            raise NoCoordinateError("Tree fit unexpectedly binds a TabPFN checkpoint")
    elif checkpoint_entries != [expected_checkpoint_sha256]:
        raise NoCoordinateError("TabPFN 2.6 fit checkpoint identity differs")

    expected_prediction_hashes = _hashes_by_basename(metadata.get("prediction_sha256", {}))
    expected_provenance_hashes = _hashes_by_basename(metadata.get("provenance_sha256", {}))
    if len(expected_prediction_hashes) != expected_fits or len(
        expected_provenance_hashes
    ) != expected_fits:
        raise NoCoordinateError("Fresh fit metadata shard catalogs are incomplete")
    validation_order_sha = _ordered_ids_sha(validation["POI_ID"].astype(str).tolist())
    for stem in _expected_stems(expected_models, regions):
        prediction_path = root / "predictions" / f"{stem}_clean.parquet"
        provenance_path = root / "work" / f"{stem}_fit_provenance.json"
        marker_path = root / "work" / f"{stem}.json"
        prediction_sha = snapshot[str(prediction_path.resolve())]["sha256"]
        provenance_sha = snapshot[str(provenance_path.resolve())]["sha256"]
        marker = _bound_json(marker_path, snapshot)
        provenance = _bound_json(provenance_path, snapshot)
        model, remainder = stem.split("_no_coordinates_", 1)
        seed_text, region = remainder.split("_", 1)
        if (
            marker.get("fingerprint_sha256") != fingerprint
            or marker.get("files_sha256") != {
                str(prediction_path.relative_to(root)): prediction_sha,
                str(provenance_path.relative_to(root)): provenance_sha,
            }
            or expected_prediction_hashes.get(prediction_path.name) != prediction_sha
            or expected_provenance_hashes.get(provenance_path.name) != provenance_sha
            or provenance.get("format") != "poi-no-coordinate-actual-fit-provenance-v1"
            or provenance.get("model") != model
            or provenance.get("seed") != int(seed_text)
            or provenance.get("region") != region
            or provenance.get("ablation") != "no_coordinates"
            or provenance.get("attack_condition") != "clean"
            or provenance.get("features") != "|".join(FEATURES)
            or provenance.get("prediction_sha256") != prediction_sha
            or provenance.get("source_fingerprint_sha256") != fingerprint
            or provenance.get("train_rows") != expected_train_rows
            or provenance.get("validation_rows") != len(validation)
            or provenance.get("validation_poi_order_sha256") != validation_order_sha
        ):
            raise NoCoordinateError(f"Fresh fit shard provenance differs: {stem}")
        attestation = provenance.get("fit_attestation")
        if (
            not isinstance(attestation, dict)
            or attestation.get("model_id") != model
            or attestation.get("seed") != int(seed_text)
            or attestation.get("feature_order") != list(FEATURES)
            or attestation.get("train_rows") != expected_train_rows
            or not isinstance(attestation.get("truth_sha256"), str)
            or len(attestation["truth_sha256"]) != 64
        ):
            raise NoCoordinateError(f"Fresh fit attestation differs: {stem}")
        if model in TREE_MODELS and (
            attestation.get("format") != "poi-tree-fit-attestation-v1"
            or not attestation.get("sample_weight_sha256")
            or not attestation.get("transformer_fitted_state_sha256")
            or not attestation.get("estimator_fitted_state_sha256")
        ):
            raise NoCoordinateError(f"Tree fitted-state binding differs: {stem}")
        if model == "tabpfn_v2_6" and (
            attestation.get("format") != "poi-tabpfn25-fit-replay-v1"
            or attestation.get("official_checkpoint_sha256") != expected_checkpoint_sha256
            or attestation.get("n_estimators") != 8
        ):
            raise NoCoordinateError(f"TabPFN 2.6 fitted-state binding differs: {stem}")
    return {
        "root": str(root.resolve()), "mode": expected_mode,
        "fingerprint_sha256": fingerprint, "fit_count": expected_fits,
        "metadata_sha256": metadata_sha, "source_snapshot": source_binding,
        "training_weight_policy": (
            "balanced_binary_sample_weight_n_over_2_class_count"
            if expected_mode == "trees" else "TabPFN_default_no_tree_sample_weight"
        ),
    }


def _validate_imported_artifact(
    root: Path,
    expected_models: Sequence[str],
    validation: pd.DataFrame,
    regions: Sequence[str],
    data_dir: Path,
    source_archive: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from .clean_ablation_subset_report import marker_fingerprint

    identity = _bound_json(root / "input_identity.json", snapshot)
    metadata = _bound_json(root / "metadata.json", snapshot)
    fingerprint = identity.get("sha256")
    if (
        not isinstance(fingerprint, str) or len(fingerprint) != 64
        or metadata.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or metadata.get("test_used") is not False
        or metadata.get("attack_sha256") is not None
        or metadata.get("config", {}).get("model_ids") != list(expected_models)
        or metadata.get("config", {}).get("seeds") != list(SEEDS)
        or metadata.get("config", {}).get("feature_sets", {}).get("no_coordinates")
        != list(FEATURES)
    ):
        raise NoCoordinateError("Imported R2 identity/metadata contract differs")
    archive_payload = _bound_bytes(source_archive, snapshot)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload)) as archive:
            feature_source = archive.extractfile("src/poi/feature_ablation.py")
            factory_source = archive.extractfile("src/poi/selected_explain.py")
            if feature_source is None or factory_source is None:
                raise NoCoordinateError("Imported executed-source archive lacks required code")
            code_sha = hashlib.sha256(feature_source.read()).hexdigest()
            factory_sha = hashlib.sha256(factory_source.read()).hexdigest()
    except (tarfile.TarError, KeyError) as exc:
        raise NoCoordinateError("Imported executed-source archive is invalid") from exc
    clean_sha = str(snapshot[str((data_dir / "poi_data_region.csv").resolve())]["sha256"])
    manifest_sha = str(snapshot[str((data_dir / "sample_manifest.csv").resolve())]["sha256"])
    if metadata.get("clean_sha256") != clean_sha:
        raise NoCoordinateError("Imported R2 clean input hash differs")
    reproduced_fingerprint = marker_fingerprint(
        metadata["config"], clean_sha, manifest_sha, code_sha, factory_sha,
    )
    if reproduced_fingerprint != fingerprint:
        raise NoCoordinateError(
            "Imported R2 identity is not linked to archived executed source and clean inputs"
        )
    for stem in _expected_stems(expected_models, regions):
        prediction_path = root / "predictions" / f"{stem}_clean.parquet"
        marker = _bound_json(root / "work" / f"{stem}.json", snapshot)
        prediction_sha = snapshot[str(prediction_path.resolve())]["sha256"]
        model, remainder = stem.split("_no_coordinates_", 1)
        seed_text, region = remainder.split("_", 1)
        metrics = marker.get("metrics")
        if (
            marker.get("fingerprint") != fingerprint
            or marker.get("files") != {
                str(prediction_path.relative_to(root)): prediction_sha,
            }
            or not isinstance(metrics, list) or len(metrics) != 1
            or metrics[0].get("model") != model
            or metrics[0].get("seed") != int(seed_text)
            or metrics[0].get("region") != region
            or metrics[0].get("ablation") != "no_coordinates"
            or metrics[0].get("attack_condition") != "clean"
            or metrics[0].get("features") != "|".join(FEATURES)
        ):
            raise NoCoordinateError(f"Imported R2 shard marker differs: {stem}")
    return {
        "root": str(root.resolve()), "status": "PARTIAL",
        "fingerprint_sha256": fingerprint,
        "fit_count": len(expected_models) * len(SEEDS) * len(regions),
        "metadata_sha256": snapshot[str((root / "metadata.json").resolve())]["sha256"],
        "executed_source_archive": str(source_archive.resolve()),
        "executed_source_archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "per_fit_attestation": "unavailable_by_original_run_design_not_reconstructed",
        "training_weight_policy": "TabPFN_default_no_tree_sample_weight",
    }


def _read_prediction_shards(
    directory: Path,
    expected_models: Sequence[str],
    validation: pd.DataFrame,
    regions: Sequence[str],
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(directory.glob("*_no_coordinates_*_clean.parquet"))
    expected_count = len(expected_models) * len(SEEDS) * len(regions)
    if len(paths) != expected_count:
        raise NoCoordinateError(
            f"Prediction shard count differs at {directory}: {len(paths)} != {expected_count}"
        )
    expected_ids = set(validation["POI_ID"].astype(str))
    lookup = validation.set_index("POI_ID")["region"].astype(str)
    parts = []
    for path in paths:
        frame = (
            pd.read_parquet(io.BytesIO(_bound_bytes(path, snapshot)))
            if snapshot is not None else pd.read_parquet(path)
        )
        required = {
            "model", "ablation", "seed", "region", "attack_condition", "features",
            "POI_ID", "y_true", "y_score", "y_pred",
        }
        if not required.issubset(frame):
            raise NoCoordinateError(f"Prediction shard lacks columns: {path}")
        values = {name: frame[name].drop_duplicates().tolist() for name in (
            "model", "ablation", "seed", "region", "attack_condition", "features",
        )}
        if (
            len(frame) != len(validation)
            or values["model"][0] not in expected_models or len(values["model"]) != 1
            or values["ablation"] != ["no_coordinates"]
            or values["seed"][0] not in SEEDS or len(values["seed"]) != 1
            or values["region"][0] not in regions or len(values["region"]) != 1
            or values["attack_condition"] != ["clean"]
            or values["features"] != ["|".join(FEATURES)]
            or set(frame["POI_ID"].astype(str)) != expected_ids
            or frame["POI_ID"].astype(str).duplicated().any()
        ):
            raise NoCoordinateError(f"Prediction shard identity differs: {path}")
        frame = frame.copy()
        frame["POI_ID"] = frame["POI_ID"].astype(str)
        truth = frame["POI_ID"].map(lookup).eq(str(values["region"][0])).astype(np.int8)
        raw_truth = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(np.float64)
        raw_prediction = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy(np.float64)
        score = pd.to_numeric(frame["y_score"], errors="raise").to_numpy(np.float64)
        if (
            not np.isfinite(raw_truth).all() or not np.isin(raw_truth, [0.0, 1.0]).all()
            or not np.array_equal(raw_truth, truth.to_numpy(np.float64))
            or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any()
            or not np.isfinite(raw_prediction).all()
            or not np.isin(raw_prediction, [0.0, 1.0]).all()
            or not np.array_equal(raw_prediction, (score >= 0.5).astype(np.float64))
        ):
            raise NoCoordinateError(f"Prediction values/truth differ: {path}")
        frame["y_true"] = raw_truth.astype(np.int8)
        frame["y_score"] = score
        parts.append(frame[["POI_ID", "model", "seed", "region", "y_true", "y_score"]])
    result = pd.concat(parts, ignore_index=True)
    expected_groups = {
        (model, seed, region) for model in expected_models for seed in SEEDS for region in regions
    }
    actual_groups = set(result[["model", "seed", "region"]].itertuples(index=False, name=None))
    if actual_groups != expected_groups or result.duplicated(
        ["model", "seed", "region", "POI_ID"]
    ).any():
        raise NoCoordinateError(f"Prediction Cartesian product differs: {directory}")
    return result, paths


def _baseline_predictions(
    train: pd.DataFrame, validation: pd.DataFrame, regions: Sequence[str], *, alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fitted = fit_joint_frequency_baseline(train, regions, alpha=alpha)
    frequency, fallback = predict_joint_frequency(fitted, validation)
    parts = []
    for seed in SEEDS:
        for region_index, region in enumerate(regions):
            truth = validation["region"].astype(str).eq(region).astype(np.int8).to_numpy()
            for model, score in (
                ("constant_negative", np.zeros(len(validation), dtype=np.float64)),
                ("train_joint_category_frequency", frequency[:, region_index]),
            ):
                parts.append(pd.DataFrame({
                    "POI_ID": validation["POI_ID"].astype(str).to_numpy(),
                    "model": model, "seed": seed, "region": region,
                    "y_true": truth, "y_score": score,
                }))
    provenance = {
        key: value for key, value in fitted.items()
        if key not in {"counts", "prior_probability"}
    }
    provenance.update({
        "smoothing": "Laplace over all 17 region outcomes within each exact train tuple",
        "unseen_tuple_fallback": "Laplace-smoothed train-only regional prior",
        "validation_fallback_rows": int(fallback.sum()),
        "validation_fallback_poi_ids_sha256": _ordered_ids_sha(
            validation.loc[fallback, "POI_ID"].astype(str).tolist()
        ),
        "constant_negative_score": 0.0,
        "constant_negative_decision": "always negative in both reporting regimes",
        "seed_replication": "deterministic baselines repeated over seeds for paired aggregation",
    })
    return pd.concat(parts, ignore_index=True), provenance


def _bootstrap_tables(
    evaluation: pd.DataFrame,
    *,
    evaluation_ids: Sequence[str],
    repeats: int,
    seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    if repeats < 2000:
        raise ValueError("At least 2,000 paired POI bootstrap repeats are required")
    ids = np.asarray(list(map(str, evaluation_ids)), dtype=object)
    weights = make_bootstrap_weights(len(ids), repeats, seed)
    aggregate: dict[tuple[str, str], np.ndarray] = {}
    point_rows = []
    for (model, model_seed, region), group in evaluation.groupby(
        ["model", "seed", "region"], sort=True,
    ):
        indexed = group.set_index("POI_ID", verify_integrity=True).loc[ids]
        truth = indexed["y_true"].to_numpy(np.int8)
        score = indexed["y_score"].to_numpy(np.float64)
        prediction = indexed["y_pred_calibrated"].to_numpy(np.int8)
        values = weighted_metric_replicates(truth, score, prediction, weights)
        values["average_precision"] = weighted_average_precision_replicates(
            truth, score, weights,
        )
        point = _point_metrics(truth, score, float(indexed["threshold"].iloc[0]))
        point_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            **{metric: point[metric] for metric in METRICS},
        })
        for metric in METRICS:
            aggregate.setdefault((str(model), metric), np.zeros(repeats, dtype=np.float64))
            aggregate[(str(model), metric)] += values[metric]
    expected_strata = len(SEEDS) * evaluation["region"].nunique()
    replicate_rows = []
    for (model, metric), values in aggregate.items():
        values = values / expected_strata
        if np.isinf(values).any() or not np.isfinite(values).any():
            raise NoCoordinateError(f"Bootstrap has no finite values for {model}/{metric}")
        replicate_rows.extend({
            "model": model, "metric": metric, "replicate": index,
            "value": float(value),
        } for index, value in enumerate(values))
    replicates = pd.DataFrame(replicate_rows)
    points = pd.DataFrame(point_rows)
    alpha = (1 - confidence_level) / 2
    macro_point = points.groupby("model", sort=False)[list(METRICS)].mean()
    summary_rows = []
    for (model, metric), group in replicates.groupby(["model", "metric"], sort=True):
        values = group.sort_values("replicate")["value"].to_numpy()
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise NoCoordinateError(f"Bootstrap interval has no finite values for {model}/{metric}")
        summary_rows.append({
            "model": model, "metric": metric,
            "estimate": float(macro_point.loc[model, metric]),
            "bootstrap_mean": float(finite.mean()),
            "ci_low": float(np.quantile(finite, alpha)),
            "ci_high": float(np.quantile(finite, 1 - alpha)),
            "confidence_level": confidence_level, "bootstrap_repeats": repeats,
            "finite_repeats": len(finite), "undefined_repeats": repeats - len(finite),
            "bootstrap_unit": "POI_ID",
            "aggregation": "equal_mean_over_3_model_seeds_and_17_regions",
            "interval_method": "paired_POI_percentile_conditional_on_fitted_models_and_thresholds",
        })
    summary = pd.DataFrame(summary_rows)
    indexed = replicates.set_index(["metric", "model", "replicate"])["value"]
    pair_rows = []
    for metric in METRICS:
        for model_a, model_b in combinations(ALL_PIPELINES, 2):
            a = indexed.loc[metric, model_a].sort_index().to_numpy()
            b = indexed.loc[metric, model_b].sort_index().to_numpy()
            finite = np.isfinite(a) & np.isfinite(b)
            delta = a[finite] - b[finite]
            if not len(delta):
                raise NoCoordinateError(
                    f"Pairwise bootstrap has no finite draws: {metric}/{model_a}/{model_b}"
                )
            pair_rows.append({
                "model_a": model_a, "model_b": model_b, "metric": metric,
                "estimate_difference_a_minus_b": float(
                    macro_point.loc[model_a, metric] - macro_point.loc[model_b, metric]
                ),
                "bootstrap_mean_difference": float(delta.mean()),
                "ci_low": float(np.quantile(delta, alpha)),
                "ci_high": float(np.quantile(delta, 1 - alpha)),
                "probability_difference_gt_zero": float(np.mean(delta > 0)),
                "confidence_level": confidence_level, "bootstrap_repeats": repeats,
                "finite_paired_repeats": len(delta),
                "undefined_paired_repeats": repeats - len(delta),
                "pairing": "shared evaluation POI multiplicities for all pipelines",
                "p_value": np.nan,
                "p_value_interpretation": "not_computed_bootstrap_interval_is_not_null_calibrated",
            })
    return points, replicates, summary, pd.DataFrame(pair_rows), weights


def _save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plots(
    metrics: pd.DataFrame,
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    model_order = list(ALL_PIPELINES)
    for metric in METRICS:
        heat = metrics.pivot_table(
            index="region", columns="model", values=metric, aggfunc="mean",
        ).reindex(columns=model_order)
        fig, axis = plt.subplots(figsize=(15, 9))
        sns.heatmap(heat, vmin=0, vmax=1, cmap="viridis", ax=axis)
        axis.set_title(f"No-coordinate evaluation | calibrated {metric} | seed mean")
        _save_figure(fig, figures / f"regional_{metric}_heatmap.png")

        interval = summary.loc[summary["metric"].eq(metric)].set_index("model").loc[
            model_order
        ].reset_index()
        fig, axis = plt.subplots(figsize=(11, 6))
        axis.hlines(interval["model"], interval["ci_low"], interval["ci_high"])
        axis.scatter(interval["estimate"], interval["model"])
        axis.set(
            xlabel=f"Macro {metric}", ylabel="Pipeline",
            title="Evaluation-half paired POI bootstrap 95% CI",
            xlim=(0, 1),
        )
        axis.grid(axis="x", alpha=.25)
        _save_figure(fig, figures / f"macro_{metric}_bootstrap_ci.png")

    visual_seed = min(SEEDS)
    for region in sorted(evaluation["region"].astype(str).unique()):
        subset = evaluation.loc[
            evaluation["region"].eq(region) & evaluation["seed"].eq(visual_seed)
        ]
        for curve in ("roc", "pr"):
            fig, axis = plt.subplots(figsize=(10, 7))
            for model in model_order:
                part = subset.loc[subset["model"].eq(model)]
                if curve == "roc":
                    x, y, _ = roc_curve(part["y_true"], part["y_score"])
                    value = roc_auc_score(part["y_true"], part["y_score"])
                    axis.set(xlabel="False positive rate", ylabel="True positive rate")
                else:
                    y, x, _ = precision_recall_curve(part["y_true"], part["y_score"])
                    value = average_precision_score(part["y_true"], part["y_score"])
                    axis.set(xlabel="Recall", ylabel="Precision")
                axis.plot(x, y, label=f"{model} {value:.3f}")
            if curve == "roc":
                axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=.7)
            axis.set_title(f"{region} no-coordinate {curve.upper()} | evaluation | seed {visual_seed}")
            axis.legend(fontsize=7)
            _save_figure(fig, figures / f"{curve}_{region.lower()}_seed{visual_seed}.png")


def _aggregate(cfg: Mapping[str, Any]) -> None:
    started = time.time()
    output = Path(cfg["aggregate_output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate output: {output}")
    tables, figures, work, provenance = (
        output / "tables", output / "figures", output / "work", output / "provenance"
    )
    for directory in (tables, figures, work, provenance):
        directory.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        *(Path(__file__).with_name(name).resolve() for name in AGGREGATE_SOURCE_MODULES),
        Path(cfg["_config_path"]), Path(cfg["executed_ablation_source_archive"]).resolve(),
    ]
    data_dir = Path(cfg["data_dir"]).resolve()
    clean_input_paths = [
        data_dir / "poi_data_region.csv", data_dir / "sample_manifest.csv",
        data_dir / "report.json",
    ]
    partition_paths = [Path(path).resolve() for path in cfg["partition_files"]]
    tree_source_snapshot = Path(cfg["tree_fit_source_snapshot"]).resolve()
    tabpfn26_source_snapshot = Path(cfg["tabpfn26_fit_source_snapshot"]).resolve()
    static_paths = [*source_paths, *clean_input_paths, *partition_paths]
    static_paths.extend(
        path for root in (tree_source_snapshot, tabpfn26_source_snapshot)
        for path in root.rglob("*") if path.is_file()
    )
    initial_snapshot = _snapshot_files(static_paths)
    if initial_snapshot[str(Path(cfg["_config_path"]).resolve())] != cfg.get(
        "_config_load_snapshot"
    ):
        raise NoCoordinateError("Aggregate config changed after its bound initial parse")
    _atomic_json({
        "format": "poi-no-coordinate-aggregate-input-snapshot-v1",
        "phase": "before_clean_or_prediction_read",
        "files": initial_snapshot,
    }, provenance / "input_snapshot_before_read.json")
    source_hashes = _snapshot_digest_map(source_paths, initial_snapshot)
    clean, _ = _load_clean_splits_bound(
        data_dir, ("train", "validation"), initial_snapshot,
    )
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    if (
        len(train) != int(cfg["expected_train_rows"])
        or len(validation) != int(cfg["expected_validation_rows"])
        or train["POI_ID"].astype(str).duplicated().any()
        or validation["POI_ID"].astype(str).duplicated().any()
        or set(train["POI_ID"].astype(str)) & set(validation["POI_ID"].astype(str))
    ):
        raise NoCoordinateError("Clean train/validation cardinality or POI identity differs")
    regions = tuple(sorted(train["region"].astype(str).unique()))
    if (
        len(regions) != int(cfg["expected_regions"])
        or tuple(sorted(validation["region"].astype(str).unique())) != regions
    ):
        raise NoCoordinateError("Clean regional cardinality differs")
    calibration_ids, evaluation_ids, partition = load_common_partition(
        validation, cfg["partition_files"],
        expected_calibration=int(cfg["expected_calibration_rows"]),
        expected_evaluation=int(cfg["expected_evaluation_rows"]),
        snapshot=initial_snapshot,
    )
    partition_sha = initial_snapshot[str(partition_paths[0])]["sha256"]
    if any(initial_snapshot[str(path)]["sha256"] != partition_sha for path in partition_paths):
        raise NoCoordinateError("Common partition files differ at the byte level")
    tree_root = Path(cfg["tree_output"]).resolve()
    imported_root = Path(cfg["imported_tabpfn_output"]).resolve()
    tabpfn26_root = Path(cfg["tabpfn26_output"]).resolve()
    archive_path = Path(cfg["executed_ablation_source_archive"]).resolve()
    artifact_paths = [
        *_fresh_artifact_paths(tree_root, TREE_MODELS, regions, tree_source_snapshot),
        *_imported_artifact_paths(
            imported_root, ("tabpfn_v2_5", "tabpfn_v3"), regions, archive_path,
        ),
        *_fresh_artifact_paths(
            tabpfn26_root, ("tabpfn_v2_6",), regions, tabpfn26_source_snapshot,
        ),
    ]
    all_input_paths = sorted(set(map(Path.resolve, [*static_paths, *artifact_paths])), key=str)
    input_snapshot = _snapshot_files(all_input_paths)
    if any(input_snapshot.get(path) != value for path, value in initial_snapshot.items()):
        raise NoCoordinateError("A source, clean-data, or partition input changed during setup")
    _atomic_json({
        "format": "poi-no-coordinate-aggregate-input-snapshot-v1",
        "phase": "before_prediction_or_provenance_read",
        "files": input_snapshot,
    }, provenance / "full_input_snapshot_before_read.json")
    tree_artifact = _validate_fresh_artifact(
        tree_root, TREE_MODELS, validation, regions, data_dir, tree_source_snapshot,
        input_snapshot, expected_mode="trees", expected_train_rows=len(train),
        expected_partition_sha256=partition_sha,
    )
    imported_artifact = _validate_imported_artifact(
        imported_root, ("tabpfn_v2_5", "tabpfn_v3"), validation, regions,
        data_dir, archive_path, input_snapshot,
    )
    tabpfn26_artifact = _validate_fresh_artifact(
        tabpfn26_root, ("tabpfn_v2_6",), validation, regions, data_dir,
        tabpfn26_source_snapshot, input_snapshot, expected_mode="tabpfn26",
        expected_train_rows=len(train), expected_partition_sha256=partition_sha,
        expected_checkpoint_sha256=str(
            cfg["tabpfn_v2_6"]["expected_checkpoint_sha256"]
        ),
    )
    tree, tree_paths = _read_prediction_shards(
        tree_root / "predictions", TREE_MODELS, validation, regions, input_snapshot,
    )
    imported, imported_paths = _read_prediction_shards(
        imported_root / "predictions", ("tabpfn_v2_5", "tabpfn_v3"),
        validation, regions, input_snapshot,
    )
    tabpfn26, tabpfn26_paths = _read_prediction_shards(
        tabpfn26_root / "predictions", ("tabpfn_v2_6",), validation, regions,
        input_snapshot,
    )
    baseline, baseline_provenance = _baseline_predictions(
        train, validation, regions, alpha=float(cfg["joint_frequency_alpha"]),
    )
    predictions = pd.concat([tree, imported, tabpfn26, baseline], ignore_index=True)
    expected_groups = len(ALL_PIPELINES) * len(SEEDS) * len(regions)
    expected_prediction_rows = expected_groups * len(validation)
    if (
        predictions.groupby(["model", "seed", "region"]).ngroups != expected_groups
        or set(predictions["model"]) != set(ALL_PIPELINES)
        or len(predictions) != expected_prediction_rows
        or predictions.duplicated(["model", "seed", "region", "POI_ID"]).any()
        or not predictions.groupby(["model", "seed", "region"]).size().eq(
            len(validation)
        ).all()
    ):
        raise NoCoordinateError("Combined prediction design is incomplete")
    partition_lookup = {value: "calibration" for value in calibration_ids}
    partition_lookup.update({value: "evaluation" for value in evaluation_ids})
    predictions["partition"] = predictions["POI_ID"].map(partition_lookup)
    if predictions["partition"].isna().any():
        raise NoCoordinateError("A prediction POI is outside the frozen partition")

    threshold_rows = []
    thresholds: dict[tuple[str, int, str], float] = {}
    for (model, model_seed, region), group in predictions.groupby(
        ["model", "seed", "region"], sort=True,
    ):
        calibration = group.loc[group["partition"].eq("calibration")]
        if model == "constant_negative":
            selected = {
                "threshold": float(np.nextafter(0.0, np.inf)), "f1": 0.0,
                "precision": 0.0, "recall": 0.0, "accepted": 0,
                "tp": 0, "fp": 0, "fn": int(calibration["y_true"].sum()),
                "rows": len(calibration), "positive_truth": int(calibration["y_true"].sum()),
                "predicted_positive_rate": 0.0, "candidate_count": 1,
                "decision_rule": "fixed_always_negative_baseline",
                "tie_break": "not_applicable",
            }
            source = "fixed_baseline_no_selection"
        else:
            selected = select_f1_threshold(
                calibration["y_true"].to_numpy(), calibration["y_score"].to_numpy(),
            )
            source = "validation_calibration_ids_only"
        key = (str(model), int(model_seed), str(region))
        thresholds[key] = float(selected["threshold"])
        threshold_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            "threshold_source": source, **selected,
        })
    thresholds_frame = pd.DataFrame(threshold_rows)
    predictions["threshold"] = [
        thresholds[(str(model), int(seed), str(region))]
        for model, seed, region in predictions[["model", "seed", "region"]].itertuples(
            index=False, name=None,
        )
    ]
    predictions["y_pred_calibrated"] = (
        predictions["y_score"].to_numpy(np.float64)
        >= predictions["threshold"].to_numpy(np.float64)
    ).astype(np.int8)
    constant = predictions["model"].eq("constant_negative")
    if predictions.loc[constant, "y_pred_calibrated"].any():
        raise NoCoordinateError("Constant-negative baseline emitted a positive decision")
    predictions["y_pred_fixed_0p5"] = (
        predictions["y_score"].to_numpy(np.float64) >= float(cfg["fixed_threshold"])
    ).astype(np.int8)

    evaluation_rows, evaluation_fixed_rows, threshold_effect_rows, full_rows = [], [], [], []
    for (model, model_seed, region), group in predictions.groupby(
        ["model", "seed", "region"], sort=True,
    ):
        evaluation = group.loc[group["partition"].eq("evaluation")]
        threshold = float(evaluation["threshold"].iloc[0])
        calibrated_metrics = _point_metrics(
            evaluation["y_true"], evaluation["y_score"], threshold,
        )
        fixed_metrics = _point_metrics(
            evaluation["y_true"], evaluation["y_score"], float(cfg["fixed_threshold"]),
        )
        evaluation_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            "cohort": "validation_evaluation_ids", "threshold": threshold,
            "threshold_source": (
                "fixed_baseline_no_selection" if model == "constant_negative"
                else "validation_calibration_ids_only"
            ),
            **calibrated_metrics,
        })
        evaluation_fixed_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            "cohort": "validation_evaluation_ids", "threshold": float(cfg["fixed_threshold"]),
            "threshold_source": "fixed_0p5_no_selection",
            **fixed_metrics,
        })
        threshold_effect_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            "cohort": "same_validation_evaluation_ids",
            "calibrated_threshold": threshold,
            "fixed_threshold": float(cfg["fixed_threshold"]),
            **{
                f"delta_{metric}_calibrated_minus_fixed_0p5": (
                    calibrated_metrics[metric] - fixed_metrics[metric]
                ) for metric in METRICS
            },
            "delta_predicted_positive_rate_calibrated_minus_fixed_0p5": (
                calibrated_metrics["predicted_positive_rate"]
                - fixed_metrics["predicted_positive_rate"]
            ),
            "evaluation_rows": len(evaluation),
        })
        full_rows.append({
            "model": model, "seed": int(model_seed), "region": region,
            "cohort": "full_validation_descriptive", "threshold": float(cfg["fixed_threshold"]),
            "threshold_source": "fixed_0p5_no_selection",
            **_point_metrics(
                group["y_true"], group["y_score"], float(cfg["fixed_threshold"])
            ),
        })
    evaluation_metrics = pd.DataFrame(evaluation_rows)
    evaluation_fixed_metrics = pd.DataFrame(evaluation_fixed_rows)
    threshold_effects = pd.DataFrame(threshold_effect_rows)
    full_metrics = pd.DataFrame(full_rows)
    evaluation_predictions = predictions.loc[
        predictions["partition"].eq("evaluation")
    ].copy()
    bootstrap_points, replicates, intervals, pairwise, weights = _bootstrap_tables(
        evaluation_predictions,
        evaluation_ids=evaluation_ids,
        repeats=int(cfg["bootstrap_repeats"]), seed=int(cfg["bootstrap_seed"]),
        confidence_level=float(cfg["confidence_level"]),
    )
    if not np.allclose(
        bootstrap_points.sort_values(["model", "seed", "region"])[list(METRICS)],
        evaluation_metrics.sort_values(["model", "seed", "region"])[list(METRICS)],
        rtol=0, atol=1e-12,
    ):
        raise NoCoordinateError("Bootstrap point metrics differ from evaluation metrics")
    expected_metric_rows = expected_groups
    if (
        len(thresholds_frame) != expected_metric_rows
        or len(evaluation_metrics) != expected_metric_rows
        or len(evaluation_fixed_metrics) != expected_metric_rows
        or len(threshold_effects) != expected_metric_rows
        or len(full_metrics) != expected_metric_rows
        or len(evaluation_predictions) != expected_groups * len(evaluation_ids)
        or len(replicates) != len(ALL_PIPELINES) * len(METRICS) * int(
            cfg["bootstrap_repeats"]
        )
        or len(intervals) != len(ALL_PIPELINES) * len(METRICS)
        or len(pairwise) != len(tuple(combinations(ALL_PIPELINES, 2))) * len(METRICS)
    ):
        raise NoCoordinateError("Aggregate table cardinality differs from the frozen design")

    _atomic_frame(predictions, tables / "all_validation_predictions.parquet")
    _atomic_frame(thresholds_frame, tables / "calibration_thresholds.csv")
    _atomic_frame(evaluation_metrics, tables / "evaluation_metrics_by_region_seed.csv")
    _atomic_frame(
        evaluation_fixed_metrics, tables / "evaluation_fixed_0p5_metrics_by_region_seed.csv"
    )
    _atomic_frame(threshold_effects, tables / "evaluation_threshold_effects.csv")
    _atomic_frame(full_metrics, tables / "full_validation_fixed_0p5_metrics.csv")
    _atomic_frame(replicates, tables / "bootstrap_macro_metric_replicates.parquet")
    _atomic_frame(intervals, tables / "bootstrap_macro_confidence_intervals.csv")
    _atomic_frame(pairwise, tables / "bootstrap_pairwise_differences.csv")
    _atomic_json(baseline_provenance, provenance / "baseline_fit_provenance.json")
    np.save(work / "bootstrap_poi_weights.npy", weights, allow_pickle=False)
    pd.DataFrame({"POI_ID": evaluation_ids}).to_csv(
        work / "bootstrap_poi_order.csv", index=False,
    )
    _plots(evaluation_metrics, evaluation_predictions, intervals, figures)
    if list(figures.glob("*.svg")):
        raise NoCoordinateError("SVG figures are forbidden")
    _assert_snapshot_unchanged(input_snapshot)
    _atomic_json({
        "format": "poi-no-coordinate-aggregate-input-snapshot-v1",
        "phase": "verified_unchanged_after_all_computation",
        "files": input_snapshot,
    }, provenance / "full_input_snapshot_after_read.json")
    prediction_paths = [*tree_paths, *imported_paths, *tabpfn26_paths]
    prediction_hashes = _snapshot_digest_map(prediction_paths, input_snapshot)
    input_hashes = {
        path: str(values["sha256"]) for path, values in input_snapshot.items()
    }
    metadata = {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scientific_completion": False,
        "format": "poi-no-coordinate-eight-model-comparison-v1",
        "scope": "clean train/validation no-coordinate comparison only",
        "features": list(FEATURES), "learned_models": list(LEARNED_MODELS),
        "baselines": list(BASELINES), "seeds": list(SEEDS), "regions": list(regions),
        "prediction_rows": len(predictions), "expected_prediction_rows": expected_prediction_rows,
        "model_seed_region_groups": expected_groups,
        "calibration_rows": len(calibration_ids), "evaluation_rows": len(evaluation_ids),
        "metrics": list(METRICS), "partition": partition,
        "threshold_selection": (
            "per model/seed/region on calibration IDs only; deterministic max F1 tie policy; "
            "zero thresholds and all-positive rates retained"
        ),
        "threshold_effect_comparison": (
            "calibrated and fixed-0.5 decisions compared on the exact same 1,707 evaluation IDs"
        ),
        "full_validation_0p5": (
            "separate 3,400-row historical reproduction; never used as the threshold-effect comparator"
        ),
        "bootstrap": {
            "repeats": int(cfg["bootstrap_repeats"]), "seed": int(cfg["bootstrap_seed"]),
            "unit": "POI_ID", "same_weights_all_models": True,
            "conditional_on_fitted_models_and_calibration_selected_thresholds": True,
            "weights_sha256": sha256(work / "bootstrap_poi_weights.npy"),
        },
        "baseline_provenance": baseline_provenance,
        "source_sha256": source_hashes,
        "prediction_input_sha256": prediction_hashes,
        "all_input_sha256": input_hashes,
        "fresh_tree_artifact": tree_artifact,
        "fresh_tabpfn26_artifact": tabpfn26_artifact,
        "imported_tabpfn_provenance": {
            **imported_artifact,
            "statement": (
                "Original R2 v2.5/v3 shards are bound to every original marker, metadata, "
                "input identity, clean inputs, and the archived executed feature-ablation and "
                "factory source. That historical run predates per-fit attestation, so no "
                "per-fit provenance is reconstructed."
            ),
        },
        "training_weight_policy": {
            "tree_models": (
                "per OvR fit balanced binary sample weights: n/(2*positive_count) for positives "
                "and n/(2*negative_count) for negatives"
            ),
            "tabpfn_models": "TabPFN default fit without the tree sample-weight policy",
            "interpretation": (
                "Fixed-0.5 F1 and positive-rate differences are not evidence of greater feature "
                "information; calibration-matched evaluation and ranking metrics ROC-AUC/AP are "
                "reported alongside them."
            ),
        },
        "test_rows_read": 0, "test_features_read": 0,
        "historical_test_access": (
            "PARTIAL: prior unrelated processing materialized all 34,000 coordinates; "
            "this validation-only run does not restore pristine historical isolation"
        ),
        "limitations": [
            "Validation estimates remain conditional on the fitted models and selected thresholds.",
            "The POI bootstrap does not resample training, spatial clusters, or model-selection steps.",
            "Imported v2.5/v3 predictions have PARTIAL per-fit provenance by original-run design.",
            "Trees use class-balanced fit weights while TabPFN uses its default unweighted fit; "
            "fixed-0.5 F1 and positive rates are therefore not direct information-content evidence.",
            "No test rows or attack robustness outcomes are part of this artifact.",
            "Independent review is pending.",
        ],
        "git": _git_state(), "python": platform.python_version(),
        "versions": _runtime_versions([
            "numpy", "pandas", "numba", "scikit-learn", "matplotlib", "seaborn", "pyarrow",
        ]),
        "seconds": time.time() - started,
    }
    _atomic_json(metadata, output / "metadata.json")
    (output / "config.yaml").write_bytes(_bound_bytes(cfg["_config_path"], input_snapshot))
    (output / "report.md").write_text(
        "# No-coordinate validation comparison\n\n"
        "Eight learned models and two declared baselines use only the three category codes. "
        "Thresholds are selected independently for each model, seed and OvR region on the 1,693 "
        "calibration POIs; the 1,707 evaluation POIs supply the reported calibrated metrics and "
        "2,000-draw shared paired-POI confidence intervals. The full 3,400 validation rows at "
        "threshold 0.5 are reported separately and never enter threshold selection. Threshold "
        "effects compare calibrated and fixed-0.5 decisions on those same 1,707 evaluation POIs.\n\n"
        "The constant-negative baseline always predicts negative. The joint-frequency baseline is "
        "fit on clean train only with Laplace smoothing and a smoothed train regional-prior fallback "
        "for unseen category tuples. Tree fits use balanced binary sample weights n/(2 class count), "
        "whereas TabPFN uses its default fit without those weights. Fixed-0.5 F1 and positive-rate "
        "differences therefore do not establish a feature-information advantage; the report retains "
        "calibration-matched evaluation and ROC-AUC/AP. R2 TabPFN 2.5/v3 per-fit provenance remains "
        "PARTIAL and is not reconstructed retrospectively. Historical test isolation remains PARTIAL; "
        "no test row was read here. Independent review is pending.\n"
    )
    _assert_snapshot_unchanged(input_snapshot)
    _atomic_json({
        "status": metadata["status"], "metadata_sha256": sha256(output / "metadata.json"),
        "prediction_sha256": sha256(tables / "all_validation_predictions.parquet"),
        "interval_sha256": sha256(tables / "bootstrap_macro_confidence_intervals.csv"),
        "test_rows_read": 0,
    }, output / "_GENERATED_PENDING_REVIEW.json")


def load_config(
    path: str | Path,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config_bytes = (
        _bound_bytes(config_path, snapshot) if snapshot is not None else config_path.read_bytes()
    )
    cfg = yaml.safe_load(config_bytes)
    cfg["_config_path"] = str(config_path)
    if tuple(cfg.get("features", ())) != FEATURES:
        raise ValueError("No-coordinate feature order must be the three frozen category codes")
    if tuple(map(int, cfg.get("seeds", ()))) != SEEDS:
        raise ValueError("No-coordinate experiment requires seeds 42, 202, 340")
    if tuple(cfg.get("learned_models", ())) != LEARNED_MODELS:
        raise ValueError("No-coordinate experiment requires the eight planned learned models")
    if tuple(cfg.get("baselines", ())) != BASELINES:
        raise ValueError("No-coordinate experiment requires both declared baselines")
    return cfg


def run(config_path: str | Path, mode: str) -> None:
    if mode == "aggregate":
        config_path = Path(config_path).resolve()
        config_snapshot = _snapshot_files([config_path])
        cfg = load_config(config_path, config_snapshot)
        cfg["_config_load_snapshot"] = config_snapshot[str(config_path)]
    else:
        cfg = load_config(config_path)
    if mode in {"trees", "tabpfn26"}:
        _fit_models(cfg, mode)
    elif mode == "aggregate":
        _aggregate(cfg)
    else:
        raise ValueError("mode must be trees, tabpfn26, or aggregate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("trees", "tabpfn26", "aggregate"), required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.mode)


if __name__ == "__main__":
    main()
