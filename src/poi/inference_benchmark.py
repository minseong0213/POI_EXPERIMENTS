"""Validation-only inference timing and fitted-model reproducibility benchmark.

This runner refits the eight planned models on the frozen clean train split, predicts the full
clean validation split, and times prediction only.  It never reads test features and it does not
modify the separately configured common downstream classifier.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import CATEGORICAL, FEATURES, load_clean_splits, sha256


MODELS = (
    "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
    "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3",
)
TABPFN_MODELS = frozenset({"tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"})
TREE_MODELS = frozenset(set(MODELS) - TABPFN_MODELS)
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
IDENTITY = ("model", "seed", "region", "POI_ID")
FIT_IDENTITY = ("model", "seed", "region")
_SHA256_LENGTH = 64


class BenchmarkContractError(RuntimeError):
    """The benchmark configuration or evidence violates the frozen contract."""


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray, dtype: Any | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _ordered_text_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    """Frozen float32 matrix contract used only by TabPFN families."""
    matrix = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all():
        raise BenchmarkContractError("Model matrix must be finite with the five frozen features")
    return np.ascontiguousarray(matrix)


def _raw_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash exact ordered feature text and dtypes before any numeric coercion."""
    from .selected_explain import _dataframe_sha256
    return _dataframe_sha256(frame, FEATURES)


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise BenchmarkContractError("Git identity is required for the inference benchmark") from error
    return {
        "commit": commit, "dirty": bool(status.strip()),
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status_porcelain_lines": len(status.splitlines()),
    }


def validate_checkpoint_contracts(
    checkpoint_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve and byte-check all three explicitly configured official checkpoints."""
    if set(checkpoint_contracts) != set(TABPFN_MODELS):
        raise BenchmarkContractError("Exactly the three planned TabPFN checkpoints are required")
    result: dict[str, dict[str, Any]] = {}
    for model in sorted(checkpoint_contracts):
        raw = checkpoint_contracts[model]
        path = Path(str(raw.get("path", ""))).expanduser()
        expected = str(raw.get("sha256", "")).lower()
        if (
            not path.is_absolute() or path.is_symlink() or not path.is_file()
            or len(expected) != _SHA256_LENGTH
        ):
            raise BenchmarkContractError(f"Missing/invalid explicit checkpoint for {model}: {path}")
        observed = sha256(path)
        if observed != expected:
            raise BenchmarkContractError(f"Checkpoint SHA-256 mismatch for {model}")
        result[model] = {
            "path": str(path.resolve()), "sha256": observed,
            "size_bytes": path.stat().st_size,
        }
    return result


def validate_tree_thread_contract(tree_config_path: str | Path, cpu_threads: int) -> dict[str, Any]:
    """Require a common two-thread ceiling and explicit estimator thread parameters."""
    if int(cpu_threads) != 2:
        raise BenchmarkContractError("Tree benchmark CPU thread budget must be exactly 2")
    path = Path(tree_config_path)
    if not path.is_file():
        raise BenchmarkContractError(f"Tree model configuration is absent: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = raw.get("models", {}) if isinstance(raw, Mapping) else {}
    if set(models) != set(TREE_MODELS):
        raise BenchmarkContractError("Tree configuration must contain exactly five planned models")
    required = {
        "random_forest": ("n_jobs", 2),
        "xgboost": ("n_jobs", 2),
        "lightgbm": ("n_jobs", 2),
        "catboost": ("thread_count", 2),
    }
    for model, (field, expected) in required.items():
        if models[model].get(field) != expected:
            raise BenchmarkContractError(f"{model}.{field} must equal {expected}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "models": models}


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed design without opening clean/test data."""
    cfg = dict(raw)
    if cfg.get("format") != "poi-eight-model-inference-benchmark-v1":
        raise BenchmarkContractError("Inference benchmark format is invalid")
    if tuple(cfg.get("models", ())) != MODELS:
        raise BenchmarkContractError("Model order must contain the exact eight planned models")
    if list(cfg.get("seeds", ())) != [42, 202, 340]:
        raise BenchmarkContractError("Model seeds must be exactly [42, 202, 340]")
    if int(cfg.get("expected_regions", 0)) != 17:
        raise BenchmarkContractError("Exactly 17 OvR regions are required")
    if int(cfg.get("expected_train_rows", 0)) != 23_800:
        raise BenchmarkContractError("Expected train rows must be 23,800")
    if int(cfg.get("expected_validation_rows", 0)) != 3_400:
        raise BenchmarkContractError("Expected validation rows must be 3,400")
    if float(cfg.get("classification_threshold", -1)) != 0.5:
        raise BenchmarkContractError("The frozen independent OvR threshold must be 0.5")
    timing = cfg.get("timing")
    if not isinstance(timing, Mapping) or timing.get("warmup_repeats") != 1 or timing.get(
        "measured_repeats"
    ) != 5 or timing.get("rows_per_repeat") != 3_400:
        raise BenchmarkContractError("Timing requires one warmup and five full-validation repeats")
    selection = cfg.get("selection_rule")
    if (
        not isinstance(selection, Mapping)
        or float(selection.get("absolute_f1_tolerance", -1)) != 0.005
        or selection.get("performance_metric") != "clean_validation_macro_f1"
        or selection.get("candidate_boundary") != "inclusive"
        or selection.get("timing_statistic")
        != "equal_mean_of_seed_region_fit_median_validation_batch_seconds"
    ):
        raise BenchmarkContractError("Selection rule differs from the frozen protocol")
    if not str(cfg.get("device", "")).startswith("cuda"):
        raise BenchmarkContractError("The executable benchmark requires one explicit CUDA device")
    if int(cfg.get("tabpfn_n_estimators", 0)) <= 0:
        raise BenchmarkContractError("TabPFN n_estimators must be positive")
    if cfg.get("existing_common_downstream") != "tabpfn_v2_5":
        raise BenchmarkContractError("The separately fixed common downstream disclosure is missing")
    comparison = cfg.get("prediction_comparison")
    if (
        not isinstance(comparison, Mapping)
        or float(comparison.get("absolute_tolerance", -1)) < 0
        or float(comparison.get("relative_tolerance", -1)) < 0
        or float(cfg.get("frozen_metric_tolerance", -1)) < 0
    ):
        raise BenchmarkContractError("Prediction/metric comparison tolerances must be nonnegative")
    return cfg


def measure_prediction_repeats(
    predictor: Callable[[Any], np.ndarray],
    prediction_input: Any,
    *,
    measured_repeats: int = 5,
    repeat_absolute_tolerance: float = 0.0,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Warm once, then time deterministic full-batch prediction with synchronization."""
    if (
        measured_repeats != 5 or len(prediction_input) == 0
        or repeat_absolute_tolerance < 0 or not math.isfinite(repeat_absolute_tolerance)
    ):
        raise BenchmarkContractError("Exactly five measured repeats and non-empty input are required")

    synchronize()
    warm = np.asarray(predictor(prediction_input), dtype=np.float64)
    synchronize()
    if warm.shape != (len(prediction_input),) or not np.isfinite(warm).all():
        raise BenchmarkContractError("Warmup prediction is non-finite or has wrong shape")
    warm_sha = _array_sha256(warm, np.float64)

    repeats: list[dict[str, Any]] = []
    primary: np.ndarray | None = None
    maximum_repeat_difference = 0.0
    all_repeats_exact = True
    for repeat in range(measured_repeats):
        synchronize()
        start = clock()
        prediction = np.asarray(predictor(prediction_input), dtype=np.float64)
        synchronize()
        elapsed = float(clock() - start)
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise BenchmarkContractError("Measured prediction duration must be finite and positive")
        if prediction.shape != warm.shape or not np.isfinite(prediction).all():
            raise BenchmarkContractError("Measured prediction is non-finite or has wrong shape")
        digest = _array_sha256(prediction, np.float64)
        exact = bool(digest == warm_sha and np.array_equal(prediction, warm))
        difference = float(np.abs(prediction - warm).max(initial=0.0))
        maximum_repeat_difference = max(maximum_repeat_difference, difference)
        all_repeats_exact &= exact
        if not np.allclose(
            prediction, warm, atol=repeat_absolute_tolerance, rtol=0, equal_nan=False
        ):
            raise BenchmarkContractError(
                "Repeated prediction differs from the excluded warmup beyond configured tolerance"
            )
        if primary is None:
            primary = prediction.copy()
        repeats.append({
            "repeat": repeat + 1,
            "validation_batch_seconds": elapsed,
            "prediction_sha256": digest,
            "exactly_equal_to_warmup": exact,
            "max_absolute_difference_from_warmup": difference,
        })
    durations = np.asarray([row["validation_batch_seconds"] for row in repeats], dtype=float)
    mean = float(durations.mean())
    summary = {
        "warmup_repeats": 1,
        "measured_repeats": measured_repeats,
        "warmup_excluded_from_primary_time": True,
        "prediction_sha256": warm_sha,
        "all_repeats_exactly_equal": all_repeats_exact,
        "maximum_repeat_absolute_difference": maximum_repeat_difference,
        "repeat_absolute_tolerance": float(repeat_absolute_tolerance),
        "median_validation_batch_seconds": float(np.median(durations)),
        "mean_validation_batch_seconds": mean,
        "std_validation_batch_seconds": float(durations.std(ddof=1)),
        "timing_cv": float(durations.std(ddof=1) / mean),
        "min_validation_batch_seconds": float(durations.min()),
        "max_validation_batch_seconds": float(durations.max()),
        "validation_rows": len(prediction_input),
        "median_microseconds_per_row_amortized": float(
            np.median(durations) * 1e6 / len(prediction_input)
        ),
    }
    assert primary is not None
    return primary, summary, repeats


def compare_fit_predictions(
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Align one model/seed/region fit and describe exact/tolerant/different reproduction."""
    required = {"POI_ID", "y_true", "y_score", "y_pred"}
    if required - set(actual) or required - set(reference):
        raise BenchmarkContractError("Prediction comparison lacks required columns")
    if actual.POI_ID.duplicated().any() or reference.POI_ID.duplicated().any():
        raise BenchmarkContractError("Prediction comparison has duplicate POI_ID")
    if set(actual.POI_ID.astype(str)) != set(reference.POI_ID.astype(str)):
        raise BenchmarkContractError("Actual/reference prediction POI sets differ")
    actual_aligned = actual.assign(POI_ID=actual.POI_ID.astype(str)).set_index("POI_ID").sort_index()
    reference_aligned = reference.assign(
        POI_ID=reference.POI_ID.astype(str)
    ).set_index("POI_ID").sort_index()
    if not np.array_equal(
        actual_aligned.y_true.to_numpy(dtype=int),
        reference_aligned.y_true.to_numpy(dtype=int),
    ):
        raise BenchmarkContractError("Actual/reference prediction truth differs after alignment")
    actual_score = actual_aligned.y_score.to_numpy(dtype=np.float64)
    reference_score = reference_aligned.y_score.to_numpy(dtype=np.float64)
    if not np.isfinite(actual_score).all() or not np.isfinite(reference_score).all():
        raise BenchmarkContractError("Actual/reference prediction score is non-finite")
    difference = np.abs(actual_score - reference_score)
    exact = bool(np.array_equal(actual_score, reference_score))
    within = bool(np.allclose(
        actual_score, reference_score, atol=absolute_tolerance, rtol=relative_tolerance,
        equal_nan=False,
    ))
    label_mismatches = int(np.sum(
        actual_aligned.y_pred.to_numpy(dtype=int)
        != reference_aligned.y_pred.to_numpy(dtype=int)
    ))
    if exact and label_mismatches == 0:
        status = "EXACT_SAME_SCORES_AND_LABELS"
    elif within and label_mismatches == 0:
        status = "DIFFERENT_SCORES_WITHIN_TOLERANCE_SAME_LABELS"
    elif within:
        status = "DIFFERENT_SCORES_WITHIN_TOLERANCE_DIFFERENT_LABELS"
    else:
        status = "DIFFERENT_OUTSIDE_TOLERANCE"
    return {
        "reproduction_status": status,
        "scores_exactly_equal": exact,
        "scores_within_configured_tolerance": within,
        "prediction_label_mismatches": label_mismatches,
        "max_absolute_score_difference": float(difference.max(initial=0.0)),
        "mean_absolute_score_difference": float(difference.mean()) if len(difference) else 0.0,
        "actual_score_sha256": _array_sha256(actual_score, np.float64),
        "reference_score_sha256": _array_sha256(reference_score, np.float64),
        "compared_rows": len(actual_score),
        "score_absolute_tolerance": float(absolute_tolerance),
        "score_relative_tolerance": float(relative_tolerance),
    }


def summarize_prediction_reproduction(comparison: pd.DataFrame) -> dict[str, bool]:
    """Summarize score and label fidelity from the comparison's recorded fields."""
    required = {"scores_exactly_equal", "prediction_label_mismatches"}
    missing = required - set(comparison)
    if missing or comparison.empty:
        raise BenchmarkContractError(
            f"Prediction reproduction summary is incomplete: missing={sorted(missing)}"
        )
    mismatches = pd.to_numeric(comparison["prediction_label_mismatches"], errors="raise")
    if not np.isfinite(mismatches.to_numpy(dtype=np.float64)).all() or (mismatches < 0).any():
        raise BenchmarkContractError("Prediction label mismatch counts must be finite and nonnegative")
    return {
        "all_scores_exact": bool(comparison["scores_exactly_equal"].astype(bool).all()),
        "all_labels_same": bool(mismatches.eq(0).all()),
    }


def aggregate_timing_rows(
    timing_rows: pd.DataFrame,
    *,
    models: Sequence[str],
    expected_fits_per_model: int,
) -> pd.DataFrame:
    """Give every seed/region fit equal weight after taking its five-repeat median."""
    required = {
        "model", "seed", "region", "median_validation_batch_seconds", "fit_seconds",
        "timing_cv", "validation_rows",
    }
    if required - set(timing_rows):
        raise BenchmarkContractError("Timing table lacks required fields")
    if timing_rows.duplicated(list(FIT_IDENTITY)).any():
        raise BenchmarkContractError("Timing table has duplicate model/seed/region fits")
    rows: list[dict[str, Any]] = []
    for model in models:
        part = timing_rows.loc[timing_rows.model.eq(model)].copy()
        if len(part) != expected_fits_per_model:
            raise BenchmarkContractError(
                f"{model} timing coverage {len(part)} != {expected_fits_per_model}"
            )
        for field in ("median_validation_batch_seconds", "fit_seconds", "timing_cv"):
            values = part[field].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise BenchmarkContractError(f"{model} has invalid {field}")
        if (part["median_validation_batch_seconds"] <= 0).any():
            raise BenchmarkContractError(f"{model} has non-positive inference time")
        if part.validation_rows.nunique() != 1:
            raise BenchmarkContractError(f"{model} timing fits use different row counts")
        medians = part.median_validation_batch_seconds.to_numpy(dtype=float)
        cvs = part.timing_cv.to_numpy(dtype=float)
        fit_seconds = part.fit_seconds.to_numpy(dtype=float)
        rows.append({
            "model": model,
            "fit_count": len(part),
            "validation_rows_per_fit": int(part.validation_rows.iloc[0]),
            "primary_inference_seconds": float(medians.mean()),
            "primary_aggregation": "equal_mean_of_seed_region_fit_median_validation_batch_seconds",
            "median_of_fit_median_seconds": float(np.median(medians)),
            "std_of_fit_median_seconds": float(medians.std(ddof=1)),
            "min_fit_median_seconds": float(medians.min()),
            "max_fit_median_seconds": float(medians.max()),
            "mean_per_fit_timing_cv": float(cvs.mean()),
            "median_per_fit_timing_cv": float(np.median(cvs)),
            "max_per_fit_timing_cv": float(cvs.max()),
            "mean_fit_seconds": float(fit_seconds.mean()),
            "total_fit_seconds": float(fit_seconds.sum()),
            "fit_time_excluded_from_selection": True,
        })
    return pd.DataFrame(rows)


def select_by_protocol_rule(
    performance: pd.DataFrame,
    timing_summary: pd.DataFrame,
    *,
    models: Sequence[str] = MODELS,
    absolute_f1_tolerance: float = 0.005,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the inclusive 0.005 Macro-F1 band, then choose the shortest inference time."""
    expected = set(models)
    if set(performance.model) != expected or set(timing_summary.model) != expected:
        raise BenchmarkContractError("Selection tables must each cover exactly all eight models")
    if performance.model.duplicated().any() or timing_summary.model.duplicated().any():
        raise BenchmarkContractError("Selection tables must contain one row per model")
    values = performance[["model", "macro_f1"]].merge(
        timing_summary[["model", "primary_inference_seconds"]], on="model", validate="one_to_one"
    )
    if (
        not np.isfinite(values.macro_f1.to_numpy(dtype=float)).all()
        or not np.isfinite(values.primary_inference_seconds.to_numpy(dtype=float)).all()
        or (values.primary_inference_seconds <= 0).any()
    ):
        raise BenchmarkContractError("Selection inputs must be finite with positive timing")
    best_f1 = float(values.macro_f1.max())
    values["absolute_f1_gap_from_best"] = best_f1 - values.macro_f1
    values["within_inclusive_0p005_candidate_band"] = (
        values.absolute_f1_gap_from_best <= float(absolute_f1_tolerance) + 1e-15
    )
    values["performance_rank"] = values.macro_f1.rank(method="min", ascending=False).astype(int)
    candidates = values.loc[values.within_inclusive_0p005_candidate_band].sort_values(
        ["primary_inference_seconds", "model"], kind="mergesort"
    )
    if candidates.empty:
        raise BenchmarkContractError("Selection candidate band is unexpectedly empty")
    selected = str(candidates.iloc[0].model)
    values["candidate_inference_rank"] = np.nan
    values.loc[candidates.index, "candidate_inference_rank"] = np.arange(1, len(candidates) + 1)
    values["selected_by_protocol_rule"] = values.model.eq(selected)
    values = values.sort_values(["performance_rank", "model"], kind="mergesort").reset_index(drop=True)
    result = {
        "status": "DEVELOPMENT_SELECTION_RECOMMENDATION_PENDING_INDEPENDENT_REVIEW",
        "selected_model": selected,
        "highest_macro_f1": best_f1,
        "absolute_f1_tolerance": float(absolute_f1_tolerance),
        "candidate_boundary": "inclusive",
        "candidate_models": candidates.model.astype(str).tolist(),
        "candidate_count": len(candidates),
        "tie_breaker": "shortest equal-weight mean of per-fit median full-validation inference seconds",
        "deterministic_final_tie_breaker": "lexicographic model id",
        "performance_scope": "immutable clean validation comparison",
        "test_used": False,
        "does_not_modify_explicit_common_downstream_choice": True,
    }
    return values, result


def _five_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    prediction = (score >= threshold).astype(int)
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, score)),
    }


def _runtime_environment(device: str, cpu_threads: int) -> tuple[dict[str, Any], Callable[[], None]]:
    import torch

    if not torch.cuda.is_available():
        raise BenchmarkContractError("CUDA is unavailable; timing cannot use the required GPU host")
    resolved = torch.device(device)
    torch.cuda.set_device(resolved)
    distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions[str(name).lower()] = distribution.version
    runtime = {
        "format": "poi-eight-model-inference-runtime-v1",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "tree_cpu_thread_budget": int(cpu_threads),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device": str(resolved),
        "gpu_device_name": torch.cuda.get_device_name(resolved),
        "installed_distributions": dict(sorted(distributions.items())),
        "test_feature_rows_read": 0,
    }

    def synchronize() -> None:
        torch.cuda.synchronize(resolved)

    return runtime, synchronize


def _validate_reference(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    from .clean_comparison import validate_predictions

    groups = validate_predictions(
        predictions, validation, cfg["models"], cfg["seeds"],
        float(cfg["classification_threshold"]),
    )
    expected_groups = len(cfg["models"]) * len(cfg["seeds"]) * int(cfg["expected_regions"])
    if groups != expected_groups or len(predictions) != expected_groups * len(validation):
        raise BenchmarkContractError("Immutable comparison prediction cardinality differs")
    rows = []
    for key, group in predictions.groupby(list(FIT_IDENTITY), sort=True, observed=True):
        rows.append(dict(
            zip(FIT_IDENTITY, key),
            **_five_metrics(
                group.y_true.to_numpy(dtype=int), group.y_score.to_numpy(dtype=float),
                float(cfg["classification_threshold"]),
            ),
        ))
    return pd.DataFrame(rows)


def _performance_from_frozen_summary(
    summary: pd.DataFrame,
    recomputed_metrics: pd.DataFrame,
    models: Sequence[str],
    tolerance: float,
) -> pd.DataFrame:
    if set(summary.model) != set(models) or summary.model.duplicated().any() or "f1_mean" not in summary:
        raise BenchmarkContractError("Frozen clean summary lacks exact eight-model F1 coverage")
    recomputed = recomputed_metrics.groupby("model", observed=True).f1.mean().rename(
        "recomputed_macro_f1"
    ).reset_index()
    result = summary[["model", "f1_mean"]].rename(columns={"f1_mean": "macro_f1"}).merge(
        recomputed, on="model", validate="one_to_one"
    )
    result["summary_minus_recomputed_f1"] = result.macro_f1 - result.recomputed_macro_f1
    if not np.allclose(
        result.macro_f1, result.recomputed_macro_f1, atol=tolerance, rtol=0, equal_nan=False
    ):
        raise BenchmarkContractError("Frozen Macro F1 summary differs from its paired predictions")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_tree_fidelity_validation(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """CPU-only seed-42 reproduction using exact raw DataFrame tree inference.

    This is a focused launch gate for the five tree families.  It neither runs the
    eight-model timing benchmark nor reads test features.
    """
    execution_started_utc = datetime.now(timezone.utc).isoformat()
    git_start = _git_state()
    config_path = Path(config_path).resolve()
    cfg = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    output = Path(output_path).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite fidelity output/staging: {output} / {staging}")
    validate_tree_thread_contract(cfg["tree_config"], cfg["tree_cpu_threads"])
    reference_path = Path(cfg["existing_clean_predictions"]).resolve()
    if not reference_path.is_file():
        raise BenchmarkContractError("Immutable clean comparison predictions are absent")
    data_dir = Path(os.environ.get("DATA_DIR", cfg["data_dir"])).expanduser().resolve()
    dataset_paths = [data_dir / name for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")]
    source_paths = [
        Path(__file__).resolve(), Path(__file__).with_name("selected_explain.py").resolve(),
        Path(__file__).with_name("data.py").resolve(), Path(__file__).with_name("ablation.py").resolve(),
        Path(__file__).with_name("benchmark.py").resolve(), config_path,
        Path(cfg["tree_config"]).resolve(), reference_path,
    ]
    guarded = [*source_paths, *dataset_paths]
    if any(not path.is_file() for path in guarded):
        raise BenchmarkContractError("Tree fidelity source/input snapshot is incomplete")
    initial_hashes = {str(path): sha256(path) for path in guarded}
    train, _ = load_clean_splits(data_dir, splits=("train",))
    validation, _ = load_clean_splits(data_dir, splits=("validation",))
    regions = sorted(train.region.astype(str).unique())
    if len(train) != 23_800 or len(validation) != 3_400 or len(regions) != 17:
        raise BenchmarkContractError("Tree fidelity validation requires the frozen development design")
    reference = pd.read_parquet(reference_path)
    reference = reference.loc[
        reference.model.isin(TREE_MODELS) & reference.seed.eq(42)
    ].copy()
    expected_rows = len(TREE_MODELS) * len(regions) * len(validation)
    if len(reference) != expected_rows:
        raise BenchmarkContractError("Seed-42 tree reference grid is incomplete")
    reference_groups = reference.groupby(list(FIT_IDENTITY), observed=True, sort=False)

    staging.mkdir(parents=True)
    tables = staging / "tables"
    provenance = staging / "provenance"
    snapshots = provenance / "source_snapshots"
    for directory in (tables, provenance, snapshots):
        directory.mkdir()
    manifest = []
    for index, source in enumerate(source_paths[:-1]):
        target = snapshots / f"{index:02d}_{source.name}"
        shutil.copyfile(source, target)
        manifest.append({
            "source_path": str(source),
            "snapshot_path": str(output / "provenance" / "source_snapshots" / target.name),
            "sha256": sha256(target), "size_bytes": target.stat().st_size,
        })
    _write_json(provenance / "execution_source_manifest.json", manifest)
    runtime = {
        "format": "poi-tree-fidelity-cpu-runtime-v1",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "tree_cpu_thread_budget": int(cfg["tree_cpu_threads"]),
        "device_assignment": "five trees on CPU; no GPU used by this focused reconstruction",
        "installed_distributions": dict(sorted(
            (str(d.metadata["Name"]).lower(), d.version)
            for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
        "recorded_before_fitting": True,
    }
    _write_json(provenance / "runtime_environment.json", runtime)
    from .selected_explain import ModelFactory
    from threadpoolctl import threadpool_limits

    predictions = []
    comparisons = []
    attestations = []
    raw_train_sha = _raw_frame_sha256(train)
    raw_validation_sha = _raw_frame_sha256(validation)
    with threadpool_limits(limits=2):
        for region in regions:
            truth_train = train.region.astype(str).eq(region).astype(int).to_numpy()
            truth_validation = validation.region.astype(str).eq(region).astype(int).to_numpy()
            for model in MODELS:
                if model not in TREE_MODELS:
                    continue
                factory = ModelFactory(model, {
                    "seed": 42, "tree_config": str(Path(cfg["tree_config"]).resolve()),
                    "repeat_prediction_absolute_tolerance": float(
                        cfg["prediction_comparison"]["absolute_tolerance"]
                    ),
                }, train)
                predictor = factory.fit(FEATURES, truth_train)
                score = np.asarray(predictor(validation), dtype=np.float64)
                if score.shape != (len(validation),) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
                    raise BenchmarkContractError("Tree fidelity probability is invalid")
                actual = pd.DataFrame({
                    "POI_ID": validation.POI_ID.astype(str), "model": model, "seed": 42,
                    "region": region, "condition": "clean", "split": "validation",
                    "y_true": truth_validation, "y_score": score,
                    "y_pred": (score >= float(cfg["classification_threshold"])).astype(int),
                })
                comparison = compare_fit_predictions(
                    actual, reference_groups.get_group((model, 42, region)),
                    absolute_tolerance=float(cfg["prediction_comparison"]["absolute_tolerance"]),
                    relative_tolerance=float(cfg["prediction_comparison"]["relative_tolerance"]),
                )
                attestation = dict(factory.last_fit_attestation)
                if (
                    attestation.get("format") != "poi-tree-fit-attestation-v1"
                    or attestation.get("raw_train_frame_sha256") != raw_train_sha
                    or attestation.get("truth_sha256") != _array_sha256(truth_train, np.int8)
                ):
                    raise BenchmarkContractError("Focused tree fit attestation is incomplete")
                identity = {"model": model, "seed": 42, "region": region}
                predictions.append(actual)
                comparisons.append({**identity, **comparison})
                attestations.append({
                    **identity, "model_factory_attestation": attestation,
                    "train_raw_feature_frame_sha256": raw_train_sha,
                    "validation_raw_feature_frame_sha256": raw_validation_sha,
                    "test_feature_rows_read": 0,
                })
                print(json.dumps({**identity, "max_absolute_score_difference": comparison["max_absolute_score_difference"],
                                  "status": comparison["reproduction_status"]}), flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    comparison_frame = pd.DataFrame(comparisons)
    if len(prediction_frame) != expected_rows or len(comparison_frame) != len(TREE_MODELS) * len(regions):
        raise BenchmarkContractError("Focused tree fidelity output grid is incomplete")
    prediction_frame.to_parquet(tables / "fresh_tree_predictions_seed42.parquet", index=False)
    comparison_frame.to_csv(tables / "prediction_reproduction_seed42.csv", index=False)
    with (tables / "fit_attestations_seed42.jsonl").open("w", encoding="utf-8") as stream:
        for row in attestations:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    final_hashes = {path: sha256(path) for path in initial_hashes}
    if final_hashes != initial_hashes:
        raise BenchmarkContractError("A source/input changed during tree fidelity validation")
    execution_completed_utc = datetime.now(timezone.utc).isoformat()
    git_end = _git_state()
    if git_end["commit"] != git_start["commit"]:
        raise BenchmarkContractError("Git commit changed during tree fidelity validation")
    reproduction_counts = comparison_frame.reproduction_status.value_counts().to_dict()
    by_model = comparison_frame.groupby("model", observed=True).agg(
        fits=("region", "size"),
        max_absolute_score_difference=("max_absolute_score_difference", "max"),
        prediction_label_mismatches=("prediction_label_mismatches", "sum"),
    ).reset_index()
    by_model.to_csv(tables / "reproduction_summary_by_model.csv", index=False)
    reproduction_summary = summarize_prediction_reproduction(comparison_frame)
    metadata = {
        "format": "poi-tree-input-fidelity-validation-v1",
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "scientific_completion": False,
        "scope": "clean train fit and clean validation prediction; five trees; seed42; 17 OvR",
        "models": sorted(TREE_MODELS), "seed": 42, "regions": regions,
        "fits": len(comparison_frame), "prediction_rows": len(prediction_frame),
        "input_representation": "raw_dataframe_full_precision",
        "category_mapping": "exact_train_string_or_unique_numeric_equivalent; ambiguity_fails",
        "reproduction_status_counts": reproduction_counts,
        **reproduction_summary,
        "train_raw_feature_frame_sha256": raw_train_sha,
        "validation_raw_feature_frame_sha256": raw_validation_sha,
        "source_input_sha256": initial_hashes,
        "runtime_environment_sha256": sha256(provenance / "runtime_environment.json"),
        "runtime_environment_path": str(output / "provenance" / "runtime_environment.json"),
        "execution_started_utc": execution_started_utc,
        "execution_completed_utc": execution_completed_utc,
        "git_at_start": git_start, "git_at_end": git_end,
        "test_feature_rows_read": 0,
        "limitations": [
            "This CPU gate covers five tree families at seed 42 only; it is not the full timing run.",
            "Fresh refits can differ from frozen library/runtime fits and every difference remains explicit.",
            "Independent review is pending and no Stage-06 completion is claimed.",
        ],
    }
    _write_json(staging / "metadata.json", metadata)
    _write_json(staging / "_GENERATED.json", {
        "status": metadata["status"], "metadata_sha256": sha256(staging / "metadata.json"),
        "predictions_sha256": sha256(tables / "fresh_tree_predictions_seed42.parquet"),
        "comparison_sha256": sha256(tables / "prediction_reproduction_seed42.csv"),
        "attestations_sha256": sha256(tables / "fit_attestations_seed42.jsonl"),
        "test_feature_rows_read": 0,
    })
    (staging / "report.md").write_text(
        "# Tree input-fidelity validation\n\n"
        "Five tree families were freshly fitted for seed 42 and 17 OvR regions using clean train. "
        "Clean validation inference consumed the original full-precision DataFrame and exact "
        "clean-train category strings. Actual fitted transformer, transformed matrix, estimator, "
        "truth, weights, and probe hashes are retained. This is pending independent review and "
        "does not read test features or complete the full inference timing stage.\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return metadata


def run(config_path: str | Path) -> None:
    """Execute the full validation-only benchmark. This is intentionally expensive."""
    execution_started_utc = datetime.now(timezone.utc).isoformat()
    git_start = _git_state()
    config_path = Path(config_path).resolve()
    cfg = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    output = Path(cfg["output"])
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite output/staging path: {output} / {staging}")
    checkpoint_contracts = validate_checkpoint_contracts(cfg["tabpfn_checkpoints"])
    tree_contract = validate_tree_thread_contract(cfg["tree_config"], cfg["tree_cpu_threads"])
    reference_path = Path(cfg["existing_clean_predictions"])
    summary_path = Path(cfg["existing_clean_macro_summary"])
    if not reference_path.is_file() or not summary_path.is_file():
        raise BenchmarkContractError("Immutable Stage06 clean comparison inputs are absent")

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("selected_explain.py").resolve(),
        Path(__file__).with_name("data.py").resolve(),
        Path(__file__).with_name("ablation.py").resolve(),
        Path(__file__).with_name("benchmark.py").resolve(),
        Path(__file__).with_name("clean_comparison.py").resolve(),
        Path(__file__).with_name("attack_evaluate.py").resolve(),
        config_path,
        Path(cfg["tree_config"]).resolve(),
    ]
    if any(not path.is_file() for path in source_paths):
        raise BenchmarkContractError("Execution source/config snapshot is incomplete")
    guarded_inputs = [*source_paths, reference_path.resolve(), summary_path.resolve()]
    initial_hashes = {str(path): sha256(path) for path in guarded_inputs}
    initial_hashes.update({value["path"]: value["sha256"] for value in checkpoint_contracts.values()})

    data_dir = Path(os.environ.get("DATA_DIR", cfg["data_dir"])).expanduser().resolve()
    dataset_files = [data_dir / name for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")]
    if any(not path.is_file() for path in dataset_files):
        raise BenchmarkContractError(f"DATA_DIR lacks frozen clean inputs: {data_dir}")
    initial_hashes.update({str(path): sha256(path) for path in dataset_files})

    staging.mkdir(parents=True)
    tables = staging / "tables"
    provenance = staging / "provenance"
    snapshots = provenance / "source_snapshots"
    for directory in (tables, provenance, snapshots):
        directory.mkdir()
    source_manifest = []
    for index, source in enumerate(source_paths):
        target = snapshots / f"{index:02d}_{source.name}"
        shutil.copyfile(source, target)
        source_manifest.append({
            "source_path": str(source),
            "snapshot_path": str(output / "provenance" / "source_snapshots" / target.name),
            "sha256": sha256(target), "size_bytes": target.stat().st_size,
        })
    source_manifest_path = provenance / "execution_source_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    source_manifest_sha = sha256(source_manifest_path)

    runtime, synchronize = _runtime_environment(str(cfg["device"]), int(cfg["tree_cpu_threads"]))
    runtime.update({
        "execution_started_utc": execution_started_utc,
        "git_commit_at_start": git_start["commit"],
        "git_dirty_at_start": git_start["dirty"],
        "git_status_porcelain_sha256_at_start": git_start["status_porcelain_sha256"],
        "git_status_porcelain_lines_at_start": git_start["status_porcelain_lines"],
    })
    runtime_path = provenance / "runtime_environment.json"
    _write_json(runtime_path, runtime)
    runtime_sha = sha256(runtime_path)

    train, _ = load_clean_splits(data_dir, splits=("train",))
    validation, _ = load_clean_splits(data_dir, splits=("validation",))
    regions = sorted(train.region.astype(str).unique())
    if (
        len(train) != int(cfg["expected_train_rows"])
        or len(validation) != int(cfg["expected_validation_rows"])
        or len(regions) != int(cfg["expected_regions"])
        or sorted(validation.region.astype(str).unique()) != regions
    ):
        raise BenchmarkContractError("Clean train/validation rows or region coverage differ")
    train_matrix = _matrix(train)
    validation_matrix = _matrix(validation)
    train_raw_sha = _raw_frame_sha256(train)
    validation_raw_sha = _raw_frame_sha256(validation)
    input_identity = {
        "format": "poi-inference-input-identity-v1",
        "data_dir": str(data_dir),
        "dataset_sha256": {str(path): initial_hashes[str(path)] for path in dataset_files},
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL),
        "train_rows": len(train), "validation_rows": len(validation),
        "regions": regions,
        "train_poi_order_sha256": _ordered_text_sha256(train, ["POI_ID"]),
        "validation_poi_order_sha256": _ordered_text_sha256(validation, ["POI_ID"]),
        "train_label_source_sha256": _ordered_text_sha256(train, ["POI_ID", "region"]),
        "validation_label_source_sha256": _ordered_text_sha256(
            validation, ["POI_ID", "region"]
        ),
        "raw_dataframe_contract": (
            "tree fit/inference preserves ordered source strings and maps only synthetic numeric "
            "categories to a unique exact clean-train vocabulary value"
        ),
        "train_raw_feature_frame_sha256": train_raw_sha,
        "validation_raw_feature_frame_sha256": validation_raw_sha,
        "raw_feature_dtypes": {name: str(train[name].dtype) for name in FEATURES},
        "tabpfn_train_float32_matrix_sha256": _array_sha256(train_matrix, np.float32),
        "tabpfn_validation_float32_matrix_sha256": _array_sha256(validation_matrix, np.float32),
        "test_feature_rows_read": 0,
    }
    input_identity_path = provenance / "input_identity.json"
    _write_json(input_identity_path, input_identity)
    input_identity_sha = sha256(input_identity_path)

    reference = pd.read_parquet(reference_path)
    frozen_metrics = _validate_reference(reference, validation, cfg)
    frozen_summary = pd.read_csv(summary_path)
    performance = _performance_from_frozen_summary(
        frozen_summary, frozen_metrics, cfg["models"],
        float(cfg["frozen_metric_tolerance"]),
    )
    frozen_metrics.to_csv(tables / "frozen_reference_recomputed_metrics.csv", index=False)

    reference_groups = reference.groupby(list(FIT_IDENTITY), sort=False, observed=True)
    from .selected_explain import ModelFactory
    from threadpoolctl import threadpool_limits
    import pyarrow as pa
    import pyarrow.parquet as pq

    predictions_path = tables / "actual_predictions.parquet"
    parquet_writer: pq.ParquetWriter | None = None
    timing_rows: list[dict[str, Any]] = []
    timing_repeat_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    attestation_path = tables / "fit_attestations.jsonl"
    expected_fits = len(cfg["models"]) * len(cfg["seeds"]) * len(regions)
    try:
        with attestation_path.open("w", encoding="utf-8") as attestation_stream:
            for seed in cfg["seeds"]:
                for region in regions:
                    truth_train = train.region.astype(str).eq(region).astype(int).to_numpy()
                    truth_validation = validation.region.astype(str).eq(region).astype(int).to_numpy()
                    if set(np.unique(truth_train)) != {0, 1} or set(np.unique(truth_validation)) != {0, 1}:
                        raise BenchmarkContractError(f"OvR labels lack both classes for {region}")
                    for model in cfg["models"]:
                        model_cfg: dict[str, Any] = {
                            "seed": int(seed),
                            "tree_config": str(Path(cfg["tree_config"]).resolve()),
                            "device": str(cfg["device"]),
                            "n_estimators": int(cfg["tabpfn_n_estimators"]),
                            "require_checkpoint_attestation": model in TABPFN_MODELS,
                            "repeat_prediction_absolute_tolerance": float(
                                cfg["prediction_comparison"]["absolute_tolerance"]
                            ),
                        }
                        if model in TABPFN_MODELS:
                            model_cfg.update({
                                "official_checkpoint": checkpoint_contracts[model]["path"],
                                "expected_checkpoint_sha256": checkpoint_contracts[model]["sha256"],
                            })
                        factory = ModelFactory(model, model_cfg, train)
                        context = threadpool_limits(limits=2) if model in TREE_MODELS else nullcontext()
                        with context:
                            synchronize()
                            fit_started = time.perf_counter()
                            predictor = factory.fit(FEATURES, truth_train)
                            synchronize()
                            fit_seconds = float(time.perf_counter() - fit_started)
                            if fit_seconds <= 0 or not math.isfinite(fit_seconds):
                                raise BenchmarkContractError("Fit duration must be finite and positive")
                            score, timing, repeat_rows = measure_prediction_repeats(
                                predictor,
                                validation if model in TREE_MODELS else validation_matrix,
                                measured_repeats=int(cfg["timing"]["measured_repeats"]),
                                repeat_absolute_tolerance=float(
                                    cfg["prediction_comparison"]["absolute_tolerance"]
                                ),
                                synchronize=synchronize,
                            )
                        if ((score < 0) | (score > 1)).any():
                            raise BenchmarkContractError("Model probability lies outside [0,1]")
                        prediction = (score >= float(cfg["classification_threshold"])).astype(int)
                        actual = pd.DataFrame({
                            "POI_ID": validation.POI_ID.astype(str).to_numpy(),
                            "model": model, "seed": int(seed), "region": region,
                            "condition": "clean", "split": "validation",
                            "y_true": truth_validation, "y_score": score, "y_pred": prediction,
                        })
                        reference_fit = reference_groups.get_group((model, int(seed), region))
                        comparison = compare_fit_predictions(
                            actual, reference_fit,
                            absolute_tolerance=float(cfg["prediction_comparison"]["absolute_tolerance"]),
                            relative_tolerance=float(cfg["prediction_comparison"]["relative_tolerance"]),
                        )
                        identity = {"model": model, "seed": int(seed), "region": region}
                        comparison_rows.append({**identity, **comparison})
                        metrics = _five_metrics(
                            truth_validation, score, float(cfg["classification_threshold"])
                        )
                        metric_rows.append({**identity, "condition": "clean", **metrics})
                        timing_rows.append({**identity, "fit_seconds": fit_seconds, **timing})
                        timing_repeat_rows.extend({**identity, **row} for row in repeat_rows)

                        table = pa.Table.from_pandas(actual, preserve_index=False)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(predictions_path, table.schema)
                        parquet_writer.write_table(table)

                        raw_factory_attestation = dict(factory.last_fit_attestation)
                        if model in TABPFN_MODELS and (
                            raw_factory_attestation.get("model_id") != model
                            or raw_factory_attestation.get("official_checkpoint_sha256")
                            != checkpoint_contracts[model]["sha256"]
                            or raw_factory_attestation.get("feature_order") != list(FEATURES)
                            or raw_factory_attestation.get("train_rows") != len(train)
                            or raw_factory_attestation.get("truth_sha256")
                            != _array_sha256(truth_train, np.int8)
                        ):
                            raise BenchmarkContractError(
                                f"Actual ModelFactory attestation is incomplete for {model}/{seed}/{region}"
                            )
                        if model in TREE_MODELS and (
                            raw_factory_attestation.get("format") != "poi-tree-fit-attestation-v1"
                            or raw_factory_attestation.get("model_id") != model
                            or raw_factory_attestation.get("feature_order") != list(FEATURES)
                            or raw_factory_attestation.get("train_rows") != len(train)
                            or raw_factory_attestation.get("raw_train_frame_sha256") != train_raw_sha
                            or raw_factory_attestation.get("truth_sha256")
                            != _array_sha256(truth_train, np.int8)
                            or not raw_factory_attestation.get("transformed_train", {}).get("sha256")
                            or not raw_factory_attestation.get("transformer_fitted_state_sha256")
                            or not raw_factory_attestation.get("estimator_fitted_state_sha256")
                        ):
                            raise BenchmarkContractError(
                                f"Actual tree fit attestation is incomplete for {model}/{seed}/{region}"
                            )
                        attestation = {
                            "format": "poi-inference-fit-attestation-v1",
                            **identity,
                            "factory_class": "poi.selected_explain.ModelFactory",
                            "model_factory_attestation": raw_factory_attestation,
                            "model_factory_runtime": dict(factory.tabpfn_runtime),
                            "execution_source_manifest_sha256": source_manifest_sha,
                            "runtime_environment_sha256": runtime_sha,
                            "input_identity_sha256": input_identity_sha,
                            "train_poi_order_sha256": input_identity["train_poi_order_sha256"],
                            "validation_poi_order_sha256": input_identity[
                                "validation_poi_order_sha256"
                            ],
                            "fit_input_representation": (
                                "raw_dataframe_full_precision" if model in TREE_MODELS
                                else "tabpfn_float32_matrix"
                            ),
                            "train_fit_input_sha256": (
                                input_identity["train_raw_feature_frame_sha256"]
                                if model in TREE_MODELS else
                                input_identity["tabpfn_train_float32_matrix_sha256"]
                            ),
                            "validation_inference_input_sha256": (
                                input_identity["validation_raw_feature_frame_sha256"]
                                if model in TREE_MODELS else
                                input_identity["tabpfn_validation_float32_matrix_sha256"]
                            ),
                            "train_label_source": "ordered clean train region == target_region",
                            "validation_label_source": (
                                "ordered clean validation region == target_region"
                            ),
                            "train_truth_sha256": _array_sha256(truth_train, np.int8),
                            "validation_truth_sha256": _array_sha256(
                                truth_validation, np.int8
                            ),
                            "fit_seconds": fit_seconds,
                            "timing": timing,
                            "reference_comparison": comparison,
                            "test_feature_rows_read": 0,
                        }
                        attestation_stream.write(json.dumps(attestation, sort_keys=True) + "\n")
                        del predictor, factory
                        gc.collect()
        if parquet_writer is None:
            raise BenchmarkContractError("No prediction fits were written")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if len(timing_rows) != expected_fits or len(metric_rows) != expected_fits:
        raise BenchmarkContractError("Benchmark did not complete the exact 8×3×17 fit grid")
    timing_frame = pd.DataFrame(timing_rows)
    repeat_frame = pd.DataFrame(timing_repeat_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metric_reproduction = metrics_frame.merge(
        frozen_metrics,
        on=list(FIT_IDENTITY),
        suffixes=("_actual", "_frozen"),
        validate="one_to_one",
    )
    for metric in METRICS:
        metric_reproduction[f"{metric}_actual_minus_frozen"] = (
            metric_reproduction[f"{metric}_actual"]
            - metric_reproduction[f"{metric}_frozen"]
        )
    timing_summary = aggregate_timing_rows(
        timing_frame, models=cfg["models"],
        expected_fits_per_model=len(cfg["seeds"]) * len(regions),
    )
    candidate_table, selection = select_by_protocol_rule(
        performance[["model", "macro_f1"]], timing_summary,
        models=cfg["models"],
        absolute_f1_tolerance=float(cfg["selection_rule"]["absolute_f1_tolerance"]),
    )
    reproduction_counts = comparisons.reproduction_status.value_counts().to_dict()
    selection.update({
        "immutable_performance_file": str(summary_path),
        "immutable_performance_file_sha256": initial_hashes[str(summary_path.resolve())],
        "immutable_predictions_file": str(reference_path),
        "immutable_predictions_file_sha256": initial_hashes[str(reference_path.resolve())],
        "all_fresh_scores_within_configured_tolerance": bool(
            comparisons.scores_within_configured_tolerance.all()
        ),
        "fresh_prediction_reproduction_status_counts": reproduction_counts,
        "existing_common_downstream": cfg["existing_common_downstream"],
    })

    timing_frame.to_csv(tables / "per_fit_timing.csv", index=False)
    repeat_frame.to_csv(tables / "timing_repeats.csv", index=False)
    timing_summary.to_csv(tables / "model_timing_summary.csv", index=False)
    metrics_frame.to_csv(tables / "actual_recomputed_metrics.csv", index=False)
    metric_reproduction.to_csv(tables / "metric_reproduction.csv", index=False)
    comparisons.to_csv(tables / "prediction_reproduction.csv", index=False)
    performance.to_csv(tables / "frozen_performance_validation.csv", index=False)
    candidate_table.to_csv(tables / "selection_candidates.csv", index=False)
    _write_json(staging / "selection_recommendation.json", selection)

    final_hashes = {path: sha256(path) for path in initial_hashes}
    if final_hashes != initial_hashes:
        changed = sorted(path for path in initial_hashes if final_hashes[path] != initial_hashes[path])
        raise BenchmarkContractError(f"Source/input/checkpoint changed during benchmark: {changed}")

    execution_completed_utc = datetime.now(timezone.utc).isoformat()
    git_end = _git_state()
    if git_end["commit"] != git_start["commit"]:
        raise BenchmarkContractError("Git commit changed during inference benchmark")
    execution_metadata = {
        "format": "poi-inference-execution-metadata-v1",
        "execution_started_utc": execution_started_utc,
        "execution_completed_utc": execution_completed_utc,
        "git_at_start": git_start,
        "git_at_end": git_end,
        "git_commit_unchanged": True,
        "dirty_state_can_include_new_staging_evidence": True,
        "test_feature_rows_read": 0,
    }
    execution_metadata_path = provenance / "execution_metadata.json"
    _write_json(execution_metadata_path, execution_metadata)
    execution_metadata_sha = sha256(execution_metadata_path)

    metadata = {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scope": "clean train fitting and clean validation inference only",
        "models": list(cfg["models"]), "seeds": list(cfg["seeds"]), "regions": regions,
        "fits": expected_fits, "measured_prediction_repeats_per_fit": 5,
        "validation_rows_per_repeat": len(validation),
        "timing_primary": (
            "median of five synchronized full-validation prediction calls per fit, then equal "
            "mean over 3 seeds x 17 regions"
        ),
        "warmup_and_fit_excluded_from_primary_inference_time": True,
        "tree_cpu_thread_budget": int(cfg["tree_cpu_threads"]),
        "device": cfg["device"],
        "runtime_environment_sha256": runtime_sha,
        "execution_source_manifest_sha256": source_manifest_sha,
        "execution_metadata_sha256": execution_metadata_sha,
        "execution_started_utc": execution_started_utc,
        "execution_completed_utc": execution_completed_utc,
        "git_commit": git_start["commit"],
        "git_dirty_at_start": git_start["dirty"],
        "git_dirty_at_end": git_end["dirty"],
        "input_identity_sha256": input_identity_sha,
        "checkpoint_contracts": checkpoint_contracts,
        "tree_contract": {key: value for key, value in tree_contract.items() if key != "models"},
        "selection": selection,
        "test_feature_rows_read": 0,
        "explicit_common_downstream_modified": False,
        "limitations": [
            "Times are five repeated full-batch samples on one host, not online request latency.",
            "Amortized per-row time is not a p50/p95/p99 latency distribution.",
            "Sequential fitting can retain cache, thermal, and system-load effects despite warmup.",
            "One GPU host does not establish cross-hardware or cross-runtime performance.",
            "Per-fit timing CV is descriptive with only five measured repeats.",
            "Tree estimators share a two-thread ceiling; a single-threaded algorithm may use less.",
            "Selection remains conditional on the same clean validation data and its fitted models.",
            "Fresh prediction differences are disclosed and never relabeled as exact reproduction.",
            "This recommendation does not overwrite the explicit common TabPFN2.5 downstream choice.",
            "Historical locked-test coordinate access remains disclosed; no test features are read here.",
        ],
    }
    _write_json(staging / "metadata.json", metadata)
    shutil.copyfile(config_path, staging / "config.yaml")
    (staging / "report.md").write_text(
        "# Eight-model inference timing and reproducibility\n\n"
        "This validation-only benchmark refits 8 models × 3 seeds × 17 OvR tasks on one fixed "
        "host. Each fit receives one excluded warmup and five synchronized predictions over all "
        "3,400 validation rows. The primary model time equally averages the 51 per-fit medians; "
        "fit time is reported separately.\n\n"
        f"The protocol-rule recommendation is `{selection['selected_model']}`. Candidates are "
        "models within an inclusive 0.005 absolute Macro-F1 gap from the immutable clean-validation "
        "leader; timing only breaks ties inside that band. This is pending independent review and "
        "does not modify the explicit common TabPFN2.5 downstream choice.\n\n"
        "Fresh predictions, five metrics, input hashes, per-fit ModelFactory attestations, raw "
        "timings, CVs, and exact/tolerant/different reproduction labels are retained. These times "
        "are full-batch samples from one host and are not a request-latency distribution. No locked "
        "test features were read.\n",
        encoding="utf-8",
    )
    _write_json(staging / "_GENERATED.json", {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "metadata_sha256": sha256(staging / "metadata.json"),
        "selection_recommendation_sha256": sha256(staging / "selection_recommendation.json"),
        "actual_predictions_sha256": sha256(predictions_path),
        "fit_attestations_sha256": sha256(attestation_path),
        "execution_metadata_sha256": execution_metadata_sha,
        "test_feature_rows_read": 0,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--tree-fidelity-output",
        help="Run the CPU-only five-tree seed-42 input-fidelity gate instead of full timing",
    )
    args = parser.parse_args()
    if args.tree_fidelity_output:
        result = run_tree_fidelity_validation(args.config, args.tree_fidelity_output)
        print(json.dumps(result, indent=2))
    else:
        run(args.config)


if __name__ == "__main__":
    main()
