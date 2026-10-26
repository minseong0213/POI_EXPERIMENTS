"""Prospective clean comparison for three frozen feature sets.

This module is intentionally separate from the legacy clean and ablation
runners.  It freezes an aggregate source contract, authenticates the imported
and newly fitted prediction sources, and aggregates the three feature sets on
one validation calibration/evaluation partition.  It does not fit models and
never reads test features.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import yaml

from .data import sha256
from .no_coordinate_experiment import (
    _atomic_frame,
    _atomic_json,
    _bound_bytes,
    _load_clean_splits_bound,
    _ordered_ids_sha,
    _snapshot_files,
    load_common_partition,
    select_f1_threshold,
    weighted_average_precision_replicates,
)
from .uncertainty import make_bootstrap_weights, weighted_metric_replicates


FEATURE_SETS = {
    "full": ("X_COORD", "Y_COORD", "ASORT_LCLASDC", "ASORT_MLSFCDC", "ASORT_SDASDC"),
    "coordinates_only": ("X_COORD", "Y_COORD"),
    "no_coordinates": ("ASORT_LCLASDC", "ASORT_MLSFCDC", "ASORT_SDASDC"),
}
TREE_MODELS = ("decision_tree", "random_forest", "xgboost", "lightgbm", "catboost")
MODELS = (*TREE_MODELS, "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3")
SEEDS = (42, 202, 340)
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc", "average_precision")
SOURCE_FORMAT = "poi-unified-clean-aggregate-source-freeze-v1"
FIT_FORMAT = "poi-unified-clean-actual-fit-v1"
AGGREGATE_FORMAT = "poi-unified-clean-three-feature-comparison-v1"


class UnifiedCleanError(RuntimeError):
    """Raised when the prospective clean comparison fails closed."""


def _canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _strict_binary(values: Any, name: str) -> np.ndarray:
    raw = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(np.float64)
    if not np.isfinite(raw).all() or not np.isin(raw, [0.0, 1.0]).all():
        raise UnifiedCleanError(f"{name} must contain exact finite binary values before casting")
    return raw.astype(np.int8)


def _strict_scores(values: Any) -> np.ndarray:
    score = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(np.float64)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise UnifiedCleanError("Scores must be finite probabilities in [0, 1]")
    return score


def _read_yaml_bound(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = {str(path.resolve()): {"sha256": sha256(path), "size_bytes": path.stat().st_size}}
    payload = _bound_bytes(path, before)
    value = yaml.safe_load(payload)
    if not isinstance(value, dict):
        raise UnifiedCleanError("Configuration must be a YAML mapping")
    value["_config_path"] = str(path.resolve())
    value["_config_load_snapshot"] = before[str(path.resolve())]
    return value, before[str(path.resolve())]


def _env_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply path-only deployment overrides; bytes remain bound by the fit identity."""
    cfg = dict(cfg)
    if os.environ.get("UNIFIED_RUN_ROOT"):
        cfg["run_root"] = os.environ["UNIFIED_RUN_ROOT"]
    if os.environ.get("DATA_DIR"):
        cfg["data_dir"] = os.environ["DATA_DIR"]
    if os.environ.get("TABPFN26_CHECKPOINT"):
        cfg["tabpfn_v2_6"] = dict(cfg["tabpfn_v2_6"])
        cfg["tabpfn_v2_6"]["official_checkpoint"] = os.environ["TABPFN26_CHECKPOINT"]
    if os.environ.get("UNIFIED_SOURCE_FREEZE"):
        cfg["source_freeze_manifest"] = os.environ["UNIFIED_SOURCE_FREEZE"]
    if os.environ.get("UNIFIED_SOURCE_REVIEW"):
        cfg["source_review_verdict"] = os.environ["UNIFIED_SOURCE_REVIEW"]
    root = Path(cfg["run_root"])
    cfg["tree_output"] = str(root / cfg["outputs"]["tree_relative"])
    cfg["tabpfn26_output"] = str(root / cfg["outputs"]["tabpfn26_relative"])
    cfg["full_v3_output"] = str(root / cfg["outputs"]["full_v3_relative"])
    cfg["aggregate_output"] = str(root / cfg["outputs"]["aggregate_relative"])
    return cfg


def _source_catalog(cfg: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    catalog: dict[str, Path] = {}
    for label in cfg["source_files"]:
        path = (root / str(label)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source dependency absent: {path}")
        catalog[str(label)] = path
    config = Path(str(cfg["_config_path"])).resolve()
    catalog[f"config::{config.name}"] = config
    protocol = Path(str(cfg["protocol_file"])).resolve()
    catalog[f"protocol::{protocol.name}"] = protocol
    return catalog


def freeze_source(cfg: Mapping[str, Any]) -> dict[str, Any]:
    catalog = _source_catalog(cfg)
    files = {
        label: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for label, path in sorted(catalog.items())
    }
    payload = {
        "format": SOURCE_FORMAT,
        "status": "FROZEN_PENDING_INDEPENDENT_SOURCE_REVIEW",
        "scientific_completion": False,
        "files": files,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "test_rows_read": 0,
        "test_features_read": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = Path(str(cfg["source_freeze_manifest"]))
    if target.exists():
        old = json.loads(target.read_text())
        comparable = {key: value for key, value in old.items() if key != "created_utc"}
        wanted = {key: value for key, value in payload.items() if key != "created_utc"}
        if comparable != wanted:
            raise UnifiedCleanError("Existing source freeze differs; use a new run root")
        return old
    _atomic_json(payload, target)
    return payload


def verify_source_gate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(cfg["source_freeze_manifest"]))
    verdict_path = Path(str(cfg["source_review_verdict"]))
    if not manifest_path.is_file() or not verdict_path.is_file():
        raise UnifiedCleanError("Source freeze and fresh independent PASS verdict are required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != SOURCE_FORMAT:
        raise UnifiedCleanError("Source freeze format differs")
    current = _source_catalog(cfg)
    if set(current) != set(manifest.get("files", {})):
        raise UnifiedCleanError("Source freeze catalog differs from configured dependencies")
    for label, path in current.items():
        entry = manifest["files"][label]
        if entry.get("sha256") != sha256(path) or int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise UnifiedCleanError(f"Frozen source bytes changed: {label}")
    verdict = json.loads(verdict_path.read_text())
    status = verdict.get("status", verdict.get("overall_verdict", verdict.get("verdict")))
    if status != "PASS" or verdict.get("review_scope") != "unified_clean_aggregate_source_before_run":
        raise UnifiedCleanError("Fresh independent source review did not PASS the exact scope")
    if verdict.get("reviewed_source_manifest_sha256") != sha256(manifest_path):
        raise UnifiedCleanError("Source review is not bound to the current freeze manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "verdict_path": str(verdict_path.resolve()),
        "verdict_sha256": sha256(verdict_path),
    }


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    validation: pd.DataFrame,
    expected_models: Sequence[str],
) -> pd.DataFrame:
    """Validate one complete feature-set design before integer coercion."""
    required = {"POI_ID", "model", "seed", "region", "y_true", "y_score"}
    if not required.issubset(frame.columns):
        raise UnifiedCleanError(f"Prediction columns absent: {sorted(required - set(frame))}")
    result = frame.copy()
    result["POI_ID"] = result["POI_ID"].astype(str)
    if "ablation" in result and not result["ablation"].astype(str).eq(feature_set).all():
        raise UnifiedCleanError("Legacy ablation label differs from requested feature set")
    for column in ("condition", "attack_condition"):
        if column in result and not result[column].astype(str).eq("clean").all():
            raise UnifiedCleanError("Only clean prediction rows may enter this stage")
    if "split" in result and not result["split"].astype(str).eq("validation").all():
        raise UnifiedCleanError("Only validation prediction rows may enter this stage")
    result["y_true"] = _strict_binary(result["y_true"], "y_true")
    result["y_score"] = _strict_scores(result["y_score"])
    if "y_pred" in result:
        raw = _strict_binary(result["y_pred"], "y_pred")
        if not np.array_equal(raw, (result["y_score"].to_numpy() >= 0.5).astype(np.int8)):
            raise UnifiedCleanError("Legacy y_pred differs from the fixed 0.5 rule")
    if "y_pred_fixed_0p5" in result:
        raw = _strict_binary(result["y_pred_fixed_0p5"], "y_pred_fixed_0p5")
        if not np.array_equal(raw, (result["y_score"].to_numpy() >= 0.5).astype(np.int8)):
            raise UnifiedCleanError("Fixed 0.5 prediction column differs")
    expected_ids = set(validation["POI_ID"].astype(str))
    regions = tuple(sorted(validation["region"].astype(str).unique()))
    expected_groups = {(model, seed, region) for model in expected_models for seed in SEEDS for region in regions}
    actual_groups = set(result[["model", "seed", "region"]].itertuples(index=False, name=None))
    if actual_groups != expected_groups:
        raise UnifiedCleanError("Model/seed/region Cartesian product differs")
    if result.duplicated(["model", "seed", "region", "POI_ID"]).any():
        raise UnifiedCleanError("Duplicate prediction identity")
    sizes = result.groupby(["model", "seed", "region"], observed=True).size()
    if not sizes.eq(len(validation)).all():
        raise UnifiedCleanError("Prediction group row coverage differs")
    seen = result.groupby(["model", "seed", "region"], observed=True)["POI_ID"].agg(lambda x: set(x))
    if any(values != expected_ids for values in seen):
        raise UnifiedCleanError("Prediction group POI coverage differs")
    region_lookup = validation.set_index("POI_ID")["region"].astype(str)
    expected_truth = result["POI_ID"].map(region_lookup).eq(result["region"].astype(str)).astype(np.int8)
    if expected_truth.isna().any() or not np.array_equal(result["y_true"].to_numpy(), expected_truth.to_numpy()):
        raise UnifiedCleanError("Prediction truth differs from validation identity")
    result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(np.int64)
    result["feature_set"] = feature_set
    result["condition"] = "clean"
    result["split"] = "validation"
    result["y_pred_fixed_0p5"] = (result["y_score"].to_numpy() >= 0.5).astype(np.int8)
    return result[["POI_ID", "feature_set", "model", "seed", "region", "condition", "split", "y_true", "y_score", "y_pred_fixed_0p5"]]


def _read_parquet_bound(path: Path, snapshot: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_bound_bytes(path, snapshot)))


def _shard_paths(root: Path, models: Sequence[str], feature_set: str, regions: Sequence[str]) -> tuple[list[Path], list[Path]]:
    predictions = [root / "predictions" / f"{model}_{feature_set}_{seed}_{region}_clean.parquet" for model in models for seed in SEEDS for region in regions]
    markers = [root / "work" / f"{model}_{feature_set}_{seed}_{region}.json" for model in models for seed in SEEDS for region in regions]
    return predictions, markers


def _read_shards(
    root: Path,
    models: Sequence[str],
    feature_set: str,
    validation: pd.DataFrame,
    snapshot: Mapping[str, Mapping[str, Any]],
    *,
    legacy: bool,
) -> pd.DataFrame:
    regions = tuple(sorted(validation["region"].astype(str).unique()))
    prediction_paths, marker_paths = _shard_paths(root, models, feature_set, regions)
    identity = json.loads(_bound_bytes(root / "input_identity.json", snapshot))
    fingerprint = identity.get("sha256") if legacy else identity.get("fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise UnifiedCleanError("Coordinate fit identity fingerprint absent")
    parts = []
    for prediction_path, marker_path in zip(prediction_paths, marker_paths):
        marker = json.loads(_bound_bytes(marker_path, snapshot))
        marker_fingerprint = marker.get("fingerprint") if legacy else marker.get("fingerprint_sha256")
        if marker_fingerprint != fingerprint:
            raise UnifiedCleanError(f"Coordinate marker fingerprint differs: {marker_path}")
        relative = str(prediction_path.relative_to(root))
        files = marker.get("files") if legacy else marker.get("files_sha256")
        if not isinstance(files, dict) or files.get(relative) != snapshot[str(prediction_path.resolve())]["sha256"]:
            raise UnifiedCleanError(f"Coordinate marker does not bind prediction: {prediction_path}")
        parts.append(_read_parquet_bound(prediction_path, snapshot))
    return normalize_prediction_frame(pd.concat(parts, ignore_index=True), feature_set=feature_set, validation=validation, expected_models=models)


def _point_row(truth: np.ndarray, score: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "f1": f1_score(truth, prediction, zero_division=0),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "accuracy": accuracy_score(truth, prediction),
        "roc_auc": roc_auc_score(truth, score),
        "average_precision": average_precision_score(truth, score),
        "tn": int(((truth == 0) & (prediction == 0)).sum()),
        "fp": int(((truth == 0) & (prediction == 1)).sum()),
        "fn": int(((truth == 1) & (prediction == 0)).sum()),
        "tp": int(((truth == 1) & (prediction == 1)).sum()),
        "predicted_positive_rate": float(prediction.mean()),
    }


def analyze_feature_predictions(
    frame: pd.DataFrame,
    *,
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Select thresholds on calibration IDs and score/boot evaluation IDs."""
    partition = {str(value): "calibration" for value in calibration_ids}
    partition.update({str(value): "evaluation" for value in evaluation_ids})
    frame = frame.copy()
    frame["partition"] = frame["POI_ID"].map(partition)
    if frame["partition"].isna().any():
        raise UnifiedCleanError("Prediction POI is outside the common partition")
    threshold_rows, metric_rows = [], []
    aggregate: dict[tuple[str, str], np.ndarray] = {}
    eval_ids = np.asarray(list(map(str, evaluation_ids)), dtype=object)
    feature_set = str(frame["feature_set"].iloc[0])
    for (model, model_seed, region), group in frame.groupby(["model", "seed", "region"], sort=True):
        calibration = group.loc[group["partition"].eq("calibration")]
        selected = select_f1_threshold(calibration["y_true"].to_numpy(), calibration["y_score"].to_numpy())
        threshold = float(selected["threshold"])
        threshold_rows.append({"feature_set": feature_set, "model": model, "seed": int(model_seed), "region": region, "threshold_source": "common_validation_calibration_ids_only", **selected})
        full_truth = group["y_true"].to_numpy(np.int8)
        full_score = group["y_score"].to_numpy(np.float64)
        metric_rows.append({"feature_set": feature_set, "model": model, "seed": int(model_seed), "region": region, "partition": "full_validation", "threshold_rule": "fixed_0p5_full_validation", "threshold": 0.5, "rows": len(full_truth), "coverage": 1.0, **_point_row(full_truth, full_score, (full_score >= 0.5).astype(np.int8))})
        evaluation = group.loc[group["partition"].eq("evaluation")].set_index("POI_ID", verify_integrity=True).loc[eval_ids]
        truth = evaluation["y_true"].to_numpy(np.int8)
        score = evaluation["y_score"].to_numpy(np.float64)
        for rule, prediction in (
            ("calibrated_f1", (score >= threshold).astype(np.int8)),
            ("fixed_0p5", (score >= 0.5).astype(np.int8)),
        ):
            metric_rows.append({"feature_set": feature_set, "model": model, "seed": int(model_seed), "region": region, "partition": "evaluation", "threshold_rule": rule, "threshold": threshold if rule == "calibrated_f1" else 0.5, "rows": len(truth), "coverage": 1.0, **_point_row(truth, score, prediction)})
        prediction = (score >= threshold).astype(np.int8)
        values = weighted_metric_replicates(truth, score, prediction, weights)
        values["average_precision"] = weighted_average_precision_replicates(truth, score, weights)
        for metric in METRICS:
            aggregate.setdefault((f"{feature_set}|{model}", metric), np.zeros(weights.shape[0], dtype=np.float64))
            aggregate[(f"{feature_set}|{model}", metric)] += values[metric]
    strata = len(SEEDS) * frame["region"].nunique()
    aggregate = {key: value / strata for key, value in aggregate.items()}
    return pd.DataFrame(threshold_rows), pd.DataFrame(metric_rows), aggregate


def summarize_bootstrap(
    metrics: pd.DataFrame,
    replicates: Mapping[tuple[str, str], np.ndarray],
    *,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = (1 - confidence_level) / 2
    calibrated = metrics.loc[metrics["threshold_rule"].eq("calibrated_f1")]
    macro = calibrated.groupby(["feature_set", "model"], sort=True)[list(METRICS)].mean()
    replicate_rows, interval_rows = [], []
    for (pipeline, metric), values in sorted(replicates.items()):
        feature_set, model = pipeline.split("|", 1)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise UnifiedCleanError(f"No finite bootstrap draws: {pipeline}/{metric}")
        replicate_rows.extend({"feature_set": feature_set, "model": model, "metric": metric, "replicate": index, "value": float(value)} for index, value in enumerate(values))
        interval_rows.append({
            "feature_set": feature_set,
            "model": model,
            "metric": metric,
            "estimate": float(macro.loc[(feature_set, model), metric]),
            "bootstrap_mean": float(finite.mean()),
            "ci_low": float(np.quantile(finite, alpha)),
            "ci_high": float(np.quantile(finite, 1 - alpha)),
            "confidence_level": confidence_level,
            "bootstrap_repeats": len(values),
            "finite_repeats": len(finite),
            "undefined_repeats": len(values) - len(finite),
            "bootstrap_unit": "POI_ID",
            "pairing": "shared evaluation POI multiplicities across all 24 pipelines",
            "aggregation": "equal_mean_over_3_model_seeds_and_17_regions",
            "interval_method": "paired_POI_percentile_conditional_on_fits_and_calibration_thresholds",
        })
    pair_rows = []
    pipelines = [(feature_set, model) for feature_set in FEATURE_SETS for model in MODELS]
    for metric in METRICS:
        for (feature_a, model_a), (feature_b, model_b) in combinations(pipelines, 2):
            a = replicates[(f"{feature_a}|{model_a}", metric)]
            b = replicates[(f"{feature_b}|{model_b}", metric)]
            finite = np.isfinite(a) & np.isfinite(b)
            delta = a[finite] - b[finite]
            if not len(delta):
                raise UnifiedCleanError("A paired pipeline contrast has no finite draws")
            pair_rows.append({
                "feature_set_a": feature_a,
                "model_a": model_a,
                "feature_set_b": feature_b,
                "model_b": model_b,
                "metric": metric,
                "estimate_difference_a_minus_b": float(macro.loc[(feature_a, model_a), metric] - macro.loc[(feature_b, model_b), metric]),
                "bootstrap_mean_difference": float(delta.mean()),
                "ci_low": float(np.quantile(delta, alpha)),
                "ci_high": float(np.quantile(delta, 1 - alpha)),
                "probability_difference_gt_zero": float(np.mean(delta > 0)),
                "confidence_level": confidence_level,
                "bootstrap_repeats": len(a),
                "finite_paired_repeats": len(delta),
                "undefined_paired_repeats": len(a) - len(delta),
                "p_value": np.nan,
                "p_value_interpretation": "not_computed_bootstrap_interval_is_not_null_calibrated",
            })
    return pd.DataFrame(replicate_rows), pd.DataFrame(interval_rows), pd.DataFrame(pair_rows)


def _save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)


def _plots(metrics: pd.DataFrame, intervals: pd.DataFrame, coverage: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    calibrated = metrics.loc[metrics["threshold_rule"].eq("calibrated_f1")]
    pipeline_order = [f"{feature}|{model}" for feature in FEATURE_SETS for model in MODELS]
    for metric in METRICS:
        subset = intervals.loc[intervals["metric"].eq(metric)].copy()
        subset["pipeline"] = subset["feature_set"] + "|" + subset["model"]
        subset = subset.set_index("pipeline").loc[pipeline_order].reset_index()
        fig, axis = plt.subplots(figsize=(11, 10))
        y = np.arange(len(subset))
        axis.hlines(y, subset["ci_low"], subset["ci_high"], color="black", linewidth=1)
        axis.scatter(subset["estimate"], y, c=pd.Categorical(subset["feature_set"], categories=list(FEATURE_SETS)).codes, cmap="viridis")
        axis.set(yticks=y, yticklabels=subset["pipeline"], xlabel=metric, title=f"Clean evaluation | calibrated {metric} | paired POI 95% CI", xlim=(0, 1))
        axis.grid(axis="x", alpha=.25)
        _save_figure(fig, figures / f"macro_{metric}_bootstrap_ci.png")
        for feature_set in FEATURE_SETS:
            part = calibrated.loc[calibrated["feature_set"].eq(feature_set)]
            heat = part.pivot_table(index="region", columns="model", values=metric, aggfunc="mean").reindex(columns=MODELS)
            fig, axis = plt.subplots(figsize=(14, 9))
            sns.heatmap(heat, vmin=0, vmax=1, cmap="viridis", ax=axis)
            axis.set_title(f"{feature_set} | evaluation calibrated {metric} | seed mean")
            _save_figure(fig, figures / f"regional_{feature_set}_{metric}_heatmap.png")
    fixed = metrics.loc[metrics["threshold_rule"].eq("fixed_0p5")].groupby(["feature_set", "model"])[list(METRICS)].mean()
    selected = calibrated.groupby(["feature_set", "model"])[list(METRICS)].mean()
    for metric in METRICS:
        delta = (selected[metric] - fixed[metric]).unstack("feature_set").reindex(index=MODELS, columns=list(FEATURE_SETS))
        fig, axis = plt.subplots(figsize=(10, 6))
        delta.plot(kind="bar", ax=axis)
        axis.set(ylabel=f"calibrated minus fixed 0.5 {metric}", title="Same evaluation POIs: threshold-rule effect")
        axis.axhline(0, color="black", linewidth=.8)
        _save_figure(fig, figures / f"threshold_effect_{metric}.png")
    fig, axis = plt.subplots(figsize=(12, 5))
    plot = coverage.copy()
    plot["pipeline"] = plot["feature_set"] + "|" + plot["model"]
    sns.barplot(data=plot, x="pipeline", y="coverage", hue="partition", ax=axis)
    axis.set(ylim=(0, 1.05), title="Clean prediction coverage by pipeline and partition")
    axis.tick_params(axis="x", rotation=80)
    _save_figure(fig, figures / "pipeline_coverage.png")


def _write_report(
    path: Path,
    macro_seed_summary: pd.DataFrame,
    paired_differences: pd.DataFrame,
    cardinality: Mapping[str, int],
) -> None:
    """Write the bounded clean result interpretation from generated tables."""
    calibrated = macro_seed_summary.loc[
        macro_seed_summary["partition"].eq("evaluation")
        & macro_seed_summary["threshold_rule"].eq("calibrated_f1")
    ].copy()
    full = calibrated.loc[calibrated["feature_set"].eq("full")].sort_values(
        "f1_mean", ascending=False,
    )
    leader = full.iloc[0]
    runner_up = full.iloc[1]
    difference = float(leader["f1_mean"] - runner_up["f1_mean"])
    pair = paired_differences.loc[
        paired_differences["metric"].eq("f1")
        & paired_differences["feature_set_a"].eq("full")
        & paired_differences["feature_set_b"].eq("full")
        & (
            (paired_differences["model_a"].eq(leader["model"]) & paired_differences["model_b"].eq(runner_up["model"]))
            | (paired_differences["model_b"].eq(leader["model"]) & paired_differences["model_a"].eq(runner_up["model"]))
        )
    ]
    if len(pair) != 1:
        raise UnifiedCleanError("Full-feature leader/runner-up paired interval is absent")
    pair_row = pair.iloc[0]
    if pair_row["model_a"] == leader["model"]:
        ci_low, ci_high = float(pair_row["ci_low"]), float(pair_row["ci_high"])
    else:
        ci_low, ci_high = -float(pair_row["ci_high"]), -float(pair_row["ci_low"])
    ranking_lines = [
        f"| {row.model} | {row.f1_mean:.9f} | {row.f1_std:.9f} |"
        for row in full.itertuples(index=False)
    ]
    text = "\n".join([
        "# Unified clean three-feature comparison",
        "",
        "**Status: GENERATED_PENDING_INDEPENDENT_REVIEW.** This report is exploratory validation evidence from a previously observed development partition. It is not pristine confirmatory or locked-test evidence.",
        "",
        "## Design and thresholds",
        "",
        "The comparison uses full five features, coordinates only, and the three categorical features across eight models, 17 one-vs-rest regions, and seeds 42/202/340. Every threshold is selected on the same 1,693 clean calibration POIs. Calibrated and fixed-0.5 results are both scored on the same 1,707 evaluation POIs; fixed-0.5 full-validation reproduction is reported separately.",
        "",
        "Tree models use balanced one-vs-rest sample weights. TabPFN fits are unweighted, so fixed-0.5 F1 differences are not interpreted as pure feature-information effects. Prediction coverage is 100% for every required stratum.",
        "",
        "## Full-feature calibrated point estimates",
        "",
        "| Model | Mean F1 over three seed macros | Sample SD across three seed macros |",
        "|---|---:|---:|",
        *ranking_lines,
        "",
        f"The user-specified highest point-estimate rule retains **{leader['model']}** as the common downstream classifier (mean F1 {leader['f1_mean']:.9f}). The runner-up is {runner_up['model']} and the point difference is {difference:.9f}; its paired conditional 95% interval for leader minus runner-up is [{ci_low:.9f}, {ci_high:.9f}]. This small difference is not described as statistical superiority.",
        "",
        "## Uncertainty scope",
        "",
        "The 2,000 paired bootstrap draws resample original evaluation POI IDs with shared multiplicities across all models, feature sets, regions, and seeds. Each draw averages the same 17 regions and three fixed model seeds before differences are formed. The percentile intervals are marginal and conditional on the 51 fitted strata and calibration-selected thresholds. They exclude training-algorithm, calibration-selection, spatial-dependence, model-selection, and locked-test uncertainty and are not simultaneous intervals or null-calibrated tests.",
        "",
        "## Provenance and limits",
        "",
        "Historical v2.5/v3 feature-ablation predictions come from one authenticated archived run but lack per-fit checkpoint attestations (PARTIAL). Prior whole-dataset coordinate access remains a historical isolation limitation (PARTIAL). New full-v3 persisted states are the attack/defense replay baseline and are compared diagnostically with historical full-v3 predictions; they do not silently replace the historical feature-table source.",
        "",
        f"Generated cardinalities: {json.dumps(dict(cardinality), sort_keys=True)}.",
        "",
        "No attack, defense, locked-test, or unseen-test performance claim is made here.",
        "",
    ])
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _artifact_paths(cfg: Mapping[str, Any], regions: Sequence[str]) -> tuple[list[Path], dict[str, Any]]:
    legacy = cfg["legacy_inputs"]
    full_prediction = Path(str(legacy["full"]["predictions"])).resolve()
    no_prediction = Path(str(legacy["no_coordinates"]["predictions"])).resolve()
    paths = [full_prediction, no_prediction]
    paths.extend(Path(str(path)).resolve() for path in legacy["full"]["evidence"])
    paths.extend(Path(str(path)).resolve() for path in legacy["no_coordinates"]["evidence"])
    ablation_root = Path(str(legacy["ablation_root"])).resolve()
    ablation_archive = Path(str(legacy["ablation_source_archive"])).resolve()
    paths.extend([ablation_root / "metadata.json", ablation_root / "input_identity.json", ablation_archive])
    for feature_set in FEATURE_SETS:
        legacy_predictions, legacy_markers = _shard_paths(ablation_root, ("tabpfn_v2_5", "tabpfn_v3"), feature_set, regions)
        paths.extend([*legacy_predictions, *legacy_markers])
    fresh = {}
    for name, root_value, models, feature_set in (
        ("trees", cfg["tree_output"], TREE_MODELS, "coordinates_only"),
        ("tabpfn26", cfg["tabpfn26_output"], ("tabpfn_v2_6",), "coordinates_only"),
        ("full_v3_states", cfg["full_v3_output"], ("tabpfn_v3",), "full"),
    ):
        root = Path(str(root_value)).resolve()
        predictions, markers = _shard_paths(root, models, feature_set, regions)
        provenance = [root / "work" / f"{model}_{feature_set}_{seed}_{region}_fit_provenance.json" for model in models for seed in SEEDS for region in regions]
        paths.extend([root / "metadata.json", root / "input_identity.json", root / "config.yaml", root / "_GENERATED_PENDING_REVIEW.json", *predictions, *markers, *provenance])
        if name == "full_v3_states":
            paths.extend([
                root / "replay_contract.json",
                root / "tables" / "calibration_thresholds.csv",
                root / "tables" / "calibration_scores.parquet",
                *[root / "states" / f"modelseed{seed}__region{index:02d}.tabpfn_fit" for seed in SEEDS for index in range(len(regions))],
            ])
        fresh[name] = root
    return paths, {"full": full_prediction, "no_coordinates": no_prediction, "ablation": ablation_root, "ablation_archive": ablation_archive, **fresh}


def _validate_fit_identity_contract(
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
    models: Sequence[str],
    expected_mode: str,
    feature_set: str,
    expected_features: Sequence[str],
    current_inventory: Sequence[tuple[str, int]],
) -> str:
    identity_snapshot = identity.get("input_snapshot")
    if not isinstance(identity_snapshot, dict) or metadata.get("input_snapshot") != identity_snapshot:
        raise UnifiedCleanError("Fresh fit metadata/identity input snapshots differ")
    recomputed_fingerprint = _canonical_json_sha({
        "format": FIT_FORMAT,
        "mode": expected_mode,
        "models": list(models),
        "feature_set": feature_set,
        "features": list(expected_features),
        "seeds": list(SEEDS),
        "source_gate": identity["source_gate"],
        "input_snapshot": identity_snapshot,
    })
    if identity.get("fingerprint_sha256") != recomputed_fingerprint:
        raise UnifiedCleanError("Fresh fit identity fingerprint does not match its recorded contract")
    recorded_inventory = sorted(
        (str(value.get("sha256")), int(value.get("size_bytes", -1)))
        for value in identity_snapshot.values()
    )
    if len(identity_snapshot) != len(current_inventory) or recorded_inventory != sorted(current_inventory):
        raise UnifiedCleanError("Fresh fit identity is not bound to the current data/config/partition/checkpoint bytes")
    return recomputed_fingerprint


def _validate_fresh_completion(
    root: Path,
    models: Sequence[str],
    regions: Sequence[str],
    fit_source_gate: Mapping[str, str],
    partition: Mapping[str, Any],
    cfg: Mapping[str, Any],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    metadata = json.loads(_bound_bytes(root / "metadata.json", snapshot))
    marker = json.loads(_bound_bytes(root / "_GENERATED_PENDING_REVIEW.json", snapshot))
    identity = json.loads(_bound_bytes(root / "input_identity.json", snapshot))
    expected = len(models) * len(SEEDS) * 17
    feature_set = str(metadata.get("feature_set"))
    expected_mode = {
        tuple(TREE_MODELS): "coordinate-trees",
        ("tabpfn_v2_6",): "coordinate-tabpfn26",
        ("tabpfn_v3",): "full-tabpfn3-states",
    }.get(tuple(models))
    expected_features = list(FEATURE_SETS[feature_set]) if feature_set in FEATURE_SETS else None
    expected_states = expected if list(models) == ["tabpfn_v3"] and feature_set == "full" else 0
    if (
        metadata.get("format") != FIT_FORMAT
        or metadata.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or metadata.get("fit_count") != expected
        or metadata.get("models") != list(models)
        or metadata.get("mode") != expected_mode
        or metadata.get("features") != expected_features
        or metadata.get("seeds") != list(SEEDS)
        or metadata.get("regions") != list(regions)
        or metadata.get("persisted_state_count") != expected_states
        or identity.get("format") != FIT_FORMAT
        or identity.get("mode") != expected_mode
        or not isinstance(identity.get("fingerprint_sha256"), str)
        or len(identity["fingerprint_sha256"]) != 64
        or identity.get("source_gate", {}).get("manifest_sha256") != fit_source_gate["manifest_sha256"]
        or identity.get("source_gate", {}).get("verdict_sha256") != fit_source_gate["verdict_sha256"]
        or metadata.get("source_gate") != identity.get("source_gate")
    ):
        raise UnifiedCleanError(f"Fresh coordinate fit metadata differs: {root}")
    contract = cfg["fit_input_contract"]
    data_dir = Path(str(cfg["data_dir"])).resolve()
    current_inputs = [
        data_dir / "poi_data_region.csv",
        data_dir / "sample_manifest.csv",
        data_dir / "report.json",
        *[Path(str(path)).resolve() for path in cfg["partition_files"]],
        Path(str(contract["tree_config"])).resolve(),
        Path(str(cfg["fit_source_freeze_manifest"])).resolve(),
        Path(str(cfg["fit_source_review_verdict"])).resolve(),
    ]
    if tuple(models) == ("tabpfn_v2_6",):
        current_inputs.append(Path(str(contract["tabpfn_v2_6_checkpoint"])).resolve())
    elif tuple(models) == ("tabpfn_v3",):
        current_inputs.append(Path(str(contract["tabpfn_v3_checkpoint"])).resolve())
    current_inventory = sorted(
        (snapshot[str(path.resolve())]["sha256"], snapshot[str(path.resolve())]["size_bytes"])
        for path in current_inputs
    )
    _validate_fit_identity_contract(
        identity, metadata, models, expected_mode, feature_set,
        expected_features, current_inventory,
    )
    if (
        marker.get("metadata_sha256") != snapshot[str((root / "metadata.json").resolve())]["sha256"]
        or marker.get("fit_count") != expected
        or marker.get("persisted_state_count") != expected_states
    ):
        raise UnifiedCleanError(f"Fresh coordinate completion marker differs: {root}")
    if metadata.get("test_rows_read") != 0 or metadata.get("test_features_read") != 0:
        raise UnifiedCleanError("Fresh coordinate fit accessed test data")
    fingerprint = identity["fingerprint_sha256"]
    prediction_hashes, provenance_hashes, marker_hashes, state_hashes = [], [], [], []
    for model in models:
        for seed in SEEDS:
            for region_index, region in enumerate(regions):
                stem = f"{model}_{feature_set}_{seed}_{region}"
                prediction = root / "predictions" / f"{stem}_clean.parquet"
                provenance_path = root / "work" / f"{stem}_fit_provenance.json"
                marker_path = root / "work" / f"{stem}.json"
                shard = json.loads(_bound_bytes(marker_path, snapshot))
                provenance = json.loads(_bound_bytes(provenance_path, snapshot))
                expected_files = {
                    str(prediction.relative_to(root)): snapshot[str(prediction.resolve())]["sha256"],
                    str(provenance_path.relative_to(root)): snapshot[str(provenance_path.resolve())]["sha256"],
                }
                if expected_states:
                    state = root / "states" / f"modelseed{seed}__region{region_index:02d}.tabpfn_fit"
                    expected_files[str(state.relative_to(root))] = snapshot[str(state.resolve())]["sha256"]
                    state_hashes.append(snapshot[str(state.resolve())]["sha256"])
                if shard.get("fingerprint_sha256") != fingerprint or shard.get("files_sha256") != expected_files:
                    raise UnifiedCleanError(f"Fresh fit shard marker differs: {marker_path}")
                if (
                    provenance.get("format") != "poi-unified-clean-actual-fit-provenance-v1"
                    or provenance.get("source_fingerprint_sha256") != fingerprint
                    or provenance.get("model") != model
                    or provenance.get("feature_set") != feature_set
                    or provenance.get("seed") != seed
                    or provenance.get("region") != region
                    or provenance.get("region_index") != region_index
                    or provenance.get("features") != expected_features
                    or provenance.get("train_rows") != int(cfg["expected_train_rows"])
                    or provenance.get("validation_rows") != int(cfg["expected_validation_rows"])
                    or provenance.get("prediction_sha256") != expected_files[str(prediction.relative_to(root))]
                    or provenance.get("test_rows_read") != 0
                    or provenance.get("test_features_read") != 0
                ):
                    raise UnifiedCleanError(f"Fresh fit provenance differs: {provenance_path}")
                provenance_partition = provenance.get("partition", {})
                partition_hashes = set(partition.get("sha256", {}).values())
                if (
                    len(partition_hashes) != 1
                    or provenance_partition.get("sha256") != next(iter(partition_hashes))
                    or provenance_partition.get("calibration_rows") != partition.get("calibration_rows")
                    or provenance_partition.get("evaluation_rows") != partition.get("evaluation_rows")
                    or provenance_partition.get("calibration_poi_order_sha256") != partition.get("calibration_poi_order_sha256")
                    or provenance_partition.get("evaluation_poi_order_sha256") != partition.get("evaluation_poi_order_sha256")
                ):
                    raise UnifiedCleanError(f"Fresh fit provenance partition differs: {provenance_path}")
                attestation = provenance.get("fit_attestation", {})
                expected_policy = (
                    "balanced_OvR_sample_weight_n_over_2class_count"
                    if model in TREE_MODELS else "TabPFN_default_unweighted_fit"
                )
                if (
                    attestation.get("model_id") != model
                    or attestation.get("seed") != seed
                    or attestation.get("feature_order") != expected_features
                    or attestation.get("train_rows") != int(cfg["expected_train_rows"])
                    or provenance.get("training_weight_policy") != expected_policy
                    or not isinstance(attestation.get("truth_sha256"), str)
                    or len(attestation["truth_sha256"]) != 64
                    or attestation.get("probe_indices") != sorted(attestation.get("probe_indices", []))
                    or not attestation.get("probe_indices")
                    or attestation.get("probe_prediction_sha256") != attestation.get("probe_repeat_prediction_sha256")
                ):
                    raise UnifiedCleanError(f"Fresh fit attestation differs: {provenance_path}")
                if model in TREE_MODELS:
                    transformed = attestation.get("transformed_train", {})
                    if (
                        attestation.get("format") != "poi-tree-fit-attestation-v1"
                        or attestation.get("probe_predictions_within_tolerance") is not True
                        or transformed.get("shape", [None])[0] != int(cfg["expected_train_rows"])
                        or not isinstance(attestation.get("sample_weight_sha256"), str)
                        or len(attestation["sample_weight_sha256"]) != 64
                    ):
                        raise UnifiedCleanError(f"Fresh tree fit attestation differs: {provenance_path}")
                else:
                    checkpoint_key = f"{model}_checkpoint_sha256"
                    if (
                        attestation.get("format") != "poi-tabpfn25-fit-replay-v1"
                        or attestation.get("n_estimators") != 8
                        or attestation.get("train_matrix_dtype") != "float32"
                        or attestation.get("official_checkpoint_sha256") != contract[checkpoint_key]
                        or attestation.get("cuda_available") is not True
                        or attestation.get("requested_device") != "cuda"
                    ):
                        raise UnifiedCleanError(f"Fresh TabPFN fit attestation differs: {provenance_path}")
                prediction_hashes.append(expected_files[str(prediction.relative_to(root))])
                provenance_hashes.append(expected_files[str(provenance_path.relative_to(root))])
                marker_hashes.append(snapshot[str(marker_path.resolve())]["sha256"])
    recorded = (
        ("prediction_sha256", prediction_hashes),
        ("provenance_sha256", provenance_hashes),
        ("marker_sha256", marker_hashes),
        ("state_sha256", state_hashes),
    )
    for key, expected_hashes in recorded:
        values = metadata.get(key)
        if not isinstance(values, dict) or sorted(values.values()) != sorted(expected_hashes):
            raise UnifiedCleanError(f"Fresh fit metadata {key} inventory differs: {root}")
    if expected_states:
        contract_path = root / "replay_contract.json"
        if metadata.get("replay_contract_sha256") != snapshot[str(contract_path.resolve())]["sha256"]:
            raise UnifiedCleanError("Fresh fit metadata does not bind its replay contract")


def _validate_legacy_evidence(cfg: Mapping[str, Any], snapshot: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for feature_set in ("full", "no_coordinates"):
        entry = cfg["legacy_inputs"][feature_set]
        evidence = [Path(str(path)).resolve() for path in entry["evidence"]]
        by_name = {path.name: path for path in evidence}
        metadata = json.loads(_bound_bytes(by_name["metadata.json"], snapshot))
        verdict = json.loads(_bound_bytes(by_name["independent_review_verdict.json"], snapshot))
        if metadata.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW":
            raise UnifiedCleanError(f"Legacy {feature_set} metadata status differs")
        if feature_set == "full":
            accepted = verdict.get("clean_comparison_arithmetic_verdict") == "PASS"
            limitation = verdict.get("overall_verdict")
        else:
            accepted = verdict.get("bounded_computational_scope") == "PASS"
            limitation = verdict.get("full_scope")
        if not accepted:
            raise UnifiedCleanError(f"Legacy {feature_set} bounded result review did not PASS")
        result[feature_set] = {
            "metadata_path": str(by_name["metadata.json"]),
            "metadata_sha256": snapshot[str(by_name["metadata.json"])]["sha256"],
            "verdict_path": str(by_name["independent_review_verdict.json"]),
            "verdict_sha256": snapshot[str(by_name["independent_review_verdict.json"])]["sha256"],
            "accepted_bounded_field": "clean_comparison_arithmetic_verdict" if feature_set == "full" else "bounded_computational_scope",
            "broader_scope_verdict": limitation,
        }
    return result


def _validate_fit_source_evidence(
    cfg: Mapping[str, Any], snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    manifest_path = Path(str(cfg["fit_source_freeze_manifest"])).resolve()
    verdict_path = Path(str(cfg["fit_source_review_verdict"])).resolve()
    manifest = json.loads(_bound_bytes(manifest_path, snapshot))
    verdict = json.loads(_bound_bytes(verdict_path, snapshot))
    manifest_sha = snapshot[str(manifest_path)]["sha256"]
    verdict_sha = snapshot[str(verdict_path)]["sha256"]
    if (
        manifest.get("format") != "poi-unified-clean-fit-source-freeze-v1"
        or manifest.get("status") != "FROZEN_PENDING_INDEPENDENT_SOURCE_REVIEW"
        or verdict.get("status") != "PASS"
        or verdict.get("review_scope") != "unified_clean_fit_source_before_fits"
        or verdict.get("reviewed_source_manifest_sha256") != manifest_sha
    ):
        raise UnifiedCleanError("Fresh fit source freeze/review evidence differs")
    return {"manifest_sha256": manifest_sha, "verdict_sha256": verdict_sha}


def _validate_ablation_source(
    root: Path,
    archive_path: Path,
    data_dir: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from .clean_ablation_subset_report import marker_fingerprint

    identity = json.loads(_bound_bytes(root / "input_identity.json", snapshot))
    metadata = json.loads(_bound_bytes(root / "metadata.json", snapshot))
    fingerprint = identity.get("sha256")
    configured = metadata.get("config", {})
    if (
        not isinstance(fingerprint, str) or len(fingerprint) != 64
        or metadata.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or metadata.get("test_used") is not False
        or metadata.get("attack_sha256") is not None
        or configured.get("model_ids") != ["tabpfn_v2_5", "tabpfn_v3"]
        or configured.get("seeds") != list(SEEDS)
        or any(configured.get("feature_sets", {}).get(name) != list(features) for name, features in FEATURE_SETS.items())
    ):
        raise UnifiedCleanError("Original ablation metadata/identity differs")
    archive_payload = _bound_bytes(archive_path, snapshot)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload)) as archive:
            feature_source = archive.extractfile("src/poi/feature_ablation.py")
            factory_source = archive.extractfile("src/poi/selected_explain.py")
            if feature_source is None or factory_source is None:
                raise UnifiedCleanError("Original ablation archive lacks executed source")
            feature_sha = hashlib.sha256(feature_source.read()).hexdigest()
            factory_sha = hashlib.sha256(factory_source.read()).hexdigest()
    except (tarfile.TarError, KeyError) as exc:
        raise UnifiedCleanError("Original ablation source archive is invalid") from exc
    clean_sha = snapshot[str((data_dir / "poi_data_region.csv").resolve())]["sha256"]
    manifest_sha = snapshot[str((data_dir / "sample_manifest.csv").resolve())]["sha256"]
    reproduced = marker_fingerprint(configured, clean_sha, manifest_sha, feature_sha, factory_sha)
    if metadata.get("clean_sha256") != clean_sha or reproduced != fingerprint:
        raise UnifiedCleanError("Original ablation fingerprint is not bound to source/data bytes")
    return {
        "status": "PARTIAL_no_per_fit_attestation",
        "fingerprint_sha256": fingerprint,
        "source_archive_path": str(archive_path),
        "source_archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "executed_feature_source_sha256": feature_sha,
        "executed_factory_source_sha256": factory_sha,
    }


def _validate_replay_contract(
    root: Path,
    regions: Sequence[str],
    partition: Mapping[str, Any],
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    validation: pd.DataFrame,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = json.loads(_bound_bytes(root / "replay_contract.json", snapshot))
    if (
        contract.get("format") != "poi-unified-clean-tabpfn3-replay-contract-v1"
        or contract.get("status") != "GENERATED_PENDING_INDEPENDENT_REVIEW"
        or contract.get("model_id") != "tabpfn_v3"
        or contract.get("feature_order") != list(FEATURE_SETS["full"])
        or contract.get("seeds") != list(SEEDS)
        or contract.get("regions") != list(regions)
        or contract.get("fit_count") != 51
        or contract.get("test_rows_read") != 0
        or contract.get("test_features_read") != 0
        or contract.get("state_load_rule") != "TabPFNClassifier.load_from_fit_state exactly once per seed/region and reuse for all clean/attack/defense inference"
    ):
        raise UnifiedCleanError("Full-v3 replay contract format/cardinality differs")
    replay_partition = contract.get("partition")
    partition_hashes = set(partition.get("sha256", {}).values())
    if (
        not isinstance(replay_partition, dict)
        or len(partition_hashes) != 1
        or replay_partition.get("sha256") != next(iter(partition_hashes))
        or replay_partition.get("calibration_rows") != partition.get("calibration_rows")
        or replay_partition.get("evaluation_rows") != partition.get("evaluation_rows")
        or replay_partition.get("calibration_poi_order_sha256") != partition.get("calibration_poi_order_sha256")
        or replay_partition.get("evaluation_poi_order_sha256") != partition.get("evaluation_poi_order_sha256")
        or partition.get("calibration_poi_order_sha256") != _ordered_ids_sha(calibration_ids)
        or partition.get("evaluation_poi_order_sha256") != _ordered_ids_sha(evaluation_ids)
    ):
        raise UnifiedCleanError("Full-v3 replay partition differs from the common validation partition")
    expected = {(seed, region, index) for seed in SEEDS for index, region in enumerate(regions)}
    observed = {(int(row["model_seed"]), str(row["region"]), int(row["region_index"])) for row in contract.get("fits", [])}
    if observed != expected or len(contract.get("fits", [])) != len(expected):
        raise UnifiedCleanError("Full-v3 replay contract seed/region grid differs")
    for relative, key in (("tables/calibration_thresholds.csv", "calibration_thresholds_sha256"), ("tables/calibration_scores.parquet", "calibration_scores_sha256")):
        if snapshot[str((root / relative).resolve())]["sha256"] != contract.get(key):
            raise UnifiedCleanError("Full-v3 calibration evidence differs from replay contract")
    thresholds = pd.read_csv(io.BytesIO(_bound_bytes(root / "tables/calibration_thresholds.csv", snapshot)))
    calibration_scores = pd.read_parquet(io.BytesIO(_bound_bytes(root / "tables/calibration_scores.parquet", snapshot)))
    required_scores = {"model", "feature_set", "seed", "region", "POI_ID", "partition", "y_true", "y_score"}
    if not required_scores.issubset(calibration_scores.columns):
        raise UnifiedCleanError("Full-v3 calibration score columns differ")
    calibration_scores = calibration_scores.copy()
    calibration_scores["POI_ID"] = calibration_scores["POI_ID"].astype(str)
    threshold_keys = set(thresholds[["seed", "region", "region_index"]].itertuples(index=False, name=None))
    score_groups = calibration_scores.groupby(["seed", "region"], observed=True).size()
    if (
        len(thresholds) != 51
        or threshold_keys != expected
        or thresholds.duplicated(["seed", "region"]).any()
        or not np.isfinite(pd.to_numeric(thresholds["threshold"], errors="coerce")).all()
        or len(calibration_scores) != 51 * len(calibration_ids)
        or calibration_scores.duplicated(["seed", "region", "POI_ID"]).any()
        or len(score_groups) != 51
        or not score_groups.eq(len(calibration_ids)).all()
        or not calibration_scores["partition"].astype(str).eq("calibration").all()
        or not calibration_scores["model"].astype(str).eq("tabpfn_v3").all()
        or not calibration_scores["feature_set"].astype(str).eq("full").all()
    ):
        raise UnifiedCleanError("Full-v3 calibration table cardinality/content differs")
    calibration_set = set(map(str, calibration_ids))
    validation_region = validation.set_index("POI_ID", verify_integrity=True)["region"].astype(str)
    for row in contract["fits"]:
        seed, region, index = int(row["model_seed"]), str(row["region"]), int(row["region_index"])
        expected_state = f"states/modelseed{seed}__region{index:02d}.tabpfn_fit"
        expected_prediction = f"predictions/tabpfn_v3_full_{seed}_{region}_clean.parquet"
        state = root / row["state_relative_path"]
        prediction = root / row["prediction_relative_path"]
        provenance_path = root / "work" / f"tabpfn_v3_full_{seed}_{region}_fit_provenance.json"
        provenance = json.loads(_bound_bytes(provenance_path, snapshot))
        attestation = provenance.get("fit_attestation", {})
        hashes = (
            row.get("official_checkpoint_sha256"), row.get("train_matrix_sha256"),
            row.get("train_truth_sha256"), row.get("train_poi_order_sha256"),
            row.get("probe_matrix_sha256"), row.get("original_probe_score_sha256"),
            row.get("repeat_probe_score_sha256"), row.get("restored_probe_score_sha256"),
            row.get("calibration_score_float64_sha256"), row.get("calibration_truth_int8_sha256"),
        )
        if (
            row.get("model_id") != "tabpfn_v3"
            or row.get("feature_order") != list(FEATURE_SETS["full"])
            or row.get("state_relative_path") != expected_state
            or row.get("prediction_relative_path") != expected_prediction
            or row.get("state_format") != "tabpfn_official_fit_state_v1"
            or row.get("state_api") != "TabPFNClassifier.save_fit_state/load_from_fit_state"
            or not isinstance(row.get("official_checkpoint_path"), str)
            or not row["official_checkpoint_path"]
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
            or not isinstance(row.get("probe_indices"), list)
            or row["probe_indices"] != sorted(row["probe_indices"])
        ):
            raise UnifiedCleanError("Full-v3 replay fit identity/attestation differs")
        attestation_contract = {
            "state_sha256": attestation.get("fitted_state_sha256"),
            "state_size_bytes": attestation.get("fitted_state_size_bytes"),
            "state_format": attestation.get("fitted_state_format"),
            "state_api": attestation.get("fitted_state_api"),
            "official_checkpoint_path": attestation.get("official_checkpoint_path"),
            "official_checkpoint_sha256": attestation.get("official_checkpoint_sha256"),
            "train_matrix_sha256": attestation.get("train_matrix_sha256"),
            "train_truth_sha256": attestation.get("truth_sha256"),
            "probe_indices": attestation.get("probe_indices"),
            "probe_matrix_sha256": attestation.get("probe_matrix_sha256"),
            "original_probe_score_sha256": attestation.get("probe_prediction_sha256"),
            "repeat_probe_score_sha256": attestation.get("probe_repeat_prediction_sha256"),
            "restored_probe_score_sha256": attestation.get("restored_probe_prediction_sha256"),
        }
        if any(row.get(key) != value for key, value in attestation_contract.items()):
            raise UnifiedCleanError("Full-v3 replay row differs from bound per-fit provenance")
        if (
            row.get("train_poi_order_sha256") != provenance.get("train_poi_order_sha256")
            or row.get("prediction_sha256") != provenance.get("prediction_sha256")
            or row.get("calibration_threshold") != provenance.get("calibration_threshold", {}).get("threshold")
            or row.get("calibration_score_float64_sha256") != provenance.get("calibration_score_float64_sha256")
            or row.get("calibration_truth_int8_sha256") != provenance.get("calibration_truth_int8_sha256")
        ):
            raise UnifiedCleanError("Full-v3 replay row differs from fit/calibration provenance")
        if snapshot[str(state.resolve())]["sha256"] != row["state_sha256"] or snapshot[str(state.resolve())]["size_bytes"] != row["state_size_bytes"]:
            raise UnifiedCleanError("Full-v3 persisted state bytes differ from replay contract")
        if snapshot[str(prediction.resolve())]["sha256"] != row["prediction_sha256"]:
            raise UnifiedCleanError("Full-v3 prediction bytes differ from replay contract")
        if row["original_probe_score_sha256"] != row["repeat_probe_score_sha256"] or row["original_probe_score_sha256"] != row["restored_probe_score_sha256"]:
            raise UnifiedCleanError("Full-v3 original/repeat/restored probe scores differ")
        table_group = calibration_scores.loc[
            calibration_scores["seed"].eq(seed) & calibration_scores["region"].astype(str).eq(region)
        ].copy()
        if set(table_group["POI_ID"]) != calibration_set:
            raise UnifiedCleanError("Full-v3 calibration table POI identity differs")
        table_truth = _strict_binary(table_group["y_true"], "calibration y_true")
        table_score = _strict_scores(table_group["y_score"])
        expected_truth = table_group["POI_ID"].map(validation_region).eq(region).astype(np.int8).to_numpy()
        if not np.array_equal(table_truth, expected_truth):
            raise UnifiedCleanError("Full-v3 calibration truth differs from validation identity")
        prediction_frame = _read_parquet_bound(prediction, snapshot)
        prediction_frame["POI_ID"] = prediction_frame["POI_ID"].astype(str)
        prediction_calibration = prediction_frame.loc[
            prediction_frame["partition"].astype(str).eq("calibration")
        ].copy()
        if (
            len(prediction_calibration) != len(calibration_ids)
            or prediction_calibration["POI_ID"].duplicated().any()
            or set(prediction_calibration["POI_ID"]) != calibration_set
        ):
            raise UnifiedCleanError("Full-v3 prediction calibration POI identity differs")
        prediction_truth = _strict_binary(prediction_calibration["y_true"], "prediction calibration y_true")
        prediction_score = _strict_scores(prediction_calibration["y_score"])
        aligned_table = table_group.set_index("POI_ID", verify_integrity=True).loc[prediction_calibration["POI_ID"]]
        if (
            not np.array_equal(prediction_truth, _strict_binary(aligned_table["y_true"], "aligned calibration y_true"))
            or not np.array_equal(prediction_score, _strict_scores(aligned_table["y_score"]))
            or hashlib.sha256(np.ascontiguousarray(prediction_score, dtype=np.float64).tobytes()).hexdigest() != row["calibration_score_float64_sha256"]
            or hashlib.sha256(np.ascontiguousarray(prediction_truth, dtype=np.int8).tobytes()).hexdigest() != row["calibration_truth_int8_sha256"]
        ):
            raise UnifiedCleanError("Full-v3 calibration table differs from bound prediction/provenance")
        selected = select_f1_threshold(prediction_truth, prediction_score)
        threshold_row = thresholds.loc[
            thresholds["seed"].eq(seed) & thresholds["region"].astype(str).eq(region)
        ]
        if len(threshold_row) != 1:
            raise UnifiedCleanError("Full-v3 calibration threshold identity differs")
        threshold_record = threshold_row.iloc[0]
        numeric_threshold_fields = ("threshold", "f1", "precision", "recall", "predicted_positive_rate")
        integer_threshold_fields = ("accepted", "tp", "fp", "fn", "rows", "positive_truth", "candidate_count")
        if (
            any(not np.isclose(float(threshold_record[key]), float(selected[key]), rtol=0, atol=1e-15) for key in numeric_threshold_fields)
            or any(int(threshold_record[key]) != int(selected[key]) for key in integer_threshold_fields)
            or float(row["calibration_threshold"]) != float(selected["threshold"])
        ):
            raise UnifiedCleanError("Full-v3 calibration threshold was not reproduced from bound scores")
    if len({row["official_checkpoint_sha256"] for row in contract["fits"]}) != 1:
        raise UnifiedCleanError("Full-v3 replay fits do not share one official checkpoint")
    return contract


def aggregate(cfg: Mapping[str, Any]) -> None:
    gate = verify_source_gate(cfg)
    started = time.time()
    output = Path(str(cfg["aggregate_output"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate output: {output}")
    tables, figures, provenance = output / "tables", output / "figures", output / "provenance"
    for directory in (tables, figures, provenance):
        directory.mkdir(parents=True, exist_ok=False)
    data_dir = Path(str(cfg["data_dir"])).resolve()
    fit_contract = cfg["fit_input_contract"]
    static = [
        data_dir / "poi_data_region.csv",
        data_dir / "sample_manifest.csv",
        data_dir / "report.json",
        *[Path(str(path)).resolve() for path in cfg["partition_files"]],
        Path(str(cfg["fit_source_freeze_manifest"])).resolve(),
        Path(str(cfg["fit_source_review_verdict"])).resolve(),
        Path(str(cfg["source_freeze_manifest"])).resolve(),
        Path(str(cfg["source_review_verdict"])).resolve(),
        Path(str(fit_contract["tree_config"])).resolve(),
        Path(str(fit_contract["tabpfn_v2_6_checkpoint"])).resolve(),
        Path(str(fit_contract["tabpfn_v3_checkpoint"])).resolve(),
    ]
    static_snapshot = _snapshot_files(static)
    for model in ("tabpfn_v2_6", "tabpfn_v3"):
        checkpoint = Path(str(fit_contract[f"{model}_checkpoint"])).resolve()
        if static_snapshot[str(checkpoint)]["sha256"] != fit_contract[f"{model}_checkpoint_sha256"]:
            raise UnifiedCleanError(f"Current {model} checkpoint differs from the frozen fit contract")
    clean, _ = _load_clean_splits_bound(data_dir, ("train", "validation"), static_snapshot)
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    if len(train) != int(cfg["expected_train_rows"]) or len(validation) != int(cfg["expected_validation_rows"]):
        raise UnifiedCleanError("Bound development split cardinality differs")
    regions = tuple(sorted(validation["region"].astype(str).unique()))
    if len(regions) != int(cfg["expected_regions"]):
        raise UnifiedCleanError("Bound validation region cardinality differs")
    calibration_ids, evaluation_ids, partition_evidence = load_common_partition(
        validation,
        cfg["partition_files"],
        expected_calibration=int(cfg["expected_calibration_rows"]),
        expected_evaluation=int(cfg["expected_evaluation_rows"]),
        snapshot=static_snapshot,
    )
    artifact_paths, roots = _artifact_paths(cfg, regions)
    full_snapshot = _snapshot_files([*static, *artifact_paths])
    if any(full_snapshot.get(path) != value for path, value in static_snapshot.items()):
        raise UnifiedCleanError("Static source/data/partition input changed during aggregate setup")
    _atomic_json({"format": "poi-unified-clean-input-snapshot-v1", "phase": "before_any_prediction_read", "files": full_snapshot}, provenance / "input_snapshot_before_read.json")
    expected_hashes = cfg["legacy_inputs"]["expected_prediction_sha256"]
    for feature, path in (("full", roots["full"]), ("no_coordinates", roots["no_coordinates"])):
        if full_snapshot[str(path)]["sha256"] != expected_hashes[feature]:
            raise UnifiedCleanError(f"Legacy combined prediction SHA differs: {feature}")
    legacy_evidence = _validate_legacy_evidence(cfg, full_snapshot)
    ablation_evidence = _validate_ablation_source(roots["ablation"], roots["ablation_archive"], data_dir, full_snapshot)
    fit_source_gate = _validate_fit_source_evidence(cfg, full_snapshot)
    _validate_fresh_completion(roots["trees"], TREE_MODELS, regions, fit_source_gate, partition_evidence, cfg, full_snapshot)
    _validate_fresh_completion(roots["tabpfn26"], ("tabpfn_v2_6",), regions, fit_source_gate, partition_evidence, cfg, full_snapshot)
    _validate_fresh_completion(roots["full_v3_states"], ("tabpfn_v3",), regions, fit_source_gate, partition_evidence, cfg, full_snapshot)
    replay_contract = _validate_replay_contract(
        roots["full_v3_states"], regions, partition_evidence,
        calibration_ids, evaluation_ids, validation, full_snapshot,
    )

    repeats = int(cfg["bootstrap_repeats"])
    if repeats < 2000:
        raise UnifiedCleanError("At least 2,000 paired POI bootstrap repeats are required")
    weights = make_bootstrap_weights(len(evaluation_ids), repeats, int(cfg["bootstrap_seed"]))
    threshold_parts, metric_parts = [], []
    replicate_map: dict[tuple[str, str], np.ndarray] = {}
    coverage_rows = []
    source_rows = []
    historical_full_v3 = None
    for feature_set in ("full", "coordinates_only", "no_coordinates"):
        if feature_set == "full":
            raw = _read_parquet_bound(roots["full"], full_snapshot)
            base = normalize_prediction_frame(raw.loc[raw["model"].isin((*TREE_MODELS, "tabpfn_v2_6"))].copy(), feature_set=feature_set, validation=validation, expected_models=(*TREE_MODELS, "tabpfn_v2_6"))
            legacy = _read_shards(roots["ablation"], ("tabpfn_v2_5", "tabpfn_v3"), feature_set, validation, full_snapshot, legacy=True)
            historical_full_v3 = legacy.loc[legacy["model"].eq("tabpfn_v3")].copy()
            predictions = normalize_prediction_frame(pd.concat([base, legacy], ignore_index=True), feature_set=feature_set, validation=validation, expected_models=MODELS)
        elif feature_set == "no_coordinates":
            raw = _read_parquet_bound(roots["no_coordinates"], full_snapshot)
            base = normalize_prediction_frame(raw.loc[raw["model"].isin((*TREE_MODELS, "tabpfn_v2_6"))].copy(), feature_set=feature_set, validation=validation, expected_models=(*TREE_MODELS, "tabpfn_v2_6"))
            legacy = _read_shards(roots["ablation"], ("tabpfn_v2_5", "tabpfn_v3"), feature_set, validation, full_snapshot, legacy=True)
            predictions = normalize_prediction_frame(pd.concat([base, legacy], ignore_index=True), feature_set=feature_set, validation=validation, expected_models=MODELS)
        else:
            legacy = _read_shards(roots["ablation"], ("tabpfn_v2_5", "tabpfn_v3"), feature_set, validation, full_snapshot, legacy=True)
            trees = _read_shards(roots["trees"], TREE_MODELS, feature_set, validation, full_snapshot, legacy=False)
            tabpfn26 = _read_shards(roots["tabpfn26"], ("tabpfn_v2_6",), feature_set, validation, full_snapshot, legacy=False)
            predictions = normalize_prediction_frame(pd.concat([legacy, trees, tabpfn26], ignore_index=True), feature_set=feature_set, validation=validation, expected_models=MODELS)
        lookup = {str(value): "calibration" for value in calibration_ids}
        lookup.update({str(value): "evaluation" for value in evaluation_ids})
        predictions["partition"] = predictions["POI_ID"].map(lookup)
        _atomic_frame(predictions, tables / f"{feature_set}_validation_predictions.parquet")
        thresholds, metrics, replicate = analyze_feature_predictions(predictions, calibration_ids=calibration_ids, evaluation_ids=evaluation_ids, weights=weights)
        threshold_parts.append(thresholds)
        metric_parts.append(metrics)
        replicate_map.update(replicate)
        for model in MODELS:
            if model in ("tabpfn_v2_5", "tabpfn_v3"):
                source_id = "legacy_original_feature_ablation_same_run"
                source_path = str(roots["ablation"])
                provenance_level = "PARTIAL_no_per_fit_attestation"
            elif feature_set == "coordinates_only":
                source_id = "new_review_gated_coordinate_fit"
                source_path = str(roots["trees"] if model in TREE_MODELS else roots["tabpfn26"])
                provenance_level = "actual_per_fit_attestation"
            else:
                source_id = "legacy_reviewed_clean_comparison" if feature_set == "full" else "legacy_reviewed_no_coordinate_fit"
                source_path = str(roots[feature_set])
                provenance_level = "actual_per_fit_attestation" if feature_set == "no_coordinates" else "mixed_legacy_attestation"
            source_rows.append({"feature_set": feature_set, "model": model, "source_id": source_id, "source_path": source_path, "provenance_level": provenance_level, "prediction_rows": len(validation) * len(SEEDS) * len(regions)})
        for model in MODELS:
            coverage_rows.extend([
                {"feature_set": feature_set, "model": model, "partition": "full_validation", "expected_rows_per_stratum": len(validation), "observed_rows_per_stratum": len(validation), "coverage": 1.0, "strata": len(SEEDS) * len(regions)},
                {"feature_set": feature_set, "model": model, "partition": "calibration", "expected_rows_per_stratum": len(calibration_ids), "observed_rows_per_stratum": len(calibration_ids), "coverage": 1.0, "strata": len(SEEDS) * len(regions)},
                {"feature_set": feature_set, "model": model, "partition": "evaluation", "expected_rows_per_stratum": len(evaluation_ids), "observed_rows_per_stratum": len(evaluation_ids), "coverage": 1.0, "strata": len(SEEDS) * len(regions)},
            ])
        if feature_set == "coordinates_only":
            del legacy, trees, tabpfn26
        else:
            del raw, base, legacy
        del predictions
    if historical_full_v3 is None:
        raise UnifiedCleanError("Historical full-v3 prediction registry is absent")
    current_full_v3 = _read_shards(roots["full_v3_states"], ("tabpfn_v3",), "full", validation, full_snapshot, legacy=False)
    comparison = historical_full_v3.merge(current_full_v3, on=["POI_ID", "feature_set", "model", "seed", "region", "condition", "split", "y_true"], suffixes=("_historical", "_replay"), validate="one_to_one")
    replay_delta_rows = []
    for (seed, region), group in comparison.groupby(["seed", "region"], sort=True):
        replay_delta_rows.append({
            "seed": int(seed), "region": region, "rows": len(group),
            "max_absolute_score_delta": float(np.max(np.abs(group["y_score_replay"] - group["y_score_historical"]))),
            "mean_absolute_score_delta": float(np.mean(np.abs(group["y_score_replay"] - group["y_score_historical"]))),
            "fixed_0p5_prediction_disagreement": float(np.mean(group["y_pred_fixed_0p5_replay"] != group["y_pred_fixed_0p5_historical"])),
            "historical_fixed_0p5_f1": f1_score(group["y_true"], group["y_pred_fixed_0p5_historical"], zero_division=0),
            "replay_fixed_0p5_f1": f1_score(group["y_true"], group["y_pred_fixed_0p5_replay"], zero_division=0),
            "interpretation": "diagnostic_only_new_persisted_state_fit_is_downstream_primary_not_feature_table_replacement",
        })
    replay_delta = pd.DataFrame(replay_delta_rows)
    thresholds = pd.concat(threshold_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    source_registry = pd.DataFrame(source_rows)
    replicates, intervals, pairs = summarize_bootstrap(metrics, replicate_map, confidence_level=float(cfg["confidence_level"]))
    same_model = pairs.loc[(pairs["model_a"].eq(pairs["model_b"])) & pairs["feature_set_a"].ne(pairs["feature_set_b"])].copy()
    summary_values = [*METRICS, "predicted_positive_rate"]
    seed_group = ["feature_set", "model", "partition", "threshold_rule", "seed"]
    region_counts = metrics.groupby(seed_group, observed=True).size()
    if not region_counts.eq(len(regions)).all():
        raise UnifiedCleanError("A per-seed macro does not contain exactly 17 regional metrics")
    macro_by_seed = metrics.groupby(seed_group, sort=True, observed=True)[summary_values].mean().reset_index()
    summary_group = ["feature_set", "model", "partition", "threshold_rule"]
    seed_counts = macro_by_seed.groupby(summary_group, observed=True).size()
    if not seed_counts.eq(len(SEEDS)).all():
        raise UnifiedCleanError("A seed summary does not contain exactly three seed macros")
    macro_seed_summary = macro_by_seed.groupby(summary_group, sort=True, observed=True)[summary_values].agg(["mean", "std"])
    macro_seed_summary.columns = [f"{metric}_{statistic}" for metric, statistic in macro_seed_summary.columns]
    macro_seed_summary = macro_seed_summary.reset_index()
    expected = {
        "threshold_rows": 3 * len(MODELS) * len(SEEDS) * len(regions),
        "metric_rows": 3 * len(MODELS) * len(SEEDS) * len(regions) * 3,
        "replicate_rows": 3 * len(MODELS) * len(METRICS) * repeats,
        "interval_rows": 3 * len(MODELS) * len(METRICS),
        "pair_rows": (24 * 23 // 2) * len(METRICS),
        "same_model_feature_pair_rows": len(MODELS) * 3 * len(METRICS),
        "macro_by_seed_rows": 3 * len(MODELS) * 3 * len(SEEDS),
        "macro_seed_summary_rows": 3 * len(MODELS) * 3,
    }
    observed = {"threshold_rows": len(thresholds), "metric_rows": len(metrics), "replicate_rows": len(replicates), "interval_rows": len(intervals), "pair_rows": len(pairs), "same_model_feature_pair_rows": len(same_model), "macro_by_seed_rows": len(macro_by_seed), "macro_seed_summary_rows": len(macro_seed_summary)}
    if observed != expected:
        raise UnifiedCleanError(f"Aggregate table cardinality differs: {observed} != {expected}")
    for frame, name in (
        (thresholds, "calibration_thresholds.csv"),
        (metrics, "metrics_by_feature_model_seed_region.csv"),
        (coverage, "prediction_coverage.csv"),
        (source_registry, "prediction_source_registry.csv"),
        (replay_delta, "historical_vs_new_full_v3_replay_delta.csv"),
        (intervals, "macro_bootstrap_intervals.csv"),
        (pairs, "all_pipeline_paired_differences.csv"),
        (same_model, "same_model_feature_set_paired_differences.csv"),
        (macro_by_seed, "macro_by_seed.csv"),
        (macro_seed_summary, "macro_seed_summary.csv"),
    ):
        _atomic_frame(frame, tables / name)
    _atomic_frame(replicates, tables / "macro_bootstrap_replicates.parquet")
    macro = metrics.groupby(["feature_set", "model", "partition", "threshold_rule"], sort=True)[summary_values].mean().reset_index()
    _atomic_frame(macro, tables / "macro_summary.csv")
    _write_report(output / "report.md", macro_seed_summary, pairs, observed)
    _plots(metrics, intervals, coverage, figures)
    if any(fig.suffix.lower() != ".png" for fig in figures.iterdir()) or not list(figures.glob("*.png")):
        raise UnifiedCleanError("Figure output must contain PNG only")
    if _snapshot_files([*static, *artifact_paths]) != full_snapshot:
        raise UnifiedCleanError("An aggregate input changed during execution")
    verify_source_gate(cfg)
    output_files = sorted(path for path in output.rglob("*") if path.is_file())
    metadata = {
        "format": AGGREGATE_FORMAT,
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scientific_completion": False,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "regions": list(regions),
        "validation_rows": len(validation),
        "calibration_rows": len(calibration_ids),
        "evaluation_rows": len(evaluation_ids),
        "prediction_rows": 3 * len(MODELS) * len(SEEDS) * len(regions) * len(validation),
        "common_partition": partition_evidence,
        "evaluation_interpretation": "exploratory_validation_previously_observed; not_pristine_confirmatory",
        "threshold_rule": "per feature_set/model/seed/region F1 threshold selected only on 1693 calibration IDs; inclusive score>=threshold",
        "fixed_0p5_auxiliary": "reported on the same 1707 evaluation IDs; no threshold-effect comparison mixes full validation",
        "metrics": list(METRICS),
        "bootstrap": {"repeats": repeats, "seed": int(cfg["bootstrap_seed"]), "unit": "POI_ID", "paired": True, "fit_aggregation": "equal_mean_over_3_model_seeds_and_17_regions", "uncertainty_scope": "POI_resampling_conditional_on_51_fixed_fits_and_calibration_selected_thresholds; not_a_full_training_algorithm_distribution", "conditional_on_fits_and_thresholds": True},
        "training_weight_policy": {**{model: "balanced_OvR_sample_weight_n_over_2class_count" for model in TREE_MODELS}, **{model: "TabPFN_default_unweighted_fit" for model in ("tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3")}},
        "common_downstream_selection": "user-specified highest full-feature calibrated evaluation F1 point estimate; tabpfn_v3 retained; paired conditional CI reported; no statistical-superiority claim",
        "source_gate": gate,
        "input_snapshot_sha256": sha256(provenance / "input_snapshot_before_read.json"),
        "table_cardinality": observed,
        "figure_count": len(list(figures.glob("*.png"))),
        "figure_format": "PNG_300_DPI_no_SVG",
        "execution_completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.time() - started,
        "test_rows_read": 0,
        "test_features_read": 0,
        "downstream_full_v3_replay_contract": {"path": str((roots["full_v3_states"] / "replay_contract.json").resolve()), "sha256": full_snapshot[str((roots["full_v3_states"] / "replay_contract.json").resolve())]["sha256"], "fit_count": replay_contract["fit_count"], "role": "attack_and_defense_primary; diagnostic delta only in clean feature table"},
        "legacy_evidence": legacy_evidence,
        "legacy_ablation_evidence": ablation_evidence,
        "historical_test_isolation": "PARTIAL: prior whole-dataset coordinate access is known and cannot be undone; this run reads train/validation only",
        "limitations": [
            "Clean validation comparison only; no attack or defense evidence is produced here.",
            "Tree models use balanced OvR sample weights while TabPFN fits are unweighted; fixed-0.5 F1 is therefore not a pure feature-information comparison.",
            "Intervals condition on the fitted models and calibration-selected thresholds.",
            "Intervals are marginal rather than simultaneous and omit training, threshold-selection, spatial-dependence, and model-selection uncertainty.",
            "Independent result review is pending.",
        ],
    }
    _atomic_json(metadata, output / "metadata.json")
    _atomic_json({"status": metadata["status"], "scientific_completion": False, "metadata_sha256": sha256(output / "metadata.json"), "output_sha256": {str(path.relative_to(output)): sha256(path) for path in output_files}, "test_rows_read": 0}, output / "_GENERATED_PENDING_REVIEW.json")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    cfg, snapshot = _read_yaml_bound(config_path)
    cfg = _env_paths(cfg)
    if cfg["_config_load_snapshot"] != snapshot:
        raise UnifiedCleanError("Configuration changed during load")
    if tuple(cfg["models"]) != MODELS or tuple(cfg["seeds"]) != SEEDS:
        raise UnifiedCleanError("Frozen models or seeds differ")
    if cfg["feature_sets"] != {key: list(value) for key, value in FEATURE_SETS.items()}:
        raise UnifiedCleanError("Frozen feature sets differ")
    if float(cfg["fixed_threshold"]) != 0.5 or int(cfg["expected_regions"]) != 17:
        raise UnifiedCleanError("Frozen threshold or region count differs")
    return cfg


def run(config_path: str | Path, mode: str) -> None:
    cfg = load_config(config_path)
    if mode == "freeze-source":
        freeze_source(cfg)
    elif mode == "aggregate":
        aggregate(cfg)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("freeze-source", "aggregate"))
    args = parser.parse_args()
    run(args.config, args.mode)


if __name__ == "__main__":
    main()
