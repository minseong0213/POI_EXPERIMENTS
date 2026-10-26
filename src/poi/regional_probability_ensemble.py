"""Validation-only regional probability calibration and exploratory ensembles.

This runner consumes predictions made by already-fitted regional classifiers.  It
never fits a regional classifier and never reads the locked test split.  Model
ranking, probability temperatures, and ensemble weights are fitted on the declared
validation calibration IDs.  The disjoint validation evaluation IDs are used only
after every fitted parameter has been frozen.

Attack predictions use a wide, sharded contract so all member probabilities for a
POI are paired before an ensemble is evaluated.  See ``ATTACK_REQUIRED_COLUMNS`` and
``validate_attack_manifest`` for the machine-checked source contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import yaml

from .uncertainty import make_bootstrap_weights, weighted_metric_replicates


FORMAT = "poi-regional-probability-ensemble-v1"
ATTACK_MANIFEST_FORMAT = "poi-eight-model-regional-attack-predictions-v1"
FROZEN_MODELS = (
    "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
    "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3",
)
FROZEN_SEEDS = (42, 202, 340)
FROZEN_GENERATION_SEEDS = (42, 202, 340)
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
CLEAN_REQUIRED_COLUMNS = {
    "POI_ID", "model", "seed", "region", "split", "condition", "y_true", "y_score",
}
ATTACK_REQUIRED_COLUMNS = {
    "POI_ID", "source_region", "model_seed", "generation_seed", "attack_condition",
    "target_region", "truth", "primary_evaluation_eligible", "quarantine_reason",
    "attack_method", "numeric_budget_m", "categorical_budget_l0", "source_model",
    "victim_model", "constraints_valid", "label_preserved",
}


class RegionalEnsembleError(RuntimeError):
    """A fail-closed input, partition, or evaluation contract violation."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(value: Any) -> str:
    if pd.isna(value):
        raise RegionalEnsembleError("POI_ID contains a missing value")
    return str(value)


def _score_columns(models: Sequence[str]) -> list[str]:
    return [f"score__{model}" for model in models]


def _finite_probabilities(values: Any, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RegionalEnsembleError(f"{label} must contain finite [0,1] probabilities")
    return scores


def _identity_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    normalized = frame.loc[:, columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(str)
    normalized = normalized.sort_values(list(columns), kind="mergesort")
    payload = "\n".join("\x1f".join(row) for row in normalized.itertuples(index=False, name=None))
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def load_partition(path: str | Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """Load the existing common calibration/evaluation partition without fallback."""
    source = Path(path)
    if not source.is_file():
        raise RegionalEnsembleError(f"Required frozen validation partition is missing: {source}")
    payload = json.loads(source.read_text())
    if payload.get("split") != "validation" or payload.get("overlap") != 0:
        raise RegionalEnsembleError("Partition must declare split=validation and overlap=0")
    calibration = [_canonical_id(value) for value in payload.get("calibration_ids", [])]
    evaluation = [_canonical_id(value) for value in payload.get("evaluation_ids", [])]
    if not calibration or not evaluation:
        raise RegionalEnsembleError("Calibration and evaluation ID lists must both be nonempty")
    if len(set(calibration)) != len(calibration) or len(set(evaluation)) != len(evaluation):
        raise RegionalEnsembleError("Partition ID lists contain duplicates")
    if set(calibration) & set(evaluation):
        raise RegionalEnsembleError("Calibration and evaluation ID lists overlap")
    return calibration, evaluation, {
        "path": str(source.resolve()), "sha256": sha256(source),
        "calibration_rows": len(calibration), "evaluation_rows": len(evaluation),
        "partition_seed": payload.get("partition_seed", payload.get("seed")),
    }


def validate_clean_predictions(
    frame: pd.DataFrame,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    regions: Sequence[str],
    expected_pois: int,
) -> pd.DataFrame:
    """Validate the complete clean long table and return an aligned wide table."""
    missing = CLEAN_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RegionalEnsembleError(f"Clean predictions miss columns: {sorted(missing)}")
    clean = frame.copy()
    clean["POI_ID"] = clean["POI_ID"].map(_canonical_id)
    clean["seed"] = pd.to_numeric(clean["seed"], errors="raise").astype(int)
    clean["y_true"] = pd.to_numeric(clean["y_true"], errors="raise").astype(int)
    clean["y_score"] = _finite_probabilities(clean["y_score"], "clean y_score")
    if not clean["split"].eq("validation").all() or not clean["condition"].eq("clean").all():
        raise RegionalEnsembleError("Clean artifact may contain only clean validation predictions")
    if not clean["model"].isin(models).all() or set(clean["model"]) != set(models):
        raise RegionalEnsembleError("Clean artifact model set differs from the frozen eight models")
    if set(clean["seed"]) != set(map(int, seeds)) or set(clean["region"].map(str)) != set(regions):
        raise RegionalEnsembleError("Clean artifact seed or region set differs from configuration")
    if not np.isin(clean["y_true"], [0, 1]).all():
        raise RegionalEnsembleError("Clean truth must be binary")
    key = ["POI_ID", "model", "seed", "region"]
    if clean.duplicated(key).any():
        raise RegionalEnsembleError("Clean artifact has duplicate model/seed/region/POI rows")
    expected_rows = expected_pois * len(models) * len(seeds) * len(regions)
    if len(clean) != expected_rows or clean["POI_ID"].nunique() != expected_pois:
        raise RegionalEnsembleError(
            f"Clean artifact cardinality mismatch: rows={len(clean)}, POIs={clean.POI_ID.nunique()}"
        )
    group_sizes = clean.groupby(["model", "seed", "region"], observed=True).size()
    if len(group_sizes) != len(models) * len(seeds) * len(regions) or not group_sizes.eq(expected_pois).all():
        raise RegionalEnsembleError("Clean model/seed/region Cartesian coverage is incomplete")

    truth_consistency = clean.groupby(["POI_ID", "seed", "region"], observed=True)["y_true"].nunique()
    if not truth_consistency.eq(1).all():
        raise RegionalEnsembleError("Clean truth changes across models")
    truth = clean.drop_duplicates(["POI_ID", "seed", "region"])
    positives = truth.groupby(["POI_ID", "seed"], observed=True)["y_true"].sum()
    if not positives.eq(1).all():
        raise RegionalEnsembleError("Each POI/seed must have exactly one positive OvR region")
    source = truth.loc[truth.y_true.eq(1), ["POI_ID", "seed", "region"]].rename(
        columns={"region": "source_region"}
    )
    source_nunique = source.groupby("POI_ID", observed=True)["source_region"].nunique()
    if not source_nunique.eq(1).all():
        raise RegionalEnsembleError("A clean POI changes source region across model seeds")

    index = ["POI_ID", "seed", "region", "y_true"]
    wide = clean.pivot(index=index, columns="model", values="y_score").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"seed": "model_seed", "region": "target_region", "y_true": "truth"})
    wide = wide.merge(source[["POI_ID", "seed", "source_region"]].rename(columns={"seed": "model_seed"}),
                      on=["POI_ID", "model_seed"], validate="many_to_one")
    wide = wide.rename(columns={model: f"score__{model}" for model in models})
    return wide.sort_values(["model_seed", "target_region", "POI_ID"], kind="mergesort").reset_index(drop=True)


def validate_partition_coverage(
    clean_wide: pd.DataFrame,
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    regions: Sequence[str],
) -> None:
    observed = set(clean_wide["POI_ID"])
    if set(calibration_ids) | set(evaluation_ids) != observed:
        raise RegionalEnsembleError("Frozen partition is not an exact partition of clean validation POIs")
    source = clean_wide[["POI_ID", "source_region"]].drop_duplicates()
    if source["POI_ID"].duplicated().any():
        raise RegionalEnsembleError("POI/source-region mapping is not unique")
    lookup = source.set_index("POI_ID")["source_region"]
    for label, ids in (("calibration", calibration_ids), ("evaluation", evaluation_ids)):
        if set(lookup.loc[list(ids)]) != set(regions):
            raise RegionalEnsembleError(f"{label} partition does not contain every region")


def validate_attack_manifest(
    path: str | Path,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    generation_seeds: Sequence[int],
    regions: Sequence[str],
    attack_conditions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the exact wide-shard grid and immutable hashes.

    A shard contains every validation POI crossed with every target region for one
    ``(model_seed, generation_seed, attack_condition)`` cohort.  Its eight score
    columns are named ``score__<model>``.  This prevents member probabilities from
    being joined by row order or from silently using different attacked examples.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise RegionalEnsembleError(f"Attack prediction manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("format") != ATTACK_MANIFEST_FORMAT:
        raise RegionalEnsembleError("Attack prediction manifest format is not supported")
    for field, expected in (
        ("models", list(models)), ("model_seeds", list(map(int, seeds))),
        ("generation_seeds", list(map(int, generation_seeds))),
        ("regions", list(regions)), ("attack_conditions", list(attack_conditions)),
    ):
        observed = payload.get(field)
        if observed != expected:
            raise RegionalEnsembleError(f"Attack manifest {field} differs from frozen configuration")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise RegionalEnsembleError("Attack manifest shards must be a list")
    required = {"path", "sha256", "model_seed", "generation_seed", "attack_condition", "rows", "unique_pois"}
    expected_keys = {
        (int(seed), int(generation_seed), str(condition))
        for seed in seeds for generation_seed in generation_seeds for condition in attack_conditions
    }
    seen: dict[tuple[int, int, str], dict[str, Any]] = {}
    root = manifest_path.parent
    for entry in shards:
        missing = required - set(entry)
        if missing:
            raise RegionalEnsembleError(f"Attack shard manifest entry misses: {sorted(missing)}")
        key = (int(entry["model_seed"]), int(entry["generation_seed"]), str(entry["attack_condition"]))
        if key in seen:
            raise RegionalEnsembleError(f"Duplicate attack shard key: {key}")
        shard_path = Path(entry["path"])
        if not shard_path.is_absolute():
            shard_path = (root / shard_path).resolve()
        if not shard_path.is_file():
            raise RegionalEnsembleError(f"Attack prediction shard is missing: {shard_path}")
        if sha256(shard_path) != str(entry["sha256"]):
            raise RegionalEnsembleError(f"Attack prediction shard hash mismatch: {shard_path}")
        normalized = dict(entry, path=str(shard_path))
        seen[key] = normalized
    if set(seen) != expected_keys:
        missing = sorted(expected_keys - set(seen))
        extra = sorted(set(seen) - expected_keys)
        raise RegionalEnsembleError(f"Attack shard grid mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return [seen[key] for key in sorted(seen)], {
        "path": str(manifest_path.resolve()), "sha256": sha256(manifest_path),
        "shard_count": len(seen), "prediction_scope": payload.get("prediction_scope"),
        "source_provenance": payload.get("source_provenance"),
    }


def load_attack_shard(
    entry: Mapping[str, Any],
    *,
    models: Sequence[str],
    regions: Sequence[str],
    validation_ids: set[str],
    expected_pois: int,
    expected_source_region: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if "sha256" in entry and sha256(entry["path"]) != str(entry["sha256"]):
        raise RegionalEnsembleError(f"Attack prediction shard changed after manifest validation: {entry['path']}")
    frame = pd.read_parquet(entry["path"])
    required = ATTACK_REQUIRED_COLUMNS | set(_score_columns(models))
    missing = required - set(frame.columns)
    if missing:
        raise RegionalEnsembleError(f"Attack shard misses columns: {sorted(missing)}")
    if len(frame) != int(entry["rows"]) or int(entry["unique_pois"]) != expected_pois:
        raise RegionalEnsembleError("Attack shard cardinality differs from manifest")
    result = frame.loc[:, sorted(required)].copy()
    result["POI_ID"] = result["POI_ID"].map(_canonical_id)
    for name, expected in (
        ("model_seed", int(entry["model_seed"])),
        ("generation_seed", int(entry["generation_seed"])),
        ("attack_condition", str(entry["attack_condition"])),
    ):
        if not result[name].eq(expected).all():
            raise RegionalEnsembleError(f"Attack shard {name} differs from manifest")
    if set(result["POI_ID"]) != validation_ids or result["POI_ID"].nunique() != expected_pois:
        raise RegionalEnsembleError("Attack shard POI IDs differ from clean validation")
    if set(result["target_region"].map(str)) != set(regions):
        raise RegionalEnsembleError("Attack shard target-region set differs from configuration")
    key = ["POI_ID", "target_region"]
    if result.duplicated(key).any() or len(result) != expected_pois * len(regions):
        raise RegionalEnsembleError("Attack shard must have one row per POI/target-region")
    if not np.isin(pd.to_numeric(result["truth"], errors="raise"), [0, 1]).all():
        raise RegionalEnsembleError("Attack truth must be binary")
    positives = result.groupby("POI_ID", observed=True)["truth"].sum()
    if not positives.eq(1).all():
        raise RegionalEnsembleError("Each attacked POI must have exactly one positive OvR region")
    expected_truth = result["source_region"].astype(str).eq(result["target_region"].astype(str)).astype(int)
    if not np.array_equal(result["truth"].to_numpy(dtype=int), expected_truth.to_numpy()):
        raise RegionalEnsembleError("Attack truth disagrees with source and target regions")
    if expected_source_region is not None:
        observed_source = result[["POI_ID", "source_region"]].drop_duplicates().set_index("POI_ID")["source_region"].astype(str)
        expected_source = pd.Series(expected_source_region, dtype=str).loc[observed_source.index]
        if not observed_source.equals(expected_source):
            raise RegionalEnsembleError("Attack source-region labels differ from clean validation labels")
    for column in _score_columns(models):
        result[column] = _finite_probabilities(result[column], column)
    eligible = result["primary_evaluation_eligible"]
    if not (eligible.map(type).eq(bool).all() or eligible.isin([0, 1, True, False]).all()):
        raise RegionalEnsembleError("primary_evaluation_eligible must be boolean")
    result["primary_evaluation_eligible"] = eligible.astype(bool)
    for column in ("constraints_valid", "label_preserved"):
        if not result[column].isin([0, 1, True, False]).all():
            raise RegionalEnsembleError(f"{column} must be boolean")
        result[column] = result[column].astype(bool)
    invalid_eligible = result.primary_evaluation_eligible & (~result.constraints_valid | ~result.label_preserved)
    if invalid_eligible.any():
        raise RegionalEnsembleError("An invalid or label-changing attack cannot be primary-evaluation eligible")
    if result.loc[~result.primary_evaluation_eligible, "quarantine_reason"].fillna("").eq("").any():
        raise RegionalEnsembleError("Every ineligible attack row needs a quarantine_reason")
    immutable = [
        "source_region", "primary_evaluation_eligible", "quarantine_reason", "attack_method",
        "numeric_budget_m", "categorical_budget_l0", "source_model", "victim_model",
        "constraints_valid", "label_preserved",
    ]
    for column in immutable:
        if result.groupby("POI_ID", observed=True)[column].nunique(dropna=False).gt(1).any():
            raise RegionalEnsembleError(f"Attack metadata {column} changes across target regions")
    return result.sort_values(["target_region", "POI_ID"], kind="mergesort").reset_index(drop=True)


def temperature_scale(scores: Any, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise RegionalEnsembleError("Temperature must be finite and positive")
    probabilities = _finite_probabilities(scores, "temperature input")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def _weighted_log_loss(truth: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    losses = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
    return float(np.average(losses, weights=weight))


def fit_temperature(truth: Any, score: Any, sample_weight: Any | None = None) -> float:
    truth_array = np.asarray(truth, dtype=np.int8)
    score_array = _finite_probabilities(score, "calibration score")
    if truth_array.shape != score_array.shape or truth_array.ndim != 1 or not np.isin(truth_array, [0, 1]).all():
        raise RegionalEnsembleError("Temperature fit needs aligned binary truth and scores")
    weights = np.ones(len(truth_array), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != truth_array.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise RegionalEnsembleError("Calibration weights must be aligned, finite, and positive")
    result = minimize_scalar(
        lambda log_t: _weighted_log_loss(truth_array, temperature_scale(score_array, math.exp(log_t)), weights),
        bounds=(math.log(0.05), math.log(20.0)), method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RegionalEnsembleError("Temperature calibration failed")
    return float(math.exp(result.x))


def fit_simplex_weights(truth: Any, member_scores: Any, sample_weight: Any | None = None) -> np.ndarray:
    truth_array = np.asarray(truth, dtype=np.int8)
    matrix = np.asarray(member_scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != truth_array.size or matrix.shape[1] < 2:
        raise RegionalEnsembleError("Simplex fit needs aligned rows and at least two members")
    _finite_probabilities(matrix, "simplex member scores")
    weights = np.ones(len(truth_array), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != truth_array.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise RegionalEnsembleError("Simplex sample weights are invalid")
    member_count = matrix.shape[1]
    result = minimize(
        lambda values: _weighted_log_loss(truth_array, matrix @ values, weights),
        np.full(member_count, 1 / member_count), method="SLSQP",
        bounds=[(0.0, 1.0)] * member_count,
        constraints={"type": "eq", "fun": lambda values: values.sum() - 1.0},
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RegionalEnsembleError(f"Simplex ensemble calibration failed: {result.message}")
    values = np.clip(np.asarray(result.x, dtype=np.float64), 0, 1)
    values /= values.sum()
    return values


def balanced_cohort_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give clean and every attack generation/condition cohort equal total mass."""
    required = {"calibration_source", "generation_seed", "attack_condition"}
    if not required.issubset(frame):
        raise RegionalEnsembleError("Balanced calibration weights need explicit cohort columns")
    cohort = frame[["calibration_source", "generation_seed", "attack_condition"]].astype(str).agg("\x1f".join, axis=1)
    counts = cohort.map(cohort.value_counts())
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    return weights / weights.mean()


def _binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.int8)
    score = _finite_probabilities(score, "metric score")
    if len(truth) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "accuracy": np.nan, "roc_auc": np.nan}
    prediction = score >= threshold
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, score)) if np.unique(truth).size == 2 else np.nan,
    }


def rank_members_on_calibration(
    clean_calibration: pd.DataFrame,
    models: Sequence[str],
    threshold: float,
    top_k: int,
) -> pd.DataFrame:
    """Rank members independently for each model seed using calibration IDs only."""
    rows: list[dict[str, Any]] = []
    for seed, seed_frame in clean_calibration.groupby("model_seed", sort=True):
        ranking: list[tuple[str, float]] = []
        for model in models:
            regional = []
            for _, group in seed_frame.groupby("target_region", sort=True):
                regional.append(_binary_metrics(group.truth.to_numpy(), group[f"score__{model}"].to_numpy(), threshold)["f1"])
            ranking.append((model, float(np.mean(regional))))
        ranking.sort(key=lambda value: (-value[1], list(models).index(value[0])))
        for rank, (model, value) in enumerate(ranking, start=1):
            rows.append({
                "model_seed": int(seed), "model": model, "calibration_macro_f1": value,
                "calibration_rank": rank, "selected_top_k": rank <= top_k,
                "fit_partition": "validation_calibration_half",
            })
    return pd.DataFrame(rows)


def fit_calibration_and_ensembles(
    clean_calibration: pd.DataFrame,
    attack_calibration: pd.DataFrame,
    *,
    models: Sequence[str],
    regions: Sequence[str],
    member_selection: pd.DataFrame,
    calibration_variants: Sequence[str],
    top_k: int,
    train_oof: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit temperatures and simplex weights without accessing evaluation rows."""
    allowed = {"uncalibrated", "validation_clean", "validation_clean_attack", "train_oof"}
    if not set(calibration_variants).issubset(allowed):
        raise RegionalEnsembleError("Unknown calibration variant")
    if "train_oof" in calibration_variants and train_oof is None:
        raise RegionalEnsembleError("train_oof calibration was requested without train OOF predictions")
    temperature_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for seed in sorted(clean_calibration.model_seed.unique()):
        selected = member_selection.loc[
            member_selection.model_seed.eq(seed) & member_selection.selected_top_k
        ].sort_values("calibration_rank").model.tolist()
        if len(selected) != top_k:
            raise RegionalEnsembleError("Top-k member selection is incomplete")
        clean_seed = clean_calibration.loc[clean_calibration.model_seed.eq(seed)].copy()
        attack_seed = attack_calibration.loc[attack_calibration.model_seed.eq(seed)].copy()
        if attack_seed.empty and "validation_clean_attack" in calibration_variants:
            raise RegionalEnsembleError("Clean+attack calibration needs attacked calibration rows")
        sources = {
            "validation_clean": clean_seed,
            "validation_clean_attack": pd.concat([clean_seed, attack_seed], ignore_index=True),
        }
        if train_oof is not None:
            sources["train_oof"] = train_oof.loc[train_oof.model_seed.eq(seed)].copy()
        for variant in calibration_variants:
            if variant == "uncalibrated":
                for region in regions:
                    for model in models:
                        temperature_rows.append({
                            "calibration_variant": variant, "model_seed": int(seed),
                            "target_region": region, "model": model, "temperature": 1.0,
                            "fit_rows": 0, "fit_partition": "none_predeclared_identity",
                        })
                fit_frame = sources.get("validation_clean_attack", clean_seed)
            else:
                fit_frame = sources[variant]
                if fit_frame.empty:
                    raise RegionalEnsembleError(f"{variant} has no calibration rows")
                for region in regions:
                    region_frame = fit_frame.loc[fit_frame.target_region.eq(region)]
                    if region_frame.empty:
                        raise RegionalEnsembleError(f"{variant} misses calibration region {region}")
                    sample_weight = balanced_cohort_weights(region_frame) if variant == "validation_clean_attack" else None
                    for model in models:
                        temperature_rows.append({
                            "calibration_variant": variant, "model_seed": int(seed),
                            "target_region": region, "model": model,
                            "temperature": fit_temperature(
                                region_frame.truth.to_numpy(), region_frame[f"score__{model}"].to_numpy(), sample_weight
                            ),
                            "fit_rows": len(region_frame),
                            "fit_partition": (
                                "validation_calibration_half_clean_and_attack"
                                if variant == "validation_clean_attack" else
                                "validation_calibration_half_clean" if variant == "validation_clean" else "train_oof"
                            ),
                        })
            temp_lookup = {
                (row["target_region"], row["model"]): row["temperature"]
                for row in temperature_rows
                if row["calibration_variant"] == variant and row["model_seed"] == int(seed)
            }
            # Ensemble parameters always use calibration IDs.  For the uncalibrated
            # probability variant only the member scores remain uncalibrated.
            ensemble_fit = sources.get("validation_clean_attack", clean_seed)
            for region in regions:
                region_frame = ensemble_fit.loc[ensemble_fit.target_region.eq(region)].copy()
                sample_weight = balanced_cohort_weights(region_frame)
                matrix = np.column_stack([
                    temperature_scale(region_frame[f"score__{model}"], temp_lookup[(region, model)])
                    for model in selected
                ])
                fitted = fit_simplex_weights(region_frame.truth.to_numpy(), matrix, sample_weight)
                for model, value in zip(selected, fitted):
                    weight_rows.append({
                        "calibration_variant": variant, "ensemble": "top_k_simplex",
                        "model_seed": int(seed), "target_region": region, "model": model,
                        "weight": float(value), "fit_rows": len(region_frame),
                        "fit_partition": "validation_calibration_half_clean_and_attack",
                    })
    return pd.DataFrame(temperature_rows), pd.DataFrame(weight_rows)


def apply_frozen_ensembles(
    frame: pd.DataFrame,
    *,
    models: Sequence[str],
    member_selection: pd.DataFrame,
    temperatures: pd.DataFrame,
    ensemble_weights: pd.DataFrame,
    calibration_variant: str,
) -> pd.DataFrame:
    """Apply already-fitted ensemble parameters to one evaluation cohort."""
    outputs: list[pd.DataFrame] = []
    identity = [
        column for column in (
            "POI_ID", "source_region", "model_seed", "generation_seed", "attack_condition",
            "target_region", "truth", "primary_evaluation_eligible", "quarantine_reason",
            "attack_method", "numeric_budget_m", "categorical_budget_l0", "source_model",
            "victim_model", "constraints_valid", "label_preserved",
        ) if column in frame
    ]
    for (seed, region), part in frame.groupby(["model_seed", "target_region"], sort=True):
        ranking = member_selection.loc[member_selection.model_seed.eq(seed)].sort_values("calibration_rank")
        selected = ranking.loc[ranking.selected_top_k, "model"].tolist()
        best = ranking.iloc[0].model
        temperature_part = temperatures.loc[
            temperatures.calibration_variant.eq(calibration_variant)
            & temperatures.model_seed.eq(seed) & temperatures.target_region.eq(region)
        ]
        temp = dict(zip(temperature_part.model, temperature_part.temperature))
        if set(temp) != set(models):
            raise RegionalEnsembleError("Frozen temperature lookup is incomplete")
        calibrated = {model: temperature_scale(part[f"score__{model}"], temp[model]) for model in models}
        variants: dict[str, np.ndarray] = {
            "fixed_tabpfn_v2_5_reference": part["score__tabpfn_v2_5"].to_numpy(float),
            "best_single_calibration": calibrated[best],
            "all8_equal_soft_vote": np.mean(np.column_stack([calibrated[m] for m in models]), axis=1),
            "top_k_equal_soft_vote": np.mean(np.column_stack([calibrated[m] for m in selected]), axis=1),
        }
        fitted = ensemble_weights.loc[
            ensemble_weights.calibration_variant.eq(calibration_variant)
            & ensemble_weights.model_seed.eq(seed) & ensemble_weights.target_region.eq(region)
        ]
        fitted_lookup = dict(zip(fitted.model, fitted.weight))
        if set(fitted_lookup) != set(selected) or not np.isclose(sum(fitted_lookup.values()), 1.0):
            raise RegionalEnsembleError("Frozen simplex weights are incomplete")
        variants["top_k_simplex_soft_vote"] = sum(calibrated[m] * fitted_lookup[m] for m in selected)
        for dropped in selected:
            remaining = [model for model in selected if model != dropped]
            variants[f"component_ablation_drop__{dropped}"] = np.mean(
                np.column_stack([calibrated[m] for m in remaining]), axis=1
            )
        base = part[identity].reset_index(drop=True)
        for pipeline, score in variants.items():
            result = base.copy()
            result["pipeline"] = pipeline
            result["calibration_variant"] = calibration_variant
            result["y_score"] = score
            result["selected_best_single"] = best
            result["selected_members"] = "|".join(selected)
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def metric_table(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "pipeline", "calibration_variant", "model_seed", "generation_seed",
        "attack_condition", "target_region",
    ]
    for keys, part in predictions.groupby(group_columns, sort=True, dropna=False):
        identity = dict(zip(group_columns, keys))
        for scope, mask in (
            ("all_intent_to_attack", np.ones(len(part), dtype=bool)),
            ("primary_valid", part.primary_evaluation_eligible.to_numpy(dtype=bool)),
        ):
            selected = part.loc[mask]
            values = _binary_metrics(selected.truth.to_numpy(), selected.y_score.to_numpy(), threshold)
            for metric, value in values.items():
                rows.append({**identity, "evaluation_scope": scope, "metric": metric,
                             "value": value, "n": len(selected),
                             "unique_pois": selected.POI_ID.nunique()})
    return pd.DataFrame(rows)


def paired_bootstrap_table(
    predictions: pd.DataFrame,
    *,
    evaluation_ids: Sequence[str],
    threshold: float,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Compare every ensemble to the calibration-selected best single on paired POIs."""
    if repeats < 1:
        raise RegionalEnsembleError("Bootstrap repeats must be positive")
    ordered_ids = list(map(str, evaluation_ids))
    weights = make_bootstrap_weights(len(ordered_ids), repeats, seed)
    rows: list[dict[str, Any]] = []
    strata = ["calibration_variant", "model_seed", "generation_seed", "attack_condition", "target_region"]
    for keys, part in predictions.groupby(strata, sort=True, dropna=False):
        identity = dict(zip(strata, keys))
        for scope, eligible_ids in (
            ("all_intent_to_attack", set(ordered_ids)),
            ("primary_valid", set(part.loc[part.primary_evaluation_eligible, "POI_ID"])),
        ):
            indices = [index for index, poi_id in enumerate(ordered_ids) if poi_id in eligible_ids]
            if len(indices) < 2:
                continue
            scoped_weights = weights[:, indices]
            reference = part.loc[part.pipeline.eq("best_single_calibration")].set_index("POI_ID").loc[
                [ordered_ids[index] for index in indices]
            ]
            reference_rep = weighted_metric_replicates(
                reference.truth.to_numpy(), reference.y_score.to_numpy(),
                (reference.y_score.to_numpy() >= threshold).astype(np.int8), scoped_weights,
            )
            for pipeline in sorted(set(part.pipeline) - {"best_single_calibration"}):
                candidate = part.loc[part.pipeline.eq(pipeline)].set_index("POI_ID").loc[
                    [ordered_ids[index] for index in indices]
                ]
                if not np.array_equal(candidate.truth.to_numpy(), reference.truth.to_numpy()):
                    raise RegionalEnsembleError("Paired bootstrap truth differs across pipelines")
                candidate_rep = weighted_metric_replicates(
                    candidate.truth.to_numpy(), candidate.y_score.to_numpy(),
                    (candidate.y_score.to_numpy() >= threshold).astype(np.int8), scoped_weights,
                )
                for metric in METRICS:
                    delta = candidate_rep[metric] - reference_rep[metric]
                    finite = delta[np.isfinite(delta)]
                    rows.append({
                        **identity, "evaluation_scope": scope, "pipeline": pipeline,
                        "reference_pipeline": "best_single_calibration", "metric": metric,
                        "bootstrap_repeats": repeats, "finite_repeats": len(finite),
                        "delta_mean": float(np.mean(finite)) if len(finite) else np.nan,
                        "delta_ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                        "delta_ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                        "probability_delta_gt_zero": float(np.mean(finite > 0)) if len(finite) else np.nan,
                    })
    return pd.DataFrame(rows)


def _load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or cfg.get("format") != FORMAT:
        raise RegionalEnsembleError(f"Config format must be {FORMAT}")
    return cfg


def validate_frozen_config(cfg: Mapping[str, Any]) -> None:
    if tuple(cfg.get("models", [])) != FROZEN_MODELS:
        raise RegionalEnsembleError("The regional ensemble requires the frozen eight-model order")
    if tuple(map(int, cfg.get("model_seeds", []))) != FROZEN_SEEDS:
        raise RegionalEnsembleError("Model seeds must be exactly 42,202,340")
    if tuple(map(int, cfg.get("generation_seeds", []))) != FROZEN_GENERATION_SEEDS:
        raise RegionalEnsembleError("Generation seeds must be exactly 42,202,340")
    if int(cfg.get("expected_regions", -1)) != 17 or int(cfg.get("expected_validation_pois", -1)) != 3400:
        raise RegionalEnsembleError("Frozen validation design requires 17 regions and 3400 POIs")
    if len(cfg.get("regions", [])) != 17 or len(set(cfg["regions"])) != 17:
        raise RegionalEnsembleError("Config must list the 17 unique frozen regions")
    conditions = list(cfg.get("attack_conditions", []))
    if len(conditions) != 32 or len(set(conditions)) != 32 or "clean" in conditions:
        raise RegionalEnsembleError("Config must list exactly 32 unique non-clean attack conditions")
    if int(cfg.get("top_k", 0)) < 2 or int(cfg["top_k"]) >= len(FROZEN_MODELS):
        raise RegionalEnsembleError("top_k must select between two and seven members")
    if float(cfg.get("classification_threshold", -1)) != 0.5:
        raise RegionalEnsembleError("Frozen independent OvR threshold must be 0.5")
    if int(cfg.get("bootstrap_repeats", 0)) < 2000:
        raise RegionalEnsembleError("Real paired bootstrap requires at least 2000 repeats")
    variants = list(cfg.get("calibration_variants", []))
    required = {"uncalibrated", "validation_clean", "validation_clean_attack"}
    if not required.issubset(variants):
        raise RegionalEnsembleError("Required probability-calibration variants are missing")
    if "train_oof" in variants and not cfg.get("train_oof_predictions"):
        raise RegionalEnsembleError("train_oof calibration requires train_oof_predictions")
    if cfg.get("selection_partition") != "validation_calibration_half":
        raise RegionalEnsembleError("Model selection must be calibration-half only")
    if cfg.get("scoring_partition") != "validation_evaluation_half":
        raise RegionalEnsembleError("Reported validation scores must use the disjoint evaluation half")


def validate_train_oof_predictions(
    frame: pd.DataFrame,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    regions: Sequence[str],
    validation_ids: set[str],
) -> pd.DataFrame:
    """Validate an optional train-only, out-of-fold probability calibration source."""
    required = {
        "POI_ID", "model_seed", "target_region", "truth", "split",
        "prediction_scope", "fold_id", *_score_columns(models),
    }
    missing = required - set(frame.columns)
    if missing:
        raise RegionalEnsembleError(f"Train OOF predictions miss columns: {sorted(missing)}")
    result = frame.loc[:, sorted(required)].copy()
    result["POI_ID"] = result["POI_ID"].map(_canonical_id)
    result["model_seed"] = pd.to_numeric(result["model_seed"], errors="raise").astype(int)
    if not result.split.eq("train").all() or not result.prediction_scope.eq("train_oof").all():
        raise RegionalEnsembleError("Calibration alternative must be explicitly train/train_oof")
    if result.fold_id.isna().any() or set(result.POI_ID) & validation_ids:
        raise RegionalEnsembleError("Train OOF rows need fold IDs and must be disjoint from validation")
    if set(result.model_seed) != set(map(int, seeds)) or set(result.target_region.map(str)) != set(regions):
        raise RegionalEnsembleError("Train OOF seed or region set differs from configuration")
    key = ["POI_ID", "model_seed", "target_region"]
    if result.duplicated(key).any():
        raise RegionalEnsembleError("Train OOF predictions contain duplicate identities")
    counts = result.groupby(["POI_ID", "model_seed"], observed=True).size()
    if not counts.eq(len(regions)).all():
        raise RegionalEnsembleError("Train OOF target-region coverage is incomplete")
    truth = pd.to_numeric(result.truth, errors="raise").astype(int)
    if not np.isin(truth, [0, 1]).all():
        raise RegionalEnsembleError("Train OOF truth must be binary")
    result["truth"] = truth
    if not result.groupby(["POI_ID", "model_seed"], observed=True).truth.sum().eq(1).all():
        raise RegionalEnsembleError("Each train OOF POI/seed needs one positive region")
    for column in _score_columns(models):
        result[column] = _finite_probabilities(result[column], column)
    return result


def preflight(config_path: str | Path) -> dict[str, Any]:
    """Check source availability and frozen design without creating outputs."""
    cfg = _load_config(config_path)
    validate_frozen_config(cfg)
    required_paths = {
        "clean_predictions": Path(cfg["clean_predictions"]),
        "validation_partition": Path(cfg["validation_partition"]),
        "attack_prediction_manifest": Path(cfg["attack_prediction_manifest"]),
    }
    if "train_oof" in cfg["calibration_variants"]:
        required_paths["train_oof_predictions"] = Path(cfg["train_oof_predictions"])
    missing = {name: str(path) for name, path in required_paths.items() if not path.is_file()}
    result = {
        "format": FORMAT, "status": "NOT_RUN", "missing_inputs": missing,
        "test_rows_read": 0, "output_created": False,
        "selection_partition": cfg["selection_partition"],
        "scoring_partition": cfg["scoring_partition"],
        "common_downstream_primary_model_unchanged": True,
        "exploratory_ensemble": True,
    }
    if missing:
        result["status"] = "NOT_RUN"
        return result
    models = list(cfg["models"])
    regions = list(map(str, cfg["regions"]))
    clean = validate_clean_predictions(
        pd.read_parquet(cfg["clean_predictions"]), models=models,
        seeds=cfg["model_seeds"], regions=regions,
        expected_pois=int(cfg["expected_validation_pois"]),
    )
    calibration_ids, evaluation_ids, partition = load_partition(cfg["validation_partition"])
    validate_partition_coverage(clean, calibration_ids, evaluation_ids, regions)
    shards, manifest = validate_attack_manifest(
        cfg["attack_prediction_manifest"], models=models, seeds=cfg["model_seeds"],
        generation_seeds=cfg["generation_seeds"], regions=regions,
        attack_conditions=cfg["attack_conditions"],
    )
    if "train_oof" in cfg["calibration_variants"]:
        validate_train_oof_predictions(
            pd.read_parquet(cfg["train_oof_predictions"]), models=models,
            seeds=cfg["model_seeds"], regions=regions,
            validation_ids=set(calibration_ids) | set(evaluation_ids),
        )
    result.update({
        "status": "READY_FOR_EXECUTION_PENDING_INDEPENDENT_REVIEW",
        "clean_prediction_rows": int(len(clean) * len(models)),
        "clean_wide_rows": len(clean), "attack_shards": len(shards),
        "partition": partition, "attack_manifest": manifest,
    })
    return result


def _clean_eval_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["generation_seed"] = -1
    result["attack_condition"] = "clean"
    result["primary_evaluation_eligible"] = True
    result["quarantine_reason"] = ""
    result["attack_method"] = "clean"
    result["numeric_budget_m"] = 0.0
    result["categorical_budget_l0"] = 0
    result["source_model"] = "clean"
    result["victim_model"] = "regional_classifier"
    result["constraints_valid"] = True
    result["label_preserved"] = True
    result["calibration_source"] = "clean"
    return result


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_config(config_path)
    validate_frozen_config(cfg)
    readiness = preflight(config_path)
    if readiness["status"] != "READY_FOR_EXECUTION_PENDING_INDEPENDENT_REVIEW":
        raise RegionalEnsembleError("NOT_RUN: required upstream prediction artifacts are missing")
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    models = list(cfg["models"])
    regions = list(map(str, cfg["regions"]))
    clean_path = Path(cfg["clean_predictions"])
    immutable_input_hashes = {
        "clean_predictions": sha256(clean_path),
        "validation_partition": sha256(cfg["validation_partition"]),
        "attack_prediction_manifest": sha256(cfg["attack_prediction_manifest"]),
        "config": sha256(config_path),
    }
    clean = validate_clean_predictions(
        pd.read_parquet(clean_path), models=models, seeds=cfg["model_seeds"], regions=regions,
        expected_pois=int(cfg["expected_validation_pois"]),
    )
    calibration_ids, evaluation_ids, partition_meta = load_partition(cfg["validation_partition"])
    validate_partition_coverage(clean, calibration_ids, evaluation_ids, regions)
    validation_ids = set(calibration_ids) | set(evaluation_ids)
    shards, manifest_meta = validate_attack_manifest(
        cfg["attack_prediction_manifest"], models=models, seeds=cfg["model_seeds"],
        generation_seeds=cfg["generation_seeds"], regions=regions,
        attack_conditions=cfg["attack_conditions"],
    )

    clean = _clean_eval_cohort(clean)
    source_region_lookup = (
        clean[["POI_ID", "source_region"]].drop_duplicates().set_index("POI_ID")["source_region"].astype(str).to_dict()
    )
    clean_cal = clean.loc[clean.POI_ID.isin(calibration_ids)].copy()
    clean_eval = clean.loc[clean.POI_ID.isin(evaluation_ids)].copy()
    attack_cal_parts: list[pd.DataFrame] = []
    for entry in shards:
        part = load_attack_shard(
            entry, models=models, regions=regions, validation_ids=validation_ids,
            expected_pois=int(cfg["expected_validation_pois"]),
            expected_source_region=source_region_lookup,
        )
        # Geometry/label-invalid rows remain in all-intent evaluation denominators,
        # but cannot fit a robustness calibration parameter.
        selected = part.loc[
            part.POI_ID.isin(calibration_ids) & part.primary_evaluation_eligible
        ].copy()
        selected["calibration_source"] = "attack"
        attack_cal_parts.append(selected)
    attack_cal = pd.concat(attack_cal_parts, ignore_index=True)
    selection = rank_members_on_calibration(
        clean_cal, models, float(cfg["classification_threshold"]), int(cfg["top_k"]),
    )
    train_oof = None
    if "train_oof" in cfg["calibration_variants"]:
        train_oof = validate_train_oof_predictions(
            pd.read_parquet(cfg["train_oof_predictions"]), models=models,
            seeds=cfg["model_seeds"], regions=regions, validation_ids=validation_ids,
        )
        train_oof["calibration_source"] = "train_oof"
        train_oof["generation_seed"] = -2
        train_oof["attack_condition"] = "train_oof"
    temperatures, weights = fit_calibration_and_ensembles(
        clean_cal, attack_cal, models=models, regions=regions, member_selection=selection,
        calibration_variants=cfg["calibration_variants"], top_k=int(cfg["top_k"]),
        train_oof=train_oof,
    )

    tables = output / "tables"
    prediction_dir = tables / "prediction_shards"
    provenance = output / "provenance"
    for directory in (tables, prediction_dir, provenance):
        directory.mkdir(parents=True, exist_ok=False)
    selection.to_csv(tables / "member_selection.csv", index=False)
    temperatures.to_csv(tables / "probability_calibration_fits.csv", index=False)
    weights.to_csv(tables / "ensemble_weight_fits.csv", index=False)

    all_metrics: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    prediction_manifest: list[dict[str, Any]] = []
    cohorts: list[tuple[dict[str, Any] | None, pd.DataFrame]] = [
        (None, part.copy()) for _, part in clean_eval.groupby("model_seed", sort=True)
    ]
    for entry in shards:
        part = load_attack_shard(
            entry, models=models, regions=regions, validation_ids=validation_ids,
            expected_pois=int(cfg["expected_validation_pois"]),
            expected_source_region=source_region_lookup,
        )
        cohorts.append((entry, part.loc[part.POI_ID.isin(evaluation_ids)].copy()))
    start = time.monotonic()
    for entry, cohort in cohorts:
        for calibration_variant in cfg["calibration_variants"]:
            predictions = apply_frozen_ensembles(
                cohort, models=models, member_selection=selection, temperatures=temperatures,
                ensemble_weights=weights, calibration_variant=calibration_variant,
            )
            all_metrics.append(metric_table(predictions, float(cfg["classification_threshold"])))
            all_bootstrap.append(paired_bootstrap_table(
                predictions, evaluation_ids=evaluation_ids,
                threshold=float(cfg["classification_threshold"]),
                repeats=int(cfg["bootstrap_repeats"]), seed=int(cfg["bootstrap_seed"]),
            ))
            first = predictions.iloc[0]
            filename = (
                f"modelseed{int(first.model_seed)}__genseed{int(first.generation_seed)}__"
                f"{str(first.attack_condition)}__{calibration_variant}.parquet"
            ).replace("/", "_")
            destination = prediction_dir / filename
            predictions.to_parquet(destination, index=False)
            prediction_manifest.append({
                "path": str(destination.resolve()), "sha256": sha256(destination),
                "rows": len(predictions), "unique_pois": predictions.POI_ID.nunique(),
                "model_seed": int(first.model_seed), "generation_seed": int(first.generation_seed),
                "attack_condition": str(first.attack_condition),
                "calibration_variant": calibration_variant,
                "row_identity_sha256": _identity_sha256(
                    predictions, ["POI_ID", "target_region", "pipeline", "calibration_variant"]
                ),
            })
    metrics = pd.concat(all_metrics, ignore_index=True)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    metrics.to_csv(tables / "metrics_by_region.csv", index=False)
    bootstrap.to_csv(tables / "paired_poi_bootstrap.csv", index=False)
    (tables / "prediction_manifest.json").write_text(json.dumps({
        "format": "poi-regional-ensemble-prediction-shards-v1", "shards": prediction_manifest,
    }, indent=2) + "\n")

    source_files = [Path(__file__), Path(config_path)]
    for source in source_files:
        shutil.copyfile(source, provenance / source.name)
    observed_after = {
        "clean_predictions": sha256(clean_path),
        "validation_partition": sha256(cfg["validation_partition"]),
        "attack_prediction_manifest": sha256(cfg["attack_prediction_manifest"]),
        "config": sha256(config_path),
    }
    if observed_after != immutable_input_hashes:
        raise RegionalEnsembleError("An input changed during regional ensemble execution")
    metadata = {
        "format": FORMAT, "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "test_rows_read": 0, "test_evaluation_performed": False,
        "scientific_completion": False, "independent_review": "PENDING",
        "exploratory_ensemble": True, "common_downstream_primary_model_unchanged": True,
        "selection_partition": "validation_calibration_half",
        "scoring_partition": "validation_evaluation_half",
        "partition": partition_meta, "attack_manifest": manifest_meta,
        "clean_predictions": {"path": str(clean_path.resolve()), "sha256": immutable_input_hashes["clean_predictions"]},
        "immutable_input_sha256": immutable_input_hashes,
        "models": models, "model_seeds": list(cfg["model_seeds"]),
        "generation_seeds": list(cfg["generation_seeds"]),
        "attack_conditions": list(cfg["attack_conditions"]), "regions": regions,
        "calibration_variants": list(cfg["calibration_variants"]),
        "calibration_attack_fit_scope": "primary_evaluation_eligible_only",
        "bootstrap_repeats": int(cfg["bootstrap_repeats"]),
        "prediction_shards": len(prediction_manifest), "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap), "wall_seconds": time.monotonic() - start,
        "limitations": [
            "Validation calibration/evaluation halves are development data, not a locked test.",
            "The eight frozen member fits may carry their own earlier validation-development selection; this runner prevents additional ensemble-selection leakage into its evaluation half.",
            "Historical exposure of old test coordinates is not erased by this validation-only experiment.",
            "No Stage-14 LOAO, LOSO, adaptive-evasion, or detector ensemble status is changed here.",
            "Attack-score gating, proposed-plus-external-defense ensembling, and model inference resource comparisons remain separate Stage-12 experiments.",
            "Regional ensembles are exploratory and do not replace the common frozen TabPFN 2.5 downstream.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "_GENERATED.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Regional probability calibration and ensemble\n\n"
        "All selection, temperature fitting, and simplex fitting used only the frozen validation "
        "calibration IDs. Reported metrics and paired-POI bootstrap intervals use the disjoint "
        "validation evaluation IDs. The fixed TabPFN 2.5 reference is retained separately. "
        "This exploratory validation result is pending independent review and is not a locked-test result.\n"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    result = preflight(args.config) if args.preflight else run(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
