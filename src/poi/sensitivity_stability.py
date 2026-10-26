"""Stage 14 sensitivity, stability, ranking, and detector-ensemble evaluation.

The stage consumes Stage 10 prediction shards.  Ensemble coefficients and every
clean-retention threshold are fitted on the calibration half of validation, frozen,
and then applied to the disjoint validation-evaluation half.  Invalid or quarantined
attack attempts remain visible in coverage diagnostics; only ``primary_valid`` rows
enter robustness metrics.  Clean predictions are stored once per model seed and
shared across the balanced 3x3 model/generation-seed design.  The locked test split
is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
import yaml

from .defense_evaluate import binary_metrics, paired_defense_metrics, select_detection_threshold
from .uncertainty import make_bootstrap_weights


PROPOSED = "proposed_tabpfn25_attack_head"
BASE_KEY = [
    "POI_ID", "source_region", "seed", "generation_seed",
    "attack_condition", "attack_label",
]
PROVENANCE = [
    "primary_evaluation_eligible", "quarantine_reason", "attack_method",
    "numeric_budget_m", "categorical_budget_l0", "source_model", "victim_model",
    "victim_evaluation_scope", "constraints_valid", "label_preserved",
    "attack_attempted", "attack_changed_any", "transfer_evaluated", "targeted",
    "valid_numeric_train_range", "valid_category_codebook", "valid_category_combo",
    "within_budget", "clean_boundary_status", "attack_boundary_status",
]
REGION_COLUMNS = [
    *BASE_KEY, *PROVENANCE, "target_region", "truth",
    "fixed_unmodified_score", "shared_nearest_train_repair_score",
    "proposed_native_full_coverage_score",
]
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
PROJECTED_REPAIR_COLUMN = "proposed_projected_repair_fixed_score"
CANONICAL_CLEAN_GENERATION_SEED = -1


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_identity_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    digest = hashlib.sha256()
    columns = list(columns)
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_shard_identity(frame: pd.DataFrame, manifest_row: Any, path: Path) -> None:
    expected_condition = str(manifest_row.condition)
    if set(frame["seed"].astype(int)) != {int(manifest_row.seed)}:
        raise ValueError(f"Prediction shard model seed differs from manifest: {path}")
    if set(frame["generation_seed"].astype(int)) != {int(manifest_row.generation_seed)}:
        raise ValueError(f"Prediction shard generation seed differs from manifest: {path}")
    if set(frame["attack_condition"].astype(str)) != {expected_condition}:
        raise ValueError(f"Prediction shard condition differs from manifest: {path}")
    expected_label = 0 if expected_condition == "clean" else 1
    if set(frame["attack_label"].astype(int)) != {expected_label}:
        raise ValueError(f"Prediction shard attack label differs from condition: {path}")
    poi = frame[["POI_ID", "source_region"]].drop_duplicates("POI_ID", keep="first")
    if int(manifest_row.unique_pois) != frame["POI_ID"].nunique():
        raise ValueError(f"Prediction shard POI count differs from manifest: {path}")
    observed_poi = _ordered_identity_sha256(poi, ["POI_ID", "source_region"])
    observed_rows = _ordered_identity_sha256(
        frame, ["POI_ID", "source_region", "target_region", "truth", "detector"]
    )
    if observed_poi != str(manifest_row.poi_identity_sha256):
        raise ValueError(f"Prediction shard POI identity differs from manifest: {path}")
    if observed_rows != str(manifest_row.row_identity_sha256):
        raise ValueError(f"Prediction shard row identity differs from manifest: {path}")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def nested_attack_mask(poi_ids: Iterable[str], ratio: float, seed: int) -> np.ndarray:
    """Select a deterministic nested fraction using only POI identity."""
    ids = np.asarray(list(map(str, poi_ids)), dtype=object)
    if not 0 <= ratio <= 1:
        raise ValueError("Attack mixture ratio must lie in [0, 1]")
    keys = np.asarray([
        hashlib.sha256(f"{seed}:{value}".encode()).hexdigest() for value in ids
    ])
    order = np.argsort(keys, kind="stable")
    selected = np.zeros(len(ids), dtype=bool)
    selected[order[:round(len(ids) * ratio)]] = True
    return selected


def preflight(config_path: str | Path) -> pd.DataFrame:
    """Inspect prerequisites without creating an output directory."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    rows = []
    required = {
        "stage10_generated_marker": Path(cfg["stage10_output"]) / "_GENERATED.json",
        "prediction_manifest": Path(cfg["prediction_manifest"]),
        "calibration_detector_scores": Path(cfg["calibration_detector_scores"]),
        "ablation_matrix": Path(cfg["ablation_matrix"]),
    }
    for name, path in required.items():
        rows.append({
            "requirement": name, "path": str(path), "exists": path.is_file(),
            "blocking": name != "ablation_matrix",
        })
    return pd.DataFrame(rows)


def _require_sources(config_path: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    status = preflight(config_path)
    missing = status.loc[status["blocking"] & ~status["exists"]]
    if len(missing):
        paths = ", ".join(missing["path"].astype(str))
        raise FileNotFoundError(
            "Stage 14 is NOT_RUN because required Stage 10 evidence is absent: " + paths
        )
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 14 output: {output}")
    return status


def _load_prediction_shards(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(cfg["prediction_manifest"])
    required = {
        "path", "sha256", "seed", "generation_seed", "condition", "rows",
        "unique_pois", "poi_identity_sha256", "row_identity_sha256",
    }
    if required - set(manifest):
        raise ValueError(f"Prediction manifest lacks: {sorted(required - set(manifest))}")
    model_seeds = list(map(int, cfg["model_seeds"]))
    generation_seeds = list(map(int, cfg["generation_seeds"]))
    expected_conditions = int(cfg["expected_attack_conditions"])
    if manifest.duplicated(["seed", "generation_seed", "condition"]).any():
        raise ValueError("Prediction manifest contains duplicate seed/condition keys")
    clean_manifest = manifest.loc[manifest["condition"].eq("clean")]
    attack_manifest = manifest.loc[manifest["condition"].ne("clean")]
    if (
        set(map(int, clean_manifest["seed"])) != set(model_seeds)
        or set(map(int, clean_manifest["generation_seed"]))
        != {CANONICAL_CLEAN_GENERATION_SEED}
        or len(clean_manifest) != len(model_seeds)
    ):
        raise ValueError("Stage 10 must store one canonical clean shard per model seed")
    attack_counts = attack_manifest.groupby(["seed", "generation_seed"], observed=True)[
        "condition"
    ].nunique()
    expected_pairs = {(left, right) for left in model_seeds for right in generation_seeds}
    if set(map(tuple, attack_counts.index.tolist())) != expected_pairs or not attack_counts.eq(
        expected_conditions
    ).all():
        raise ValueError("Stage 10 attack manifest is not the complete model/generation cross")
    frames = []
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or _sha256(path) != row.sha256:
            raise ValueError(f"Prediction shard missing or hash-mismatched: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(row.rows):
            raise ValueError(f"Prediction shard row count differs from manifest: {path}")
        _validate_shard_identity(frame, row, path)
        frames.append(frame)
    evaluation = _broadcast_canonical_clean(
        pd.concat(frames, ignore_index=True), generation_seeds, "evaluation"
    )
    calibration = _broadcast_canonical_clean(
        pd.read_parquet(cfg["calibration_detector_scores"]), generation_seeds, "calibration"
    )
    return evaluation, calibration


def _broadcast_canonical_clean(
    frame: pd.DataFrame, generation_seeds: Iterable[int], label: str
) -> pd.DataFrame:
    """Broadcast one stored clean reference per model seed across the attack-seed cross."""
    required = {"seed", "generation_seed", "attack_condition", "attack_label", "POI_ID"}
    if required - set(frame):
        raise ValueError(f"{label} lacks canonical-clean columns: {sorted(required - set(frame))}")
    clean_mask = frame["attack_label"].eq(0) & frame["attack_condition"].eq("clean")
    if not clean_mask.any():
        raise ValueError(f"{label} has no canonical clean rows")
    if not frame.loc[clean_mask, "generation_seed"].eq(CANONICAL_CLEAN_GENERATION_SEED).all():
        raise ValueError(f"{label} clean rows are not canonical generation_seed=-1")
    if frame.loc[~clean_mask, "generation_seed"].eq(CANONICAL_CLEAN_GENERATION_SEED).any():
        raise ValueError(f"{label} attack rows use the reserved canonical clean seed")
    canonical = frame.loc[clean_mask].copy()
    attacks = frame.loc[~clean_mask].copy()
    expanded = []
    for generation_seed in map(int, generation_seeds):
        clone = canonical.copy()
        clone["generation_seed"] = generation_seed
        expanded.append(clone)
    result = pd.concat([attacks, *expanded], ignore_index=True)
    return result


def _validate_sources(evaluation: pd.DataFrame, calibration: pd.DataFrame,
                      cfg: Mapping[str, Any]) -> pd.DataFrame:
    eval_required = set(REGION_COLUMNS) | {"detector", "detector_score"}
    cal_required = set(BASE_KEY + PROVENANCE) | {"detector", "detector_score"}
    missing_eval = eval_required - set(evaluation)
    missing_cal = cal_required - set(calibration)
    if missing_eval or missing_cal:
        raise ValueError(
            f"Stage 10 schema incomplete; evaluation={sorted(missing_eval)}, "
            f"calibration={sorted(missing_cal)}"
        )
    expected_seeds = set(map(int, cfg["model_seeds"]))
    observed_model = set(map(int, evaluation["seed"].unique()))
    observed_generation = set(map(int, evaluation["generation_seed"].unique()))
    if observed_model != expected_seeds or observed_generation != set(map(int, cfg["generation_seeds"])):
        raise ValueError("Model or attack-generation seed coverage is incomplete")
    if evaluation["target_region"].nunique() != int(cfg["expected_regions"]):
        raise ValueError("Region coverage is incomplete")
    expected_regions = set(evaluation["target_region"].astype(str).unique())
    if set(evaluation["source_region"].astype(str).unique()) != expected_regions:
        raise ValueError("Source and target region vocabularies differ")
    expected_truth = (
        evaluation["source_region"].astype(str).to_numpy()
        == evaluation["target_region"].astype(str).to_numpy()
    ).astype(np.int8)
    if not np.array_equal(evaluation["truth"].to_numpy(dtype=np.int8), expected_truth):
        raise ValueError("Stage 14 paired truth differs from source-region identity")
    if (evaluation.groupby("POI_ID", observed=True)["source_region"].nunique() != 1).any():
        raise ValueError("A POI changes source region across clean/attack shards")
    target_counts = evaluation.groupby(
        [*BASE_KEY, "detector"], observed=True
    )["target_region"].agg(["nunique", "size"])
    expected_target_count = int(cfg["expected_regions"])
    if not (
        target_counts["nunique"].eq(expected_target_count).all()
        and target_counts["size"].eq(expected_target_count).all()
    ):
        raise ValueError("A prediction key lacks exactly one row per target region")
    attack_conditions = set(evaluation.loc[evaluation["attack_label"].eq(1),
                                           "attack_condition"].astype(str))
    if len(attack_conditions) != int(cfg["expected_attack_conditions"]):
        raise ValueError("Attack-condition coverage is incomplete")
    members = set(cfg["detector_members"])
    if set(evaluation["detector"].unique()) != members or set(calibration["detector"].unique()) != members:
        raise ValueError("Detector members differ between config, calibration, and evaluation")
    if evaluation.duplicated([*BASE_KEY, "target_region", "detector"]).any():
        raise ValueError("Evaluation predictions contain duplicate keys")
    if calibration.duplicated([*BASE_KEY, "detector"]).any():
        raise ValueError("Calibration detector scores contain duplicate keys")
    if (calibration.groupby("POI_ID", observed=True)["source_region"].nunique() != 1).any():
        raise ValueError("A calibration POI changes source region across conditions")
    for frame, label in ((evaluation, "evaluation"), (calibration, "calibration")):
        clean_ids = set(frame.loc[frame["attack_label"].eq(0), "POI_ID"].astype(str))
        condition_sets = frame.loc[frame["attack_label"].eq(1)].groupby(
            ["seed", "generation_seed", "attack_condition"], observed=True
        )["POI_ID"].agg(lambda values: set(map(str, values)))
        if any(ids != clean_ids for ids in condition_sets):
            raise ValueError(f"{label} clean/attack POI identity is not paired")
    for frame, label in [(evaluation, "evaluation"), (calibration, "calibration")]:
        scores = frame["detector_score"].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{label} detector score is outside [0,1]")
        eligible = frame["primary_evaluation_eligible"].astype(bool)
        if (eligible & ~frame["constraints_valid"].astype(bool)).any():
            raise ValueError(f"{label} primary-valid row violates attack constraints")
    calibration_ids = set(calibration["POI_ID"].astype(str))
    evaluation_ids = set(evaluation["POI_ID"].astype(str))
    if calibration_ids & evaluation_ids:
        raise ValueError("Calibration and evaluation POIs overlap")

    pairs = evaluation[["seed", "generation_seed"]].drop_duplicates().sort_values(
        ["seed", "generation_seed"]
    )
    model_seeds = sorted(map(int, cfg["model_seeds"]))
    generation_seeds = sorted(map(int, cfg["generation_seeds"]))
    full = {(left, right) for left in model_seeds for right in generation_seeds}
    observed = set(map(tuple, pairs.to_numpy(dtype=int)))
    if observed == full:
        design = "full_model_by_generation_seed_cross"
        confounded = False
    elif observed == {(value, value) for value in model_seeds}:
        design = "matched_diagonal_model_generation_seeds"
        confounded = True
    else:
        design = "incomplete_seed_design"
        confounded = True
    for frame, label in ((evaluation, "evaluation"), (calibration, "calibration")):
        attack = frame.loc[frame["attack_label"].eq(1)]
        pair_conditions = attack.groupby(
            ["seed", "generation_seed"], observed=True
        )["attack_condition"].nunique()
        frame_pairs = set(map(tuple, pair_conditions.index.tolist()))
        if frame_pairs != full or not pair_conditions.eq(int(cfg["expected_attack_conditions"])).all():
            raise ValueError(
                f"{label} does not contain every attack condition for the full seed cross"
            )
    pairs["seed_design"] = design
    pairs["model_and_generation_seed_effects_confounded"] = confounded
    return pairs.reset_index(drop=True)


def _collapse_detector_scores(frame: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    duplicate_check = frame.groupby([*BASE_KEY, "detector"], observed=True)["detector_score"].agg(
        ["min", "max", "size"]
    )
    if not np.allclose(duplicate_check["min"], duplicate_check["max"]):
        raise ValueError("Detector score changes across repeated OvR rows")
    unique = frame.drop_duplicates([*BASE_KEY, "detector"]).copy()
    meta = unique.drop(columns=["detector", "detector_score"]).drop_duplicates(BASE_KEY)
    metadata_variation = unique.groupby(BASE_KEY, observed=True)[PROVENANCE].nunique(dropna=False)
    if (metadata_variation > 1).any().any() or meta.duplicated(BASE_KEY).any():
        raise ValueError("Detector metadata changes between ensemble members")
    scores = unique.pivot(index=BASE_KEY, columns="detector", values="detector_score").reset_index()
    if scores[members].isna().any().any():
        raise ValueError("A detector member is missing from a paired score row")
    return meta.merge(scores, on=BASE_KEY, validate="one_to_one")


def fit_detector_ensembles(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    members: list[str],
    retentions: list[float],
    logistic_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit ensemble scores only on calibration and apply frozen fits to evaluation."""
    cal = _collapse_detector_scores(calibration, members)
    eva = _collapse_detector_scores(evaluation, members)
    cal["predefined_mean_score"] = cal[members].mean(axis=1)
    eva["predefined_mean_score"] = eva[members].mean(axis=1)
    variants = list(members) + ["predefined_mean_score"]
    fit_rows = [{
        "detector_variant": "predefined_mean_score", "fit_type": "predefined_equal_mean",
        "fit_partition": "validation_calibration_half", "evaluation_partition": "disjoint_validation_evaluation_half",
        "coefficients": json.dumps([1 / len(members)] * len(members)), "intercept": 0.0,
        "fit_rows": 0, "fit_clean_rows": 0, "fit_attack_rows": 0,
    }]
    if logistic_config.get("enabled", False):
        variant = "calibration_logistic"
        cal[variant] = np.nan
        eva[variant] = np.nan
        variants.append(variant)
        for seed, seed_rows in cal.groupby("seed", sort=True):
            clean = seed_rows.loc[seed_rows["attack_label"].eq(0)].drop_duplicates("POI_ID")
            attacks = seed_rows.loc[
                seed_rows["attack_label"].eq(1)
                & seed_rows["primary_evaluation_eligible"].astype(bool)
            ]
            usable = pd.concat([clean, attacks], ignore_index=True)
            if set(usable["attack_label"].unique()) != {0, 1}:
                raise ValueError(f"Calibration logistic lacks both classes for model seed {seed}")
            model = LogisticRegression(
                class_weight=logistic_config.get("class_weight", "balanced"),
                max_iter=int(logistic_config.get("max_iter", 1000)),
                random_state=int(logistic_config.get("random_state", 0)),
            ).fit(usable[members], usable["attack_label"])
            cal_mask = cal["seed"].eq(seed)
            eva_mask = eva["seed"].eq(seed)
            cal.loc[cal_mask, variant] = model.predict_proba(cal.loc[cal_mask, members])[:, 1]
            eva.loc[eva_mask, variant] = model.predict_proba(eva.loc[eva_mask, members])[:, 1]
            fit_rows.append({
                "seed": seed, "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
                "detector_variant": variant,
                "fit_type": "logistic_regression", "fit_partition": "validation_calibration_half",
                "evaluation_partition": "disjoint_validation_evaluation_half",
                "coefficients": json.dumps(model.coef_[0].tolist()),
                "intercept": float(model.intercept_[0]), "fit_rows": len(usable),
                "fit_clean_rows": int(usable["attack_label"].eq(0).sum()),
                "fit_attack_rows": int(usable["attack_label"].eq(1).sum()),
                "generation_seed_fit_scope": "all_generation_seeds_balanced_factorial",
            })
        if cal[variant].isna().any() or eva[variant].isna().any():
            raise ValueError("Calibration logistic was not applied to every seed pair")

    thresholds = []
    for seed, group in cal.groupby("seed", sort=True):
        clean = group.loc[group["attack_label"].eq(0)].drop_duplicates("POI_ID")
        if len(clean) == 0:
            raise ValueError("No clean calibration scores for threshold selection")
        for variant in variants:
            for retention in retentions:
                threshold = select_detection_threshold(clean[variant], retention)
                thresholds.append({
                    "seed": seed,
                    "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
                    "detector_variant": variant, "clean_retention_target": retention,
                    "threshold": threshold, "calibration_rows": len(clean),
                    "calibration_clean_retention": float((clean[variant] <= threshold).mean()),
                    "selection_partition": "validation_calibration_half",
                    "reported_partition": "disjoint_validation_evaluation_half",
                })
    return cal, eva, pd.DataFrame(thresholds), pd.DataFrame(fit_rows)


def _region_frame(evaluation: pd.DataFrame) -> pd.DataFrame:
    # The proposed row carries the native score; fixed and shared scores are common to both detectors.
    common = evaluation.groupby([*BASE_KEY, "target_region"], observed=True)[
        ["truth", "fixed_unmodified_score", "shared_nearest_train_repair_score"]
    ].nunique(dropna=False)
    if (common > 1).any().any():
        raise ValueError("Common region scores differ between detector rows")
    columns = list(REGION_COLUMNS)
    if PROJECTED_REPAIR_COLUMN in evaluation:
        columns.append(PROJECTED_REPAIR_COLUMN)
    result = evaluation.loc[evaluation["detector"].eq(PROPOSED), columns].copy()
    if result.duplicated([*BASE_KEY, "target_region"]).any():
        raise ValueError("Proposed region rows are duplicated")
    probability_columns = [
        "fixed_unmodified_score", "shared_nearest_train_repair_score",
        "proposed_native_full_coverage_score",
    ]
    values = result[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Region probability is missing or outside [0,1]")
    if PROJECTED_REPAIR_COLUMN in result:
        projected = result[PROJECTED_REPAIR_COLUMN].to_numpy(dtype=float)
        if not np.isfinite(projected).all() or ((projected < 0) | (projected > 1)).any():
            raise ValueError("Proposed projected-repair fixed-TabPFN score is invalid")
    return result


def _condition_arrays(region: pd.DataFrame, detector: pd.DataFrame,
                      seed: int, generation_seed: int, condition: str,
                      variants: list[str], regions: list[str]) -> dict[str, Any]:
    part = region.loc[
        region["seed"].eq(seed) & region["generation_seed"].eq(generation_seed)
        & region["attack_condition"].eq(condition)
    ].copy()
    if part.empty:
        raise ValueError(f"Missing region predictions for {seed}/{generation_seed}/{condition}")
    ids = sorted(part["POI_ID"].astype(str).unique())
    index = pd.MultiIndex.from_product([ids, regions], names=["POI_ID", "target_region"])
    part = part.set_index(["POI_ID", "target_region"]).loc[index].reset_index()
    n = len(ids)
    values = {
        "ids": np.asarray(ids, dtype=object),
        "truth": part["truth"].to_numpy(dtype=np.int8).reshape(n, len(regions)),
        "fixed": part["fixed_unmodified_score"].to_numpy(dtype=float).reshape(n, len(regions)),
        "shared": part["shared_nearest_train_repair_score"].to_numpy(dtype=float).reshape(n, len(regions)),
        "native": part["proposed_native_full_coverage_score"].to_numpy(dtype=float).reshape(n, len(regions)),
    }
    values["projected"] = (
        part[PROJECTED_REPAIR_COLUMN].to_numpy(dtype=float).reshape(n, len(regions))
        if PROJECTED_REPAIR_COLUMN in part else None
    )
    poi = part.drop_duplicates("POI_ID").set_index("POI_ID").loc[ids]
    values["eligible"] = poi["primary_evaluation_eligible"].to_numpy(dtype=bool)
    values["changed"] = poi["attack_changed_any"].to_numpy(dtype=bool)
    for name in ["attack_method", "numeric_budget_m", "categorical_budget_l0"]:
        unique = poi[name].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"{name} is not condition-constant for {condition}")
        values[name] = unique.iloc[0]
    det = detector.loc[
        detector["seed"].eq(seed) & detector["generation_seed"].eq(generation_seed)
        & detector["attack_condition"].eq(condition)
    ].set_index("POI_ID").loc[ids]
    values["detector"] = {variant: det[variant].to_numpy(dtype=float) for variant in variants}
    return values


def _threshold_lookup(thresholds: pd.DataFrame) -> dict[tuple[int, int, str, float], float]:
    return {
        (int(row.seed), int(row.generation_seed), str(row.detector_variant),
         float(row.clean_retention_target)): float(row.threshold)
        for row in thresholds.itertuples(index=False)
    }


def _threshold_for_pair(
    thresholds: Mapping[tuple[int, int, str, float], float],
    seed: int,
    generation_seed: int,
    variant: str,
    retention: float,
) -> float:
    pair_key = (seed, generation_seed, variant, retention)
    canonical_key = (seed, CANONICAL_CLEAN_GENERATION_SEED, variant, retention)
    if pair_key in thresholds:
        return thresholds[pair_key]
    if canonical_key in thresholds:
        return thresholds[canonical_key]
    raise KeyError(f"No threshold for {pair_key} or canonical model-seed key")


def _prefix_metrics(values: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{name}": values[name] for name in ("n", "positives", *METRICS)}


def evaluate_sensitivity(
    region_frame: pd.DataFrame,
    detector_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    variants = sorted(thresholds["detector_variant"].unique())
    regions = sorted(region_frame["target_region"].unique())
    lookup = _threshold_lookup(thresholds)
    details: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    payload: list[dict[str, Any]] = []
    primary_retention = float(cfg["primary_clean_retention"])
    ratios = list(map(float, cfg["attack_mixture_ratios"]))
    retentions = list(map(float, cfg["clean_retention_targets"]))
    classification_threshold = float(cfg["classification_threshold"])
    pairs = sorted(region_frame[["seed", "generation_seed"]].drop_duplicates().itertuples(
        index=False, name=None
    ))
    _validate_clean_broadcast_invariant(region_frame, detector_frame, variants)
    emitted_clean_model_seeds: set[int] = set()
    for seed, generation_seed in pairs:
        seed, generation_seed = int(seed), int(generation_seed)
        clean = _condition_arrays(
            region_frame, detector_frame, seed, generation_seed, "clean", variants, regions
        )
        conditions = sorted(region_frame.loc[
            region_frame["seed"].eq(seed) & region_frame["generation_seed"].eq(generation_seed)
            & region_frame["attack_label"].eq(1), "attack_condition"
        ].unique())

        # Clean scores depend on model seed only. Store and aggregate them once, while the
        # immutable in-memory reference is shared with all generation-seed attacks.
        if seed not in emitted_clean_model_seeds:
            _evaluate_one_mixture(
                clean, clean, np.zeros(len(clean["ids"]), dtype=bool), "clean_only", 0.0,
                seed, CANONICAL_CLEAN_GENERATION_SEED, regions, variants, retentions, lookup,
                classification_threshold, primary_retention, details, coverage, payload,
            )
            emitted_clean_model_seeds.add(seed)
        for condition in conditions:
            attack = _condition_arrays(
                region_frame, detector_frame, seed, generation_seed, str(condition), variants, regions
            )
            if not np.array_equal(clean["ids"], attack["ids"]):
                raise ValueError(f"Clean/attack POI pairing differs for {condition}")
            for ratio in ratios:
                if ratio == 0:
                    continue
                mask = nested_attack_mask(clean["ids"], ratio, int(cfg["mixture_seed"]))
                _evaluate_one_mixture(
                    clean, attack, mask, str(condition), ratio, seed, generation_seed,
                    regions, variants, retentions, lookup, classification_threshold,
                    primary_retention, details, coverage, payload,
                )
    return pd.DataFrame(details), pd.DataFrame(coverage), payload


def _validate_clean_broadcast_invariant(
    region_frame: pd.DataFrame, detector_frame: pd.DataFrame, variants: Iterable[str]
) -> None:
    """Ensure in-memory generation anchors are exact copies of canonical clean scores."""
    region_columns = [
        "POI_ID", "target_region", "truth", "fixed_unmodified_score",
        "shared_nearest_train_repair_score", "proposed_native_full_coverage_score",
    ]
    if PROJECTED_REPAIR_COLUMN in region_frame:
        region_columns.append(PROJECTED_REPAIR_COLUMN)
    detector_columns = ["POI_ID", *variants]
    for seed in sorted(map(int, region_frame["seed"].unique())):
        region_clean = region_frame.loc[
            region_frame["seed"].eq(seed) & region_frame["attack_condition"].eq("clean")
        ]
        detector_clean = detector_frame.loc[
            detector_frame["seed"].eq(seed) & detector_frame["attack_condition"].eq("clean")
        ]
        expected_generations = sorted(map(int, region_clean["generation_seed"].unique()))
        if expected_generations != sorted(map(int, detector_clean["generation_seed"].unique())):
            raise ValueError("Clean region and detector generation anchors differ")
        baseline_region = None
        baseline_detector = None
        for generation_seed in expected_generations:
            current_region = region_clean.loc[
                region_clean["generation_seed"].eq(generation_seed), region_columns
            ].sort_values(["POI_ID", "target_region"]).reset_index(drop=True)
            current_detector = detector_clean.loc[
                detector_clean["generation_seed"].eq(generation_seed), detector_columns
            ].sort_values("POI_ID").reset_index(drop=True)
            if baseline_region is None:
                baseline_region, baseline_detector = current_region, current_detector
            elif not current_region.equals(baseline_region) or not current_detector.equals(
                baseline_detector
            ):
                raise ValueError(
                    f"Clean broadcast differs across generation seeds for model seed {seed}"
                )


def _evaluate_one_mixture(
    clean: Mapping[str, Any], attack: Mapping[str, Any], attacked: np.ndarray,
    condition: str, ratio: float, seed: int, generation_seed: int,
    regions: list[str], variants: list[str], retentions: list[float],
    thresholds: Mapping[tuple[int, int, str, float], float],
    classification_threshold: float, primary_retention: float,
    details: list[dict[str, Any]], coverage_rows: list[dict[str, Any]],
    payload: list[dict[str, Any]],
) -> None:
    select2d = attacked[:, None]
    truth = np.where(select2d, attack["truth"], clean["truth"])
    fixed = np.where(select2d, attack["fixed"], clean["fixed"])
    shared = np.where(select2d, attack["shared"], clean["shared"])
    native = np.where(select2d, attack["native"], clean["native"])
    projected = (
        np.where(select2d, attack["projected"], clean["projected"])
        if attack["projected"] is not None and clean["projected"] is not None else None
    )
    eligible = np.where(attacked, attack["eligible"], True)
    identity = {
        "seed": seed, "generation_seed": generation_seed,
        "attack_condition": condition, "attack_method": attack["attack_method"],
        "numeric_budget_m": attack["numeric_budget_m"],
        "categorical_budget_l0": attack["categorical_budget_l0"],
        "attack_mixture_ratio": ratio, "selected_attack_pois": int(attacked.sum()),
        "total_pois": len(attacked), "primary_valid_pois": int(eligible.sum()),
        "quarantined_selected_attack_pois": int((attacked & ~attack["eligible"]).sum()),
        "unchanged_selected_attack_pois": int((attacked & ~attack["changed"]).sum()),
    }
    for pipeline, score in [
        ("fixed_unmodified", fixed),
        ("proposed_native_full_coverage", native),
    ]:
        for index, region in enumerate(regions):
            metric = binary_metrics(truth[eligible, index], score[eligible, index],
                                    classification_threshold)
            details.append({
                **identity, "target_region": region, "pipeline": pipeline,
                "detector_variant": "none", "clean_retention_target": np.nan,
                "detection_threshold": np.nan, "coverage": 1.0,
                "classification_mode": "full_coverage",
                "roc_auc_defined": bool(np.isfinite(metric["roc_auc"])), **metric,
            })
            if ratio in (0.0, 1.0):
                payload.append({
                    "pipeline": pipeline, "cohort": "clean" if ratio == 0 else "attack",
                    "condition": condition, "seed": seed, "generation_seed": generation_seed,
                    "region": region, "ids": clean["ids"], "eligible": eligible.copy(),
                    "truth": truth[:, index].copy(), "score": score[:, index].copy(),
                })

    for variant in variants:
        detection = np.where(attacked, attack["detector"][variant], clean["detector"][variant])
        for retention in retentions:
            threshold = _threshold_for_pair(
                thresholds, seed, generation_seed, variant, retention
            )
            accepted = detection <= threshold
            for scope, scope_mask in [
                ("all_intent_diagnostic", np.ones(len(eligible), dtype=bool)),
                ("primary_valid", eligible),
            ]:
                coverage_rows.append({
                    **identity, "detector_variant": variant,
                    "clean_retention_target": retention, "detection_threshold": threshold,
                    "population_scope": scope, "denominator_pois": int(scope_mask.sum()),
                    "accepted_pois": int((accepted & scope_mask).sum()),
                    "coverage": (
                        float(accepted[scope_mask].mean()) if scope_mask.any() else np.nan
                    ),
                    "empty_population": not bool(scope_mask.any()),
                    "zero_division_policy": 0,
                    "robustness_evidence_admissible": scope == "primary_valid",
                })
            pipeline = f"{variant}__shared_nearest_train_repair"
            for index, region in enumerate(regions):
                values = paired_defense_metrics(
                    truth[eligible, index], fixed[eligible, index], shared[eligible, index],
                    detection[eligible], threshold, classification_threshold,
                )
                details.append({
                    **identity, "target_region": region, "pipeline": pipeline,
                    "detector_variant": variant, "clean_retention_target": retention,
                    "detection_threshold": threshold, "coverage": values["coverage"],
                    "classification_mode": "repaired_all",
                    "roc_auc_defined": bool(np.isfinite(values["repaired_all"]["roc_auc"])),
                    "accepted_only_roc_auc_defined": bool(
                        np.isfinite(values["accepted_only"]["roc_auc"])
                    ),
                    "accepted_only_empty": values["accepted_only"]["n"] == 0,
                    "accepted_only_single_class": (
                        values["accepted_only"]["n"] > 0
                        and values["accepted_only"]["positives"]
                        in (0, values["accepted_only"]["n"])
                    ),
                    **values["repaired_all"],
                    **_prefix_metrics(values["unfiltered"], "unfiltered"),
                    **_prefix_metrics(values["accepted_only"], "accepted_only"),
                    "accepted_and_correct_fraction": values["accepted_and_correct_fraction"],
                })
                if retention == primary_retention and ratio in (0.0, 1.0):
                    final_score = np.where(accepted, fixed[:, index], shared[:, index])
                    payload.append({
                        "pipeline": pipeline, "cohort": "clean" if ratio == 0 else "attack",
                        "condition": condition, "seed": seed, "generation_seed": generation_seed,
                        "region": region, "ids": clean["ids"], "eligible": eligible.copy(),
                        "truth": truth[:, index].copy(), "score": final_score.copy(),
                    })
            if variant == PROPOSED and projected is not None:
                projected_pipeline = (
                    "proposed_tabpfn25_attack_head__"
                    "proposed_projected_repair_fixed_tabpfn25"
                )
                for index, region in enumerate(regions):
                    projected_values = paired_defense_metrics(
                        truth[eligible, index], fixed[eligible, index],
                        projected[eligible, index], detection[eligible], threshold,
                        classification_threshold,
                    )
                    details.append({
                        **identity, "target_region": region, "pipeline": projected_pipeline,
                        "detector_variant": variant, "clean_retention_target": retention,
                        "detection_threshold": threshold,
                        "coverage": projected_values["coverage"],
                        "classification_mode": "proposed_projected_repair_fixed_tabpfn25",
                        "roc_auc_defined": bool(
                            np.isfinite(projected_values["repaired_all"]["roc_auc"])
                        ),
                        "accepted_only_roc_auc_defined": bool(
                            np.isfinite(projected_values["accepted_only"]["roc_auc"])
                        ),
                        "accepted_only_empty": projected_values["accepted_only"]["n"] == 0,
                        "accepted_only_single_class": (
                            projected_values["accepted_only"]["n"] > 0
                            and projected_values["accepted_only"]["positives"]
                            in (0, projected_values["accepted_only"]["n"])
                        ),
                        **projected_values["repaired_all"],
                        **_prefix_metrics(projected_values["unfiltered"], "unfiltered"),
                        **_prefix_metrics(projected_values["accepted_only"], "accepted_only"),
                        "accepted_and_correct_fraction": projected_values[
                            "accepted_and_correct_fraction"
                        ],
                    })
                    if retention == primary_retention and ratio in (0.0, 1.0):
                        final_projected = np.where(
                            accepted, fixed[:, index], projected[:, index]
                        )
                        payload.append({
                            "pipeline": projected_pipeline,
                            "cohort": "clean" if ratio == 0 else "attack",
                            "condition": condition, "seed": seed,
                            "generation_seed": generation_seed, "region": region,
                            "ids": clean["ids"], "eligible": eligible.copy(),
                            "truth": truth[:, index].copy(),
                            "score": final_projected.copy(),
                        })


def summarize_sensitivity(details: pd.DataFrame, primary_retention: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = [
        "attack_mixture_ratio", "pipeline", "detector_variant", "clean_retention_target",
        "classification_mode",
    ]
    summary = details.groupby(identity, dropna=False, observed=True)[
        [*METRICS, "coverage"]
    ].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = ["_".join(filter(None, value if isinstance(value, tuple) else (value,)))
                       for value in summary.columns]
    budget = details.loc[
        details["attack_mixture_ratio"].eq(1.0)
        & ((details["clean_retention_target"].eq(primary_retention))
           | details["clean_retention_target"].isna())
    ].groupby([
        "attack_method", "numeric_budget_m", "categorical_budget_l0", "pipeline"
    ], dropna=False, observed=True)[[*METRICS, "coverage"]].mean().reset_index()
    return summary, budget


def rank_stability(
    details: pd.DataFrame, primary_retention: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = details.loc[
        details["attack_mixture_ratio"].isin([0.0, 1.0])
        & ((details["clean_retention_target"].eq(primary_retention))
           | details["clean_retention_target"].isna())
    ]
    clean_group = ["pipeline", "seed", "target_region"]
    attack_group = ["pipeline", "seed", "generation_seed", "target_region"]
    clean = selected.loc[selected["attack_mixture_ratio"].eq(0)].groupby(
        clean_group, observed=True
    ).f1.mean().rename("clean_f1").reset_index()
    attack = selected.loc[selected["attack_mixture_ratio"].eq(1)].groupby(
        attack_group, observed=True
    ).f1.mean().rename("attack_f1").reset_index()
    ranks = attack.merge(clean, on=clean_group, how="left", validate="many_to_one")
    if ranks.isna().any().any():
        raise ValueError("Clean/attack ranking inputs are incomplete")
    ranks["robust_f1"] = (ranks["clean_f1"] + ranks["attack_f1"]) / 2
    ranks["rank"] = ranks.groupby(["seed", "generation_seed", "target_region"])[
        "robust_f1"
    ].rank(ascending=False, method="average")
    summary = ranks.groupby("pipeline").agg(
        mean_rank=("rank", "mean"), mean_robust_f1=("robust_f1", "mean"),
        std_robust_f1=("robust_f1", "std"), min_robust_f1=("robust_f1", "min"),
        max_robust_f1=("robust_f1", "max"),
    ).sort_values(["mean_rank", "mean_robust_f1"], ascending=[True, False]).reset_index()
    correlations = []
    pairs = sorted(ranks[["seed", "generation_seed"]].drop_duplicates().itertuples(index=False, name=None))
    for left_index, left in enumerate(pairs):
        left_values = (ranks.loc[
            ranks.seed.eq(left[0]) & ranks.generation_seed.eq(left[1])
        ].set_index(["pipeline", "target_region"])["rank"])
        for right in pairs[left_index + 1:]:
            right_values = (ranks.loc[
                ranks.seed.eq(right[0]) & ranks.generation_seed.eq(right[1])
            ].set_index(["pipeline", "target_region"])["rank"])
            common = left_values.index.intersection(right_values.index)
            rho, spearman_p = spearmanr(left_values.loc[common], right_values.loc[common])
            tau, kendall_p = kendalltau(left_values.loc[common], right_values.loc[common])
            comparison_type = seed_comparison_type(left, right)
            correlations.append({
                "left_model_seed": left[0], "left_generation_seed": left[1],
                "right_model_seed": right[0], "right_generation_seed": right[1],
                "comparison_type": comparison_type,
                "compared_pipeline_region_ranks": len(common), "spearman_rho": rho,
                "spearman_nominal_p": spearman_p, "kendall_tau": tau,
                "kendall_nominal_p": kendall_p,
                "p_value_role": "descriptive_only_no_multiplicity_or_independence_claim",
                "interpretation": (
                    "balanced-factorial descriptive rank stability; regions are not independent "
                    "experiments and three seeds do not support confirmatory inference"
                ),
            })
    marginals, variance = factorial_seed_stability(ranks)
    return ranks, summary, pd.DataFrame(correlations), marginals, variance


def seed_comparison_type(left: tuple[int, int], right: tuple[int, int]) -> str:
    if left[1] == right[1] and left[0] != right[0]:
        return "model_seed_effect_same_generation_seed"
    if left[0] == right[0] and left[1] != right[1]:
        return "generation_seed_effect_same_model_seed"
    return "both_seeds_changed_not_effect_isolating"


def factorial_seed_stability(ranks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate model/generation marginals and descriptive balanced-factorial variation."""
    model_seeds = sorted(map(int, ranks["seed"].unique()))
    generation_seeds = sorted(map(int, ranks["generation_seed"].unique()))
    expected = {(left, right) for left in model_seeds for right in generation_seeds}
    observed = set(map(tuple, ranks[["seed", "generation_seed"]].drop_duplicates().to_numpy()))
    if observed != expected:
        raise ValueError("Factorial stability requires the complete model/generation seed cross")

    marginal_rows: list[dict[str, Any]] = []
    for pipeline, group in ranks.groupby("pipeline", sort=True):
        for factor, column, other in (
            ("model_seed", "seed", "generation_seed"),
            ("generation_seed", "generation_seed", "seed"),
        ):
            for level, part in group.groupby(column, sort=True):
                marginal_rows.append({
                    "pipeline": pipeline,
                    "factor": factor,
                    "seed_level": int(level),
                    "marginal_mean_robust_f1": float(part["robust_f1"].mean()),
                    "marginal_std_robust_f1": float(part["robust_f1"].std()),
                    "other_seed_levels": int(part[other].nunique()),
                    "regions": int(part["target_region"].nunique()),
                    "descriptive_only": True,
                })

    variance_rows: list[dict[str, Any]] = []
    for pipeline, group in ranks.groupby("pipeline", sort=True):
        cells = group.groupby(["seed", "generation_seed"], observed=True)[
            "robust_f1"
        ].mean().unstack("generation_seed").loc[model_seeds, generation_seeds]
        values = cells.to_numpy(dtype=float)
        grand = float(values.mean())
        model_means = values.mean(axis=1)
        generation_means = values.mean(axis=0)
        ss_model = len(generation_seeds) * float(np.square(model_means - grand).sum())
        ss_generation = len(model_seeds) * float(
            np.square(generation_means - grand).sum()
        )
        residual = values - model_means[:, None] - generation_means[None, :] + grand
        ss_residual = float(np.square(residual).sum())
        ss_total = float(np.square(values - grand).sum())
        variance_rows.append({
            "pipeline": pipeline,
            "metric": "robust_f1_mean_over_regions",
            "model_seed_levels": len(model_seeds),
            "generation_seed_levels": len(generation_seeds),
            "balanced_cells": values.size,
            "ss_model_seed": ss_model,
            "ss_generation_seed": ss_generation,
            "ss_residual_including_interaction": ss_residual,
            "ss_total": ss_total,
            "model_seed_fraction": ss_model / ss_total if ss_total else 0.0,
            "generation_seed_fraction": ss_generation / ss_total if ss_total else 0.0,
            "residual_fraction": ss_residual / ss_total if ss_total else 0.0,
            "inferential_test_performed": False,
            "limitation": (
                "descriptive balanced 3x3 decomposition; one aggregate per cell cannot separate "
                "interaction from residual and regions are not independent replicates"
            ),
        })
    return pd.DataFrame(marginal_rows), pd.DataFrame(variance_rows)


def stage14_expected_counts(
    *,
    model_seeds: int,
    generation_seeds: int,
    attack_conditions: int,
    regions: int,
    ratios: int,
    detector_variants: int,
    retentions: int,
    projected_repair: bool,
) -> dict[str, int]:
    """Exact cardinalities after canonical clean de-duplication."""
    seed_pairs = model_seeds * generation_seeds
    mixture_units = model_seeds + seed_pairs * attack_conditions * (ratios - 1)
    projected_per_mixture = retentions if projected_repair else 0
    rows_per_mixture = regions * (
        2 + detector_variants * retentions + projected_per_mixture
    )
    coverage_per_mixture = detector_variants * retentions * 2
    pipelines = 2 + detector_variants + int(projected_repair)
    payload_per_endpoint = regions * pipelines
    return {
        "model_generation_seed_pairs": seed_pairs,
        "canonical_clean_mixtures": model_seeds,
        "attack_or_mixed_mixtures": seed_pairs * attack_conditions * (ratios - 1),
        "sensitivity_detail_rows": mixture_units * rows_per_mixture,
        "coverage_rows": mixture_units * coverage_per_mixture,
        "ranking_payload_strata": (
            model_seeds + seed_pairs * attack_conditions
        ) * payload_per_endpoint,
        "rank_rows": pipelines * seed_pairs * regions,
        "rank_pair_correlations": seed_pairs * (seed_pairs - 1) // 2,
        "seed_marginal_rows": pipelines * (model_seeds + generation_seeds),
        "variance_decomposition_rows": pipelines,
        "canonical_threshold_rows": model_seeds * detector_variants * retentions,
    }


def _weighted_f1(truth: np.ndarray, score: np.ndarray, eligible: np.ndarray,
                 weights: np.ndarray, threshold: float) -> np.ndarray:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(score) >= threshold
    mask = np.asarray(eligible, dtype=bool)
    current = weights[:, mask]
    truth, predicted = truth[mask], predicted[mask]
    tp = current @ (truth & predicted).astype(np.int64)
    fp = current @ (~truth & predicted).astype(np.int64)
    fn = current @ (truth & ~predicted).astype(np.int64)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    )


def bootstrap_ranking(payload: list[dict[str, Any]], cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not payload:
        raise ValueError("No ranking payload")
    ids = payload[0]["ids"]
    if any(not np.array_equal(item["ids"], ids) for item in payload):
        raise ValueError("Paired bootstrap requires the same ordered evaluation POIs")
    ranking_cfg = cfg["ranking"]
    weights = make_bootstrap_weights(
        len(ids), int(ranking_cfg["bootstrap_repeats"]), int(ranking_cfg["bootstrap_seed"])
    )
    accumulated: dict[tuple[str, str], np.ndarray] = {}
    finite_counts: dict[tuple[str, str], np.ndarray] = {}
    intended_strata: dict[tuple[str, str], int] = {}
    point_accumulated: dict[tuple[str, str], float] = {}
    point_counts: dict[tuple[str, str], int] = {}
    point_weights = np.ones((1, len(ids)), dtype=np.uint16)
    for item in payload:
        key = (item["pipeline"], item["cohort"])
        values = _weighted_f1(
            item["truth"], item["score"], item["eligible"], weights,
            float(cfg["classification_threshold"]),
        )
        finite = np.isfinite(values)
        accumulated[key] = accumulated.get(key, np.zeros(len(weights))) + np.nan_to_num(values)
        finite_counts[key] = finite_counts.get(key, np.zeros(len(weights), dtype=int)) + finite
        intended_strata[key] = intended_strata.get(key, 0) + 1
        point = _weighted_f1(
            item["truth"], item["score"], item["eligible"], point_weights,
            float(cfg["classification_threshold"]),
        )[0]
        if not np.isfinite(point):
            raise ValueError("Point-estimate F1 violated the frozen zero-division policy")
        point_accumulated[key] = point_accumulated.get(key, 0.0) + float(point)
        point_counts[key] = point_counts.get(key, 0) + 1
    alpha = (1 - float(ranking_cfg["confidence_level"])) / 2
    robust: dict[str, np.ndarray] = {}
    rows = []
    pipelines = sorted({key[0] for key in accumulated})
    for pipeline in pipelines:
        clean = np.divide(
            accumulated[(pipeline, "clean")], finite_counts[(pipeline, "clean")],
            out=np.full(len(weights), np.nan),
            where=finite_counts[(pipeline, "clean")] == intended_strata[(pipeline, "clean")],
        )
        attack = np.divide(
            accumulated[(pipeline, "attack")], finite_counts[(pipeline, "attack")],
            out=np.full(len(weights), np.nan),
            where=finite_counts[(pipeline, "attack")] == intended_strata[(pipeline, "attack")],
        )
        values = (clean + attack) / 2
        robust[pipeline] = values
        point_clean = (
            point_accumulated.get((pipeline, "clean"), np.nan)
            / point_counts.get((pipeline, "clean"), np.nan)
        )
        point_attack = (
            point_accumulated.get((pipeline, "attack"), np.nan)
            / point_counts.get((pipeline, "attack"), np.nan)
        )
        point_robust = (point_clean + point_attack) / 2
        clean_point_required = intended_strata[(pipeline, "clean")]
        attack_point_required = intended_strata[(pipeline, "attack")]
        if (point_counts.get((pipeline, "clean"), 0) != clean_point_required
                or point_counts.get((pipeline, "attack"), 0) != attack_point_required):
            raise ValueError("Point and bootstrap ranking strata differ")
        finite = np.isfinite(values)
        rows.append({
            "pipeline": pipeline,
            "point_estimate_robust_f1": point_robust,
            "bootstrap_mean_robust_f1": np.nanmean(values) if finite.any() else np.nan,
            "ci_low": np.nanquantile(values, alpha) if finite.any() else np.nan,
            "ci_high": np.nanquantile(values, 1 - alpha) if finite.any() else np.nan,
            "bootstrap_repeats": len(weights), "total_draws": len(weights),
            "confidence_level": ranking_cfg["confidence_level"],
            "finite_draws": int(finite.sum()), "undefined_draws": int((~finite).sum()),
            "interval_defined": bool(finite.any()),
            "clean_strata_required_per_draw": intended_strata[(pipeline, "clean")],
            "attack_strata_required_per_draw": intended_strata[(pipeline, "attack")],
            "clean_point_strata_finite": point_counts[(pipeline, "clean")],
            "attack_point_strata_finite": point_counts[(pipeline, "attack")],
            "f1_zero_division_policy": 0,
            "bootstrap_unit": "POI_ID", "selection_conditional": True,
            "interval_method": "marginal percentile; not simultaneous",
        })
    reference = str(ranking_cfg["reference_pipeline"])
    if reference not in robust:
        raise ValueError(f"Configured ranking reference is absent: {reference}")
    pairwise = []
    for pipeline, values in robust.items():
        if pipeline == reference:
            continue
        delta = values - robust[reference]
        finite = np.isfinite(delta)
        pairwise.append({
            "pipeline": pipeline, "reference_pipeline": reference,
            "point_estimate_difference": (
                next(row["point_estimate_robust_f1"] for row in rows if row["pipeline"] == pipeline)
                - next(row["point_estimate_robust_f1"] for row in rows if row["pipeline"] == reference)
            ),
            "bootstrap_mean_difference": np.nanmean(delta) if finite.any() else np.nan,
            "ci_low": np.nanquantile(delta, alpha) if finite.any() else np.nan,
            "ci_high": np.nanquantile(delta, 1 - alpha) if finite.any() else np.nan,
            "probability_difference_gt_zero": np.mean(delta[finite] > 0) if finite.any() else np.nan,
            "total_draws": len(delta), "finite_draws": int(finite.sum()),
            "undefined_draws": int((~finite).sum()), "interval_defined": bool(finite.any()),
            "bootstrap_unit": "POI_ID", "selection_conditional": True,
            "limitation": "conditional on primary-valid attacks and validation-selected pipelines",
        })
    return pd.DataFrame(rows), pd.DataFrame(pairwise)


def assess_ablations(cfg: Mapping[str, Any]) -> pd.DataFrame:
    matrix = Path(cfg["ablation_matrix"])
    if not matrix.is_file():
        return pd.DataFrame([{
            "variant": "ablation_matrix", "kind": "unknown", "configured_to_train": False,
            "output": "", "status": "NOT_RUN", "missing": str(matrix),
        }])
    rows = []
    required = list(cfg.get("ablation_required_files", ["metadata.json"]))
    matrix_config = yaml.safe_load(matrix.read_text())
    variants = matrix_config.get("variants", [])
    if not variants:
        raise ValueError("Ablation matrix contains no variants")
    for variant in variants:
        name = str(variant["name"])
        train = bool(variant.get("train", True))
        output = Path(variant["output"])
        missing = [value for value in required if not (output / value).is_file()]
        rows.append({
            "variant": name, "kind": str(variant["kind"]),
            "configured_to_train": train, "output": str(output),
            "status": "AVAILABLE_PENDING_REVIEW" if not missing else "NOT_RUN",
            "missing": "|".join(missing),
            "changes": "|".join(variant.get("changes", [])),
        })
    return pd.DataFrame(rows)


def assess_extended_plan(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Record required Stage-14 experiments that this derived evaluator does not execute."""
    required = {
        "leave_one_attack_out_generalization", "leave_one_source_model_out_transfer",
        "adaptive_detector_evasion_attack", "regional_probability_ensemble_calibration",
    }
    if set(map(str, cfg.get("required_extended_experiments", []))) != required:
        raise ValueError("Stage 14 required extended-experiment blockers are incomplete")
    source_models = list(map(str, cfg.get("available_attack_source_models", ["surrogate_mlp"])))
    rows = [
        {
            "requirement_id": "leave_one_attack_out_generalization",
            "status": "NOT_RUN", "blocking_for_test_unlock": True,
            "implemented_by_this_runner": False,
            "observed_scope": "derived sensitivity over existing Stage-10 prediction shards",
            "limitation": "requires retraining/evaluation with each attack family held out",
        },
        {
            "requirement_id": "leave_one_source_model_out_transfer",
            "status": "NOT_RUN", "blocking_for_test_unlock": True,
            "implemented_by_this_runner": False,
            "observed_scope": "available attack source models: " + ",".join(source_models),
            "limitation": (
                "LOSO transfer is not identifiable with fewer than two independent attack "
                "source models; the current configured source is surrogate_mlp only"
            ),
        },
        {
            "requirement_id": "adaptive_detector_evasion_attack",
            "status": "NOT_RUN", "blocking_for_test_unlock": True,
            "implemented_by_this_runner": False,
            "observed_scope": "no detector-aware adaptive attack generation in Stage 14",
            "limitation": "requires a separately generated and constraint-validated attack set",
        },
        {
            "requirement_id": "regional_probability_ensemble_calibration",
            "status": "NOT_RUN", "blocking_for_test_unlock": True,
            "implemented_by_this_runner": False,
            "observed_scope": "detector-score ensemble only",
            "limitation": (
                "Section-12 regional model probability ensemble/calibration is distinct from "
                "the proposed/URE detector-score ensemble evaluated here"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _save_figures(details: pd.DataFrame, coverage: pd.DataFrame, ranks: pd.DataFrame,
                  budget: pd.DataFrame, cfg: Mapping[str, Any], figures: Path) -> None:
    primary = float(cfg["primary_clean_retention"])
    view = details.loc[
        details["clean_retention_target"].eq(primary)
        | details["clean_retention_target"].isna()
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=view, x="attack_mixture_ratio", y="f1", hue="pipeline",
                 estimator="mean", errorbar=None, marker="o", ax=ax)
    ax.set(title="Primary-valid attack-mixture sensitivity", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figures / "attack_mixture_f1.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    primary_coverage = coverage.loc[coverage["population_scope"].eq("primary_valid")]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=primary_coverage, x="clean_retention_target", y="coverage",
                 hue="detector_variant", estimator="mean", errorbar=None, marker="o", ax=ax)
    ax.set(title="Clean-retention sensitivity of primary-valid coverage", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figures / "retention_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    heat = ranks.pivot_table(index="target_region", columns="pipeline", values="rank", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="rocket_r", ax=ax)
    ax.set_title("Descriptive mean pipeline rank by region")
    fig.tight_layout(); fig.savefig(figures / "region_rank_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    numeric = budget.loc[budget["numeric_budget_m"].notna()]
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(numeric):
        sns.lineplot(data=numeric, x="numeric_budget_m", y="f1", hue="pipeline",
                     style="attack_method", marker="o", ax=ax)
    ax.set(title="Primary-valid numeric/mixed budget sensitivity", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figures / "attack_budget_f1.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    if output_override is not None:
        cfg["output"] = str(output_override)
    if tuple(map(int, cfg["model_seeds"])) != (42, 202, 340):
        raise ValueError("Stage 14 requires model seeds 42, 202, and 340")
    if tuple(map(int, cfg["generation_seeds"])) != (42, 202, 340):
        raise ValueError("Stage 14 requires generation seeds 42, 202, and 340")
    if cfg.get("seed_design") != "full_model_by_generation_seed_cross":
        raise ValueError("Stage 14 requires the complete model/generation seed cross")
    if int(cfg.get("canonical_clean_generation_seed")) != CANONICAL_CLEAN_GENERATION_SEED:
        raise ValueError("Stage 14 canonical clean generation seed must be -1")
    if tuple(map(float, cfg["attack_mixture_ratios"])) != (0, .1, .25, .5, .75, 1):
        raise ValueError("Stage 14 attack-mixture ratios differ from the frozen plan")
    if tuple(map(float, cfg["clean_retention_targets"])) != (.9, .95, .975, .99):
        raise ValueError("Stage 14 clean-retention targets differ from the frozen plan")
    preflight_status = _require_sources(config_path, cfg)
    evaluation, calibration = _load_prediction_shards(cfg)
    seed_design = _validate_sources(evaluation, calibration, cfg)

    output = Path(cfg["output"]); tables = output / "tables"; figures = output / "figures"
    members = list(map(str, cfg["detector_members"]))
    _cal_wide, eval_wide, thresholds, fits = fit_detector_ensembles(
        calibration, evaluation, members, list(map(float, cfg["clean_retention_targets"])),
        cfg["detector_ensembles"]["calibration_logistic"],
    )
    region = _region_frame(evaluation)
    details, coverage, payload = evaluate_sensitivity(region, eval_wide, thresholds, cfg)
    summary, budget = summarize_sensitivity(details, float(cfg["primary_clean_retention"]))
    ranks, rank_summary, correlations, seed_marginals, variance_decomposition = rank_stability(
        details, float(cfg["primary_clean_retention"])
    )
    bootstrap, pairwise = bootstrap_ranking(payload, cfg)
    ablations = assess_ablations(cfg)
    extended_plan = assess_extended_plan(cfg)

    # Do not leave an apparently started stage when an in-memory validation or analysis fails.
    tables.mkdir(parents=True)
    figures.mkdir()

    source_rows = []
    for role, path in [
        ("stage10_generated_marker", Path(cfg["stage10_output"]) / "_GENERATED.json"),
        ("prediction_manifest", Path(cfg["prediction_manifest"])),
        ("calibration_detector_scores", Path(cfg["calibration_detector_scores"])),
    ]:
        source_rows.append({"role": role, "path": str(path), "bytes": path.stat().st_size,
                            "sha256": _sha256(path)})
    outputs = {
        "preflight.csv": preflight_status, "source_provenance.csv": pd.DataFrame(source_rows),
        "seed_design.csv": seed_design, "detector_ensemble_fits.csv": fits,
        "detector_thresholds.csv": thresholds, "attempt_coverage.csv": coverage,
        "sensitivity_by_region.csv": details, "sensitivity_summary.csv": summary,
        "budget_sensitivity.csv": budget, "rank_by_seed_region.csv": ranks,
        "rank_summary.csv": rank_summary, "rank_stability.csv": correlations,
        "seed_effect_marginals.csv": seed_marginals,
        "seed_variance_decomposition.csv": variance_decomposition,
        "ranking_bootstrap_intervals.csv": bootstrap,
        "ranking_pairwise_vs_reference.csv": pairwise,
        "ablation_assessment.csv": ablations,
        "extended_plan_assessment.csv": extended_plan,
    }
    for name, frame in outputs.items():
        _atomic_csv(frame, tables / name)
    _save_figures(details, coverage, ranks, budget, cfg, figures)

    confounded = bool(seed_design["model_and_generation_seed_effects_confounded"].iloc[0])
    missing_ablations = ablations.loc[ablations["status"].eq("NOT_RUN"), "variant"].tolist()
    extended_blockers = extended_plan.loc[
        extended_plan["blocking_for_test_unlock"] & extended_plan["status"].ne("PASS"),
        "requirement_id",
    ].tolist()
    variants = int(thresholds["detector_variant"].nunique())
    expected_counts = stage14_expected_counts(
        model_seeds=len(cfg["model_seeds"]),
        generation_seeds=len(cfg["generation_seeds"]),
        attack_conditions=int(cfg["expected_attack_conditions"]),
        regions=int(cfg["expected_regions"]),
        ratios=len(cfg["attack_mixture_ratios"]),
        detector_variants=variants,
        retentions=len(cfg["clean_retention_targets"]),
        projected_repair=PROJECTED_REPAIR_COLUMN in evaluation,
    )
    observed_counts = {
        "model_generation_seed_pairs": len(seed_design),
        "canonical_clean_mixtures": int(
            details.loc[details["attack_mixture_ratio"].eq(0), ["seed"]].drop_duplicates().shape[0]
        ),
        "attack_or_mixed_mixtures": int(
            details.loc[details["attack_mixture_ratio"].gt(0), [
                "seed", "generation_seed", "attack_condition", "attack_mixture_ratio"
            ]].drop_duplicates().shape[0]
        ),
        "sensitivity_detail_rows": len(details),
        "coverage_rows": len(coverage),
        "ranking_payload_strata": len(payload),
        "rank_rows": len(ranks),
        "rank_pair_correlations": len(correlations),
        "seed_marginal_rows": len(seed_marginals),
        "variance_decomposition_rows": len(variance_decomposition),
        "canonical_threshold_rows": len(thresholds),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Stage-14 cross-design cardinality mismatch: expected={expected_counts}, "
            f"observed={observed_counts}"
        )
    metadata = {
        "status": "generated_pending_independent_review",
        "scope": "validation_calibration_fit_then_disjoint_validation_evaluation",
        "test_rows_read": 0,
        "model_seeds": cfg["model_seeds"], "generation_seeds": cfg["generation_seeds"],
        "seed_design": seed_design["seed_design"].iloc[0],
        "model_generation_seed_effects_confounded": confounded,
        "canonical_clean_generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
        "clean_macro_aggregation": (
            "one immutable clean reference per model seed; not counted once per generation seed"
        ),
        "cross_design_expected_counts": expected_counts,
        "cross_design_observed_counts": observed_counts,
        "attack_mixture_ratios": cfg["attack_mixture_ratios"],
        "clean_retention_targets": cfg["clean_retention_targets"],
        "primary_detector_ensemble": "predefined_mean_score",
        "optional_detector_ensemble": "calibration_logistic",
        "proposed_native_score_role": (
            "separate proposed own-head full-coverage baseline; not the fixed TabPFN 2.5 "
            "same-downstream comparison"
        ),
        "proposed_projected_repair_fixed_score_available": PROJECTED_REPAIR_COLUMN in evaluation,
        "ensemble_fit_partition": (
            "validation calibration half only; logistic attack rows pool the balanced generation-"
            "seed cross once per model seed and clean rows are de-duplicated"
        ),
        "reported_partition": "disjoint validation evaluation half only",
        "primary_robustness_scope": "primary_valid",
        "all_intent_scope": "diagnostic coverage only; inadmissible as robustness evidence",
        "ranking_uncertainty": "paired POI bootstrap, selection-conditional marginal percentile intervals",
        "seed_effect_interpretation": (
            "model-seed and generation-seed marginals plus balanced 3x3 descriptive sums of "
            "squares; no confirmatory seed-effect inference"
        ),
        "model_selection_limitation": (
            "The frozen downstream model was selected in the earlier clean comparison using the "
            "whole validation set; this disjoint calibration/evaluation split is honest for Stage 14 "
            "ensemble fitting but is not a fully unseen model-selection test."
        ),
        "regions_are_independent_replicates": False,
        "ablation_not_run": missing_ablations,
        "extended_plan_not_run": extended_blockers,
        "test_unlock_permitted": False,
        "stage14_scientific_completion_claim_permitted": False,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn"]},
        "seconds": time.time() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "metrics.json").write_text(json.dumps({
        "rank_summary": rank_summary.to_dict(orient="records"),
        "ranking_bootstrap": bootstrap.to_dict(orient="records"),
    }, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Stage 14 sensitivity, stability, ranking, and ensembles\n\n"
        "Detector ensembles and all retention thresholds were fitted only on the validation "
        "calibration half, frozen, and evaluated on disjoint validation-evaluation POIs. The "
        "predefined equal mean of the proposed and URE-RF detector scores is the primary ensemble; "
        "the logistic stacker is optional and calibration-fitted. The proposed native own-head "
        "score is reported as a separate full-coverage baseline, not as the same fixed-TabPFN "
        "downstream comparison. When Stage 10 supplies it, projected repair is evaluated by sending "
        "rejected rows through the same frozen TabPFN 2.5 classifier.\n\n"
        "Robustness metrics use only primary-valid attacks. All generated intents, including "
        "quarantined and unchanged attempts, appear only in diagnostic coverage tables. Regional "
        "and seed rows are descriptive strata, not independent experiments; ranking intervals use "
        "a shared paired POI bootstrap and remain conditional on validation selection. The frozen "
        "downstream model was selected in the earlier whole-validation comparison, so this is not "
        "a fully unseen model-selection test. Undefined ROC-AUC draws and single-class accepted "
        "subsets remain NaN and are counted explicitly; F1, precision, and recall use the declared "
        "zero_division=0 convention in both point estimates and bootstrap draws.\n\n"
        + ("Model-seed and attack-generation-seed effects remain confounded because the Stage 10 "
           "seed cross is incomplete.\n\n" if confounded else
           "The balanced 3x3 model-seed by attack-generation-seed cross supports separate "
           "descriptive marginals and sums-of-squares only. Three seeds and non-independent "
           "regions do not support confirmatory seed-effect inference.\n\n")
        + ("The following configured ablations remain NOT_RUN: " + ", ".join(missing_ablations)
           + ".\n\n" if missing_ablations else "")
        + "The following original-plan experiments remain NOT_RUN and block locked-test unlock: "
        + ", ".join(extended_blockers)
        + ". The regional probability ensemble/calibration requirement is separate from the "
        "detector-score ensemble reported here.\n\n"
        + "No locked-test rows were read. Results remain pending independent review.\n"
    )
    marker = {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "metadata_sha256": _sha256(output / "metadata.json"),
        "sensitivity_sha256": _sha256(tables / "sensitivity_by_region.csv"),
        "ranking_sha256": _sha256(tables / "rank_summary.csv"),
    }
    (output / "_GENERATED.json").write_text(json.dumps(marker, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stages/14_sensitivity_stability_ranking.yaml")
    parser.add_argument("--output")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        status = preflight(args.config)
        print(status.to_json(orient="records", indent=2))
        if (status["blocking"] & ~status["exists"]).any():
            raise SystemExit(2)
        return
    run(args.config, args.output)


if __name__ == "__main__":
    main()
