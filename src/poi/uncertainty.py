"""Paired POI bootstrap uncertainty for the Stage 06 validation predictions.

The resampling unit is POI_ID.  A single matrix of multinomial POI weights is
reused for every model, attack condition, OvR region, and model seed.  Region
and seed results are aggregated inside each replicate; they are never treated
as independent bootstrap observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
KEY = ["model", "condition", "seed", "region"]
PREDICTION_COLUMNS = ["POI_ID", *KEY, "split", "y_true", "y_score", "y_pred"]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_bootstrap_weights(n_pois: int, repeats: int, seed: int) -> np.ndarray:
    """Return deterministic POI multiplicities, one row per bootstrap draw."""
    if n_pois < 2 or repeats < 1:
        raise ValueError("Bootstrap needs at least two POIs and one repeat")
    probabilities = np.full(n_pois, 1.0 / n_pois)
    weights = np.random.default_rng(seed).multinomial(
        n_pois, probabilities, size=repeats
    )
    if weights.max() > np.iinfo(np.uint16).max:
        raise OverflowError("Bootstrap multiplicity does not fit uint16")
    return weights.astype(np.uint16)


@njit(cache=True, parallel=True)
def _weighted_auc_replicates(
    y_true: np.ndarray,
    y_score: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Exact weighted ROC-AUC for each bootstrap row, including score ties."""
    order = np.argsort(y_score)
    repeats = weights.shape[0]
    result = np.empty(repeats, dtype=np.float64)
    for repeat in prange(repeats):
        positive_total = 0.0
        negative_total = 0.0
        for position in range(order.size):
            index = order[position]
            if y_true[index] == 1:
                positive_total += weights[repeat, index]
            else:
                negative_total += weights[repeat, index]

        numerator = 0.0
        negative_before = 0.0
        position = 0
        while position < order.size:
            score = y_score[order[position]]
            tied_positive = 0.0
            tied_negative = 0.0
            end = position
            while end < order.size and y_score[order[end]] == score:
                index = order[end]
                if y_true[index] == 1:
                    tied_positive += weights[repeat, index]
                else:
                    tied_negative += weights[repeat, index]
                end += 1
            numerator += tied_positive * (negative_before + 0.5 * tied_negative)
            negative_before += tied_negative
            position = end

        denominator = positive_total * negative_total
        result[repeat] = numerator / denominator if denominator > 0 else np.nan
    return result


def weighted_metric_replicates(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the five requested metrics for every POI bootstrap draw."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.uint16)
    if y_true.ndim != 1 or y_score.shape != y_true.shape or y_pred.shape != y_true.shape:
        raise ValueError("Prediction arrays must be aligned one-dimensional vectors")
    if weights.ndim != 2 or weights.shape[1] != y_true.size:
        raise ValueError("Bootstrap weight columns must match the POI rows")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_true and y_pred must be binary")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains non-finite values")

    true_positive = weights @ ((y_true == 1) & (y_pred == 1)).astype(np.int64)
    false_positive = weights @ ((y_true == 0) & (y_pred == 1)).astype(np.int64)
    false_negative = weights @ ((y_true == 1) & (y_pred == 0)).astype(np.int64)
    true_negative = weights @ ((y_true == 0) & (y_pred == 0)).astype(np.int64)

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    return {
        "f1": divide(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "precision": divide(true_positive, true_positive + false_positive),
        "recall": divide(true_positive, true_positive + false_negative),
        "accuracy": divide(true_positive + true_negative,
                           true_positive + true_negative + false_positive + false_negative),
        "roc_auc": _weighted_auc_replicates(y_true, y_score, weights),
    }


def _point_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   y_pred: np.ndarray) -> dict[str, float]:
    ones = np.ones((1, len(y_true)), dtype=np.uint16)
    return {name: float(values[0]) for name, values in
            weighted_metric_replicates(y_true, y_score, y_pred, ones).items()}


def _validate_prediction_frame(
    frame: pd.DataFrame,
    source: str,
    expected_pois: int,
    expected_regions: int,
    expected_seeds: set[int],
) -> None:
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} predictions lack columns: {sorted(missing)}")
    if set(frame["split"].unique()) != {"validation"}:
        raise ValueError(f"{source} predictions must be validation-only")
    if frame["POI_ID"].nunique() != expected_pois:
        raise ValueError(f"{source} has an unexpected number of unique POIs")
    if frame["region"].nunique() != expected_regions:
        raise ValueError(f"{source} has an unexpected number of regions")
    if set(frame["seed"].unique()) != expected_seeds:
        raise ValueError(f"{source} has an unexpected seed set")
    if frame.duplicated([*KEY, "POI_ID"]).any():
        raise ValueError(f"{source} has duplicate prediction keys")
    sizes = frame.groupby(KEY, observed=True).size()
    if sizes.empty or not sizes.eq(expected_pois).all():
        raise ValueError(f"{source} prediction strata are incomplete")
    if not np.isin(frame["y_true"], [0, 1]).all() or not np.isin(frame["y_pred"], [0, 1]).all():
        raise ValueError(f"{source} has non-binary labels or predictions")
    if not np.isfinite(frame["y_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} has non-finite scores")


def _aligned_arrays(group: pd.DataFrame, poi_order: np.ndarray) -> tuple[np.ndarray, ...]:
    ids = group["POI_ID"].to_numpy()
    if not np.array_equal(ids, poi_order):
        indexed = group.set_index("POI_ID", verify_integrity=True)
        try:
            group = indexed.loc[poi_order]
        except KeyError as error:
            raise ValueError("A prediction stratum does not contain the common POI set") from error
    return (
        group["y_true"].to_numpy(dtype=np.int8),
        group["y_score"].to_numpy(dtype=np.float64),
        group["y_pred"].to_numpy(dtype=np.int8),
    )


def _check_truth_consistency(frame: pd.DataFrame, poi_order: np.ndarray) -> dict[str, np.ndarray]:
    truth = {}
    first_model = str(frame["model"].iloc[0])
    first_condition = str(frame["condition"].iloc[0])
    first_seed = int(frame["seed"].iloc[0])
    baseline = frame.loc[
        frame["model"].eq(first_model)
        & frame["condition"].eq(first_condition)
        & frame["seed"].eq(first_seed)
    ]
    for region, group in baseline.groupby("region", sort=True, observed=True):
        truth[str(region)] = _aligned_arrays(group, poi_order)[0]
    for keys, group in frame.groupby(KEY, sort=False, observed=True):
        region = str(keys[3])
        if not np.array_equal(_aligned_arrays(group, poi_order)[0], truth[region]):
            raise ValueError(f"y_true changes across prediction strata at {keys}")
    return truth


def _load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    clean_path = Path(cfg["clean_predictions"])
    attack_path = Path(cfg["attack_predictions"])
    clean = pd.read_parquet(clean_path, columns=PREDICTION_COLUMNS)
    attack = pd.read_parquet(attack_path, columns=PREDICTION_COLUMNS)
    for column in ["model", "condition", "region", "split"]:
        clean[column] = clean[column].astype("category")
        attack[column] = attack[column].astype("category")
    expected_seeds = set(map(int, cfg["expected_seeds"]))
    _validate_prediction_frame(clean, "clean", cfg["expected_pois"],
                               cfg["expected_regions"], expected_seeds)
    _validate_prediction_frame(attack, "attack", cfg["expected_pois"],
                               cfg["expected_regions"], expected_seeds)
    if clean["model"].nunique() != cfg["expected_models"] or set(clean["condition"].astype(str)) != {"clean"}:
        raise ValueError("Clean predictions do not contain the expected model comparison")
    attack_conditions = set(attack["condition"].astype(str))
    if len(attack_conditions - {cfg["reference_condition"]}) != cfg["expected_attack_conditions"]:
        raise ValueError("Attack predictions do not contain the expected attack conditions")
    if set(attack["model"].astype(str)) != {cfg["reference_model"]}:
        raise ValueError("Attack predictions are not for the configured reference model")

    reference = clean.loc[clean["model"].astype(str).eq(cfg["reference_model"])].copy()
    attack_clean = attack.loc[attack["condition"].astype(str).eq(cfg["reference_condition"])].copy()
    sort_columns = [*KEY, "POI_ID"]
    reference = reference.sort_values(sort_columns).reset_index(drop=True)
    attack_clean = attack_clean.sort_values(sort_columns).reset_index(drop=True)
    for column in ["POI_ID", *KEY, "split", "y_true", "y_pred"]:
        if not np.array_equal(reference[column].astype(str).to_numpy(),
                              attack_clean[column].astype(str).to_numpy()):
            raise ValueError(f"Clean reference mismatch between source files in {column}")
    if not np.array_equal(reference["y_score"].to_numpy(), attack_clean["y_score"].to_numpy()):
        raise ValueError("Clean reference scores differ between source files")

    first_key = next(iter(clean.groupby(KEY, sort=False, observed=True).groups))
    first = clean.groupby(KEY, sort=False, observed=True).get_group(first_key)
    poi_order = first["POI_ID"].to_numpy(copy=True)
    _check_truth_consistency(clean, poi_order)
    _check_truth_consistency(attack, poi_order)
    cardinality = pd.concat([
        clean.groupby(["model", "condition"], observed=True).agg(
            prediction_rows=("POI_ID", "size"), unique_pois=("POI_ID", "nunique"),
            regions=("region", "nunique"), seeds=("seed", "nunique")
        ).reset_index().assign(source="clean_model_comparison"),
        attack.groupby(["model", "condition"], observed=True).agg(
            prediction_rows=("POI_ID", "size"), unique_pois=("POI_ID", "nunique"),
            regions=("region", "nunique"), seeds=("seed", "nunique")
        ).reset_index().assign(source="attack_evaluation"),
    ], ignore_index=True)
    return clean, attack, poi_order, cardinality


def _source_provenance(cfg: dict) -> pd.DataFrame:
    rows = []
    for role, key in [("clean_model_comparison", "clean_predictions"),
                      ("attack_evaluation", "attack_predictions")]:
        path = Path(cfg[key])
        rows.append({
            "role": role,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return pd.DataFrame(rows)


def _analyze_groups(
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    poi_order: np.ndarray,
    weights: np.ndarray,
    reference_condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeats = weights.shape[0]
    aggregate: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    stratum_rows: list[dict] = []
    inputs = [
        ("clean_model_comparison", clean),
        ("attack_evaluation", attack.loc[
            ~attack["condition"].astype(str).eq(reference_condition)
        ]),
    ]
    for family, frame in inputs:
        group_count = frame[["model", "condition", "seed", "region"]].drop_duplicates().shape[0]
        completed = 0
        for keys, group in frame.groupby(KEY, sort=True, observed=True):
            model, condition, seed, region = str(keys[0]), str(keys[1]), int(keys[2]), str(keys[3])
            y_true, y_score, y_pred = _aligned_arrays(group, poi_order)
            replicate_metrics = weighted_metric_replicates(y_true, y_score, y_pred, weights)
            point = _point_metrics(y_true, y_score, y_pred)
            aggregate_key = (family, model, condition)
            if aggregate_key not in aggregate:
                aggregate[aggregate_key] = {
                    metric: np.zeros(repeats, dtype=np.float64) for metric in METRICS
                }
                aggregate[aggregate_key]["_strata"] = np.array([0], dtype=np.int64)
            for metric in METRICS:
                aggregate[aggregate_key][metric] += replicate_metrics[metric]
            aggregate[aggregate_key]["_strata"][0] += 1
            stratum_rows.append({
                "analysis_family": family,
                "model": model,
                "condition": condition,
                "seed": seed,
                "region": region,
                **point,
            })
            completed += 1
            if completed % 100 == 0 or completed == group_count:
                print(f"{family}: {completed}/{group_count} strata", flush=True)

    replicate_rows = []
    for (family, model, condition), values in aggregate.items():
        strata = int(values["_strata"][0])
        for metric in METRICS:
            values[metric] /= strata
        for repeat in range(repeats):
            replicate_rows.append({
                "replicate": repeat,
                "analysis_family": family,
                "model": model,
                "condition": condition,
                **{metric: values[metric][repeat] for metric in METRICS},
            })
    return pd.DataFrame(stratum_rows), pd.DataFrame(replicate_rows)


def _summaries(
    strata: pd.DataFrame,
    replicates: pd.DataFrame,
    confidence_level: float,
    repeats: int,
    n_pois: int,
) -> pd.DataFrame:
    alpha = (1.0 - confidence_level) / 2.0
    point = strata.groupby(
        ["analysis_family", "model", "condition"], observed=True
    )[list(METRICS)].mean()
    rows = []
    for keys, group in replicates.groupby(
        ["analysis_family", "model", "condition"], sort=True, observed=True
    ):
        for metric in METRICS:
            values = group[metric].to_numpy()
            rows.append({
                "analysis_family": keys[0],
                "model": keys[1],
                "condition": keys[2],
                "metric": metric,
                "estimate": point.loc[keys, metric],
                "bootstrap_mean": values.mean(),
                "ci_low": np.quantile(values, alpha),
                "ci_high": np.quantile(values, 1.0 - alpha),
                "confidence_level": confidence_level,
                "interval_method": "percentile",
                "simultaneous_coverage": False,
                "bootstrap_repeats": repeats,
                "bootstrap_unit": "POI_ID",
                "unique_pois": n_pois,
                "regions_aggregated_within_replicate": 17,
                "seeds_aggregated_within_replicate": 3,
            })
    return pd.DataFrame(rows)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _paired_comparisons(
    replicates: pd.DataFrame,
    strata: pd.DataFrame,
    reference_model: str,
    reference_condition: str,
    confidence_level: float,
) -> pd.DataFrame:
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    point = strata.groupby(
        ["analysis_family", "model", "condition"], observed=True
    )[list(METRICS)].mean()
    indexed = replicates.set_index(
        ["analysis_family", "model", "condition", "replicate"]
    ).sort_index()
    clean_reference = indexed.loc[("clean_model_comparison", reference_model,
                                   reference_condition)]
    clean_keys = point.loc["clean_model_comparison"].index.tolist()
    for model, condition in clean_keys:
        if model == reference_model and condition == reference_condition:
            continue
        comparator = indexed.loc[("clean_model_comparison", model, condition)]
        for metric in METRICS:
            delta = comparator[metric].to_numpy() - clean_reference[metric].to_numpy()
            left = (np.count_nonzero(delta <= 0) + 1) / (len(delta) + 1)
            right = (np.count_nonzero(delta >= 0) + 1) / (len(delta) + 1)
            rows.append({
                "comparison_family": "clean_model_vs_reference",
                "comparator_model": model,
                "comparator_condition": condition,
                "reference_model": reference_model,
                "reference_condition": reference_condition,
                "metric": metric,
                "estimate_difference": (
                    point.loc[("clean_model_comparison", model, condition), metric]
                    - point.loc[("clean_model_comparison", reference_model,
                                reference_condition), metric]
                ),
                "ci_low": np.quantile(delta, alpha),
                "ci_high": np.quantile(delta, 1.0 - alpha),
                "bootstrap_probability_difference_gt_zero": np.mean(delta > 0),
                "two_sided_bootstrap_p": min(1.0, 2.0 * min(left, right)),
            })
    attack_keys = point.loc["attack_evaluation"].index.tolist()
    for model, condition in attack_keys:
        comparator = indexed.loc[("attack_evaluation", model, condition)]
        for metric in METRICS:
            delta = comparator[metric].to_numpy() - clean_reference[metric].to_numpy()
            left = (np.count_nonzero(delta <= 0) + 1) / (len(delta) + 1)
            right = (np.count_nonzero(delta >= 0) + 1) / (len(delta) + 1)
            rows.append({
                "comparison_family": "attack_condition_vs_clean_reference",
                "comparator_model": model,
                "comparator_condition": condition,
                "reference_model": reference_model,
                "reference_condition": reference_condition,
                "metric": metric,
                "estimate_difference": (
                    point.loc[("attack_evaluation", model, condition), metric]
                    - point.loc[("clean_model_comparison", reference_model,
                                reference_condition), metric]
                ),
                "ci_low": np.quantile(delta, alpha),
                "ci_high": np.quantile(delta, 1.0 - alpha),
                "bootstrap_probability_difference_gt_zero": np.mean(delta > 0),
                "two_sided_bootstrap_p": min(1.0, 2.0 * min(left, right)),
            })
    result = pd.DataFrame(rows)
    result["bh_fdr_q"] = np.nan
    for _, indexes in result.groupby(["comparison_family", "metric"]).groups.items():
        result.loc[indexes, "bh_fdr_q"] = _bh_adjust(
            result.loc[indexes, "two_sided_bootstrap_p"].to_numpy()
        )
    result["confidence_level"] = confidence_level
    result["interval_method"] = "percentile"
    result["simultaneous_coverage"] = False
    result["pairing"] = "shared POI_ID bootstrap weights"
    return result


def _all_clean_model_pairs(
    replicates: pd.DataFrame,
    strata: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Return all unordered clean-model contrasts using the paired replicates."""
    alpha = (1.0 - confidence_level) / 2.0
    clean_replicates = replicates.loc[
        replicates["analysis_family"].eq("clean_model_comparison")
    ]
    clean_strata = strata.loc[
        strata["analysis_family"].eq("clean_model_comparison")
    ]
    conditions = clean_replicates["condition"].unique()
    if len(conditions) != 1:
        raise ValueError("All-model pairwise contrasts require one clean condition")
    condition = str(conditions[0])
    models = sorted(clean_replicates["model"].unique())
    point = clean_strata.groupby("model", observed=True)[list(METRICS)].mean()
    indexed = clean_replicates.set_index(["model", "replicate"]).sort_index()
    rows = []
    for model_a, model_b in combinations(models, 2):
        values_a = indexed.loc[model_a]
        values_b = indexed.loc[model_b]
        for metric in METRICS:
            delta = values_a[metric].to_numpy() - values_b[metric].to_numpy()
            left = (np.count_nonzero(delta <= 0) + 1) / (len(delta) + 1)
            right = (np.count_nonzero(delta >= 0) + 1) / (len(delta) + 1)
            rows.append({
                "comparison_family": "all_clean_model_pairs",
                "model_a": model_a,
                "model_b": model_b,
                "condition": condition,
                "metric": metric,
                "estimate_difference_a_minus_b": point.loc[model_a, metric] - point.loc[model_b, metric],
                "ci_low": np.quantile(delta, alpha),
                "ci_high": np.quantile(delta, 1.0 - alpha),
                "bootstrap_probability_difference_gt_zero": np.mean(delta > 0),
                "two_sided_bootstrap_p": min(1.0, 2.0 * min(left, right)),
                "confidence_level": confidence_level,
                "interval_method": "percentile",
                "simultaneous_coverage": False,
                "pairing": "shared POI_ID bootstrap weights",
            })
    result = pd.DataFrame(rows)
    result["bh_fdr_q"] = np.nan
    for _, indexes in result.groupby("metric").groups.items():
        result.loc[indexes, "bh_fdr_q"] = _bh_adjust(
            result.loc[indexes, "two_sided_bootstrap_p"].to_numpy()
        )
    return result


def _save_figures(summary: pd.DataFrame, comparisons: pd.DataFrame, figures: Path) -> None:
    clean = summary.loc[
        summary["analysis_family"].eq("clean_model_comparison")
        & summary["metric"].eq("f1")
    ].sort_values("estimate")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(clean["estimate"], clean["model"],
                xerr=[clean["estimate"] - clean["ci_low"],
                      clean["ci_high"] - clean["estimate"]],
                fmt="o", color="#2b6f92", capsize=3)
    ax.set(xlabel="Macro F1 (mean over 17 regions and 3 seeds)",
           ylabel="Model", title="Clean validation paired POI bootstrap 95% CIs")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "clean_model_f1_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    attack = comparisons.loc[
        comparisons["comparison_family"].eq("attack_condition_vs_clean_reference")
        & comparisons["metric"].eq("f1")
    ].sort_values("estimate_difference")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.errorbar(attack["estimate_difference"], attack["comparator_condition"],
                xerr=[attack["estimate_difference"] - attack["ci_low"],
                      attack["ci_high"] - attack["estimate_difference"]],
                fmt="o", color="#9b3a4a", capsize=2, markersize=4)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="Macro F1 difference from clean tabpfn_v3",
           ylabel="Attack condition",
           title="Paired POI bootstrap differences (95% CI)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "attack_f1_difference_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    attack_all = comparisons.loc[
        comparisons["comparison_family"].eq("attack_condition_vs_clean_reference")
    ].copy()
    attack_all["condition"] = pd.Categorical(
        attack_all["comparator_condition"],
        categories=attack["comparator_condition"].tolist(), ordered=True,
    )
    pivot = attack_all.pivot(index="condition", columns="metric",
                              values="estimate_difference")
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot[list(METRICS)], center=0.0, cmap="vlag", annot=False, ax=ax)
    ax.set_title("Attack minus clean paired metric differences")
    ax.set(xlabel="Metric", ylabel="Attack condition")
    fig.tight_layout()
    fig.savefig(figures / "attack_metric_difference_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def _git_state() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, True


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    if tuple(cfg["metrics"]) != METRICS:
        raise ValueError(f"metrics must be exactly {list(METRICS)}")
    if cfg["bootstrap_repeats"] < 2000:
        raise ValueError("At least 2,000 bootstrap repeats are required")
    output = Path(output_override or cfg["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tables, figures, work = output / "tables", output / "figures", output / "work"
    tables.mkdir(parents=True)
    figures.mkdir()
    work.mkdir()

    provenance = _source_provenance(cfg)
    clean, attack, poi_order, cardinality = _load_inputs(cfg)
    weights = make_bootstrap_weights(len(poi_order), cfg["bootstrap_repeats"],
                                     cfg["bootstrap_seed"])
    weights_path = work / "bootstrap_poi_weights.npy"
    np.save(weights_path, weights, allow_pickle=False)
    checks = pd.DataFrame({
        "replicate": np.arange(len(weights)),
        "total_poi_weight": weights.sum(axis=1),
        "nonzero_pois": np.count_nonzero(weights, axis=1),
        "maximum_poi_weight": weights.max(axis=1),
    })

    strata, replicates = _analyze_groups(
        clean, attack, poi_order, weights, cfg["reference_condition"]
    )
    summary = _summaries(strata, replicates, cfg["confidence_level"],
                         cfg["bootstrap_repeats"], len(poi_order))
    comparisons = _paired_comparisons(
        replicates, strata, cfg["reference_model"], cfg["reference_condition"],
        cfg["confidence_level"],
    )
    all_clean_pairs = _all_clean_model_pairs(
        replicates, strata, cfg["confidence_level"]
    )

    provenance.to_csv(tables / "source_provenance.csv", index=False)
    cardinality.to_csv(tables / "input_cardinality.csv", index=False)
    checks.to_csv(tables / "bootstrap_replicate_checks.csv", index=False)
    strata.to_csv(tables / "seed_region_point_metrics.csv", index=False)
    replicates.to_parquet(tables / "bootstrap_metric_replicates.parquet", index=False)
    summary.to_csv(tables / "metric_confidence_intervals.csv", index=False)
    comparisons.to_csv(tables / "paired_differences_vs_reference.csv", index=False)
    all_clean_pairs.to_csv(tables / "all_clean_model_pairwise_differences.csv", index=False)
    _save_figures(summary, comparisons, figures)

    commit, dirty = _git_state()
    elapsed = time.time() - started
    metadata = {
        "status": "executed_pending_independent_review",
        "analysis_date": "2026-10-01",
        "scope": "validation only",
        "test_split_used": False,
        "bootstrap_unit": "POI_ID",
        "bootstrap_method": "multinomial pairs bootstrap with shared POI multiplicities",
        "bootstrap_repeats": cfg["bootstrap_repeats"],
        "bootstrap_seed": cfg["bootstrap_seed"],
        "confidence_level": cfg["confidence_level"],
        "confidence_interval_method": "marginal percentile intervals; not simultaneous or multiplicity-adjusted",
        "bootstrap_weights_path": str(weights_path),
        "bootstrap_weights_sha256": sha256(weights_path),
        "aggregation": "metric per seed and OvR region, then equal mean over 17 regions and 3 seeds inside each replicate",
        "regions_and_seeds_as_independent_units": False,
        "unique_pois": len(poi_order),
        "regions": clean["region"].nunique(),
        "seeds": sorted(map(int, clean["seed"].unique())),
        "clean_models": sorted(clean["model"].astype(str).unique()),
        "attack_models": sorted(attack["model"].astype(str).unique()),
        "attack_conditions": sorted(set(attack["condition"].astype(str)) - {cfg["reference_condition"]}),
        "reference_model": cfg["reference_model"],
        "reference_condition": cfg["reference_condition"],
        "metrics": list(METRICS),
        "input_files": provenance.to_dict(orient="records"),
        "selection_bias": "tabpfn_v3 was selected using the same clean validation set; intervals are conditional descriptive uncertainty and do not correct post-selection optimism",
        "robustness_claim": "none; attacked predictions cover only tabpfn_v3 on validation data",
        "unavailable_scope": "attacked predictions for the other seven clean models",
        "source_attack_evidence_limitations": [
            "source attack artifact did not establish actual numeric bounds or spatial label preservation",
            "source attack victim_model provenance was blank",
            "attacks were generated with one attack seed (42); model seeds do not add attack-generation variation",
            "POI bootstrap does not quantify attack-generation or model-selection uncertainty",
        ],
        "git_commit": commit,
        "git_dirty_at_report_time": dirty,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "pyarrow", "numba", "matplotlib", "seaborn"]},
        "seconds": elapsed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps({
        "confidence_intervals": summary.to_dict(orient="records"),
        "paired_differences": comparisons.to_dict(orient="records"),
        "all_clean_model_pairs": all_clean_pairs.to_dict(orient="records"),
    }, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "run_command.txt").write_text(
        f"python -m poi.uncertainty --config {config_path}\n"
    )

    clean_f1 = summary.loc[
        summary["analysis_family"].eq("clean_model_comparison")
        & summary["metric"].eq("f1")
    ].sort_values("estimate", ascending=False)
    best = clean_f1.iloc[0]
    (output / "report.md").write_text(f"""# Stage 06 paired POI bootstrap uncertainty

The validation predictions were re-evaluated with {cfg['bootstrap_repeats']:,} paired POI
bootstrap draws. Each draw resampled the 3,400 unique `POI_ID` values once and reused the same
multiplicities across all models, 32 attack conditions, 17 OvR regions, and three model seeds.
For every draw, each metric was calculated at the seed-by-region level and then equally averaged
over the 17 regions and three seeds. Region and seed rows were not treated as independent samples.

The clean model with the highest point Macro F1 is `{best.model}` at {best.estimate:.6f}
(paired POI bootstrap 95% CI {best.ci_low:.6f} to {best.ci_high:.6f}). Full F1, precision,
recall, accuracy, and ROC-AUC intervals are in `tables/metric_confidence_intervals.csv`; paired
model differences from clean `tabpfn_v3` and attack-condition differences from clean
`tabpfn_v3` are in `tables/paired_differences_vs_reference.csv`. All 28 unordered clean-model
pairs for each metric are in `tables/all_clean_model_pairwise_differences.csv`. The reported 95%
percentile intervals are marginal intervals for individual contrasts; they are not simultaneous
confidence bands and their interval coverage is not multiplicity-adjusted.

These are validation-set, conditional descriptive intervals. `tabpfn_v3` was selected on the
same clean validation data, so the intervals do not correct selection bias and can be optimistic.
Only `tabpfn_v3` has attacked predictions; the other seven clean models cannot be compared under
attack from these inputs. The source attack artifact also did not establish actual numeric bounds
or spatial label preservation, had blank victim-model provenance, and used one attack-generation
seed. The POI bootstrap does not cover those uncertainties. This analysis does not validate attack
robustness and does not use the locked test split. Independent review is pending.
""")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/06_uncertainty.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
