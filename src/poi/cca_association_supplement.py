"""Stage-05 supplementary CCA loadings, fixed-direction CI, and associations.

The analysis is train-only.  For every generation seed and attack condition it
uses the same internal half split, clean-fitted encoder/PCA, and three-component
CCA design as :func:`poi.expanded_analysis.heldout_cca`.  Bootstrap replicates
resample paired inference POIs while keeping every learned direction fixed.

Region-by-category chi-square tests are cross-sectional association tests.  A
separate paired-POI bootstrap estimates the clean-to-attack change in Cramer's V;
the cross-sectional chi-square p-values are never described as paired contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
import yaml

from .data import CATEGORICAL, FEATURES, NUMERIC, sha256


FORMAT = "poi-stage05-cca-association-supplement-v1"
OVERALL = "ALL_REGIONS"


class SupplementError(RuntimeError):
    """Fail-closed Stage-05 source or statistical-contract error."""


def ordered_id_sha256(values: Sequence[Any]) -> str:
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise SupplementError("POI_ID values must be unique within an analysis scope")
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _correlation_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlations between each left column and each right column."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise SupplementError("Correlation matrices need aligned two-dimensional rows")
    lcenter = left - left.mean(axis=0)
    rcenter = right - right.mean(axis=0)
    numerator = lcenter.T @ rcenter
    denominator = np.sqrt(
        np.sum(lcenter * lcenter, axis=0)[:, None]
        * np.sum(rcenter * rcenter, axis=0)[None, :]
    )
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0,
    )


def fixed_direction_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    repeats: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    """Paired nonparametric bootstrap correlations with fixed projected scores.

    The returned array is ``repeats x components``.  It is memory-bounded by
    ``chunk_size x inference_rows`` rather than by all replicates at once.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (left.ndim != 2 or right.shape != left.shape or left.shape[0] < 2
            or repeats < 1 or chunk_size < 1):
        raise SupplementError("Invalid fixed-direction bootstrap dimensions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise SupplementError("Fixed-direction scores contain non-finite values")
    n_rows, components = left.shape
    sufficient = np.column_stack([left, right, left * left, right * right, left * right])
    output = np.empty((repeats, components), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_rows, 1.0 / n_rows)
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        weights = rng.multinomial(n_rows, probabilities, size=stop - start)
        sums = weights @ sufficient
        sum_left = sums[:, :components]
        sum_right = sums[:, components:2 * components]
        sum_left2 = sums[:, 2 * components:3 * components]
        sum_right2 = sums[:, 3 * components:4 * components]
        sum_product = sums[:, 4 * components:5 * components]
        covariance = sum_product - sum_left * sum_right / n_rows
        var_left = sum_left2 - sum_left * sum_left / n_rows
        var_right = sum_right2 - sum_right * sum_right / n_rows
        denominator = np.sqrt(np.maximum(var_left, 0) * np.maximum(var_right, 0))
        output[start:stop] = np.divide(
            covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0,
        )
    return output


def heldout_cca_supplement(
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    *,
    seed: int,
    components: int,
    pca_components_max: int,
    permutations: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    bootstrap_chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the frozen held-out CCA design and report loadings and conditional CI."""
    if len(clean) != len(attack) or len(clean) < 12:
        raise SupplementError("Held-out CCA requires at least 12 aligned pairs")
    if not clean.POI_ID.astype(str).reset_index(drop=True).equals(
        attack.POI_ID.astype(str).reset_index(drop=True)
    ):
        raise SupplementError("Clean and attacked POI order differs")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))
    fit_ids, infer_ids = np.split(order, [len(order) // 2])
    transform = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    clean_fit_encoded = transform.fit_transform(clean.iloc[fit_ids][FEATURES])
    attack_fit_encoded = transform.transform(attack.iloc[fit_ids][FEATURES])
    dimensions = min(pca_components_max, clean_fit_encoded.shape[1], len(fit_ids) - 2)
    if dimensions < components:
        raise SupplementError("Too few fit dimensions for the requested CCA components")
    pca = PCA(n_components=dimensions, random_state=seed).fit(clean_fit_encoded)
    left_fit = pca.transform(clean_fit_encoded)
    right_fit = pca.transform(attack_fit_encoded)
    valid = (left_fit.std(axis=0) > 1e-9) & (right_fit.std(axis=0) > 1e-9)
    if int(valid.sum()) < components:
        raise SupplementError("Fewer than three nonconstant joint PCA dimensions")
    cca = CCA(n_components=components, max_iter=2000).fit(left_fit[:, valid], right_fit[:, valid])

    clean_infer_encoded = transform.transform(clean.iloc[infer_ids][FEATURES])
    attack_infer_encoded = transform.transform(attack.iloc[infer_ids][FEATURES])
    left_infer = pca.transform(clean_infer_encoded)[:, valid]
    right_infer = pca.transform(attack_infer_encoded)[:, valid]
    u, v = cca.transform(left_infer, right_infer)
    centered_u = u - u.mean(axis=0)
    centered_v = v - v.mean(axis=0)
    denominator = np.sqrt(np.sum(centered_u * centered_u, axis=0) * np.sum(centered_v * centered_v, axis=0))
    observed = np.divide(
        np.sum(centered_u * centered_v, axis=0), denominator,
        out=np.full(components, np.nan), where=denominator > 0,
    )
    exceed = np.zeros(components, dtype=np.int64)
    for _ in range(permutations):
        permuted = centered_v[rng.permutation(len(centered_v))]
        null = np.divide(
            np.sum(centered_u * permuted, axis=0), denominator,
            out=np.full(components, np.nan), where=denominator > 0,
        )
        exceed += np.abs(null) >= np.abs(observed)

    bootstrap = fixed_direction_bootstrap(
        u, v, repeats=bootstrap_repeats, seed=bootstrap_seed,
        chunk_size=bootstrap_chunk_size,
    )
    correlation_rows = []
    for index in range(components):
        finite = bootstrap[:, index][np.isfinite(bootstrap[:, index])]
        correlation_rows.append({
            "component": index + 1, "correlation": float(observed[index]),
            "p_value": float((exceed[index] + 1) / (permutations + 1)),
            "ci_low": float(np.quantile(finite, .025)) if len(finite) else np.nan,
            "ci_high": float(np.quantile(finite, .975)) if len(finite) else np.nan,
            "bootstrap_mean": float(np.mean(finite)) if len(finite) else np.nan,
            "bootstrap_repeats": bootstrap_repeats, "finite_bootstrap_repeats": len(finite),
            "permutations": permutations, "status": "ESTIMATED" if np.isfinite(observed[index]) else "UNIDENTIFIABLE",
            "inference": "paired_POI_bootstrap_conditional_on_fixed_encoder_PCA_CCA_directions",
        })

    feature_names = transform.get_feature_names_out().astype(str)
    clean_loadings = _correlation_columns(clean_infer_encoded, u)
    attack_loadings = _correlation_columns(attack_infer_encoded, v)
    loading_rows = []
    for side, values in (("clean", clean_loadings), ("adversarial", attack_loadings)):
        for feature_index, feature_name in enumerate(feature_names):
            encoded_suffix = feature_name.split("__", 1)[-1]
            source_feature = next(
                (feature for feature in FEATURES
                 if encoded_suffix == feature or encoded_suffix.startswith(f"{feature}_")),
                "UNRESOLVED",
            )
            for component_index in range(components):
                value = values[feature_index, component_index]
                loading_rows.append({
                    "side": side, "encoded_feature": feature_name,
                    "source_feature_prefix": source_feature,
                    "component": component_index + 1,
                    "loading": float(value) if np.isfinite(value) else np.nan,
                    "loading_definition": "heldout_inference_Pearson_encoded_feature_vs_fixed_canonical_variate",
                    "status": "ESTIMATED" if np.isfinite(value) else "CONSTANT_IN_INFERENCE",
                })
    split = {
        "scope_rows": len(clean), "fit_rows": len(fit_ids), "inference_rows": len(infer_ids),
        "eligible_ordered_id_sha256": ordered_id_sha256(clean.POI_ID.astype(str).tolist()),
        "fit_ordered_id_sha256": ordered_id_sha256(clean.iloc[fit_ids].POI_ID.astype(str).tolist()),
        "inference_ordered_id_sha256": ordered_id_sha256(clean.iloc[infer_ids].POI_ID.astype(str).tolist()),
        "fit_inference_overlap": len(set(clean.iloc[fit_ids].POI_ID.astype(str)) & set(clean.iloc[infer_ids].POI_ID.astype(str))),
        "encoded_features": len(feature_names), "pca_dimensions": dimensions,
        "joint_nonconstant_pca_dimensions": int(valid.sum()), "cca_components": components,
        "split_seed": seed, "bootstrap_seed": bootstrap_seed,
        "transform_fit_scope": "clean_train_internal_fit_half_only",
    }
    return pd.DataFrame(correlation_rows), pd.DataFrame(loading_rows), split


def cramers_v(table: np.ndarray) -> float:
    """Bias-corrected Cramer's V for a nonempty contingency table."""
    observed = np.asarray(table, dtype=np.float64)
    if observed.ndim != 2 or (observed < 0).any() or not np.isfinite(observed).all():
        raise SupplementError("Invalid contingency table")
    observed = observed[observed.sum(axis=1) > 0][:, observed.sum(axis=0) > 0]
    n = observed.sum()
    rows, columns = observed.shape
    if n <= 1 or rows < 2 or columns < 2:
        return 0.0
    row_total = observed.sum(axis=1, keepdims=True)
    column_total = observed.sum(axis=0, keepdims=True)
    expected = row_total @ column_total / n
    statistic = np.sum(np.divide(
        (observed - expected) ** 2, expected,
        out=np.zeros_like(expected), where=expected > 0,
    ))
    phi2 = statistic / n
    corrected_phi2 = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    corrected_rows = rows - (rows - 1) ** 2 / (n - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (n - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return float(np.sqrt(corrected_phi2 / denominator)) if denominator > 0 else 0.0


def region_category_association(regions: Sequence[Any], categories: Sequence[Any]) -> dict[str, Any]:
    frame = pd.DataFrame({"region": list(map(str, regions)), "category": list(map(str, categories))})
    table = pd.crosstab(frame.region, frame.category).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "n": len(frame), "chi_square": 0.0, "degrees_of_freedom": 0,
            "p_value": 1.0, "cramers_v": 0.0, "expected_cells": int(table.size),
            "expected_cells_lt5": int(table.size), "expected_cells_lt1": int(table.size),
            "expected_lt5_fraction": 1.0, "minimum_expected_count": 0.0,
            "asymptotic_reliability": "UNRELIABLE_SINGLE_LEVEL",
        }
    statistic, p_value, degrees, expected = chi2_contingency(table, correction=False)
    sparse5 = int(np.sum(expected < 5))
    sparse1 = int(np.sum(expected < 1))
    fraction = float(np.mean(expected < 5))
    return {
        "n": len(frame), "chi_square": float(statistic),
        "degrees_of_freedom": int(degrees), "p_value": float(p_value),
        "cramers_v": cramers_v(table), "expected_cells": int(expected.size),
        "expected_cells_lt5": sparse5, "expected_cells_lt1": sparse1,
        "expected_lt5_fraction": fraction, "minimum_expected_count": float(expected.min()),
        "asymptotic_reliability": "SPARSE_EXPECTED_COUNTS" if sparse1 or fraction > .2 else "ADEQUATE_EXPECTED_COUNTS",
    }


def paired_cramers_v_bootstrap(
    regions: Sequence[Any],
    clean_categories: Sequence[Any],
    attack_categories: Sequence[Any],
    *,
    repeats: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Bootstrap marginal Cramer's V and their difference from paired POIs."""
    region_values = np.asarray(list(map(str, regions)))
    clean_values = np.asarray(list(map(str, clean_categories)))
    attack_values = np.asarray(list(map(str, attack_categories)))
    if not (len(region_values) == len(clean_values) == len(attack_values)) or len(region_values) < 2:
        raise SupplementError("Paired association bootstrap arrays are not aligned")
    region_levels, region_code = np.unique(region_values, return_inverse=True)
    category_levels = np.unique(np.concatenate([clean_values, attack_values]))
    category_lookup = {value: index for index, value in enumerate(category_levels)}
    clean_code = np.asarray([category_lookup[value] for value in clean_values], dtype=int)
    attack_code = np.asarray([category_lookup[value] for value in attack_values], dtype=int)
    category_count = len(category_levels)
    joint_code = (region_code * category_count + clean_code) * category_count + attack_code
    joint_levels, joint_counts = np.unique(joint_code, return_counts=True)
    joint_region = joint_levels // (category_count * category_count)
    remainder = joint_levels % (category_count * category_count)
    joint_clean = remainder // category_count
    joint_attack = remainder % category_count
    probabilities = joint_counts / joint_counts.sum()
    clean_values_out = np.empty(repeats, dtype=np.float64)
    attack_values_out = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(seed)
    region_count = len(region_levels)
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        draws = rng.multinomial(len(region_values), probabilities, size=stop - start)
        for local, counts in enumerate(draws):
            clean_table = np.zeros((region_count, category_count), dtype=np.int64)
            attack_table = np.zeros((region_count, category_count), dtype=np.int64)
            np.add.at(clean_table, (joint_region, joint_clean), counts)
            np.add.at(attack_table, (joint_region, joint_attack), counts)
            clean_values_out[start + local] = cramers_v(clean_table)
            attack_values_out[start + local] = cramers_v(attack_table)
    delta = attack_values_out - clean_values_out
    finite = np.isfinite(delta)
    finite_delta = delta[finite]
    less_equal = int(np.sum(finite_delta <= 0))
    greater_equal = int(np.sum(finite_delta >= 0))
    sign_tail = min(1.0, 2 * min((less_equal + 1) / (len(finite_delta) + 1),
                               (greater_equal + 1) / (len(finite_delta) + 1)))
    return {
        "clean_ci_low": float(np.quantile(clean_values_out, .025)),
        "clean_ci_high": float(np.quantile(clean_values_out, .975)),
        "adversarial_ci_low": float(np.quantile(attack_values_out, .025)),
        "adversarial_ci_high": float(np.quantile(attack_values_out, .975)),
        "delta_bootstrap_mean": float(np.mean(finite_delta)),
        "delta_ci_low": float(np.quantile(finite_delta, .025)),
        "delta_ci_high": float(np.quantile(finite_delta, .975)),
        "bootstrap_sign_tail_fraction": float(sign_tail), "bootstrap_repeats": repeats,
        "sign_tail_interpretation": "descriptive_uncentered_bootstrap_fraction_not_null_calibrated_p_value",
        "finite_bootstrap_repeats": int(finite.sum()),
        "bootstrap_unit": "paired_POI_joint_region_clean_category_attack_category",
    }


def add_fdr(frame: pd.DataFrame, p_column: str = "p_value", q_column: str = "q_value") -> pd.DataFrame:
    result = frame.copy()
    result[q_column] = np.nan
    finite = np.isfinite(pd.to_numeric(result[p_column], errors="coerce"))
    if finite.any():
        result.loc[finite, q_column] = multipletests(
            result.loc[finite, p_column].astype(float), method="fdr_bh"
        )[1]
    return result


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("format") != FORMAT:
        raise SupplementError(f"Config format must be {FORMAT}")
    if list(map(int, cfg.get("attack_seeds", []))) != [42, 202, 340]:
        raise SupplementError("Attack generation seeds must be exactly 42,202,340")
    if int(cfg.get("expected_conditions", -1)) != 32 or int(cfg.get("expected_regions", -1)) != 17:
        raise SupplementError("Frozen design requires 32 conditions and 17 regions")
    if int(cfg.get("cca_components", 0)) != 3 or int(cfg.get("pca_components_max", 0)) != 8:
        raise SupplementError("Frozen CCA design requires three components after at most eight PCs")
    if int(cfg.get("cca_permutations", 0)) != 499:
        raise SupplementError("CCA permutation design requires 499 fixed-direction permutations")
    if int(cfg.get("bootstrap_repeats", 0)) < 2000:
        raise SupplementError("Final paired bootstrap requires at least 2000 replicates")
    if int(cfg.get("thread_limit", 0)) != 1 or int(cfg.get("memory_limit_mb", 0)) > 1500:
        raise SupplementError("This runner is constrained to one thread and at most 1.5 GB")
    if cfg.get("analysis_split") != "train":
        raise SupplementError("Stage-05 supplement is train-only")


def preflight(config_path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    validate_config(cfg)
    paths = {
        "attack_table": Path(cfg["attacks_file"]),
        "clean_data": Path(cfg["data_dir"]) / "poi_data_region.csv",
        "sample_manifest": Path(cfg["data_dir"]) / "sample_manifest.csv",
    }
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    return {
        "format": FORMAT, "status": "NOT_RUN" if missing else "READY_FOR_TRAIN_ONLY_EXECUTION",
        "missing_inputs": missing, "test_features_read": 0, "output_created": False,
        "thread_limit": 1, "memory_limit_mb": int(cfg["memory_limit_mb"]),
    }


def _save_loading_rows(path: Path, frame: pd.DataFrame, first: bool) -> None:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def _forest_figures(
    cca: pd.DataFrame,
    contrasts: pd.DataFrame,
    output: Path,
    attack_seeds: Sequence[int],
) -> None:
    for generation_seed in attack_seeds:
        view = cca.loc[
            cca.generation_seed.eq(generation_seed) & cca.analysis_region.eq(OVERALL)
        ].sort_values(["attack_condition", "component"])
        fig, axis = plt.subplots(figsize=(12, 18))
        positions = np.arange(len(view))
        axis.hlines(positions, view.ci_low, view.ci_high, linewidth=.7)
        axis.plot(view.correlation, positions, "o", markersize=2)
        axis.axvline(0, color="black", linewidth=.8)
        axis.set_yticks(positions, [f"{c} | C{k} | q={q:.3g}" for c, k, q in
            zip(view.attack_condition, view.component, view.q_value)], fontsize=6)
        axis.set_xlim(-1, 1); axis.set_xlabel("Held-out canonical correlation (paired bootstrap 95% CI)")
        axis.set_title(f"Overall held-out CCA | generation seed {generation_seed}")
        fig.tight_layout(); fig.savefig(output / f"cca_overall_seed{generation_seed}.png", dpi=300)
        plt.close(fig)

        view = contrasts.loc[contrasts.generation_seed.eq(generation_seed)].sort_values(
            ["feature", "attack_condition"]
        )
        fig, axis = plt.subplots(figsize=(12, 18))
        positions = np.arange(len(view))
        axis.hlines(positions, view.delta_ci_low, view.delta_ci_high, linewidth=.7)
        axis.plot(view.cramers_v_delta, positions, "o", markersize=2)
        axis.axvline(0, color="black", linewidth=.8)
        axis.set_yticks(positions, [f"{f} | {c}" for f, c in zip(view.feature, view.attack_condition)], fontsize=6)
        axis.set_xlabel("Cramer's V change: attack − clean (paired bootstrap 95% CI)")
        axis.set_title(f"Region/category association contrast | generation seed {generation_seed}\n"
                       "Descriptive paired intervals; no null-calibrated delta p/q values")
        fig.tight_layout(); fig.savefig(output / f"cramers_v_delta_seed{generation_seed}.png", dpi=300)
        plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    from .data import load_clean_splits
    from .expanded_explain import aligned_condition, primary_mask
    import pyarrow.dataset as ds

    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    ready = preflight(config_path)
    if ready["status"] != "READY_FOR_TRAIN_ONLY_EXECUTION":
        raise SupplementError("NOT_RUN: required inputs are missing")
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)

    source_paths = [
        Path(__file__), Path(__file__).with_name("expanded_analysis.py"),
        Path(__file__).with_name("expanded_explain.py"), Path(__file__).with_name("data.py"),
        config_path,
    ]
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    input_paths = {
        "attacks_file": Path(cfg["attacks_file"]),
        "clean_data": Path(cfg["data_dir"]) / "poi_data_region.csv",
        "sample_manifest": Path(cfg["data_dir"]) / "sample_manifest.csv",
    }
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    dataset = ds.dataset(cfg["attacks_file"], format="parquet")
    condition_values = dataset.to_table(
        columns=["attack_condition"], filter=ds.field("split") == "train"
    ).column("attack_condition")
    conditions = sorted(condition_values.unique().to_pylist())
    if len(conditions) != int(cfg["expected_conditions"]):
        raise SupplementError("Attack table does not contain the frozen 32 train conditions")

    clean, _ = load_clean_splits(cfg["data_dir"], splits=("train",))
    clean = clean.reset_index(drop=True)
    for feature in CATEGORICAL:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise").astype(float).astype(str)
    regions = sorted(clean.region.astype(str).unique())
    if len(clean) != int(cfg["expected_train_pois"]) or len(regions) != int(cfg["expected_regions"]):
        raise SupplementError("Clean train cardinality differs from frozen design")

    tables = output / "tables"
    figures = output / "figures"
    provenance = output / "provenance"
    for directory in (tables, figures, provenance):
        directory.mkdir(parents=True, exist_ok=False)
    for path in source_paths:
        shutil.copyfile(path, provenance / path.name)
    (provenance / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    (provenance / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")

    columns = [
        "POI_ID", "split", "region", "attack_condition", "generation_seed", "victim_model",
        "primary_evaluation_eligible", "constraints_valid", "label_preserved",
        *[f"{feature}_adv" for feature in FEATURES],
    ]
    correlation_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    first_loading_write = True
    started = time.monotonic()

    with threadpool_limits(limits=1):
        for generation_seed in cfg["attack_seeds"]:
            for condition in conditions:
                raw = pd.read_parquet(
                    cfg["attacks_file"], columns=columns,
                    filters=[("split", "==", "train"),
                             ("generation_seed", "==", int(generation_seed)),
                             ("attack_condition", "==", condition)],
                )
                attack = aligned_condition(raw, clean, condition, int(generation_seed))
                del raw
                eligible = primary_mask(attack)
                eligible_clean = clean.loc[eligible].reset_index(drop=True)
                eligible_attack = attack.loc[eligible].reset_index(drop=True)
                if not eligible_clean.POI_ID.astype(str).equals(eligible_attack.POI_ID.astype(str)):
                    raise SupplementError("Eligible clean/attack pairing changed")
                coverage_rows.append({
                    "generation_seed": int(generation_seed), "attack_condition": condition,
                    "attempted_pairs": len(clean), "primary_valid_pairs": len(eligible_clean),
                    "quarantined_pairs": int((~eligible).sum()),
                    "primary_valid_coverage": float(np.mean(eligible)),
                })

                scopes = [(OVERALL, np.ones(len(eligible_clean), dtype=bool))]
                scopes.extend((region, eligible_clean.region.astype(str).eq(region).to_numpy()) for region in regions)
                for analysis_region, selected in scopes:
                    selected_clean = eligible_clean.loc[selected].reset_index(drop=True)
                    selected_attack = eligible_attack.loc[selected].reset_index(drop=True)
                    identity = {
                        "analysis_region": analysis_region, "attack_condition": condition,
                        "generation_seed": int(generation_seed),
                    }
                    group_seed = stable_seed(
                        int(cfg["bootstrap_seed"]), generation_seed, condition, analysis_region, "cca"
                    )
                    correlations, loadings, split = heldout_cca_supplement(
                        selected_clean, selected_attack, seed=int(cfg["cca_split_seed"]),
                        components=int(cfg["cca_components"]),
                        pca_components_max=int(cfg["pca_components_max"]),
                        permutations=int(cfg["cca_permutations"]),
                        bootstrap_repeats=int(cfg["bootstrap_repeats"]),
                        bootstrap_seed=group_seed,
                        bootstrap_chunk_size=int(cfg["bootstrap_chunk_size"]),
                    )
                    correlation_rows.extend([{**identity, **row} for row in correlations.to_dict("records")])
                    split_rows.append({**identity, **split})
                    loading_frame = loadings.assign(**identity)
                    loading_frame = loading_frame[[*identity, *[c for c in loadings.columns]]]
                    _save_loading_rows(tables / "cca_encoded_feature_loadings.csv", loading_frame, first_loading_write)
                    first_loading_write = False

                for feature in CATEGORICAL:
                    clean_stats = region_category_association(eligible_clean.region, eligible_clean[feature])
                    attack_stats = region_category_association(eligible_attack.region, eligible_attack[feature])
                    association_rows.extend([
                        {
                            "generation_seed": int(generation_seed), "attack_condition": condition,
                            "feature": feature, "state": state,
                            "test_design": "unpaired_region_by_category_cross_sectional_association",
                            "paired_attack_contrast_claimed": False, **values,
                        }
                        for state, values in (("clean", clean_stats), ("adversarial", attack_stats))
                    ])
                    paired = paired_cramers_v_bootstrap(
                        eligible_clean.region, eligible_clean[feature], eligible_attack[feature],
                        repeats=int(cfg["bootstrap_repeats"]),
                        seed=stable_seed(int(cfg["bootstrap_seed"]), generation_seed, condition, feature, "cramers_v"),
                        chunk_size=int(cfg["association_bootstrap_chunk_size"]),
                    )
                    contrast_rows.append({
                        "generation_seed": int(generation_seed), "attack_condition": condition,
                        "feature": feature, "clean_cramers_v": clean_stats["cramers_v"],
                        "adversarial_cramers_v": attack_stats["cramers_v"],
                        "cramers_v_delta": attack_stats["cramers_v"] - clean_stats["cramers_v"],
                        "test_design": "paired_POI_bootstrap_attack_minus_clean_association_effect",
                        **paired,
                    })
                print(json.dumps({
                    "generation_seed": int(generation_seed), "attack_condition": condition,
                    "cca_scopes": 18, "eligible_pairs": len(eligible_clean),
                }), flush=True)

    correlations = add_fdr(pd.DataFrame(correlation_rows))
    associations = add_fdr(pd.DataFrame(association_rows))
    contrasts = pd.DataFrame(contrast_rows)
    splits = pd.DataFrame(split_rows)
    coverage = pd.DataFrame(coverage_rows)
    correlations.to_csv(tables / "cca_correlations_bootstrap.csv", index=False)
    splits.to_csv(tables / "cca_split_identities.csv", index=False)
    associations.to_csv(tables / "region_category_associations.csv", index=False)
    contrasts.to_csv(tables / "region_category_association_contrasts.csv", index=False)
    coverage.to_csv(tables / "coverage.csv", index=False)
    _forest_figures(correlations, contrasts, figures, cfg["attack_seeds"])

    observed_sources = {str(path.resolve()): sha256(path) for path in source_paths}
    observed_inputs = {name: sha256(path) for name, path in input_paths.items()}
    if observed_sources != source_hashes or observed_inputs != input_hashes:
        raise SupplementError("A source or input changed during Stage-05 supplementary analysis")
    expected_scopes = (int(cfg["expected_regions"]) + 1) * int(cfg["expected_conditions"]) * len(cfg["attack_seeds"])
    if len(splits) != expected_scopes or len(correlations) != expected_scopes * int(cfg["cca_components"]):
        raise SupplementError("CCA output grid is incomplete")
    if len(associations) != int(cfg["expected_conditions"]) * len(cfg["attack_seeds"]) * len(CATEGORICAL) * 2:
        raise SupplementError("Region/category association grid is incomplete")
    if len(contrasts) != int(cfg["expected_conditions"]) * len(cfg["attack_seeds"]) * len(CATEGORICAL):
        raise SupplementError("Region/category paired contrast grid is incomplete")
    pngs = sorted(figures.glob("*.png"))
    if len(pngs) != 6 or list(output.rglob("*.svg")):
        raise SupplementError("Supplement must contain six 300-DPI PNG summaries and no SVG")

    metadata = {
        "format": FORMAT, "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scientific_completion": False, "independent_review": "PENDING",
        "test_features_read": 0, "analysis_split": "train",
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "attack_conditions": conditions, "generation_seeds": list(cfg["attack_seeds"]),
        "regions": regions, "cca_scopes": len(splits), "cca_components": int(cfg["cca_components"]),
        "cca_correlation_rows": len(correlations),
        "cca_loading_rows": sum(1 for _ in open(tables / "cca_encoded_feature_loadings.csv")) - 1,
        "association_rows": len(associations), "paired_association_contrast_rows": len(contrasts),
        "bootstrap_repeats": int(cfg["bootstrap_repeats"]),
        "bootstrap_interpretation": "conditional_on_fixed_train_internal_fit_half_encoder_PCA_CCA_directions",
        "chi_square_interpretation": "unpaired_cross_sectional_region_category_association_only",
        "paired_contrast_interpretation": "paired_POI_bootstrap_of_attack_minus_clean_Cramers_V",
        "thread_limit": 1, "memory_limit_mb": int(cfg["memory_limit_mb"]),
        "wall_seconds": time.monotonic() - started,
        "runtime": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib"
        )},
        "limitations": [
            "CCA bootstrap intervals condition on fitted directions and do not include direction-selection uncertainty.",
            "Encoded-feature loadings are held-out point estimates; the 2000-replicate intervals apply to canonical correlations.",
            "Sparse expected-cell diagnostics qualify every asymptotic region/category chi-square result.",
            "Association chi-square tests are not paired attack contrasts; paired changes use the separate Cramer's V bootstrap.",
            "The analysis uses train POIs only and does not change the status of any separate Stage-05 or Stage-14 requirement.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Stage 05 CCA and regional categorical association supplement\n\n"
        "Three-component CCA directions, encoding, and PCA were fitted on the internal fit half of "
        "clean train POIs. Canonical correlations and encoded-feature loadings use the disjoint train "
        "inference half. The 95% intervals resample paired inference POIs 2,000 times with directions "
        "fixed. Region/category chi-square rows are cross-sectional associations; paired clean-to-attack "
        "effect changes are reported separately as Cramer's V differences with paired-POI intervals. "
        "This generated evidence is pending independent review and does not use test features.\n"
    )
    (output / "_GENERATED.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
