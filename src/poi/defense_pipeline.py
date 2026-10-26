"""Stage 10: paired detector -> frozen TabPFN 2.5 defense evaluation.

The stage reads only clean train/validation rows.  Detection thresholds are chosen
on one region-stratified half of clean validation and every reported metric uses the
other half.  A single set of 17 train-only TabPFN classifiers and a single
train-feature nearest-neighbour repair are shared by both detectors for each model
seed, then evaluated against all three attack-generation seeds in a balanced cross.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .data import CATEGORICAL, FEATURES, NUMERIC, load_clean_splits, load_generated_attacks, sha256
from .defense_evaluate import (
    SharedNearestTrainRepair,
    binary_metrics,
    paired_defense_metrics,
    select_detection_threshold,
)
from .exact_prediction_cache import (
    ExactFloat32PredictionCache,
    build_frozen_model_identity,
)
PROPOSED_LABEL = "proposed_tabpfn25_attack_head"
URE_LABEL = "URE-RF attack detector"
DETECTOR_LABELS = (PROPOSED_LABEL, URE_LABEL)
DEFAULT_RETENTIONS = (0.90, 0.95, 0.975, 0.99)
CANONICAL_CLEAN_GENERATION_SEED = -1
ATTACK_DISTANCE_METRIC = "local_equirectangular_wgs84_approximation"
NUMERIC_BUDGET_TOLERANCE_M = 0.05
CATEGORICAL_ONLY_TOLERANCE_M = 1e-6


def cross_design_expected_counts(
    *,
    model_seeds: int,
    generation_seeds: int,
    attack_conditions: int,
    regions: int,
    detectors: int,
    retentions: int,
) -> dict[str, int]:
    """Return exact Stage-10 table/shard cardinalities for the balanced cross."""
    pairs = model_seeds * generation_seeds
    attack_evaluations = pairs * attack_conditions
    # Each detector/retention emits fixed, accepted-only, and shared-repair metrics.
    # The proposed detector additionally emits projected-fixed repair per retention,
    # and its own native head emits one threshold-free row.
    classification_rows_per_scope_cohort = regions * (
        detectors * retentions * 3 + retentions + 1
    )
    return {
        "model_generation_seed_pairs": pairs,
        "clean_prediction_shards": model_seeds,
        "attack_prediction_shards": attack_evaluations,
        "prediction_shards": model_seeds + attack_evaluations,
        "threshold_rows": model_seeds * detectors * retentions,
        "detector_metric_rows": attack_evaluations * detectors * 2 * retentions,
        "classification_metric_rows": (
            model_seeds * 2 * classification_rows_per_scope_cohort
            + attack_evaluations * 4 * classification_rows_per_scope_cohort
        ),
        "region_model_fits": model_seeds * regions,
    }


def split_validation_ids(
    clean_validation: pd.DataFrame, *, calibration_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Mirror the stage-08/09 region-stratified split without importing torch."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    calibration: list[str] = []
    evaluation: list[str] = []
    for _, group in clean_validation.groupby("region", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed)
        count = max(1, min(len(group) - 1, round(len(group) * calibration_fraction)))
        calibration.extend(shuffled.iloc[:count]["POI_ID"].astype(str).tolist())
        evaluation.extend(shuffled.iloc[count:]["POI_ID"].astype(str).tolist())
    if set(calibration).intersection(evaluation):
        raise RuntimeError("Calibration and validation-evaluation POIs overlap")
    return calibration, evaluation


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError("Defense input must be a finite N x 5 feature matrix")
    return values


def _attack_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["POI_ID", "region"]].copy()
    for feature in FEATURES:
        result[feature] = pd.to_numeric(frame[f"{feature}_adv"], errors="raise")
    _matrix(result)
    return result


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _device_synchronizer(device: str) -> Callable[[], None]:
    """Return an explicit accelerator barrier for honest prediction timings."""
    if not str(device).startswith("cuda"):
        return lambda: None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA prediction timing requires an available CUDA device")
    resolved = torch.device(device)

    def synchronize() -> None:
        torch.cuda.synchronize(resolved)

    return synchronize


def _ordered_identity_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prediction_manifest_identity(frame: pd.DataFrame) -> dict[str, Any]:
    poi = frame[["POI_ID", "source_region"]].drop_duplicates("POI_ID", keep="first")
    return {
        "unique_pois": int(frame["POI_ID"].nunique()),
        "poi_identity_sha256": _ordered_identity_sha256(poi, ["POI_ID", "source_region"]),
        "row_identity_sha256": _ordered_identity_sha256(
            frame, ["POI_ID", "source_region", "target_region", "truth", "detector"]
        ),
    }


def _retention_name(value: float) -> str:
    return f"retention_{value * 100:g}".replace(".", "p")


def _read_partition(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    if payload.get("split") != "validation" or payload.get("overlap") != 0:
        raise ValueError(f"Invalid validation calibration contract: {path}")
    calibration = [str(value) for value in payload.get("calibration_ids", [])]
    evaluation = [str(value) for value in payload.get("evaluation_ids", [])]
    if not calibration or not evaluation:
        raise ValueError(f"Empty calibration/evaluation partition: {path}")
    return calibration, evaluation


def load_calibration_partition(
    clean_validation: pd.DataFrame,
    proposed_path: str | Path,
    ure_path: str | Path,
    *,
    fraction: float = 0.5,
    seed: int = 340,
    allow_fallback: bool = True,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Load the common detector partition, or derive the declared fallback once.

    One present and one missing partition is always an error.  This prevents the two
    detector evaluations from silently using different validation POIs.
    """
    proposed_path, ure_path = Path(proposed_path), Path(ure_path)
    existence = (proposed_path.is_file(), ure_path.is_file())
    if existence[0] != existence[1]:
        raise FileNotFoundError("Both detector calibration partitions must be present together")
    if all(existence):
        proposed = _read_partition(proposed_path)
        ure = _read_partition(ure_path)
        if proposed != ure:
            raise ValueError("Proposed and URE calibration/evaluation IDs differ")
        calibration, evaluation = proposed
        source = "matching_stage08_stage09_calibration_files"
        paths = [str(proposed_path), str(ure_path)]
    else:
        if not allow_fallback:
            raise FileNotFoundError("Detector calibration partitions are required")
        calibration, evaluation = split_validation_ids(
            clean_validation, calibration_fraction=fraction, seed=seed
        )
        source = "deterministic_validation_clean_fallback"
        paths = []

    validation_ids = clean_validation["POI_ID"].astype(str).tolist()
    validation_set = set(validation_ids)
    calibration_set, evaluation_set = set(calibration), set(evaluation)
    if (len(calibration_set) != len(calibration) or len(evaluation_set) != len(evaluation)
            or calibration_set & evaluation_set
            or calibration_set | evaluation_set != validation_set):
        raise ValueError("Calibration/evaluation IDs must be a disjoint validation partition")
    lookup = clean_validation.set_index("POI_ID")["region"]
    regions = set(clean_validation["region"])
    if set(lookup.loc[calibration]) != regions or set(lookup.loc[evaluation]) != regions:
        raise ValueError("Both validation partitions must contain all regions")
    metadata = {
        "source": source,
        "paths": paths,
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "calibration_fraction": fraction,
        "partition_seed": seed,
        "threshold_selection_limitation": (
            "Thresholds use the calibration half of clean validation. It is disjoint "
            "from reported validation evaluation rows, but is not a separately sampled "
            "detector-development dataset."
        ),
    }
    return calibration, evaluation, metadata


class RegionModelFactory:
    """Fit exactly one frozen train-only OvR classifier per seed and region."""

    def __init__(
        self,
        clean_train: pd.DataFrame,
        regions: Sequence[str],
        *,
        seed: int,
        config: Mapping[str, Any],
        factory_class: type | None = None,
    ) -> None:
        if factory_class is None:
            # Lazy import avoids importing SHAP/LIME when validating inputs or unit tests.
            from .selected_explain import ModelFactory

            factory_class = ModelFactory
        if config.get("model_id", "tabpfn_v2_5") != "tabpfn_v2_5":
            raise ValueError("The common downstream model must be tabpfn_v2_5")
        if config.get("require_checkpoint_attestation") and not config.get(
            "persist_official_fitted_states"
        ):
            raise ValueError("Strict downstream replay requires official fitted-state archives")
        self.clean_train = clean_train.reset_index(drop=True)
        self.regions = tuple(regions)
        self.seed = int(seed)
        self.config = dict(config)
        self.config["seed"] = self.seed
        self._factory_class = factory_class
        self._predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self.fit_count = 0
        self.replay_attestations: list[dict[str, Any]] = []

    def fit(self) -> "RegionModelFactory":
        if self._predictors:
            raise RuntimeError("RegionModelFactory may only be fitted once")
        factory = self._factory_class(
            "tabpfn_v2_5", self.config, self.clean_train
        )
        for region_index, region in enumerate(self.regions):
            truth = self.clean_train["region"].eq(region).astype(int).to_numpy()
            if set(np.unique(truth)) != {0, 1}:
                raise ValueError(f"Train data cannot fit an OvR classifier for {region}")
            self._predictors[region] = factory.fit(FEATURES, truth)
            if self.config.get("require_checkpoint_attestation"):
                output_dir = self.config.get("fit_state_output_dir")
                if not output_dir or not hasattr(factory, "persist_last_fit_state"):
                    raise RuntimeError(
                        f"Missing official downstream fitted-state persistence for {region}"
                    )
                state_path = Path(output_dir) / (
                    f"modelseed{self.seed}__region{region_index:02d}.tabpfn_fit"
                )
                factory.persist_last_fit_state(state_path)
            fit_attestation = dict(getattr(factory, "last_fit_attestation", {}))
            if self.config.get("require_checkpoint_attestation") and (
                fit_attestation.get("format") != "poi-tabpfn25-fit-replay-v1"
                or fit_attestation.get("official_checkpoint_sha256")
                != self.config.get("expected_checkpoint_sha256")
            ):
                raise RuntimeError(f"Missing verified downstream replay attestation for {region}")
            poi_order = self.clean_train[["POI_ID"]].copy()
            fit_attestation.update({
                "region": region,
                "model_seed": self.seed,
                "train_poi_order_sha256": _ordered_identity_sha256(
                    poi_order, ["POI_ID"]
                ),
                "classification_threshold": float(
                    self.config.get("classification_threshold", 0.5)
                ),
            })
            self.replay_attestations.append(fit_attestation)
            self.fit_count += 1
        return self

    def predict(self, matrix5: np.ndarray) -> np.ndarray:
        if len(self._predictors) != len(self.regions):
            raise RuntimeError("RegionModelFactory is not fitted")
        matrix5 = np.asarray(matrix5, dtype=np.float32)
        if matrix5.ndim != 2 or matrix5.shape[1] != len(FEATURES):
            raise ValueError("Region classifier input must be N x 5")
        columns = []
        for region in self.regions:
            score = np.asarray(self._predictors[region](matrix5), dtype=float)
            if score.shape != (len(matrix5),) or not np.isfinite(score).all():
                raise ValueError(f"Invalid frozen classifier output for {region}")
            if ((score < 0) | (score > 1)).any():
                raise ValueError(f"Frozen classifier probabilities outside [0,1]: {region}")
            columns.append(score)
        return np.column_stack(columns)


def validate_attack_rows(
    attacks: pd.DataFrame,
    clean_validation: pd.DataFrame,
    *,
    expected_conditions: int,
    clean_train: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required = {
        "POI_ID", "region", "split", "attack_condition", "generation_seed",
        "constraints_valid", "label_preserved", "primary_evaluation_eligible",
        "quarantine_reason", "attack_attempted", "attack_changed_any", "attack_method", "numeric_budget_m",
        "numeric_budget_was_implicit_zero", "categorical_budget_l0",
        "distance_metric", "distance_tolerance_m", "source_model", "victim_model",
        "victim_evaluation_scope", "transfer_evaluated", "targeted",
        "valid_numeric_train_range", "valid_category_codebook",
        "valid_clean_category_combo", "valid_attack_category_combo",
        "valid_category_combo",
        "within_budget", "clean_boundary_status", "attack_boundary_status",
    }
    required.update(f"{feature}_clean" for feature in FEATURES)
    required.update(f"{feature}_adv" for feature in FEATURES)
    missing = required - set(attacks.columns)
    if missing:
        raise ValueError(f"Attack table is missing columns: {sorted(missing)}")
    if set(attacks["split"]) != {"validation"}:
        raise ValueError("Stage 10 accepts validation attacks only")
    conditions = tuple(sorted(attacks["attack_condition"].astype(str).unique()))
    if len(conditions) != int(expected_conditions):
        raise ValueError(
            f"Expected {expected_conditions} attack conditions, observed {len(conditions)}"
        )
    if attacks.duplicated(["POI_ID", "attack_condition"]).any():
        raise ValueError("Duplicate validation POI/condition attack rows")
    expected_ids = clean_validation["POI_ID"].astype(str).tolist()
    expected_set = set(expected_ids)
    for condition, group in attacks.groupby("attack_condition", sort=True):
        if len(group) != len(expected_ids) or set(group["POI_ID"].astype(str)) != expected_set:
            raise ValueError(f"Condition {condition} is not paired to every validation POI")

    clean_lookup = clean_validation.set_index("POI_ID", drop=False)
    aligned = clean_lookup.loc[attacks["POI_ID"].astype(str)]
    if (aligned["region"].to_numpy() != attacks["region"].astype(str).to_numpy()).any():
        raise ValueError("Attack region differs from the frozen clean manifest")
    for feature in FEATURES:
        clean_values = pd.to_numeric(aligned[feature], errors="raise").to_numpy(dtype=float)
        attack_clean = pd.to_numeric(attacks[f"{feature}_clean"], errors="raise").to_numpy(
            dtype=float
        )
        attack_adv = pd.to_numeric(attacks[f"{feature}_adv"], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(attack_clean).all() or not np.isfinite(attack_adv).all():
            raise ValueError(f"Non-finite attack feature: {feature}")
        if not np.array_equal(clean_values, attack_clean):
            raise ValueError(f"Attack clean anchor differs from frozen clean data: {feature}")

    eligible = attacks["primary_evaluation_eligible"].astype(bool).to_numpy()
    constrained = attacks["constraints_valid"].astype(bool).to_numpy()
    preserved = attacks["label_preserved"].astype(bool).to_numpy()
    reasons = attacks["quarantine_reason"].fillna("").astype(str).str.strip().to_numpy()
    if (eligible & ~(constrained & preserved)).any():
        raise ValueError("Primary-valid rows must be constraint-valid and label-preserved")
    if (eligible & (reasons != "")).any() or ((~eligible) & (reasons == "")).any():
        raise ValueError("Quarantine reasons must identify every ineligible row and no eligible row")
    if attacks["transfer_evaluated"].astype(bool).any():
        raise ValueError("Stage 10 does not accept unverified transfer-evaluation attribution")
    if attacks["targeted"].astype(bool).any():
        raise ValueError("Stage 10 attack table unexpectedly contains targeted attacks")
    if (attacks["source_model"].astype(str) != attacks["victim_model"].astype(str)).any():
        raise ValueError("Direct white-box source/victim provenance is inconsistent")
    if set(attacks["victim_evaluation_scope"].astype(str)) != {
            "direct_white_box_source_model"}:
        raise ValueError("Unknown victim evaluation scope")

    clean_boundary_valid = attacks["clean_boundary_status"].astype(str).eq("match").to_numpy()
    attack_boundary_valid = attacks["attack_boundary_status"].astype(str).eq("match").to_numpy()
    if not np.array_equal(preserved, clean_boundary_valid & attack_boundary_valid):
        raise ValueError("label_preserved disagrees with polygon validation statuses")

    if clean_train is not None:
        train = clean_train.copy()
        clean_values = np.column_stack(
            [pd.to_numeric(attacks[f"{feature}_clean"], errors="raise") for feature in FEATURES]
        ).astype(float)
        adv_values = np.column_stack(
            [pd.to_numeric(attacks[f"{feature}_adv"], errors="raise") for feature in FEATURES]
        ).astype(float)
        numeric_lower = train[NUMERIC].apply(pd.to_numeric, errors="raise").min().to_numpy(float)
        numeric_upper = train[NUMERIC].apply(pd.to_numeric, errors="raise").max().to_numpy(float)
        numeric_valid = (
            (adv_values[:, :2] >= numeric_lower) & (adv_values[:, :2] <= numeric_upper)
        ).all(axis=1)
        category_valid = np.ones(len(attacks), dtype=bool)
        for position, feature in enumerate(CATEGORICAL, start=2):
            vocabulary = set(pd.to_numeric(train[feature], errors="raise").astype(float))
            category_valid &= np.isin(adv_values[:, position], list(vocabulary))
        train_tuples = {
            tuple(values)
            for values in train[CATEGORICAL].apply(pd.to_numeric, errors="raise").to_numpy(float)
        }
        clean_tuple_valid = np.asarray(
            [tuple(values) in train_tuples for values in clean_values[:, 2:]], dtype=bool
        )
        attack_tuple_valid = np.asarray(
            [tuple(values) in train_tuples for values in adv_values[:, 2:]], dtype=bool
        )
        tuple_valid = clean_tuple_valid & attack_tuple_valid
        mean_latitude = np.deg2rad((clean_values[:, 1] + adv_values[:, 1]) / 2)
        dx = (adv_values[:, 0] - clean_values[:, 0]) * 111_320.0 * np.cos(mean_latitude)
        dy = (adv_values[:, 1] - clean_values[:, 1]) * 110_574.0
        distances = np.sqrt(dx ** 2 + dy ** 2)
        numeric_budget = pd.to_numeric(
            attacks["numeric_budget_m"], errors="raise"
        ).to_numpy(dtype=float)
        category_budget = pd.to_numeric(
            attacks["categorical_budget_l0"], errors="raise"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(numeric_budget).all()
            or (numeric_budget < 0).any()
            or not np.isfinite(category_budget).all()
            or (category_budget < 0).any()
            or not np.equal(category_budget, np.floor(category_budget)).all()
        ):
            raise ValueError("Attack budgets must be finite and nonnegative")
        implicit_zero = attacks["numeric_budget_was_implicit_zero"]
        if implicit_zero.isna().any() or not pd.api.types.is_bool_dtype(
            implicit_zero.dtype
        ):
            raise ValueError("numeric_budget_was_implicit_zero must be non-null boolean")
        implicit_zero_values = implicit_zero.to_numpy(dtype=bool)
        if (numeric_budget[implicit_zero_values] != 0.0).any():
            raise ValueError("Implicit numeric budgets must be stored as zero")
        metric = attacks["distance_metric"].astype(str).to_numpy()
        if not np.equal(metric, ATTACK_DISTANCE_METRIC).all():
            raise ValueError("Attack distance metric differs from the producer contract")
        expected_tolerance = np.where(
            implicit_zero_values,
            CATEGORICAL_ONLY_TOLERANCE_M,
            NUMERIC_BUDGET_TOLERANCE_M,
        )
        recorded_tolerance = pd.to_numeric(
            attacks["distance_tolerance_m"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.array_equal(recorded_tolerance, expected_tolerance):
            raise ValueError("Attack distance tolerance differs from the producer contract")
        category_changes = (clean_values[:, 2:] != adv_values[:, 2:]).sum(axis=1)
        within_budget = (
            (distances <= numeric_budget + expected_tolerance)
            & (category_changes <= category_budget)
        )
        recomputed = {
            "valid_numeric_train_range": numeric_valid,
            "valid_category_codebook": category_valid,
            "valid_clean_category_combo": clean_tuple_valid,
            "valid_attack_category_combo": attack_tuple_valid,
            "valid_category_combo": tuple_valid,
            "within_budget": within_budget,
        }
        for name, values in recomputed.items():
            if not np.array_equal(attacks[name].astype(bool).to_numpy(), values):
                raise ValueError(f"Stored {name} differs from train-only recomputation")
        expected_constraints = (
            numeric_valid & category_valid & tuple_valid & within_budget & preserved
        )
        if not np.array_equal(constrained, expected_constraints):
            raise ValueError("constraints_valid differs from train-only recomputation")
        if not np.array_equal(eligible, expected_constraints):
            raise ValueError("primary_evaluation_eligible differs from recomputed validity")

    order = {value: index for index, value in enumerate(expected_ids)}
    result = attacks.copy()
    result["_poi_order"] = result["POI_ID"].astype(str).map(order)
    if result["_poi_order"].isna().any():
        raise ValueError("Attack POI outside clean validation")
    result = result.sort_values(["attack_condition", "_poi_order"]).drop(
        columns="_poi_order"
    )
    return result.reset_index(drop=True), conditions


def _validate_probability(values: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} shape or finite values")
    if ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name} probabilities outside [0,1]")
    return result


def _proposed_output(
    predictor: Any, matrix5: np.ndarray, regions: Sequence[str], *, return_details: bool = False
) -> Any:
    output = predictor.predict(matrix5)
    attack = _validate_probability(output.get("attack_probability"), len(matrix5), "proposed")
    native = np.asarray(output.get("region_probability"), dtype=float)
    predictor_regions = tuple(str(value) for value in predictor.regions)
    if set(predictor_regions) != set(regions) or native.shape != (
        len(matrix5), len(predictor_regions)
    ):
        raise ValueError("Proposed checkpoint region heads differ from the 17-region contract")
    if not np.isfinite(native).all() or ((native < 0) | (native > 1)).any():
        raise ValueError("Invalid proposed native region probabilities")
    positions = [predictor_regions.index(region) for region in regions]
    if return_details:
        return attack, native[:, positions], output
    return attack, native[:, positions]


def _thresholds(
    detector_scores: Mapping[str, np.ndarray], retentions: Sequence[float]
) -> dict[str, dict[float, float]]:
    result: dict[str, dict[float, float]] = {}
    for detector in DETECTOR_LABELS:
        score = _validate_probability(detector_scores[detector], len(detector_scores[detector]), detector)
        result[detector] = {
            float(retention): select_detection_threshold(score, float(retention))
            for retention in retentions
        }
    return result


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
    generation_seed: int,
    condition: str,
    regions: Sequence[str],
    valid: np.ndarray,
    quarantine_reason: np.ndarray,
    detector_scores: Mapping[str, np.ndarray],
    fixed_scores: np.ndarray,
    shared_repair_scores: np.ndarray,
    proposed_native_scores: np.ndarray,
    thresholds: Mapping[str, Mapping[float, float]],
    provenance: Mapping[str, np.ndarray] | None = None,
    proposed_projected_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    n, region_count = len(frame), len(regions)
    truth = frame["region"].astype(str).to_numpy()[:, None] == np.asarray(regions)[None, :]
    parts = []
    provenance = provenance or {}
    for detector in DETECTOR_LABELS:
        detector_score = detector_scores[detector]
        part = pd.DataFrame(
            {
                "POI_ID": np.repeat(frame["POI_ID"].astype(str).to_numpy(), region_count),
                "source_region": np.repeat(frame["region"].astype(str).to_numpy(), region_count),
                "target_region": np.tile(np.asarray(regions, dtype=object), n),
                "truth": truth.reshape(-1).astype(np.int8),
                "seed": seed,
                "generation_seed": generation_seed,
                "attack_condition": condition,
                "attack_label": np.int8(condition != "clean"),
                "detector": detector,
                "detector_score": np.repeat(detector_score, region_count),
                "fixed_unmodified_score": fixed_scores.reshape(-1),
                "shared_nearest_train_repair_score": shared_repair_scores.reshape(-1),
                "proposed_native_full_coverage_score": (
                    proposed_native_scores.reshape(-1)
                    if detector == PROPOSED_LABEL else np.full(n * region_count, np.nan)
                ),
                "proposed_projected_repair_fixed_score": (
                    proposed_projected_scores.reshape(-1)
                    if detector == PROPOSED_LABEL and proposed_projected_scores is not None
                    else np.full(n * region_count, np.nan)
                ),
                "primary_evaluation_eligible": np.repeat(valid.astype(bool), region_count),
                "quarantine_reason": np.repeat(quarantine_reason.astype(str), region_count),
                "attack_method": np.repeat(
                    provenance.get("attack_method", np.full(n, "clean", dtype=object)), region_count
                ),
                "numeric_budget_m": np.repeat(
                    provenance.get("numeric_budget_m", np.full(n, np.nan)), region_count
                ),
                "categorical_budget_l0": np.repeat(
                    provenance.get("categorical_budget_l0", np.full(n, np.nan)), region_count
                ),
                "source_model": np.repeat(
                    provenance.get("source_model", np.full(n, "none", dtype=object)), region_count
                ),
                "victim_model": np.repeat(
                    provenance.get("victim_model", np.full(n, "none", dtype=object)), region_count
                ),
                "victim_evaluation_scope": np.repeat(
                    provenance.get(
                        "victim_evaluation_scope", np.full(n, "clean", dtype=object)
                    ),
                    region_count,
                ),
                "constraints_valid": np.repeat(
                    provenance.get("constraints_valid", np.ones(n, dtype=bool)), region_count
                ),
                "label_preserved": np.repeat(
                    provenance.get("label_preserved", np.ones(n, dtype=bool)), region_count
                ),
                "attack_attempted": np.repeat(
                    provenance.get("attack_attempted", np.zeros(n, dtype=bool)), region_count
                ),
                "attack_changed_any": np.repeat(
                    provenance.get("attack_changed_any", np.zeros(n, dtype=bool)), region_count
                ),
                "transfer_evaluated": np.repeat(
                    provenance.get("transfer_evaluated", np.zeros(n, dtype=bool)), region_count
                ),
                "targeted": np.repeat(
                    provenance.get("targeted", np.zeros(n, dtype=bool)), region_count
                ),
                "valid_numeric_train_range": np.repeat(
                    provenance.get("valid_numeric_train_range", np.ones(n, dtype=bool)),
                    region_count,
                ),
                "valid_category_codebook": np.repeat(
                    provenance.get("valid_category_codebook", np.ones(n, dtype=bool)),
                    region_count,
                ),
                "valid_category_combo": np.repeat(
                    provenance.get("valid_category_combo", np.ones(n, dtype=bool)), region_count
                ),
                "within_budget": np.repeat(
                    provenance.get("within_budget", np.ones(n, dtype=bool)), region_count
                ),
                "clean_boundary_status": np.repeat(
                    provenance.get("clean_boundary_status", np.full(n, "match", dtype=object)),
                    region_count,
                ),
                "attack_boundary_status": np.repeat(
                    provenance.get("attack_boundary_status", np.full(n, "match", dtype=object)),
                    region_count,
                ),
            }
        )
        for retention, threshold in thresholds[detector].items():
            part[_retention_name(retention) + "_accepted"] = np.repeat(
                detector_score <= threshold, region_count
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _provenance(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    names = (
        "attack_method", "numeric_budget_m", "categorical_budget_l0", "source_model",
        "victim_model", "victim_evaluation_scope", "constraints_valid", "label_preserved",
        "attack_attempted", "attack_changed_any", "transfer_evaluated", "targeted", "valid_numeric_train_range",
        "valid_category_codebook", "valid_category_combo", "within_budget",
        "clean_boundary_status", "attack_boundary_status",
    )
    return {name: frame[name].to_numpy() for name in names}


def _calibration_detector_rows(
    frame: pd.DataFrame,
    detector_scores: Mapping[str, np.ndarray],
    *,
    seed: int,
    generation_seed: int,
    condition: str,
    valid: np.ndarray,
    quarantine_reason: np.ndarray,
    provenance: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    provenance = provenance or {}
    parts = []
    n = len(frame)
    for detector in DETECTOR_LABELS:
        parts.append(
            pd.DataFrame(
                {
                    "POI_ID": frame["POI_ID"].astype(str).to_numpy(),
                    "source_region": frame["region"].astype(str).to_numpy(),
                    "seed": seed,
                    "generation_seed": generation_seed,
                    "attack_condition": condition,
                    "attack_label": np.int8(condition != "clean"),
                    "detector": detector,
                    "detector_score": detector_scores[detector],
                    "primary_evaluation_eligible": valid.astype(bool),
                    "quarantine_reason": quarantine_reason.astype(str),
                    "attack_method": provenance.get(
                        "attack_method", np.full(n, "clean", dtype=object)
                    ),
                    "numeric_budget_m": provenance.get(
                        "numeric_budget_m", np.full(n, np.nan)
                    ),
                    "categorical_budget_l0": provenance.get(
                        "categorical_budget_l0", np.full(n, np.nan)
                    ),
                    "source_model": provenance.get(
                        "source_model", np.full(n, "none", dtype=object)
                    ),
                    "victim_model": provenance.get(
                        "victim_model", np.full(n, "none", dtype=object)
                    ),
                    "victim_evaluation_scope": provenance.get(
                        "victim_evaluation_scope", np.full(n, "clean", dtype=object)
                    ),
                    "constraints_valid": provenance.get(
                        "constraints_valid", np.ones(n, dtype=bool)
                    ),
                    "label_preserved": provenance.get(
                        "label_preserved", np.ones(n, dtype=bool)
                    ),
                    "attack_attempted": provenance.get(
                        "attack_attempted", np.zeros(n, dtype=bool)
                    ),
                    "attack_changed_any": provenance.get(
                        "attack_changed_any", np.zeros(n, dtype=bool)
                    ),
                    "transfer_evaluated": provenance.get(
                        "transfer_evaluated", np.zeros(n, dtype=bool)
                    ),
                    "targeted": provenance.get("targeted", np.zeros(n, dtype=bool)),
                    "valid_numeric_train_range": provenance.get(
                        "valid_numeric_train_range", np.ones(n, dtype=bool)
                    ),
                    "valid_category_codebook": provenance.get(
                        "valid_category_codebook", np.ones(n, dtype=bool)
                    ),
                    "valid_category_combo": provenance.get(
                        "valid_category_combo", np.ones(n, dtype=bool)
                    ),
                    "within_budget": provenance.get(
                        "within_budget", np.ones(n, dtype=bool)
                    ),
                    "clean_boundary_status": provenance.get(
                        "clean_boundary_status", np.full(n, "match", dtype=object)
                    ),
                    "attack_boundary_status": provenance.get(
                        "attack_boundary_status", np.full(n, "match", dtype=object)
                    ),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _detector_metric_rows(
    clean_scores: Mapping[str, np.ndarray],
    attack_scores: Mapping[str, np.ndarray],
    valid: np.ndarray,
    changed: np.ndarray | None = None,
    thresholds: Mapping[str, Mapping[float, float]] | None = None,
    *,
    seed: int,
    generation_seed: int,
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed = (
        np.ones(len(valid), dtype=bool)
        if changed is None else np.asarray(changed, dtype=bool)
    )
    if changed.shape != valid.shape:
        raise ValueError("Attack-changed flags must align with validity flags")
    if thresholds is None or set(thresholds) != set(DETECTOR_LABELS):
        raise ValueError("Calibrated retention thresholds are required for detector metrics")
    for detector in DETECTOR_LABELS:
        for population_scope, mask in (
            ("all_intent", np.ones(len(valid), dtype=bool)),
            ("primary_valid", valid.astype(bool)),
        ):
            score = np.concatenate((clean_scores[detector], attack_scores[detector][mask]))
            truth = np.concatenate(
                (np.zeros(len(clean_scores[detector]), dtype=int), np.ones(mask.sum(), dtype=int))
            )
            for retention, threshold in thresholds[detector].items():
                rows.append({
                    "seed": seed,
                    "generation_seed": generation_seed,
                    "attack_condition": condition,
                    "detector": detector,
                    "population_scope": population_scope,
                    "clean_retention_target": float(retention),
                    "detection_threshold": float(threshold),
                    "detector_decision_rule": "attack_probability_strictly_gt_threshold",
                    "threshold_tie_rows": int(np.isclose(score, threshold, rtol=0, atol=0).sum()),
                    "all_intent_attack_rows": len(valid),
                    "primary_valid_attack_rows": int(valid.sum()),
                    "quarantined_attack_rows": int((~valid.astype(bool)).sum()),
                    "changed_attack_rows": int(changed.sum()),
                    "unchanged_attack_rows": int((~changed).sum()),
                    **binary_metrics(
                        truth, score, threshold=float(threshold), threshold_operator="gt"
                    ),
                })
    return rows


def _classification_metric_rows(
    truth: np.ndarray,
    fixed: np.ndarray,
    shared: np.ndarray,
    native: np.ndarray,
    detector_scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[float, float]],
    *,
    seed: int,
    generation_seed: int,
    condition: str,
    cohort: str,
    population_scope: str,
    all_intent_rows: int,
    primary_valid_rows: int,
    regions: Sequence[str],
    classification_threshold: float,
    proposed_projected: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        region_truth = truth[:, region_index].astype(int)
        for detector in DETECTOR_LABELS:
            for retention, threshold in thresholds[detector].items():
                values = paired_defense_metrics(
                    region_truth,
                    fixed[:, region_index],
                    shared[:, region_index],
                    detector_scores[detector],
                    threshold,
                    classification_threshold,
                )
                common = {
                    "seed": seed,
                    "generation_seed": generation_seed,
                    "attack_condition": condition,
                    "evaluation_cohort": cohort,
                    "population_scope": population_scope,
                    "target_region": region,
                    "detector": detector,
                    "clean_retention_target": retention,
                    "detection_threshold": threshold,
                    "coverage": values["coverage"],
                    "accepted": values["accepted"],
                    "rejected": values["rejected"],
                    "total": values["total"],
                    "accepted_and_correct_fraction": values[
                        "accepted_and_correct_fraction"
                    ],
                    "all_intent_rows": all_intent_rows,
                    "primary_valid_rows": primary_valid_rows,
                    "quarantined_rows": all_intent_rows - primary_valid_rows,
                }
                for mode, key in (
                    ("fixed_unfiltered", "unfiltered"),
                    ("detector_accepted_only", "accepted_only"),
                    ("shared_nearest_train_repair", "repaired_all"),
                ):
                    rows.append({**common, "classification_mode": mode, **values[key]})
                if detector == PROPOSED_LABEL and proposed_projected is not None:
                    projected_values = paired_defense_metrics(
                        region_truth, fixed[:, region_index], proposed_projected[:, region_index],
                        detector_scores[detector], threshold, classification_threshold)
                    rows.append({**common,
                        "classification_mode": "proposed_projected_repair_fixed_tabpfn25",
                        **projected_values["repaired_all"]})

        # Native proposed reconstruction/head is evaluated for every row, independent
        # of any detector threshold.  URE has no corresponding row by design.
        rows.append(
            {
                "seed": seed,
                "generation_seed": generation_seed,
                "attack_condition": condition,
                "evaluation_cohort": cohort,
                "population_scope": population_scope,
                "target_region": region,
                "detector": PROPOSED_LABEL,
                "clean_retention_target": np.nan,
                "detection_threshold": np.nan,
                "coverage": 1.0,
                "accepted": len(region_truth),
                "rejected": 0,
                "total": len(region_truth),
                "accepted_and_correct_fraction": float(
                    ((native[:, region_index] >= classification_threshold) == region_truth).mean()
                ) if len(region_truth) else np.nan,
                "all_intent_rows": all_intent_rows,
                "primary_valid_rows": primary_valid_rows,
                "quarantined_rows": all_intent_rows - primary_valid_rows,
                "classification_mode": "proposed_native_full_coverage",
                **binary_metrics(
                    region_truth, native[:, region_index], classification_threshold
                ),
            }
        )
    return rows


def _aligned_condition(
    attacks: pd.DataFrame, condition: str, evaluation_ids: Sequence[str]
) -> pd.DataFrame:
    rows = attacks.loc[attacks["attack_condition"].astype(str).eq(condition)].copy()
    rows = rows.set_index("POI_ID", drop=False)
    if set(evaluation_ids) - set(rows.index.astype(str)):
        raise ValueError(f"Condition {condition} lacks evaluation POIs")
    return rows.loc[list(evaluation_ids)].reset_index(drop=True)


def _cohort_metric_rows(
    clean: dict[str, Any],
    attack: dict[str, Any],
    valid: np.ndarray,
    **identity: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_mask = np.ones(len(valid), dtype=bool)
    for population_scope, attack_mask in (
        ("all_intent", all_mask),
        ("primary_valid", valid.astype(bool)),
    ):
        for cohort in ("attack", "clean_plus_attack"):
            if cohort == "attack":
                truth = attack["truth"][attack_mask]
                fixed = attack["fixed"][attack_mask]
                shared = attack["shared"][attack_mask]
                native = attack["native"][attack_mask]
                projected = attack["projected"][attack_mask] if "projected" in attack else None
                detector = {
                    label: attack["detector"][label][attack_mask] for label in DETECTOR_LABELS
                }
            else:
                truth = np.concatenate((clean["truth"], attack["truth"][attack_mask]), axis=0)
                fixed = np.concatenate((clean["fixed"], attack["fixed"][attack_mask]), axis=0)
                shared = np.concatenate((clean["shared"], attack["shared"][attack_mask]), axis=0)
                native = np.concatenate((clean["native"], attack["native"][attack_mask]), axis=0)
                projected = (np.concatenate((clean["projected"], attack["projected"][attack_mask]), axis=0)
                             if "projected" in attack and "projected" in clean else None)
                detector = {
                    label: np.concatenate(
                        (clean["detector"][label], attack["detector"][label][attack_mask])
                    )
                    for label in DETECTOR_LABELS
                }
            rows.extend(
                _classification_metric_rows(
                    truth,
                    fixed,
                    shared,
                    native,
                    detector,
                    identity["thresholds"],
                    seed=identity["seed"],
                    generation_seed=identity["generation_seed"],
                    condition=identity["condition"],
                    cohort=cohort,
                    population_scope=population_scope,
                    all_intent_rows=(
                        len(valid) if cohort == "attack" else len(clean["truth"]) + len(valid)
                    ),
                    primary_valid_rows=(
                        int(valid.sum())
                        if cohort == "attack"
                        else len(clean["truth"]) + int(valid.sum())
                    ),
                    regions=identity["regions"],
                    classification_threshold=identity["classification_threshold"],
                    proposed_projected=projected,
                )
            )
    return rows


def _clean_metric_rows(
    clean: dict[str, Any],
    *,
    seed: int,
    generation_seed: int,
    regions: Sequence[str],
    thresholds: Mapping[str, Mapping[float, float]],
    classification_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population_scope in ("all_intent", "primary_valid"):
        rows.extend(
            _classification_metric_rows(
                clean["truth"], clean["fixed"], clean["shared"], clean["native"],
                clean["detector"], thresholds,
                seed=seed, generation_seed=generation_seed, condition="clean",
                cohort="clean", population_scope=population_scope,
                all_intent_rows=len(clean["truth"]), primary_valid_rows=len(clean["truth"]),
                regions=regions, classification_threshold=classification_threshold,
                proposed_projected=clean.get("projected"),
            )
        )
    return rows


def macro_region_metrics(
    regional: pd.DataFrame, *, expected_regions: int
) -> pd.DataFrame:
    """Macro-average the five OvR metrics while disclosing undefined-region counts."""
    metrics = ("f1", "precision", "recall", "accuracy", "roc_auc")
    group_columns = [
        "seed", "generation_seed", "attack_condition", "evaluation_cohort",
        "population_scope", "detector", "clean_retention_target",
        "detection_threshold", "classification_mode",
    ]
    required = set(group_columns) | {"target_region", "coverage", "accepted", "rejected",
                                     "total", *metrics}
    if required - set(regional):
        raise ValueError(f"Regional metric table lacks: {sorted(required - set(regional))}")
    rows: list[dict[str, Any]] = []
    for key, group in regional.groupby(group_columns, dropna=False, sort=False):
        if len(group) != expected_regions or group["target_region"].nunique() != expected_regions:
            raise ValueError("Macro summary requires exactly one row per target region")
        for column in ("coverage", "accepted", "rejected", "total"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Regional rows disagree on selective {column}")
        row = dict(zip(group_columns, key))
        row.update({
            "regions_required": expected_regions,
            "denominator_pois": int(group["total"].iloc[0]),
            "accepted_pois": int(group["accepted"].iloc[0]),
            "rejected_pois": int(group["rejected"].iloc[0]),
            "coverage": float(group["coverage"].iloc[0]) if pd.notna(
                group["coverage"].iloc[0]
            ) else np.nan,
            "aggregation": "unweighted_macro_across_target_region_ovr_tasks",
            "zero_division_policy": 0,
        })
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            row[f"macro_{metric}"] = float(values[finite].mean()) if finite.any() else np.nan
            row[f"{metric}_finite_regions"] = int(finite.sum())
            row[f"{metric}_undefined_regions"] = int((~finite).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def selective_operating_summary(
    clean_scores: Mapping[str, np.ndarray],
    attack_scores: Mapping[str, np.ndarray],
    valid: np.ndarray,
    thresholds: Mapping[str, Mapping[float, float]],
    **identity: Any,
) -> pd.DataFrame:
    """Report clean retention, attack rejection, and common denominators explicitly."""
    rows = []
    valid = np.asarray(valid, dtype=bool)
    for detector in DETECTOR_LABELS:
        clean = np.asarray(clean_scores[detector], dtype=float)
        attack = np.asarray(attack_scores[detector], dtype=float)
        for scope, mask in (
            ("all_intent", np.ones(len(valid), dtype=bool)),
            ("primary_valid", valid),
        ):
            for retention, threshold in thresholds[detector].items():
                clean_accepted = clean <= threshold
                attack_accepted = attack[mask] <= threshold
                attack_total = int(mask.sum())
                rows.append({
                    **identity, "detector": detector, "population_scope": scope,
                    "clean_retention_target": float(retention),
                    "detection_threshold": float(threshold),
                    "accept_rule": "attack_probability_lte_threshold",
                    "attack_rule": "attack_probability_strictly_gt_threshold",
                    "clean_total_pois": len(clean),
                    "clean_accepted_pois": int(clean_accepted.sum()),
                    "clean_retention": float(clean_accepted.mean()) if len(clean) else np.nan,
                    "attack_total_pois": attack_total,
                    "attack_accepted_pois": int(attack_accepted.sum()),
                    "attack_rejected_pois": int((~attack_accepted).sum()),
                    "attack_rejection": (
                        float((~attack_accepted).mean()) if attack_total else np.nan
                    ),
                    "combined_total_pois": len(clean) + attack_total,
                    "combined_accepted_pois": int(clean_accepted.sum() + attack_accepted.sum()),
                    "combined_coverage": (
                        float((clean_accepted.sum() + attack_accepted.sum())
                              / (len(clean) + attack_total))
                        if len(clean) + attack_total else np.nan
                    ),
                    "empty_attack_population": attack_total == 0,
                })
    return pd.DataFrame(rows)


def binary_ovr_risk_coverage(
    truth: np.ndarray,
    region_scores: np.ndarray,
    detector_scores: np.ndarray,
    poi_ids: Sequence[str],
    calibrated_thresholds: Mapping[float, float],
    *,
    classification_threshold: float,
    max_curve_points: int = 101,
    **identity: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact tie-aware empirical selective risk with a bounded exported curve.

    Risk is the mean binary classification error over all POI × target-region OvR decisions.
    Coverage is the fraction of POIs accepted by ascending detector attack probability.
    """
    truth = np.asarray(truth, dtype=int)
    region_scores = np.asarray(region_scores, dtype=float)
    detector_scores = np.asarray(detector_scores, dtype=float)
    ids = np.asarray(list(map(str, poi_ids)), dtype=object)
    if (
        truth.ndim != 2 or truth.shape != region_scores.shape
        or detector_scores.shape != (len(truth),) or ids.shape != (len(truth),)
        or not np.isin(truth, [0, 1]).all()
        or not np.isfinite(region_scores).all() or not np.isfinite(detector_scores).all()
    ):
        raise ValueError("Risk-coverage inputs must be finite paired POI × OvR predictions")
    n, regions = truth.shape
    base = {
        **identity,
        "classification_mode": "fixed_unmodified_selective_accepted_only",
        "risk_definition": "mean_binary_ovr_0_1_error_over_accepted_poi_target_rows",
        "coverage_denominator": "eligible_pois_in_declared_population_scope",
        "selection_order": "ascending_detector_attack_probability_with_ties_kept_together",
        "classification_threshold": float(classification_threshold),
        "total_pois": n, "target_regions": regions,
    }
    if n == 0:
        curve = pd.DataFrame([{**base, "accepted_pois": 0, "coverage": np.nan,
                               "selective_risk": np.nan, "detector_score_cutoff": np.nan,
                               "calibrated_clean_retention_targets": "",
                               "empty_population": True}])
        return curve, {**base, "aurc": np.nan, "curve_points_exact": 0,
                       "curve_points_exported": 1, "empty_population": True}
    if ((region_scores < 0) | (region_scores > 1)).any() or (
        (detector_scores < 0) | (detector_scores > 1)
    ).any():
        raise ValueError("Risk-coverage scores must lie in [0,1]")
    poi_risk = ((region_scores >= classification_threshold) != truth).mean(axis=1)
    order = np.lexsort((ids, detector_scores))
    sorted_detection = detector_scores[order]
    sorted_risk = poi_risk[order]
    end_positions = np.flatnonzero(np.r_[np.diff(sorted_detection) != 0, True])
    accepted = end_positions + 1
    cumulative_risk = np.cumsum(sorted_risk)[end_positions] / accepted
    coverage = accepted / n
    cutoffs = sorted_detection[end_positions]
    increments = np.diff(np.r_[0.0, coverage])
    aurc = float(np.sum(increments * cumulative_risk))
    calibrated_by_count: dict[int, list[float]] = {}
    for target, threshold in calibrated_thresholds.items():
        count = int(np.searchsorted(sorted_detection, float(threshold), side="right"))
        calibrated_by_count.setdefault(count, []).append(float(target))
    selected = set(range(len(accepted)))
    if len(selected) > max_curve_points:
        selected = set(np.linspace(0, len(accepted) - 1, max_curve_points, dtype=int))
    for count in calibrated_by_count:
        if count > 0:
            selected.add(int(np.searchsorted(accepted, count, side="left")))
    rows = []
    if 0 in calibrated_by_count:
        rows.append({
            **base, "accepted_pois": 0, "coverage": 0.0, "selective_risk": np.nan,
            "detector_score_cutoff": min(calibrated_thresholds.values()),
            "calibrated_clean_retention_targets": "|".join(
                f"{value:g}" for value in sorted(calibrated_by_count[0])
            ),
            "empty_population": False,
        })
    for position in sorted(selected):
        count = int(accepted[position])
        rows.append({
            **base, "accepted_pois": count, "coverage": float(coverage[position]),
            "selective_risk": float(cumulative_risk[position]),
            "detector_score_cutoff": float(cutoffs[position]),
            "calibrated_clean_retention_targets": "|".join(
                f"{value:g}" for value in sorted(calibrated_by_count.get(count, []))
            ),
            "empty_population": False,
        })
    return pd.DataFrame(rows), {
        **base, "aurc": aurc, "curve_points_exact": len(accepted),
        "curve_points_exported": len(rows), "empty_population": False,
        "aurc_integration": "right_riemann_over_exact_tie_group_coverage_increments",
    }


def _load_models(cfg: Mapping[str, Any], seed: int) -> tuple[Any, Any, Path, Path]:
    proposed_path = Path(str(cfg["proposed_checkpoint_template"]).format(seed=seed))
    ure_path = Path(str(cfg["ure_checkpoint_template"]).format(seed=seed))
    if not proposed_path.is_file() or not ure_path.is_file():
        raise FileNotFoundError(
            f"Missing detector checkpoint(s) for seed {seed}: {proposed_path}, {ure_path}"
        )
    if not ure_path.with_suffix(".joblib").is_file():
        raise FileNotFoundError(f"Missing adjacent URE Random Forest: {ure_path.with_suffix('.joblib')}")
    from .proposed_defense import load_proposed_predictor
    from .ure import load_ure_attack_detector

    proposed = load_proposed_predictor(
        proposed_path,
        device=cfg["device"],
        official_checkpoint=cfg.get("official_checkpoint"),
        batch_size=int(cfg.get("proposed_batch_size", 128)),
    )
    proposed.assert_decision_contract(
        classification_threshold=float(cfg.get("classification_threshold", 0.5)),
        regional_decision_rule="independent_ovr_probability_gte_threshold",
    )
    ure = load_ure_attack_detector(ure_path, device=cfg["device"])
    return proposed, ure, proposed_path, ure_path


def _observations(
    frame: pd.DataFrame,
    proposed: Any,
    ure: Any,
    fixed: RegionModelFactory,
    repair: SharedNearestTrainRepair,
    regions: Sequence[str],
    fixed_cache: ExactFloat32PredictionCache | None = None,
) -> dict[str, Any]:
    matrix5 = _matrix(frame)
    proposed_attack, native, details = _proposed_output(proposed, matrix5, regions, return_details=True)
    native_repair = np.asarray(details.get("repaired_matrix"), dtype=np.float32)
    if native_repair.shape != matrix5.shape or not np.isfinite(native_repair).all():
        raise ValueError("Proposed checkpoint must supply finite projected N x 5 repair")
    ure_attack = _validate_probability(ure.predict_proba(matrix5)[:, 1], len(frame), "URE")
    repaired_frame = repair.repair(frame[FEATURES])
    repaired_matrix = _matrix(repaired_frame)
    truth = frame["region"].astype(str).to_numpy()[:, None] == np.asarray(regions)[None, :]
    fixed_predict = fixed_cache.predict if fixed_cache is not None else fixed.predict
    return {
        "truth": truth,
        "fixed": fixed_predict(matrix5),
        "shared": fixed_predict(repaired_matrix),
        "native": native,
        "projected": fixed_predict(native_repair),
        "projection_diagnostics": {
            "rows": len(frame),
            "numeric_out_of_bounds_rows": int(np.asarray(details["preprojection_numeric_out_of_bounds"]).any(axis=1).sum()),
            "invalid_category_tuple_rows": int(np.asarray(details["preprojection_invalid_category_tuple"]).sum()),
        },
        "detector": {PROPOSED_LABEL: proposed_attack, URE_LABEL: ure_attack},
    }


def _cache_runtime_statistics(
    cache: ExactFloat32PredictionCache,
    *,
    model_seed: int,
    region_count: int,
    expected_requests_per_seed: int,
    model_seed_count: int,
    status: str,
) -> dict[str, Any]:
    """Add call counts and a clearly limited runtime projection to exact cache statistics."""
    statistics = cache.statistics()
    if statistics["requests"] > expected_requests_per_seed or statistics["requests"] % 3:
        raise RuntimeError("Stage-10 fixed prediction request progress is inconsistent")
    mean_miss_seconds = (
        statistics["underlying_prediction_seconds"] / statistics["misses"]
        if statistics["misses"] else 0.0
    )
    statistics.update({
        "status": status,
        "model_seed": int(model_seed),
        "observation_groups_completed": statistics["requests"] // 3,
        "observation_groups_expected": expected_requests_per_seed // 3,
        "regions_per_request": int(region_count),
        "requested_region_predictor_calls": statistics["requests"] * region_count,
        "executed_region_predictor_calls": statistics["misses"] * region_count,
        "avoided_region_predictor_calls": statistics["hits"] * region_count,
        "mean_miss_prediction_seconds": mean_miss_seconds,
        "empirical_uncached_per_seed_projection_seconds": (
            mean_miss_seconds * expected_requests_per_seed
        ),
        "empirical_uncached_full_stage_projection_seconds": (
            mean_miss_seconds * expected_requests_per_seed * model_seed_count
        ),
        "forecast_limitation": (
            "Projection uses the observed exact-cache-miss mean on this seed and host; it is "
            "not a runtime guarantee and excludes detectors, fitting, metrics, plots, and I/O."
        ),
    })
    return statistics


def _timed_probability_call(
    predictor: Callable[[np.ndarray], np.ndarray],
    matrix5: np.ndarray,
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[np.ndarray, float]:
    """Time one synchronized one-region probability call without mutating its input."""
    matrix = np.asarray(matrix5)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("Timing-probe input must be an exact float32 N x 5 matrix")
    before = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    detached = np.array(matrix, dtype=np.float32, order="C", copy=True)
    detached_before = hashlib.sha256(detached.tobytes()).hexdigest()
    synchronize()
    started = clock()
    probability = np.asarray(predictor(detached), dtype=np.float64)
    synchronize()
    elapsed = clock() - started
    after = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    detached_after = hashlib.sha256(detached.tobytes()).hexdigest()
    if before != after or detached_before != detached_after:
        raise RuntimeError("Frozen predictor mutated timing-probe input")
    if (
        probability.shape != (len(matrix),)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or elapsed < 0 or not np.isfinite(elapsed)
    ):
        raise RuntimeError("Frozen timing-probe prediction or duration is invalid")
    return probability, float(elapsed)


def _plot_curves(prediction_paths: Sequence[Path], figures: Path, primary_retention: float) -> None:
    """Draw exact detector and bounded-size passed-region curves from prediction shards."""
    figures.mkdir(parents=True, exist_ok=True)
    accepted_column = _retention_name(primary_retention) + "_accepted"
    detector_data: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        label: [] for label in DETECTOR_LABELS
    }
    region_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        label: [] for label in DETECTOR_LABELS
    }
    clean_parts: dict[tuple[int, str], pd.DataFrame] = {}
    attack_parts: list[pd.DataFrame] = []
    columns = [
        "POI_ID", "seed", "attack_condition", "attack_label", "detector",
        "detector_score", "truth", "fixed_unmodified_score",
        "primary_evaluation_eligible", accepted_column,
    ]
    for path in prediction_paths:
        frame = pd.read_parquet(path, columns=columns)
        if frame["attack_condition"].eq("clean").all():
            for (seed, detector), group in frame.groupby(["seed", "detector"], sort=False):
                clean_parts[(int(seed), str(detector))] = group
        else:
            attack_parts.append(frame)
    for attack in attack_parts:
        attack = attack.loc[attack["primary_evaluation_eligible"].astype(bool)]
        for (seed, detector), group in attack.groupby(["seed", "detector"], sort=False):
            clean = clean_parts[(int(seed), str(detector))]
            # Detector scores are repeated for each OvR target; count every POI once.
            clean_detector = clean.drop_duplicates("POI_ID")
            attack_detector = group.drop_duplicates("POI_ID")
            detector_data[str(detector)].append(
                (
                    np.concatenate(
                        (np.zeros(len(clean_detector), dtype=int), np.ones(len(attack_detector), dtype=int))
                    ),
                    np.concatenate(
                        (clean_detector["detector_score"], attack_detector["detector_score"])
                    ).astype(float),
                    np.concatenate(
                        (
                            ~clean_detector[accepted_column].to_numpy(dtype=bool),
                            ~attack_detector[accepted_column].to_numpy(dtype=bool),
                        )
                    ).astype(int),
                )
            )
            accepted = group.loc[group[accepted_column].astype(bool)]
            if len(accepted):
                region_data[str(detector)].append(
                    (
                        accepted["truth"].to_numpy(dtype=int),
                        accepted["fixed_unmodified_score"].to_numpy(dtype=float),
                    )
                )

    def concatenate(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        if not parts:
            return np.array([], dtype=int), np.array([], dtype=float)
        return np.concatenate([item[0] for item in parts]), np.concatenate([item[1] for item in parts])

    def concatenate_detector(
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not parts:
            return (
                np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)
            )
        return (
            np.concatenate([item[0] for item in parts]),
            np.concatenate([item[1] for item in parts]),
            np.concatenate([item[2] for item in parts]),
        )

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, detector in zip(axes, DETECTOR_LABELS):
        truth, _, prediction = concatenate_detector(detector_data[detector])
        matrix = confusion_matrix(truth, prediction, labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(title=detector, xlabel="Predicted", ylabel="Observed", xticks=[0, 1], yticks=[0, 1])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(figures / "detector_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, detector in enumerate(DETECTOR_LABELS):
        truth, score, _ = concatenate_detector(detector_data[detector])
        if len(truth) == 0:
            for column, title in enumerate(("precision-recall", "ROC")):
                axes[row, column].text(
                    .5, .5, "No primary-valid detector rows", ha="center", va="center",
                    transform=axes[row, column].transAxes,
                )
                axes[row, column].set_title(f"{detector}: {title}")
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            continue
        precision, recall, _ = precision_recall_curve(truth, score)
        axes[row, 0].plot(recall, precision)
        axes[row, 0].set(title=f"{detector}: precision-recall", xlabel="Recall", ylabel="Precision")
        if np.unique(truth).size < 2:
            axes[row, 1].text(
                .5, .5, "ROC undefined: one observed class", ha="center", va="center",
                transform=axes[row, 1].transAxes,
            )
        else:
            false_positive, true_positive, _ = roc_curve(truth, score)
            axes[row, 1].plot(false_positive, true_positive)
        axes[row, 1].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[row, 1].set(title=f"{detector}: ROC", xlabel="False positive rate", ylabel="True positive rate")
    figure.tight_layout()
    figure.savefig(figures / "detector_pr_roc.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, detector in enumerate(DETECTOR_LABELS):
        truth, score = concatenate(region_data[detector])
        if len(truth) == 0:
            for column, title in enumerate((
                "passed-region confusion", "passed-region precision-recall", "passed-region ROC"
            )):
                axes[row, column].text(
                    0.5, 0.5, "No accepted primary-valid rows",
                    ha="center", va="center", transform=axes[row, column].transAxes,
                )
                axes[row, column].set_title(f"{detector}: {title}")
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            continue
        # Curves are visual summaries; exact metrics remain in CSV.  Bound figure memory
        # with a deterministic stride while retaining both classes.
        if len(truth) > 250_000:
            positions = np.linspace(0, len(truth) - 1, 250_000, dtype=int)
            truth, score = truth[positions], score[positions]
        matrix = confusion_matrix(truth, score >= 0.5, labels=[0, 1])
        axes[row, 0].imshow(matrix, cmap="Purples")
        for y in range(2):
            for x in range(2):
                axes[row, 0].text(x, y, str(matrix[y, x]), ha="center", va="center")
        axes[row, 0].set(title=f"{detector}: passed-region confusion", xlabel="Predicted", ylabel="Observed")
        precision, recall, _ = precision_recall_curve(truth, score)
        axes[row, 1].plot(recall, precision)
        axes[row, 1].set(title="Passed-region precision-recall", xlabel="Recall", ylabel="Precision")
        if np.unique(truth).size < 2:
            axes[row, 2].text(
                .5, .5, "ROC undefined: one observed class", ha="center", va="center",
                transform=axes[row, 2].transAxes,
            )
        else:
            false_positive, true_positive, _ = roc_curve(truth, score)
            axes[row, 2].plot(false_positive, true_positive)
        axes[row, 2].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[row, 2].set(title="Passed-region ROC", xlabel="False positive rate", ylabel="True positive rate")
    figure.tight_layout()
    figure.savefig(figures / "passed_region_confusion_pr_roc.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_risk_coverage(curves: pd.DataFrame, figures: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    view = curves.loc[
        curves["evaluation_cohort"].eq("attack")
        & curves["population_scope"].eq("primary_valid")
        & curves["coverage"].notna()
        & curves["selective_risk"].notna()
    ].copy()
    if view.empty:
        axis.text(.5, .5, "No primary-valid attack risk-coverage rows", ha="center", va="center",
                  transform=axis.transAxes)
    else:
        view["coverage_bin"] = view["coverage"].round(2)
        aggregate = view.groupby(["detector", "coverage_bin"], observed=True)[
            "selective_risk"
        ].mean().reset_index()
        for detector, group in aggregate.groupby("detector", sort=True):
            axis.plot(group["coverage_bin"], group["selective_risk"], label=detector)
        axis.legend()
    axis.set(
        title="Primary-valid attack binary-OvR risk–coverage",
        xlabel="Accepted POI coverage", ylabel="Mean binary OvR 0–1 risk",
        xlim=(0, 1), ylim=(0, 1),
    )
    figure.tight_layout()
    figure.savefig(figures / "risk_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_fixed_prediction_timing_probe(
    config_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Fit one frozen region model and time exact 1-partition prediction paths.

    This validation-only launch probe never supplies fitted state to Stage 10.  It
    measures one seed/region so the resulting full-stage projection remains a
    planning estimate rather than experimental output.
    """
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite timing probe: {output}")
    # The probe writes directly to a unique output directory so absolute fitted-state
    # paths stay valid.  Any interruption leaves a partial directory that cannot resume.
    output.mkdir(parents=True)
    staging = output
    state_dir = staging / "probe_fit_state"
    state_dir.mkdir()

    source_paths = [
        Path(__file__).resolve(), Path(__file__).with_name("exact_prediction_cache.py").resolve(),
        Path(__file__).with_name("selected_explain.py").resolve(),
        Path(__file__).with_name("defense_evaluate.py").resolve(),
        Path(__file__).with_name("data.py").resolve(),
        Path(__file__).with_name("ablation.py").resolve(),
        Path(__file__).with_name("benchmark.py").resolve(), config_path,
        *sorted((Path(__file__).with_name("proposed_defense")).glob("*.py")),
    ]
    if any(not path.is_file() for path in source_paths):
        raise RuntimeError("Timing-probe source/config inputs are incomplete")
    source_hashes = {str(path): sha256(path) for path in source_paths}

    model_seed = int(cfg["seeds"][0])
    proposed_path = Path(
        str(cfg["proposed_checkpoint_template"]).format(seed=model_seed)
    ).resolve()
    fixed_config = dict(cfg["fixed_region_model"])
    fixed_config.update({
        "seed": model_seed,
        "classification_threshold": float(cfg.get("classification_threshold", 0.5)),
        "fit_state_output_dir": str(state_dir),
    })
    fixed_checkpoint = Path(str(fixed_config["official_checkpoint"])).expanduser().resolve()
    data_dir = Path(cfg["data_dir"]).resolve()
    input_paths = [
        data_dir / "poi_data_region.csv", data_dir / "sample_manifest.csv",
        data_dir / "report.json", Path(cfg["calibration"]["proposed_ids"]).resolve(),
        Path(cfg["calibration"]["ure_ids"]).resolve(), proposed_path, fixed_checkpoint,
    ]
    if any(not path.is_file() for path in input_paths):
        raise FileNotFoundError("Timing-probe data, partition, or checkpoint input is absent")
    input_hashes = {str(path): sha256(path) for path in input_paths}
    if input_hashes[str(fixed_checkpoint)] != str(
        fixed_config["expected_checkpoint_sha256"]
    ).lower():
        raise RuntimeError("Timing probe fixed checkpoint is hash-mismatched")

    clean, _ = load_clean_splits(
        cfg["data_dir"], splits=("train", "validation"), verify_hashes=True
    )
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    regions = tuple(sorted(train["region"].astype(str).unique()))
    _, evaluation_ids, partition = load_calibration_partition(
        validation,
        cfg["calibration"]["proposed_ids"], cfg["calibration"]["ure_ids"],
        fraction=float(cfg["calibration"].get("fraction", 0.5)),
        seed=int(cfg["calibration"].get("seed", 340)),
        allow_fallback=bool(cfg["calibration"].get("allow_fallback", False)),
    )
    evaluation = validation.set_index("POI_ID", drop=False).loc[
        evaluation_ids
    ].reset_index(drop=True)
    raw_matrix = _matrix(evaluation)
    shared_matrix = _matrix(SharedNearestTrainRepair().fit(train[FEATURES]).repair(
        evaluation[FEATURES]
    ))

    region = regions[0]
    from .proposed_defense import load_proposed_predictor

    proposed = load_proposed_predictor(
        proposed_path, device=cfg["device"], official_checkpoint=cfg.get("official_checkpoint"),
        batch_size=int(cfg.get("proposed_batch_size", 128)),
    )
    proposed.assert_decision_contract(
        classification_threshold=float(cfg.get("classification_threshold", 0.5)),
        regional_decision_rule="independent_ovr_probability_gte_threshold",
    )
    _, _, proposed_details = _proposed_output(
        proposed, raw_matrix, regions, return_details=True
    )
    projected_matrix = np.asarray(proposed_details.get("repaired_matrix"), dtype=np.float32)
    if projected_matrix.shape != raw_matrix.shape or not np.isfinite(projected_matrix).all():
        raise RuntimeError("Timing probe received an invalid proposed projected matrix")

    from .selected_explain import ModelFactory

    synchronize = _device_synchronizer(str(fixed_config["device"]))
    factory = ModelFactory("tabpfn_v2_5", fixed_config, train)
    truth = train["region"].astype(str).eq(region).astype(int).to_numpy()
    synchronize()
    fit_started = time.perf_counter()
    predictor = factory.fit(FEATURES, truth)
    synchronize()
    fit_seconds = time.perf_counter() - fit_started
    state_path = state_dir / f"probe_modelseed{model_seed}__region00.tabpfn_fit"
    factory.persist_last_fit_state(state_path)
    fit_attestation = dict(factory.last_fit_attestation)
    fit_attestation.update({
        "region": region, "model_seed": model_seed,
        "train_poi_order_sha256": _ordered_identity_sha256(train, ["POI_ID"]),
        "classification_threshold": float(cfg.get("classification_threshold", 0.5)),
    })
    model_identity = build_frozen_model_identity(
        model_id="tabpfn_v2_5", model_seed=model_seed, feature_order=FEATURES,
        region_order=(region,), fit_attestations=(fit_attestation,),
        source_sha256=source_hashes,
    )

    warmup, warmup_seconds = _timed_probability_call(
        predictor, raw_matrix, synchronize=synchronize
    )
    raw, raw_seconds = _timed_probability_call(
        predictor, raw_matrix, synchronize=synchronize
    )
    shared, shared_seconds = _timed_probability_call(
        predictor, shared_matrix, synchronize=synchronize
    )
    projected, projected_seconds = _timed_probability_call(
        predictor, projected_matrix, synchronize=synchronize
    )
    if not np.array_equal(warmup, raw):
        raise RuntimeError("Timing-probe warmup and repeated raw probabilities differ")
    cache = ExactFloat32PredictionCache(
        lambda matrix: np.asarray(predictor(matrix), dtype=np.float64)[:, None],
        model_identity, feature_order=FEATURES, output_order=(region,), max_entries=4,
        max_bytes=16 * 1024 * 1024, synchronize=synchronize,
    )
    cache_miss = cache.predict(raw_matrix)
    cache_hit_started = time.perf_counter()
    cache_hit = cache.predict(raw_matrix)
    cache_hit_seconds = time.perf_counter() - cache_hit_started
    if (
        not np.array_equal(cache_miss[:, 0], raw)
        or not np.array_equal(cache_hit, cache_miss)
        or cache.statistics()["hits"] != 1
    ):
        raise RuntimeError("Timing-probe exact cache changed frozen probabilities")

    final_source_hashes = {str(path): sha256(path) for path in source_paths}
    if final_source_hashes != source_hashes:
        raise RuntimeError("Timing-probe source/config changed during execution")
    final_input_hashes = {str(path): sha256(path) for path in input_paths}
    if final_input_hashes != input_hashes:
        raise RuntimeError("Timing-probe data, partition, or checkpoint changed during execution")
    direct_seconds = [raw_seconds, shared_seconds, projected_seconds]
    mean_region_seconds = float(np.mean(direct_seconds))
    full_region_calls = (
        len(cfg["seeds"])
        * 3
        * (1 + len(cfg.get("generation_seeds", cfg["seeds"]))
             * int(cfg["expected_attack_conditions"]))
        * len(regions)
    )
    metadata = {
        "format": "poi-stage10-one-fit-timing-probe-v1",
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW_TIMING_PROBE_ONLY",
        "scientific_completion": False,
        "execution_started_utc": started_utc,
        "execution_completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one clean-train fit and validation-evaluation timing only",
        "model_seed": model_seed, "region": region,
        "evaluation_rows": len(evaluation), "test_rows_read": 0,
        "partition": partition,
        "feature_order": list(FEATURES),
        "region_order_for_full_stage": list(regions),
        "fit_seconds_including_factory_probe": fit_seconds,
        "warmup_seconds_excluded_from_projection": warmup_seconds,
        "raw_prediction_seconds": raw_seconds,
        "shared_prediction_seconds": shared_seconds,
        "projected_prediction_seconds": projected_seconds,
        "mean_direct_region_prediction_seconds": mean_region_seconds,
        "cache_miss_prediction_seconds": cache.statistics()["underlying_prediction_seconds"],
        "cache_hit_lookup_seconds": cache_hit_seconds,
        "full_stage_requested_region_predictor_calls": full_region_calls,
        "uncached_full_stage_fixed_prediction_projection_seconds": (
            mean_region_seconds * full_region_calls
        ),
        "uncached_full_stage_fixed_prediction_projection_hours": (
            mean_region_seconds * full_region_calls / 3600
        ),
        "model_identity": model_identity,
        "fit_attestation": fit_attestation,
        "cache_statistics": cache.statistics(),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "limitations": [
            "The projection uses one model seed, one region, and three clean-evaluation matrices.",
            "It excludes detector inference, remaining fits, metrics, plotting, hashing, and I/O.",
            "Actual attack matrices and later GPU load can have different prediction times.",
            "Exact cache hit rates are unknown until real matrices are observed; zero hits remain possible.",
            "Probe fitted state is evidence only and is never reused by the Stage-10 scientific run.",
            "An interrupted Stage-10 run has no disk-cache resume and must restart in a new output path.",
        ],
    }
    _atomic_json(metadata, staging / "metadata.json")
    _atomic_json({
        "status": metadata["status"],
        "metadata_sha256": sha256(staging / "metadata.json"),
        "fitted_state_sha256": sha256(state_path),
        "test_rows_read": 0,
    }, staging / "_PROBE.json")
    return metadata


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    output = Path(output_override or cfg["output"])
    if (output / "_GENERATED.json").exists():
        raise FileExistsError(f"Refusing to overwrite generated stage-10 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    predictions_dir = tables / "predictions"
    work = output / "work"
    figures = output / "figures"
    for directory in (tables, predictions_dir, work, figures):
        directory.mkdir(parents=True, exist_ok=True)
    replay_source_dir = work / "fixed_downstream_replay_sources"
    replay_source_dir.mkdir()
    replay_source_dependencies = [
        Path(__file__), Path(__file__).with_name("selected_explain.py"),
        Path(__file__).with_name("defense_evaluate.py"), Path(__file__).with_name("data.py"),
        Path(__file__).with_name("attack_evaluate.py"),
        Path(__file__).with_name("ablation.py"), Path(__file__).with_name("benchmark.py"),
        Path(__file__).with_name("exact_prediction_cache.py"),
    ]
    replay_source_snapshots: dict[Path, Path] = {}
    for source in replay_source_dependencies:
        snapshot = replay_source_dir / source.name
        snapshot.write_bytes(source.read_bytes())
        replay_source_snapshots[source] = snapshot

    seeds = tuple(int(value) for value in cfg["seeds"])
    generation_seeds = tuple(int(value) for value in cfg.get("generation_seeds", cfg["seeds"]))
    if seeds != (42, 202, 340) or generation_seeds != (42, 202, 340):
        raise ValueError("Stage 10 requires model and generation seeds [42, 202, 340]")
    retentions = tuple(float(value) for value in cfg.get("clean_retentions", DEFAULT_RETENTIONS))
    if retentions != DEFAULT_RETENTIONS:
        raise ValueError("Stage 10 requires clean retentions 90%, 95%, 97.5%, and 99%")
    primary_retention = float(cfg.get("primary_clean_retention", 0.95))
    if primary_retention not in retentions:
        raise ValueError("Primary clean retention must be one of clean_retentions")

    clean, manifest = load_clean_splits(
        cfg["data_dir"], splits=("train", "validation"), verify_hashes=True
    )
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    regions = tuple(sorted(train["region"].astype(str).unique()))
    if len(regions) != 17 or set(validation["region"].astype(str)) != set(regions):
        raise ValueError("Stage 10 requires the frozen 17-region train/validation split")
    calibration_ids, evaluation_ids, partition_metadata = load_calibration_partition(
        validation,
        cfg["calibration"]["proposed_ids"],
        cfg["calibration"]["ure_ids"],
        fraction=float(cfg["calibration"].get("fraction", 0.5)),
        seed=int(cfg["calibration"].get("seed", 340)),
        allow_fallback=bool(cfg["calibration"].get("allow_fallback", True)),
    )
    calibration = validation.set_index("POI_ID", drop=False).loc[calibration_ids].reset_index(drop=True)
    evaluation = validation.set_index("POI_ID", drop=False).loc[evaluation_ids].reset_index(drop=True)

    all_detector_rows: list[pd.DataFrame] = []
    all_classification_rows: list[pd.DataFrame] = []
    all_threshold_rows: list[pd.DataFrame] = []
    all_calibration_rows: list[pd.DataFrame] = []
    projection_rows: list[dict[str, Any]] = []
    prediction_manifest: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    risk_curve_frames: list[pd.DataFrame] = []
    risk_summary_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    region_model_fit_count = 0
    fixed_replay_attestations: list[dict[str, Any]] = []
    prediction_cache_rows: list[dict[str, Any]] = []
    attack_path = Path(cfg["attacks_file"])
    classification_threshold = float(cfg.get("classification_threshold", 0.5))
    fixed_config = dict(cfg["fixed_region_model"])
    fixed_config["classification_threshold"] = classification_threshold
    fixed_config["fit_state_output_dir"] = str(work / "fixed_downstream_fit_states")
    fixed_checkpoint = Path(str(fixed_config.get("official_checkpoint", ""))).expanduser()
    expected_fixed_sha = str(fixed_config.get("expected_checkpoint_sha256", "")).lower()
    if not fixed_checkpoint.is_file() or sha256(fixed_checkpoint) != expected_fixed_sha:
        raise RuntimeError("Resolved fixed downstream checkpoint is absent or hash-mismatched")
    cache_config = dict(cfg.get("exact_prediction_cache", {}))
    if cache_config.get("enabled") is not True:
        raise ValueError("Stage 10 requires the reviewed exact in-memory prediction cache")
    cache_max_entries = int(cache_config.get("max_entries_per_model_seed", 128))
    cache_max_bytes = int(cache_config.get("max_bytes_per_model_seed", 64 * 1024 * 1024))
    if cache_max_entries <= 0 or cache_max_bytes <= 0:
        raise ValueError("Stage-10 exact prediction cache bounds must be positive")
    cache_source_sha256 = {
        str(source.resolve()): sha256(snapshot)
        for source, snapshot in replay_source_snapshots.items()
    }
    expected_cache_requests_per_seed = 3 * (
        1 + len(generation_seeds) * int(cfg["expected_attack_conditions"])
    )
    fixed_synchronize = _device_synchronizer(str(fixed_config["device"]))

    for seed in seeds:
        proposed, ure, proposed_path, ure_path = _load_models(cfg, seed)
        checkpoint_hashes[str(proposed_path)] = sha256(proposed_path)
        checkpoint_hashes[str(ure_path)] = sha256(ure_path)
        checkpoint_hashes[str(ure_path.with_suffix(".joblib"))] = sha256(
            ure_path.with_suffix(".joblib")
        )

        calibration_matrix = _matrix(calibration)
        proposed_calibration, _ = _proposed_output(proposed, calibration_matrix, regions)
        ure_calibration = _validate_probability(
            ure.predict_proba(calibration_matrix)[:, 1], len(calibration), "URE calibration"
        )
        calibration_scores = {
            PROPOSED_LABEL: proposed_calibration,
            URE_LABEL: ure_calibration,
        }
        seed_calibration_rows = [
            _calibration_detector_rows(
                calibration,
                calibration_scores,
                seed=seed,
                generation_seed=CANONICAL_CLEAN_GENERATION_SEED,
                condition="clean",
                valid=np.ones(len(calibration), dtype=bool),
                quarantine_reason=np.full(len(calibration), "", dtype=object),
            )
        ]
        thresholds = _thresholds(calibration_scores, retentions)

        fixed = RegionModelFactory(
            train, regions, seed=seed, config=fixed_config
        ).fit()
        if fixed.fit_count != len(regions):
            raise RuntimeError("Frozen region classifier fit count differs from 17")
        region_model_fit_count += fixed.fit_count
        fixed_replay_attestations.extend(fixed.replay_attestations)
        fixed_identity = build_frozen_model_identity(
            model_id="tabpfn_v2_5",
            model_seed=seed,
            feature_order=FEATURES,
            region_order=regions,
            fit_attestations=fixed.replay_attestations,
            source_sha256=cache_source_sha256,
        )
        fixed_cache = ExactFloat32PredictionCache(
            fixed.predict,
            fixed_identity,
            feature_order=FEATURES,
            output_order=regions,
            max_entries=cache_max_entries,
            max_bytes=cache_max_bytes,
            synchronize=fixed_synchronize,
        )
        repair = SharedNearestTrainRepair().fit(train[FEATURES])
        clean_obs = _observations(
            evaluation, proposed, ure, fixed, repair, regions, fixed_cache=fixed_cache
        )
        cache_progress_path = work / f"modelseed{seed}_prediction_cache_progress.json"
        _atomic_json(
            _cache_runtime_statistics(
                fixed_cache, model_seed=seed, region_count=len(regions),
                expected_requests_per_seed=expected_cache_requests_per_seed,
                model_seed_count=len(seeds), status="running_after_clean_observation",
            ),
            cache_progress_path,
        )
        projection_rows.append(dict(
            seed=seed, generation_seed=CANONICAL_CLEAN_GENERATION_SEED,
            attack_condition="clean", **clean_obs["projection_diagnostics"]
        ))
        truth = clean_obs["truth"]
        if not np.array_equal(
            truth, evaluation["region"].astype(str).to_numpy()[:, None] == np.asarray(regions)[None, :]
        ):
            raise RuntimeError("Evaluation region labels are not aligned")
        for detector in DETECTOR_LABELS:
            curve, risk_summary = binary_ovr_risk_coverage(
                clean_obs["truth"], clean_obs["fixed"], clean_obs["detector"][detector],
                evaluation["POI_ID"].astype(str).tolist(), thresholds[detector],
                classification_threshold=classification_threshold,
                seed=seed, generation_seed=CANONICAL_CLEAN_GENERATION_SEED,
                attack_condition="clean", evaluation_cohort="clean",
                population_scope="clean", detector=detector,
            )
            risk_curve_frames.append(curve); risk_summary_rows.append(risk_summary)

        threshold_rows = []
        for detector in DETECTOR_LABELS:
            for retention, threshold in thresholds[detector].items():
                threshold_rows.append(
                    {
                        "seed": seed,
                        "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
                        "detector": detector,
                        "clean_retention_target": retention,
                        "threshold": threshold,
                        "calibration_rows": len(calibration),
                        "calibration_clean_retention": float(
                            (calibration_scores[detector] <= threshold).mean()
                        ),
                        "evaluation_clean_retention": float(
                            (clean_obs["detector"][detector] <= threshold).mean()
                        ),
                        "selection_partition": "validation_calibration_half",
                        "reported_partition": "validation_evaluation_half",
                        "threshold_selection_limitation": partition_metadata[
                            "threshold_selection_limitation"
                        ],
                    }
                )
        all_threshold_rows.append(pd.DataFrame(threshold_rows))

        clean_predictions = _prediction_rows(
            evaluation,
            seed=seed,
            generation_seed=CANONICAL_CLEAN_GENERATION_SEED,
            condition="clean",
            regions=regions,
            valid=np.ones(len(evaluation), dtype=bool),
            quarantine_reason=np.full(len(evaluation), "", dtype=object),
            detector_scores=clean_obs["detector"],
            fixed_scores=clean_obs["fixed"],
            shared_repair_scores=clean_obs["shared"],
            proposed_native_scores=clean_obs["native"],
            proposed_projected_scores=clean_obs["projected"],
            thresholds=thresholds,
        )
        clean_path = predictions_dir / f"modelseed{seed}__clean.parquet"
        _atomic_frame(clean_predictions, clean_path)
        prediction_manifest.append(
            {"seed": seed, "generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
             "condition": "clean", "rows": len(clean_predictions),
             "path": str(clean_path), "sha256": sha256(clean_path),
             **_prediction_manifest_identity(clean_predictions)}
        )
        classification_rows = _clean_metric_rows(
            clean_obs,
            seed=seed,
            generation_seed=CANONICAL_CLEAN_GENERATION_SEED,
            regions=regions,
            thresholds=thresholds,
            classification_threshold=classification_threshold,
        )
        detector_rows: list[dict[str, Any]] = []
        observed_conditions: tuple[str, ...] | None = None

        for generation_seed in generation_seeds:
            attacks = load_generated_attacks(
                attack_path, split="validation", generation_seed=generation_seed,
                valid_only=False,
            )
            attacks, conditions = validate_attack_rows(
                attacks,
                validation,
                expected_conditions=int(cfg["expected_attack_conditions"]),
                clean_train=train,
            )
            if observed_conditions is None:
                observed_conditions = conditions
            elif conditions != observed_conditions:
                raise ValueError("Attack-condition labels differ across generation seeds")

            for condition in conditions:
                attack_rows = _aligned_condition(attacks, condition, evaluation_ids)
                attack_frame = _attack_feature_frame(attack_rows)
                attack_obs = _observations(
                    attack_frame, proposed, ure, fixed, repair, regions,
                    fixed_cache=fixed_cache,
                )
                _atomic_json(
                    _cache_runtime_statistics(
                        fixed_cache, model_seed=seed, region_count=len(regions),
                        expected_requests_per_seed=expected_cache_requests_per_seed,
                        model_seed_count=len(seeds), status="running_attack_observations",
                    ),
                    cache_progress_path,
                )
                projection_rows.append(dict(
                    seed=seed, generation_seed=generation_seed,
                    attack_condition=condition, **attack_obs["projection_diagnostics"]
                ))
                valid = attack_rows["primary_evaluation_eligible"].astype(bool).to_numpy()
                quarantine = attack_rows["quarantine_reason"].fillna("").astype(str).to_numpy()
                predictions = _prediction_rows(
                    attack_frame,
                    seed=seed,
                    generation_seed=generation_seed,
                    condition=condition,
                    regions=regions,
                    valid=valid,
                    quarantine_reason=quarantine,
                    detector_scores=attack_obs["detector"],
                    fixed_scores=attack_obs["fixed"],
                    shared_repair_scores=attack_obs["shared"],
                    proposed_native_scores=attack_obs["native"],
                    proposed_projected_scores=attack_obs["projected"],
                    thresholds=thresholds,
                    provenance=_provenance(attack_rows),
                )
                prediction_path = predictions_dir / (
                    f"modelseed{seed}__genseed{generation_seed}__{condition}.parquet"
                )
                _atomic_frame(predictions, prediction_path)
                prediction_manifest.append(
                    {"seed": seed, "generation_seed": generation_seed,
                     "condition": condition, "rows": len(predictions),
                     "path": str(prediction_path), "sha256": sha256(prediction_path),
                     **_prediction_manifest_identity(predictions)}
                )
                detector_rows.extend(
                    _detector_metric_rows(
                        clean_obs["detector"], attack_obs["detector"], valid,
                        changed=attack_rows["attack_changed_any"].astype(bool).to_numpy(),
                        thresholds=thresholds,
                        seed=seed, generation_seed=generation_seed, condition=condition,
                    )
                )
                operating_rows.extend(selective_operating_summary(
                    clean_obs["detector"], attack_obs["detector"], valid,
                    thresholds, seed=seed, generation_seed=generation_seed,
                    attack_condition=condition,
                ).to_dict(orient="records"))
                for detector in DETECTOR_LABELS:
                    for population_scope, attack_mask in (
                        ("all_intent", np.ones(len(valid), dtype=bool)),
                        ("primary_valid", valid.astype(bool)),
                    ):
                        for cohort in ("attack", "clean_plus_attack"):
                            if cohort == "attack":
                                cohort_truth = attack_obs["truth"][attack_mask]
                                cohort_fixed = attack_obs["fixed"][attack_mask]
                                cohort_detection = attack_obs["detector"][detector][attack_mask]
                                cohort_ids = attack_frame.loc[attack_mask, "POI_ID"].astype(str).tolist()
                            else:
                                cohort_truth = np.concatenate(
                                    (clean_obs["truth"], attack_obs["truth"][attack_mask]), axis=0
                                )
                                cohort_fixed = np.concatenate(
                                    (clean_obs["fixed"], attack_obs["fixed"][attack_mask]), axis=0
                                )
                                cohort_detection = np.concatenate((
                                    clean_obs["detector"][detector],
                                    attack_obs["detector"][detector][attack_mask],
                                ))
                                cohort_ids = (
                                    [f"clean::{value}" for value in evaluation["POI_ID"].astype(str)]
                                    + [f"attack::{value}" for value in
                                       attack_frame.loc[attack_mask, "POI_ID"].astype(str)]
                                )
                            curve, risk_summary = binary_ovr_risk_coverage(
                                cohort_truth, cohort_fixed, cohort_detection, cohort_ids,
                                thresholds[detector],
                                classification_threshold=classification_threshold,
                                seed=seed, generation_seed=generation_seed,
                                attack_condition=condition, evaluation_cohort=cohort,
                                population_scope=population_scope, detector=detector,
                            )
                            risk_curve_frames.append(curve)
                            risk_summary_rows.append(risk_summary)
                classification_rows.extend(
                    _cohort_metric_rows(
                        clean_obs,
                        attack_obs,
                        valid,
                        seed=seed,
                        generation_seed=generation_seed,
                        condition=condition,
                        regions=regions,
                        thresholds=thresholds,
                        classification_threshold=classification_threshold,
                    )
                )
                calibration_attack_rows = _aligned_condition(
                    attacks, condition, calibration_ids
                )
                calibration_attack_frame = _attack_feature_frame(calibration_attack_rows)
                calibration_attack_matrix = _matrix(calibration_attack_frame)
                calibration_proposed, _ = _proposed_output(
                    proposed, calibration_attack_matrix, regions
                )
                calibration_ure = _validate_probability(
                    ure.predict_proba(calibration_attack_matrix)[:, 1],
                    len(calibration_attack_frame),
                    "URE calibration attack",
                )
                seed_calibration_rows.append(
                    _calibration_detector_rows(
                        calibration_attack_frame,
                        {PROPOSED_LABEL: calibration_proposed, URE_LABEL: calibration_ure},
                        seed=seed,
                        generation_seed=generation_seed,
                        condition=condition,
                        valid=calibration_attack_rows[
                            "primary_evaluation_eligible"
                        ].astype(bool).to_numpy(),
                        quarantine_reason=calibration_attack_rows[
                            "quarantine_reason"
                        ].fillna("").astype(str).to_numpy(),
                        provenance=_provenance(calibration_attack_rows),
                    )
                )

        seed_detector = pd.DataFrame(detector_rows)
        seed_classification = pd.DataFrame(classification_rows)
        _atomic_frame(seed_detector, work / f"modelseed{seed}_detector_metrics.csv")
        _atomic_frame(seed_classification, work / f"modelseed{seed}_classification_metrics.csv")
        all_detector_rows.append(seed_detector)
        all_classification_rows.append(seed_classification)
        all_calibration_rows.append(pd.concat(seed_calibration_rows, ignore_index=True))
        cache_statistics = _cache_runtime_statistics(
            fixed_cache, model_seed=seed, region_count=len(regions),
            expected_requests_per_seed=expected_cache_requests_per_seed,
            model_seed_count=len(seeds), status="model_seed_complete",
        )
        if cache_statistics["requests"] != expected_cache_requests_per_seed:
            raise RuntimeError(
                "Stage-10 fixed prediction cache request count differs from the full design"
            )
        prediction_cache_rows.append(cache_statistics)
        _atomic_json(cache_statistics, cache_progress_path)
        _atomic_json(
            {
                "status": "evaluated_pending_independent_review",
                "model_seed": seed,
                "generation_seeds": list(generation_seeds),
                "seed_design": "full_model_by_generation_seed_cross",
                "region_models_fitted": fixed.fit_count,
                "conditions": list(observed_conditions or ()),
                "evaluation_rows": len(evaluation),
                "prediction_shards": 1 + len(generation_seeds) * len(observed_conditions or ()),
                "exact_prediction_cache": cache_statistics,
            },
            work / f"modelseed{seed}_metadata.json",
        )

    detector_metrics = pd.concat(all_detector_rows, ignore_index=True)
    classification_metrics = pd.concat(all_classification_rows, ignore_index=True)
    thresholds_frame = pd.concat(all_threshold_rows, ignore_index=True)
    macro_metrics = macro_region_metrics(classification_metrics, expected_regions=len(regions))
    operating_summary = pd.DataFrame(operating_rows)
    risk_coverage_curve = pd.concat(risk_curve_frames, ignore_index=True)
    risk_coverage_summary = pd.DataFrame(risk_summary_rows)
    _atomic_frame(pd.DataFrame(projection_rows), tables / "native_repair_projection_diagnostics.csv")
    calibration_detector_scores = pd.concat(all_calibration_rows, ignore_index=True)
    prediction_manifest_frame = pd.DataFrame(prediction_manifest)
    expected_counts = cross_design_expected_counts(
        model_seeds=len(seeds),
        generation_seeds=len(generation_seeds),
        attack_conditions=int(cfg["expected_attack_conditions"]),
        regions=len(regions),
        detectors=len(DETECTOR_LABELS),
        retentions=len(retentions),
    )
    observed_counts = {
        "model_generation_seed_pairs": int(
            detector_metrics[["seed", "generation_seed"]].drop_duplicates().shape[0]
        ),
        "clean_prediction_shards": int(
            prediction_manifest_frame["condition"].eq("clean").sum()
        ),
        "attack_prediction_shards": int(
            prediction_manifest_frame["condition"].ne("clean").sum()
        ),
        "prediction_shards": len(prediction_manifest_frame),
        "threshold_rows": len(thresholds_frame),
        "detector_metric_rows": len(detector_metrics),
        "classification_metric_rows": len(classification_metrics),
        "region_model_fits": region_model_fit_count,
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Stage-10 cross-design cardinality mismatch: expected={expected_counts}, "
            f"observed={observed_counts}"
        )
    section10_summary_counts = {
        "classification_macro_rows": len(macro_metrics),
        "selective_operating_rows": len(operating_summary),
        "risk_coverage_summary_rows": len(risk_coverage_summary),
        "risk_coverage_curve_rows": len(risk_coverage_curve),
    }
    expected_macro_rows = expected_counts["classification_metric_rows"] // len(regions)
    expected_risk_summaries = (
        len(seeds) * len(DETECTOR_LABELS)
        + len(seeds) * len(generation_seeds) * int(cfg["expected_attack_conditions"])
        * len(DETECTOR_LABELS) * 2 * 2
    )
    if (
        len(macro_metrics) != expected_macro_rows
        or len(operating_summary) != expected_counts["detector_metric_rows"]
        or len(risk_coverage_summary) != expected_risk_summaries
        or risk_coverage_curve.empty
    ):
        raise RuntimeError("Stage-10 macro/operating/risk-coverage cardinality mismatch")
    _atomic_frame(detector_metrics, tables / "detector_metrics.csv")
    _atomic_frame(classification_metrics, tables / "classification_metrics.csv")
    _atomic_frame(thresholds_frame, tables / "detection_thresholds.csv")
    _atomic_frame(
        calibration_detector_scores, tables / "calibration_detector_scores.parquet"
    )
    _atomic_frame(prediction_manifest_frame, tables / "prediction_manifest.csv")
    _atomic_frame(macro_metrics, tables / "classification_macro_metrics.csv")
    _atomic_frame(operating_summary, tables / "selective_operating_summary.csv")
    _atomic_frame(risk_coverage_curve, tables / "risk_coverage_curve.csv")
    _atomic_frame(risk_coverage_summary, tables / "risk_coverage_summary.csv")
    prediction_cache_summary = {
        "format": "poi-stage10-exact-prediction-cache-summary-v1",
        "scope": "process_memory_only_no_disk_resume",
        "per_model_seed": prediction_cache_rows,
        "requests": sum(row["requests"] for row in prediction_cache_rows),
        "hits": sum(row["hits"] for row in prediction_cache_rows),
        "misses": sum(row["misses"] for row in prediction_cache_rows),
        "requested_region_predictor_calls": sum(
            row["requested_region_predictor_calls"] for row in prediction_cache_rows
        ),
        "executed_region_predictor_calls": sum(
            row["executed_region_predictor_calls"] for row in prediction_cache_rows
        ),
        "avoided_region_predictor_calls": sum(
            row["avoided_region_predictor_calls"] for row in prediction_cache_rows
        ),
        "underlying_prediction_seconds": sum(
            row["underlying_prediction_seconds"] for row in prediction_cache_rows
        ),
        "avoided_prediction_seconds_estimate": sum(
            row["avoided_prediction_seconds_estimate"] for row in prediction_cache_rows
        ),
        "restart_policy": (
            "No disk cache or cache resume. An interrupted run preserves partial evidence and "
            "must restart from the beginning in a new output directory."
        ),
    }
    if (
        prediction_cache_summary["requests"]
        != expected_cache_requests_per_seed * len(seeds)
        or prediction_cache_summary["requested_region_predictor_calls"]
        != expected_cache_requests_per_seed * len(seeds) * len(regions)
    ):
        raise RuntimeError("Stage-10 exact prediction cache totals differ from the full design")
    prediction_cache_summary_path = tables / "exact_prediction_cache_statistics.json"
    _atomic_json(prediction_cache_summary, prediction_cache_summary_path)

    for source, snapshot in replay_source_snapshots.items():
        if sha256(source) != sha256(snapshot):
            raise RuntimeError(f"Replay source changed during Stage 10 execution: {source}")
    fit_runtime_rows = [{
        key: fit.get(key) for key in (
            "tabpfn_version", "torch_version", "cuda_version", "cuda_available",
            "gpu_device_name", "requested_device", "restored_on_device",
        )
    } for fit in fixed_replay_attestations]
    if not fit_runtime_rows or any(row != fit_runtime_rows[0] for row in fit_runtime_rows[1:]):
        raise RuntimeError("Fixed downstream fit runtime differs across seed/region states")
    requested_device = str(fit_runtime_rows[0]["requested_device"])
    if (
        requested_device != str(fit_runtime_rows[0]["restored_on_device"])
        or (requested_device.startswith("cuda") and not fit_runtime_rows[0]["cuda_available"])
    ):
        raise RuntimeError("Fixed downstream fitted-state restore did not use the resolved device")
    installed_distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_distributions[str(name).lower()] = distribution.version
    runtime_environment = {
        "format": "poi-stage10-runtime-environment-v1",
        "execution_scope": "stage10_gpu_model_fitting_and_validation_evaluation",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        **fit_runtime_rows[0],
        "installed_distributions": dict(sorted(installed_distributions.items())),
        "test_rows_read": 0,
    }
    runtime_environment_path = output / "runtime_environment.json"
    _atomic_json(runtime_environment, runtime_environment_path)
    replay_contract = {
        "format": "poi-fixed-downstream-replay-contract-v1",
        "status": "fitted_validation_pipeline_pending_independent_review",
        "model_id": "tabpfn_v2_5",
        "official_checkpoint_path": str(fixed_checkpoint.resolve()),
        "official_checkpoint_sha256": expected_fixed_sha,
        "state_binding": (
            "official TabPFN save_fit_state archives for every seed/region fit, reloaded and "
            "verified against fixed probe prediction hashes"
        ),
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL),
        "model_seeds": list(seeds),
        "regions": list(regions),
        "n_estimators": int(fixed_config["n_estimators"]),
        "classification_threshold": classification_threshold,
        "fit_count": region_model_fit_count,
        "expected_fit_count": len(seeds) * len(regions),
        "train_rows": len(train),
        "raw_data_sha256": sha256(Path(cfg["data_dir"]) / "poi_data_region.csv"),
        "sample_manifest_sha256": sha256(Path(cfg["data_dir"]) / "sample_manifest.csv"),
        "train_poi_order_sha256": _ordered_identity_sha256(train, ["POI_ID"]),
        "train_feature_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(_matrix(train), dtype=np.float32).tobytes()
        ).hexdigest(),
        "calibration_poi_order_sha256": _ordered_identity_sha256(
            calibration, ["POI_ID"]
        ),
        "evaluation_poi_order_sha256": _ordered_identity_sha256(
            evaluation, ["POI_ID"]
        ),
        "thresholds_sha256": sha256(tables / "detection_thresholds.csv"),
        "calibration_scores_sha256": sha256(
            tables / "calibration_detector_scores.parquet"
        ),
        "prediction_manifest_sha256": sha256(tables / "prediction_manifest.csv"),
        "exact_prediction_cache_statistics_sha256": sha256(
            prediction_cache_summary_path
        ),
        "stage_config_sha256": sha256(config_path),
        "runtime_environment_sha256": sha256(runtime_environment_path),
        "calibration_partition_files_sha256": {
            str(Path(cfg["calibration"][name])): sha256(Path(cfg["calibration"][name]))
            for name in ("proposed_ids", "ure_ids")
        },
        "source_snapshots_sha256": {
            str(snapshot): sha256(snapshot)
            for snapshot in replay_source_snapshots.values()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python_executable": sys.executable,
            **fit_runtime_rows[0],
        },
        "fits": fixed_replay_attestations,
        "test_rows_read": 0,
    }
    if len(fixed_replay_attestations) != replay_contract["expected_fit_count"]:
        raise RuntimeError("Fixed downstream replay attestations are incomplete")
    replay_contract_path = output / "fixed_downstream_replay_contract.json"
    _atomic_json(replay_contract, replay_contract_path)

    prediction_paths = [Path(value) for value in prediction_manifest_frame["path"]]
    _plot_curves(prediction_paths, figures, primary_retention)
    _plot_risk_coverage(risk_coverage_curve, figures)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    input_hashes = {
        str(config_path): sha256(config_path),
        str(attack_path): sha256(attack_path),
        str(fixed_checkpoint.resolve()): expected_fixed_sha,
        str(Path(cfg["data_dir"]) / "poi_data_region.csv"): sha256(
            Path(cfg["data_dir"]) / "poi_data_region.csv"
        ),
        str(Path(cfg["data_dir"]) / "sample_manifest.csv"): sha256(
            Path(cfg["data_dir"]) / "sample_manifest.csv"
        ),
        **checkpoint_hashes,
    }
    metadata = {
        "status": "generated_pending_independent_review",
        "scope": "train_fit_validation_evaluation_only",
        "test_rows_read": 0,
        "detectors": list(DETECTOR_LABELS),
        "detector_labels": {"normal": 0, "attack": 1},
        "attack_label_semantics": (
            "attack=1 denotes every generated attack intent row, including a row whose "
            "optimizer proposed no feature change; attack_changed_any distinguishes those rows."
        ),
        "accept_rule": "attack_probability <= calibrated_threshold",
        "model_seeds": list(seeds),
        "generation_seeds": list(generation_seeds),
        "seed_design": "full_model_by_generation_seed_cross",
        "model_generation_seed_pairs": len(seeds) * len(generation_seeds),
        "model_generation_seed_effects_confounded": False,
        "canonical_clean_generation_seed": CANONICAL_CLEAN_GENERATION_SEED,
        "canonical_clean_storage": (
            "one clean prediction and calibration shard per model seed; Stage 14 broadcasts "
            "the immutable clean reference across generation seeds in memory"
        ),
        "threshold_rows_are_canonical_per_model_seed": True,
        "cross_design_expected_counts": expected_counts,
        "cross_design_observed_counts": observed_counts,
        "section10_summary_counts": section10_summary_counts,
        "macro_metric_semantics": (
            "unweighted mean across the 17 target-region binary OvR tasks; finite and undefined "
            "region counts are reported separately for every metric"
        ),
        "selective_operating_semantics": (
            "clean retention and attack rejection use POI denominators; primary-valid and "
            "all-intent attack populations remain separate"
        ),
        "risk_coverage_semantics": (
            "coverage is accepted POIs over the declared cohort; per-POI risk is mean 0-1 "
            "classification error across 17 OvR targets using the fixed TabPFN2.5 score; AURC "
            "uses exact detector-score tie groups"
        ),
        "fixed_downstream_replay_contract": {
            "path": str(replay_contract_path),
            "sha256": sha256(replay_contract_path),
            "actual_checkpoint_path": str(fixed_checkpoint.resolve()),
            "actual_checkpoint_sha256": expected_fixed_sha,
        },
        "regions": list(regions),
        "attack_conditions": int(cfg["expected_attack_conditions"]),
        "clean_train_rows": len(train),
        "validation_calibration_rows": len(calibration),
        "validation_evaluation_rows": len(evaluation),
        "calibration": partition_metadata,
        "retentions": list(retentions),
        "primary_retention": primary_retention,
        "fixed_region_model": (
            "tabpfn_v2_5 fit once per model seed/region on clean train and shared across all "
            "three attack-generation seeds"
        ),
        "exact_prediction_cache": {
            "path": str(prediction_cache_summary_path),
            "sha256": sha256(prediction_cache_summary_path),
            "mode": "bounded_process_memory_exact_float32_full_matrix",
            "max_entries_per_model_seed": cache_max_entries,
            "max_bytes_per_model_seed": cache_max_bytes,
            "requests": prediction_cache_summary["requests"],
            "hits": prediction_cache_summary["hits"],
            "misses": prediction_cache_summary["misses"],
            "requested_region_predictor_calls": prediction_cache_summary[
                "requested_region_predictor_calls"
            ],
            "executed_region_predictor_calls": prediction_cache_summary[
                "executed_region_predictor_calls"
            ],
            "avoided_region_predictor_calls": prediction_cache_summary[
                "avoided_region_predictor_calls"
            ],
            "restart_policy": prediction_cache_summary["restart_policy"],
        },
        "shared_repair": "nearest clean train feature vector; no region label or paired clean row",
        "native_repair": (
            "proposed_native_full_coverage uses the proposed model's own region heads. "
            "proposed_projected_repair_fixed_tabpfn25 keeps accepted fixed predictions and "
            "passes only rejected rows through the proposed train-support-projected repair "
            "and the SAME fixed TabPFN2.5 classifiers. URE-RF has no native reconstruction. "
            "The shared nearest-train repair comparator is reported for both detectors."
        ),
        "quarantine_policy": (
            "All rows remain in prediction shards and all-intent metrics; primary-valid metrics "
            "exclude rows only by primary_evaluation_eligible."
        ),
        "metric_interpretation": (
            "all_intent is diagnostic because it retains cross-boundary and otherwise invalid "
            "rows. Only primary_valid is admissible robustness evidence."
        ),
        "input_sha256": input_hashes,
        "python": platform.python_version(),
        "seconds": time.time() - started,
    }
    _atomic_json(metadata, output / "metadata.json")
    _atomic_json(
        {
            "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
            "metadata_sha256": sha256(output / "metadata.json"),
            "detector_metrics_sha256": sha256(tables / "detector_metrics.csv"),
            "classification_macro_metrics_sha256": sha256(
                tables / "classification_macro_metrics.csv"
            ),
            "selective_operating_summary_sha256": sha256(
                tables / "selective_operating_summary.csv"
            ),
            "risk_coverage_curve_sha256": sha256(tables / "risk_coverage_curve.csv"),
            "risk_coverage_summary_sha256": sha256(tables / "risk_coverage_summary.csv"),
            "classification_metrics_sha256": sha256(tables / "classification_metrics.csv"),
            "prediction_manifest_sha256": sha256(tables / "prediction_manifest.csv"),
            "exact_prediction_cache_statistics_sha256": sha256(
                prediction_cache_summary_path
            ),
            "calibration_detector_scores_sha256": sha256(
                tables / "calibration_detector_scores.parquet"
            ),
        },
        output / "_GENERATED.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--fixed-prediction-timing-probe-output",
        help="Run one validation-only frozen-fit timing probe instead of full Stage 10",
    )
    arguments = parser.parse_args()
    if arguments.fixed_prediction_timing_probe_output:
        result = run_fixed_prediction_timing_probe(
            arguments.config, arguments.fixed_prediction_timing_probe_output
        )
        print(json.dumps(result, indent=2))
    else:
        run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
