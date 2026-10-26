"""Stage-08 paired data reader for the regenerated attack table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from poi.data import CATEGORICAL, FEATURES, NUMERIC, load_clean_splits

ATTACK_PROVENANCE = {
    "POI_ID",
    "split",
    "region",
    "attack_method",
    "attack_condition",
    "source_model",
    "victim_model",
    "constraints_valid",
    "label_preserved",
    "generation_seed",
    "attack_attempted",
    "attack_changed_any",
    "attack_success_source",
    "primary_evaluation_eligible",
    "numeric_budget_m",
    "numeric_budget_was_implicit_zero",
    "categorical_budget_l0",
    "distance_m",
    "distance_metric",
    "distance_tolerance_m",
    "categorical_l0",
    "valid_numeric_train_range",
    "valid_category_codebook",
    "valid_clean_category_combo",
    "valid_attack_category_combo",
    "valid_category_combo",
    "within_budget",
    "quarantine_reason",
}


def load_stage08_data(
    data_dir: str | Path,
    attack_table: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read clean train/validation and paired attacks without opening locked test."""
    clean, _ = load_clean_splits(data_dir, splits=("train", "validation"))
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    path = Path(attack_table)
    if path.suffix == ".parquet":
        attacks = pd.read_parquet(path)
    elif path.suffix == ".csv":
        attacks = pd.read_csv(path)
    else:
        raise ValueError("Stage 08 attack table must be CSV or Parquet")
    required = set(ATTACK_PROVENANCE)
    required.update(f"{feature}_clean" for feature in FEATURES)
    required.update(f"{feature}_adv" for feature in FEATURES)
    required.update(f"changed_{feature.lower()}" for feature in FEATURES)
    missing = required.difference(attacks.columns)
    if missing:
        raise ValueError(f"Stage 08 attack table is missing columns: {sorted(missing)}")
    if not set(attacks["split"]).issubset({"train", "validation"}):
        raise RuntimeError("Locked test rows entered stage-08 attack data")
    if set(attacks["split"]) != {"train", "validation"}:
        raise ValueError("Stage 08 requires both train and validation attacks")
    if attacks.empty:
        raise ValueError("No attack-intent rows remain")
    if not set(attacks["POI_ID"]).issubset(set(clean["POI_ID"])):
        raise ValueError("Attack table has POI_ID values outside clean train/validation")
    lookup = clean.set_index("POI_ID")[["split", "region"]]
    aligned = lookup.loc[attacks["POI_ID"]]
    if not np.array_equal(aligned["split"].to_numpy(), attacks["split"].to_numpy()):
        raise ValueError("Attack split differs from frozen clean manifest")
    if not np.array_equal(aligned["region"].to_numpy(), attacks["region"].to_numpy()):
        raise ValueError("Attack region differs from frozen clean manifest")
    for feature in FEATURES:
        attacks[f"{feature}_clean"] = pd.to_numeric(
            attacks[f"{feature}_clean"], errors="raise"
        )
        attacks[f"{feature}_adv"] = pd.to_numeric(
            attacks[f"{feature}_adv"], errors="raise"
        )
    if not np.isfinite(
        attacks[[*(f"{feature}_clean" for feature in FEATURES), *(f"{feature}_adv" for feature in FEATURES)]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite stage-08 feature value")
    _validate_attack_integrity(clean, attacks)
    return clean.reset_index(drop=True), attacks.reset_index(drop=True)


def _validate_attack_integrity(clean: pd.DataFrame, attacks: pd.DataFrame) -> None:
    """Fail closed on paired-source, mask, budget, and train-support disagreement."""
    if clean["POI_ID"].duplicated().any():
        raise ValueError("Clean POI_ID must be unique for paired attack validation")
    lookup = clean.set_index("POI_ID")
    aligned = lookup.loc[attacks["POI_ID"]]
    paired = raw_feature_matrix(attacks, "_clean")
    manifest = raw_feature_matrix(aligned.reset_index())
    if not np.array_equal(paired, manifest):
        raise ValueError("Attack clean features differ from frozen clean source rows")

    attacked = raw_feature_matrix(attacks, "_adv")
    actual_changed = _recomputed_changed_mask(paired, attacked)
    declared = changed_mask(attacks).astype(bool)
    if not np.array_equal(actual_changed, declared):
        raise ValueError("Attack changed-feature masks differ from paired features")
    mean_latitude = np.deg2rad((paired[:, 1] + attacked[:, 1]) / 2)
    dx = (attacked[:, 0] - paired[:, 0]) * 111_320.0 * np.cos(mean_latitude)
    dy = (attacked[:, 1] - paired[:, 1]) * 110_574.0
    distances = np.sqrt(dx**2 + dy**2)
    category_l0 = actual_changed[:, 2:].sum(axis=1)
    changed_any = (distances > 1e-6) | (category_l0 > 0)
    if not np.array_equal(_strict_boolean(attacks, "attack_changed_any"), changed_any):
        raise ValueError("attack_changed_any differs from paired features")
    recorded_distance = pd.to_numeric(attacks["distance_m"], errors="raise").to_numpy(float)
    if not np.allclose(recorded_distance, distances, atol=0.05, rtol=0):
        raise ValueError("Recorded attack distance differs from paired coordinates")
    recorded_l0 = pd.to_numeric(attacks["categorical_l0"], errors="raise").to_numpy(float)
    if not np.array_equal(recorded_l0, category_l0.astype(float)):
        raise ValueError("Recorded categorical_l0 differs from paired categories")

    train = clean.loc[clean["split"].eq("train")]
    train_values = raw_feature_matrix(train)
    lower = train_values[:, :2].min(axis=0)
    upper = train_values[:, :2].max(axis=0)
    numeric_valid = ((attacked[:, :2] >= lower) & (attacked[:, :2] <= upper)).all(axis=1)
    category_valid = np.ones(len(attacks), dtype=bool)
    for position, feature in enumerate(CATEGORICAL, start=2):
        category_valid &= np.isin(attacked[:, position], train[feature].to_numpy(float))
    train_tuples = {tuple(row) for row in train_values[:, 2:]}
    clean_tuple_valid = np.asarray([tuple(row) in train_tuples for row in paired[:, 2:]])
    attack_tuple_valid = np.asarray([tuple(row) in train_tuples for row in attacked[:, 2:]])
    tuple_valid = clean_tuple_valid & attack_tuple_valid

    numeric_budget = pd.to_numeric(attacks["numeric_budget_m"], errors="raise").to_numpy(float)
    category_budget = pd.to_numeric(attacks["categorical_budget_l0"], errors="raise").to_numpy(float)
    if (not np.isfinite(numeric_budget).all() or (numeric_budget < 0).any()
            or not np.isfinite(category_budget).all() or (category_budget < 0).any()
            or not np.equal(category_budget, np.floor(category_budget)).all()):
        raise ValueError("Attack budgets must be finite, nonnegative, and categorical-integral")
    implicit = _strict_boolean(attacks, "numeric_budget_was_implicit_zero")
    if (numeric_budget[implicit] != 0).any():
        raise ValueError("Implicit numeric budgets must be stored as zero")
    expected_tolerance = np.where(implicit, 1e-6, 0.05)
    recorded_tolerance = pd.to_numeric(attacks["distance_tolerance_m"], errors="raise").to_numpy(float)
    if not np.array_equal(recorded_tolerance, expected_tolerance):
        raise ValueError("Attack distance tolerance differs from producer contract")
    if not attacks["distance_metric"].astype(str).eq(
        "local_equirectangular_wgs84_approximation"
    ).all():
        raise ValueError("Attack distance metric differs from producer contract")
    within_budget = (distances <= numeric_budget + expected_tolerance) & (category_l0 <= category_budget)
    recomputed = {
        "valid_numeric_train_range": numeric_valid,
        "valid_category_codebook": category_valid,
        "valid_clean_category_combo": clean_tuple_valid,
        "valid_attack_category_combo": attack_tuple_valid,
        "valid_category_combo": tuple_valid,
        "within_budget": within_budget,
    }
    for column, values in recomputed.items():
        if not np.array_equal(_strict_boolean(attacks, column), values):
            raise ValueError(f"Stored {column} differs from train-only recomputation")
    preserved = _strict_boolean(attacks, "label_preserved")
    expected_constraints = numeric_valid & category_valid & tuple_valid & within_budget & preserved
    if not np.array_equal(_strict_boolean(attacks, "constraints_valid"), expected_constraints):
        raise ValueError("constraints_valid differs from train-only recomputation")
    if not np.array_equal(_strict_boolean(attacks, "primary_evaluation_eligible"), expected_constraints):
        raise ValueError("primary_evaluation_eligible differs from train-only recomputation")


def attack_intent_diagnostics(
    attacks: pd.DataFrame, clean: pd.DataFrame
) -> dict[str, int | str]:
    """Count ambiguous intent labels without removing them from detector fitting.

    Every generated attack row is an attack-intent example, including failed or
    unchanged attempts.  Such a row may have exactly the same five inputs as its clean
    normal pair (or another clean POI), producing unavoidable feature/label collisions.
    These counts make that limitation visible rather than selecting only successful
    attacks.
    """
    clean_values = raw_feature_matrix(attacks, "_clean")
    attack_values = raw_feature_matrix(attacks, "_adv")
    actual_changed = _recomputed_changed_mask(clean_values, attack_values)
    declared_changed = changed_mask(attacks).astype(bool)
    unchanged = ~actual_changed.any(axis=1)
    attempted = _strict_boolean(attacks, "attack_attempted")
    changed_any = _strict_boolean(attacks, "attack_changed_any")
    clean_feature_set = {
        tuple(row) for row in raw_feature_matrix(clean)
    }
    collisions = np.fromiter(
        (
            tuple(row) in clean_feature_set
            for row in attack_values.astype(np.float64, copy=False)
        ),
        dtype=bool,
        count=len(attacks),
    )
    diagnostics: dict[str, int | str] = {
        "attack_intent_rows": int(len(attacks)),
        "attack_attempted_rows": int(attempted.sum()),
        "unattempted_attack_intent_rows": int((~attempted).sum()),
        "unchanged_attack_rows": int(unchanged.sum()),
        "attempted_unchanged_rows": int((attempted & unchanged).sum()),
        "clean_attack_feature_collision_rows": int(collisions.sum()),
        "changed_mask_disagreement_rows": int(
            (actual_changed != declared_changed).any(axis=1).sum()
        ),
        "attack_changed_any_disagreement_rows": int(
            (changed_any != actual_changed.any(axis=1)).sum()
        ),
        "constraint_invalid_rows": int(
            (~_strict_boolean(attacks, "constraints_valid")).sum()
        ),
        "label_not_preserved_rows": int(
            (~_strict_boolean(attacks, "label_preserved")).sum()
        ),
        "attack_label_policy": "all_generated_attack_intent_rows_are_label_1",
        "feature_collision_limitation": (
            "Identical five-feature vectors can carry clean=0 and attack-intent=1; "
            "detector separability is therefore bounded."
        ),
    }
    if "primary_evaluation_eligible" in attacks:
        eligible = _strict_boolean(attacks, "primary_evaluation_eligible")
        constrained = _strict_boolean(attacks, "constraints_valid")
        preserved = _strict_boolean(attacks, "label_preserved")
        training_eligible = eligible & constrained & preserved
        diagnostics["primary_evaluation_ineligible_rows"] = int((~eligible).sum())
        diagnostics["eligible_training_pool_rows"] = int(training_eligible.sum())
        diagnostics["excluded_from_training_rows"] = int((~training_eligible).sum())
        diagnostics["unchanged_valid_training_rows"] = int(
            (unchanged & training_eligible).sum()
        )
        diagnostics["attempted_unchanged_valid_training_rows"] = int(
            (attempted & unchanged & training_eligible).sum()
        )
        success = _strict_boolean(attacks, "attack_success_source")
        diagnostics["source_attack_success_rows_in_training_pool"] = int(
            (success & training_eligible).sum()
        )
        diagnostics["source_attack_failure_rows_retained_in_training_pool"] = int(
            ((~success) & training_eligible).sum()
        )
    return diagnostics


def eligible_training_attacks(attacks: pd.DataFrame) -> pd.DataFrame:
    """Select valid supervised attacks without conditioning on attack success.

    Invalid or boundary-changing rows remain available to all-intent audit and
    evaluation tables, but cannot supply a valid clean-region target for defense
    fitting.  The selection deliberately does not inspect ``attack_success_source``;
    unchanged and failed source attacks remain whenever they satisfy the primary
    validity contract.
    """
    eligible = _strict_boolean(attacks, "primary_evaluation_eligible")
    constrained = _strict_boolean(attacks, "constraints_valid")
    preserved = _strict_boolean(attacks, "label_preserved")
    result = attacks.loc[eligible & constrained & preserved].copy()
    if result.empty:
        raise ValueError("No primary-valid, constraint-valid, label-preserved attacks")
    return result.reset_index(drop=True)


def _strict_boolean(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Attack table lacks required boolean column: {column}")
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"Attack column {column} must be non-null boolean")
    return values.to_numpy(dtype=bool)


def feature_matrix(frame: pd.DataFrame, suffix: str = "") -> np.ndarray:
    """Return float32 model input only after raw integrity checks have completed."""
    columns = [f"{feature}{suffix}" for feature in FEATURES]
    return frame[columns].to_numpy(dtype=np.float32)


def raw_feature_matrix(frame: pd.DataFrame, suffix: str = "") -> np.ndarray:
    """Return original-precision float64 features for integrity and constraints."""
    columns = [f"{feature}{suffix}" for feature in FEATURES]
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Raw POI features must be a finite [rows, 5] matrix")
    return values


def changed_mask(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"changed_{feature.lower()}" for feature in FEATURES]
    result = np.empty((len(frame), len(columns)), dtype=np.float32)
    for index, column in enumerate(columns):
        if column not in frame:
            raise ValueError(f"Attack table lacks changed mask column: {column}")
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"Attack mask {column} must be non-null Boolean or exact 0/1")
        if pd.api.types.is_bool_dtype(values.dtype):
            result[:, index] = values.to_numpy(dtype=np.float32)
            continue
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(f"Attack mask {column} must not use string/object values")
        numeric = values.to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"Attack mask {column} must contain only finite exact 0/1")
        result[:, index] = numeric.astype(np.float32)
    return result


def _recomputed_changed_mask(clean: np.ndarray, attacked: np.ndarray) -> np.ndarray:
    if clean.shape != attacked.shape or clean.ndim != 2 or clean.shape[1] != 5:
        raise ValueError("Paired feature arrays must both have shape [rows, 5]")
    return np.column_stack(
        (
            np.abs(clean[:, :2] - attacked[:, :2]) > 1e-12,
            clean[:, 2:] != attacked[:, 2:],
        )
    )


def stratified_context(
    clean_train: pd.DataFrame,
    rows: int,
    *,
    seed: int,
) -> pd.DataFrame:
    if rows < clean_train["region"].nunique():
        raise ValueError("context_rows must include every region")
    rng = np.random.default_rng(seed)
    regions = sorted(clean_train["region"].unique())
    base = rows // len(regions)
    remainder = rows % len(regions)
    selected = []
    for index, region in enumerate(regions):
        group = clean_train.loc[clean_train["region"].eq(region)]
        count = base + (index < remainder)
        if len(group) < count:
            raise ValueError(f"Region {region} has too few context rows")
        selected.extend(rng.choice(group.index.to_numpy(), size=count, replace=False))
    return clean_train.loc[selected].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_validation_ids(
    clean_validation: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Make region-stratified calibration/evaluation ID sets before scoring."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    calibration: list[str] = []
    evaluation: list[str] = []
    for _, group in clean_validation.groupby("region", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed)
        count = max(1, min(len(group) - 1, round(len(group) * calibration_fraction)))
        calibration.extend(shuffled.iloc[:count]["POI_ID"].tolist())
        evaluation.extend(shuffled.iloc[count:]["POI_ID"].tolist())
    if set(calibration).intersection(evaluation):
        raise RuntimeError("Calibration and validation-evaluation POIs overlap")
    return calibration, evaluation


def one_attack_per_poi(
    attacks: pd.DataFrame,
    *,
    seed: int,
    allowed_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    frame = attacks
    if allowed_ids is not None:
        frame = frame.loc[frame["POI_ID"].isin(set(allowed_ids))]
    if frame.empty:
        raise ValueError("No attacks available for requested POIs")
    return (
        frame.groupby("POI_ID", group_keys=False, sort=True)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )


def balanced_attack_sample(
    attacks: pd.DataFrame,
    *,
    seed: int,
    max_pois: int | None,
    allowed_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Choose at most one attack per POI, balanced by condition and generation seed.

    Every selected row keeps its explicit ``attack_condition`` and ``generation_seed``;
    clean/attack pairs are never concatenated as independent observations.
    """
    frame = attacks
    if allowed_ids is not None:
        frame = frame.loc[frame["POI_ID"].isin(set(allowed_ids))]
    key = ["POI_ID", "generation_seed", "attack_condition"]
    if frame.empty or frame.duplicated(key).any():
        raise ValueError("Attack rows must be unique by POI/generation seed/condition")
    if max_pois is not None and max_pois <= 0:
        raise ValueError("max_pois must be positive or null")
    target = min(max_pois or frame["POI_ID"].nunique(), frame["POI_ID"].nunique())
    strata_frame = (
        frame[["generation_seed", "attack_condition"]]
        .drop_duplicates()
        .sort_values(["generation_seed", "attack_condition"])
    )
    strata = list(strata_frame.itertuples(index=False, name=None))
    if not strata:
        raise ValueError("No attack strata available")
    # Greedily cycle through strata and take an unused POI that has that generated
    # attack-intent row.  This preserves pair identity, avoids duplicate POIs, and keeps
    # stratum counts within one whenever available rows permit it.
    groups = {
        stratum: group.sample(frac=1.0, random_state=seed + index).reset_index(drop=True)
        for index, (stratum, group) in enumerate(
            frame.groupby(["generation_seed", "attack_condition"], sort=True)
        )
    }
    pointers = {stratum: 0 for stratum in strata}
    used: set[str] = set()
    selected_rows = []
    ordered_strata = strata[seed % len(strata) :] + strata[: seed % len(strata)]
    while len(selected_rows) < target:
        progressed = False
        for stratum in ordered_strata:
            group = groups[stratum]
            pointer = pointers[stratum]
            while pointer < len(group) and group.iloc[pointer]["POI_ID"] in used:
                pointer += 1
            pointers[stratum] = pointer
            if pointer >= len(group):
                continue
            row = group.iloc[pointer]
            pointers[stratum] += 1
            used.add(row["POI_ID"])
            selected_rows.append(row)
            progressed = True
            if len(selected_rows) == target:
                break
        if not progressed:
            break
    if len(selected_rows) < target:
        raise ValueError(
            f"Only {len(selected_rows)} unique POIs have attack rows; requested {target}"
        )
    return pd.DataFrame(selected_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
