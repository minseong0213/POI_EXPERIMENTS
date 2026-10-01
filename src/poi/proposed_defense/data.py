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
    if not attacks["label_preserved"].fillna(False).all():
        attacks = attacks.loc[attacks["label_preserved"].fillna(False)].copy()
    attacks = attacks.loc[attacks["constraints_valid"].fillna(False)].copy()
    if attacks.empty:
        raise ValueError("No constraint-valid, label-preserved attacks remain")
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
    return clean.reset_index(drop=True), attacks.reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, suffix: str = "") -> np.ndarray:
    columns = [f"{feature}{suffix}" for feature in FEATURES]
    return frame[columns].to_numpy(dtype=np.float32)


def changed_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"changed_{feature.lower()}" for feature in FEATURES]].to_numpy(
        dtype=np.float32
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
