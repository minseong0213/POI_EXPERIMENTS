"""Stage-09 URE training and URE-RF detector serialization."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from poi.proposed_defense.data import (
    feature_matrix,
    load_stage08_data,
    attack_intent_diagnostics,
    eligible_training_attacks,
    balanced_attack_sample,
    split_validation_ids,
)
from poi.proposed_defense.schema import region_indices
from poi.proposed_defense.train import _require_current_data_preflight

from .detector import UREAttackDetector, save_ure_attack_detector
from .model import URERegionModel
from .preprocessing import (
    UREPreprocessor,
    build_poi_additive_transition_costs,
    load_transition_costs,
)
from .source import IMPLEMENTATION_NOTES, OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, PAPER_URL
from .training import URETrainingConfig, train_universal_robust_embeddings


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    _require_current_data_preflight(cfg)
    output = Path(output_override or cfg["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stage-09 URE output: {output}")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / "tables").mkdir()
    clean, attacks = load_stage08_data(cfg["data_dir"], cfg["attack_table"])
    clean_train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    clean_validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    regions = tuple(sorted(clean_train["region"].unique()))
    if len(regions) != 17:
        raise RuntimeError("URE region supervision requires exactly 17 regions")
    calibration_ids, evaluation_ids = split_validation_ids(
        clean_validation,
        calibration_fraction=float(cfg["calibration_fraction"]),
        seed=int(cfg["calibration_seed"]),
    )
    (output / "calibration_ids.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "calibration_ids": calibration_ids,
                "evaluation_ids": evaluation_ids,
                "overlap": 0,
            },
            indent=2,
        )
        + "\n"
    )
    predictions = []
    seed_metadata = []
    for seed in cfg["seeds"]:
        detector, metadata = _fit_seed(
            cfg,
            int(seed),
            clean_train,
            clean_validation,
            attacks,
            regions,
            calibration_ids,
            evaluation_ids,
            output,
        )
        seed_predictions = _validation_predictions(
            detector, int(seed), clean_validation, attacks, calibration_ids, evaluation_ids
        )
        threshold = _clean_retention_threshold(
            seed_predictions.loc[
                seed_predictions["condition"].eq("clean")
                & seed_predictions["partition"].eq("calibration"),
                "attack_score",
            ].to_numpy(),
            float(cfg["clean_retention"]),
        )
        seed_predictions["threshold"] = threshold
        seed_predictions["accepted"] = seed_predictions["attack_score"] <= threshold
        predictions.append(seed_predictions)
        metadata["threshold"] = threshold
        seed_metadata.append(metadata)
    pd.concat(predictions, ignore_index=True).to_parquet(
        output / "tables" / "ure_rf_validation_predictions.parquet", index=False
    )
    (output / "seed_metadata.json").write_text(json.dumps(seed_metadata, indent=2) + "\n")
    summary = {
        "status": "trained_not_independently_reviewed",
        "scope": "URE-RF external detector; train/validation only",
        "result_label": UREAttackDetector.result_label,
        "seeds": cfg["seeds"],
        "calibration_pois": len(calibration_ids),
        "evaluation_pois": len(evaluation_ids),
        "test_rows_read": 0,
        "model_selection": "none_fixed_epochs",
        "evaluation_ids_used_for_training_or_selection": 0,
        "official_source": {
            "paper": PAPER_URL,
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
        },
        "method_correspondence_and_deviations": IMPLEMENTATION_NOTES,
        "constraint_limitation": (
            "URE uses additive independent per-field costs and cannot encode the POI "
            "three-category valid-tuple relation inside its relaxed inner attack. Final "
            "evaluation must still use only constraint-valid generated attacks."
        ),
        "attack_intent_diagnostics": attack_intent_diagnostics(attacks, clean),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seconds": time.time() - started,
        "config_sha256": _sha256(config_path),
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _fit_seed(
    cfg: dict[str, Any],
    seed: int,
    clean_train: pd.DataFrame,
    clean_validation: pd.DataFrame,
    attacks: pd.DataFrame,
    regions: tuple[str, ...],
    calibration_ids: list[str],
    evaluation_ids: list[str],
    output: Path,
) -> tuple[UREAttackDetector, dict[str, Any]]:
    _seed(seed)
    device = torch.device(cfg["device"])
    preprocessor = UREPreprocessor.fit(
        clean_train, numeric_bins=int(cfg["ure"]["numeric_bins"])
    )
    if cfg["ure"].get("transition_costs_npz"):
        costs = load_transition_costs(
            cfg["ure"]["transition_costs_npz"], preprocessor.cardinalities
        )
        cost_source = cfg["ure"]["transition_costs_npz"]
    elif cfg["ure"].get("build_poi_additive_costs", False):
        costs = build_poi_additive_transition_costs(
            preprocessor,
            clean_train,
            numeric_cost_unit_m=float(cfg["ure"]["numeric_cost_unit_m"]),
            categorical_change_cost=float(cfg["ure"]["categorical_change_cost"]),
        )
        cost_path = output / f"transition_costs_seed{seed}.npz"
        np.savez(cost_path, **{f"feature_{i}": matrix for i, matrix in enumerate(costs)})
        cost_source = str(cost_path)
    else:
        raise ValueError(
            "URE needs transition_costs_npz or explicitly configured POI additive "
            "cost construction; refusing to fabricate the robust threat model"
        )
    region_model = URERegionModel(
        preprocessor.cardinalities,
        region_count=len(regions),
        embedding_dim=int(cfg["ure"]["embedding_dim"]),
        hidden_dim=int(cfg["ure"]["hidden_dim"]),
    )
    train_config = URETrainingConfig(**cfg["ure"]["training"])
    train_indices = preprocessor.transform(feature_matrix(clean_train))
    train_labels = region_indices(clean_train["region"], regions)
    history = train_universal_robust_embeddings(
        region_model,
        train_indices,
        train_labels,
        costs,
        config=train_config,
        device=device,
        seed=seed,
    )
    pd.DataFrame(history).assign(seed=seed).to_csv(
        output / "tables" / f"ure_training_seed{seed}.csv", index=False
    )
    train_attack_intent = attacks.loc[attacks["split"].eq("train")].reset_index(drop=True)
    train_eligible_attacks = eligible_training_attacks(train_attack_intent)
    train_attacks = balanced_attack_sample(
        train_eligible_attacks,
        seed=seed,
        max_pois=cfg["random_forest"].get("attack_training_pois"),
    )
    detector_matrix = np.concatenate(
        (feature_matrix(clean_train), feature_matrix(train_attacks, "_adv")), axis=0
    )
    detector_labels = np.concatenate(
        (np.zeros(len(clean_train), dtype=np.int64), np.ones(len(train_attacks), dtype=np.int64))
    )
    detector = UREAttackDetector(
        preprocessor,
        region_model.embedding,
        random_state=seed,
        n_estimators=int(cfg["random_forest"]["n_estimators"]),
        min_samples_leaf=int(cfg["random_forest"]["min_samples_leaf"]),
        merge_percentile=float(cfg["ure"]["merge_percentile"]),
        device=device,
    ).fit(detector_matrix, detector_labels)
    checkpoint = output / "checkpoints" / f"ure_rf_seed{seed}.pt"
    metadata = {
        "seed": seed,
        "region_supervision_rows": len(clean_train),
        "detector_clean_rows": len(clean_train),
        "detector_attack_rows": len(train_attacks),
        "calibration_pois": len(calibration_ids),
        "evaluation_pois": len(evaluation_ids),
        "transition_cost_source": cost_source,
        "transition_cost_sha256": _sha256(cost_source),
        "ure_training_config": dataclasses.asdict(train_config),
        "embedding_merging": detector.merger.to_dict() | {
            "merged_weights": "stored_in_checkpoint",
            "cluster_labels": [labels.tolist() for labels in detector.merger.cluster_labels],
            "fit_split": "clean_train_only",
        },
        "test_rows_read": 0,
        "attack_sampling": "balanced_by_generation_seed_and_condition_one_per_poi",
        "attack_generation_seeds": sorted(
            int(value)
            for value in attacks.loc[
                attacks["split"].eq("train"), "generation_seed"
            ].unique()
        ),
        "attack_intent_diagnostics": attack_intent_diagnostics(
            train_attack_intent, clean_train
        ),
        "source_attack_success_used_as_training_filter": False,
        "evaluation_ids_used_for_training_or_selection": 0,
    }
    checkpoint, forest = save_ure_attack_detector(checkpoint, detector, metadata=metadata)
    metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "forest": str(forest),
            "forest_sha256": _sha256(forest),
        }
    )
    return detector, metadata


def _validation_predictions(
    detector: UREAttackDetector,
    seed: int,
    clean_validation: pd.DataFrame,
    attacks: pd.DataFrame,
    calibration_ids: list[str],
    evaluation_ids: list[str],
) -> pd.DataFrame:
    id_partition = {value: "calibration" for value in calibration_ids}
    id_partition.update({value: "evaluation" for value in evaluation_ids})
    clean_score = detector.predict_proba(feature_matrix(clean_validation))[:, 1]
    clean_rows = pd.DataFrame(
        {
            "POI_ID": clean_validation["POI_ID"],
            "region": clean_validation["region"],
            "condition": "clean",
            "attack_method": "clean",
            "attack_condition": "clean",
            "generation_seed": -1,
            "attack_label": 0,
            "attack_score": clean_score,
        }
    )
    validation_attacks = attacks.loc[attacks["split"].eq("validation")].reset_index(drop=True)
    attack_score = detector.predict_proba(feature_matrix(validation_attacks, "_adv"))[:, 1]
    provenance_columns = [
        column
        for column in validation_attacks.columns
        if column in {
            "POI_ID", "region", "attack_method", "attack_condition",
            "generation_seed", "source_model", "victim_model",
            "constraints_valid", "label_preserved", "primary_evaluation_eligible",
            "attack_attempted", "attack_changed_any", "attack_success_source",
            "quarantine_reason", "budget", "budget_m", "numeric_budget_m",
            "categorical_budget", "attack_steps", "query_count",
        }
    ]
    attack_rows = validation_attacks[provenance_columns].copy()
    attack_rows["condition"] = "attack"
    attack_rows["attack_label"] = 1
    attack_rows["attack_score"] = attack_score
    result = pd.concat((clean_rows, attack_rows), ignore_index=True)
    result["partition"] = result["POI_ID"].map(id_partition)
    if result["partition"].isna().any():
        raise RuntimeError("URE validation row lacks a calibration/evaluation partition")
    result["model"] = UREAttackDetector.result_label
    result["seed"] = seed
    result["split"] = "validation"
    return result


def _clean_retention_threshold(scores: np.ndarray, retention: float) -> float:
    if not 0 < retention < 1 or len(scores) == 0:
        raise ValueError("Invalid clean retention calibration")
    try:
        return float(np.quantile(scores, retention, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(scores, retention, interpolation="higher"))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/09_ure_rf_detection.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
