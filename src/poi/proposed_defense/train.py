"""GPU stage-08 runner for the structural TabPFN v2.5 defense."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import (
    changed_mask,
    feature_matrix,
    load_stage08_data,
    attack_intent_diagnostics,
    eligible_training_attacks,
    balanced_attack_sample,
    split_validation_ids,
    stratified_context,
)
from .losses import DefenseLossWeights, paired_defense_loss
from .model import ProposedTabPFNDefense, load_official_tabpfn25
from .preflight import _validate_producer_lineage
from .predictor import REGIONAL_DECISION_RULE, build_checkpoint_payload
from .schema import POIFeatureCodebook, region_indices


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("regional_decision_rule") != REGIONAL_DECISION_RULE:
        raise ValueError("Stage 08 requires the frozen independent OvR decision rule")
    if not 0.0 < float(cfg.get("classification_threshold", float("nan"))) < 1.0:
        raise ValueError("Stage 08 classification_threshold must be strictly between zero and one")
    _require_current_data_preflight(cfg)
    output = Path(output_override or cfg["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stage-08 output: {output}")
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Stage 08 requires the configured CUDA device")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()

    clean, attacks = load_stage08_data(cfg["data_dir"], cfg["attack_table"])
    clean_train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
    clean_validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
    regions = tuple(sorted(clean_train["region"].unique()))
    if len(regions) != 17 or set(clean_validation["region"].unique()) != set(regions):
        raise RuntimeError("Stage 08 requires the frozen 17-region train/validation split")
    codebook = POIFeatureCodebook.fit(clean_train)
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

    histories = []
    for seed in cfg["seeds"]:
        history, metadata = _train_seed(
            cfg,
            int(seed),
            clean_train,
            clean_validation,
            attacks,
            codebook,
            regions,
            calibration_ids,
            evaluation_ids,
            output,
        )
        histories.extend(history)
        (output / f"seed_{seed}_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    metadata = {
        "status": "trained_not_independently_reviewed",
        "scope": "stage08_train_validation_only",
        "seeds": cfg["seeds"],
        "regions": list(regions),
        "clean_train_rows": len(clean_train),
        "attack_intent_rows": len(attacks),
        "attack_intent_diagnostics": attack_intent_diagnostics(attacks, clean),
        "calibration_pois": len(calibration_ids),
        "evaluation_pois": len(evaluation_ids),
        "test_rows_read": 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": _sha256(config_path),
        "seconds": time.time() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _train_seed(
    cfg: dict[str, Any],
    seed: int,
    clean_train: pd.DataFrame,
    clean_validation: pd.DataFrame,
    attacks: pd.DataFrame,
    codebook: POIFeatureCodebook,
    regions: tuple[str, ...],
    calibration_ids: list[str],
    evaluation_ids: list[str],
    output: Path,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(cfg["device"])
    base, provenance = load_official_tabpfn25(
        checkpoint_path=cfg.get("official_checkpoint"),
        device=device,
        download_if_missing=bool(cfg.get("download_checkpoint_if_missing", False)),
    )
    expected_checkpoint = cfg.get("expected_official_checkpoint_sha256")
    if expected_checkpoint and provenance.get("checkpoint_sha256") != expected_checkpoint:
        raise RuntimeError(
            "Official TabPFN v2.5 checkpoint differs from the frozen stage-08 config"
        )
    model = ProposedTabPFNDefense(
        base,
        codebook,
        regions,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        numeric_scales=tuple(float(v) for v in cfg["model"]["numeric_scales"]),
        activation_checkpointing=bool(cfg["model"]["activation_checkpointing"]),
        use_reliability_gate=bool(cfg["model"].get("use_reliability_gate", True)),
        use_reconstruction_path=bool(
            cfg["model"].get("use_reconstruction_path", True)
        ),
        use_multiscale_encoder=bool(cfg["model"].get("use_multiscale_encoder", True)),
        use_feature_encoders=bool(cfg["model"].get("use_feature_encoders", True)),
        use_category_relations=bool(cfg["model"].get("use_category_relations", True)),
        use_corruption_heads=bool(cfg["model"].get("use_corruption_heads", True)),
    ).to(device)
    unfrozen_blocks = int(cfg["model"].get("unfreeze_last_blocks", 0))
    if unfrozen_blocks:
        model.unfreeze_last_blocks(unfrozen_blocks)
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(("adapter.", "region_"))
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": float(cfg["optimizer"]["learning_rate"])}
    ]
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("base_model.")
    ]
    if base_parameters:
        parameter_groups.append(
            {"params": base_parameters, "lr": float(cfg["optimizer"]["base_learning_rate"])}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(cfg["optimizer"]["weight_decay"])
    )

    context = stratified_context(clean_train, int(cfg["context_rows"]), seed=seed)
    context_ids = set(context["POI_ID"])
    fit_ids = set(clean_train["POI_ID"]).difference(context_ids)
    context_encoded = codebook.encode(feature_matrix(context), device=device)
    context_labels = torch.as_tensor(
        region_indices(context["region"], regions), dtype=torch.long, device=device
    )
    train_attack_intent = attacks.loc[
        attacks["split"].eq("train") & attacks["POI_ID"].isin(fit_ids)
    ].reset_index(drop=True)
    train_attacks = eligible_training_attacks(train_attack_intent)
    calibration_attack_intent = attacks.loc[
        attacks["split"].eq("validation")
        & attacks["POI_ID"].isin(set(calibration_ids))
    ].reset_index(drop=True)
    validation_attacks = eligible_training_attacks(calibration_attack_intent)
    if set(calibration_ids).intersection(fit_ids) or set(evaluation_ids).intersection(fit_ids):
        raise RuntimeError("Validation POIs entered stage-08 fitting")
    weights = DefenseLossWeights(**cfg["loss_weights"])
    if not model.adapter.use_reconstruction_path and weights.reconstruction != 0:
        raise ValueError(
            "A disabled reconstruction path requires loss_weights.reconstruction=0"
        )
    if not model.adapter.use_corruption_heads and weights.feature_detection != 0:
        raise ValueError(
            "Disabled corruption heads require loss_weights.feature_detection=0"
        )
    history: list[dict[str, float]] = []
    regional_history: list[dict[str, Any]] = []
    best_clean_macro_f1 = float("-inf")
    best_validation_loss = float("inf")
    fine_tuning_started_epoch: int | None = None
    checkpoint_path = output / "checkpoints" / f"proposed_tabpfn25_seed{seed}.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["mixed_precision"]))
    for epoch in range(int(cfg["epochs"])):
        epoch_attacks = balanced_attack_sample(
            train_attacks,
            seed=seed + epoch,
            max_pois=int(cfg["train_pairs_per_epoch"]),
        )
        model.train()
        totals: dict[str, float] = {name: 0.0 for name in (
            "loss", "clean_region", "attack_region", "consistency",
            "feature_detection", "reconstruction", "attack_detection"
        )}
        batches = 0
        for start in range(0, len(epoch_attacks), int(cfg["query_batch_size"])):
            batch = epoch_attacks.iloc[start : start + int(cfg["query_batch_size"])]
            clean_query = codebook.encode(feature_matrix(batch, "_clean"), device=device)
            attack_query = codebook.encode(feature_matrix(batch, "_adv"), device=device)
            query_labels = torch.as_tensor(
                region_indices(batch["region"], regions), dtype=torch.long, device=device
            )
            feature_mask = torch.as_tensor(changed_mask(batch), device=device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if bool(cfg["mixed_precision"])
                else contextlib.nullcontext()
            )
            with autocast:
                clean_output = model(
                    torch.cat((context_encoded, clean_query), dim=0), context_labels
                )
                attack_output = model(
                    torch.cat((context_encoded, attack_query), dim=0), context_labels
                )
                loss, terms = paired_defense_loss(
                    clean_output,
                    attack_output,
                    query_region_indices=query_labels,
                    clean_encoded_query=clean_query,
                    changed_feature_mask=feature_mask,
                    weights=weights,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(cfg["optimizer"]["gradient_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for name, value in terms.items():
                totals[name] += float(value.detach())
            batches += 1
        validation_frame = balanced_attack_sample(
            validation_attacks,
            seed=seed,
            max_pois=int(cfg["validation_pairs_per_epoch"]),
        )
        validation_loss = _validation_loss(
            model,
            validation_frame,
            codebook,
            regions,
            context_encoded,
            context_labels,
            weights,
            int(cfg["query_batch_size"]),
        )
        calibration_clean = clean_validation.loc[
            clean_validation["POI_ID"].isin(set(calibration_ids))
        ].reset_index(drop=True)
        clean_macro_f1, clean_metrics = _regional_validation_metrics(
            model, calibration_clean, codebook, regions, context_encoded,
            context_labels, int(cfg["query_batch_size"]), condition="clean",
            classification_threshold=float(cfg["classification_threshold"]),
        )
        _, attack_metrics = _regional_validation_metrics(
            model, validation_frame, codebook, regions, context_encoded,
            context_labels, int(cfg["query_batch_size"]), condition="attack",
            classification_threshold=float(cfg["classification_threshold"]),
        )
        for metric in (*clean_metrics, *attack_metrics):
            regional_history.append({"seed": seed, "epoch": epoch + 1, **metric})
        row = {
            "seed": float(seed),
            "epoch": float(epoch + 1),
            **{name: total / max(batches, 1) for name, total in totals.items()},
            "validation_loss": validation_loss,
            "validation_clean_macro_f1": clean_macro_f1,
            "unfrozen_blocks": float(model.unfrozen_block_count),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = clean_macro_f1 > best_clean_macro_f1 or (
            clean_macro_f1 == best_clean_macro_f1
            and validation_loss < best_validation_loss
        )
        if improved:
            best_clean_macro_f1 = clean_macro_f1
            best_validation_loss = validation_loss
            payload = build_checkpoint_payload(
                model,
                context_encoded=context_encoded,
                context_region_indices=context_labels,
                official_provenance=provenance,
                training_metadata={
                    "seed": seed,
                    "epoch": epoch + 1,
                    "best_validation_loss": validation_loss,
                    "best_validation_clean_macro_f1": clean_macro_f1,
                    "model_selection_metric": "calibration_clean_macro_f1",
                    "regional_decision_rule": REGIONAL_DECISION_RULE,
                    "classification_threshold": float(cfg["classification_threshold"]),
                    "context_poi_ids": context["POI_ID"].tolist(),
                    "fit_poi_count": len(fit_ids),
                    "calibration_ids_path": str(output / "calibration_ids.json"),
                    "calibration_poi_count": len(calibration_ids),
                    "evaluation_poi_count": len(evaluation_ids),
                    "model_selection_split": "validation_calibration_ids",
                    "model_selection_attack_sample_seed": seed,
                    "evaluation_ids_used_for_model_selection": 0,
                    "test_rows_read": 0,
                    "attack_sampling": cfg["attack_sampling"],
                    "train_pairs_per_epoch": int(cfg["train_pairs_per_epoch"]),
                    "validation_pairs_per_epoch": int(cfg["validation_pairs_per_epoch"]),
                    "run_label": cfg.get("run_label", "full_model"),
                    "ablation_kind": cfg.get("ablation_kind", "reference"),
                    "ablation_changes": cfg.get("ablation_changes", []),
                    "architecture_flags": model.architecture_flags,
                    "loss_weights": cfg["loss_weights"],
                    "unfrozen_blocks": model.unfrozen_block_count,
                    "attack_generation_seeds": sorted(
                        int(value) for value in train_attacks["generation_seed"].unique()
                    ),
                    "attack_conditions": sorted(train_attacks["attack_condition"].unique()),
                    "attack_intent_diagnostics": attack_intent_diagnostics(
                        train_attack_intent,
                        clean_train.loc[clean_train["POI_ID"].isin(fit_ids)],
                    ),
                },
            )
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
        fine_tuning_started_epoch = _maybe_start_staged_fine_tuning(
            model, optimizer, cfg, history, epoch + 1, fine_tuning_started_epoch
        )
    pd.DataFrame(regional_history).to_csv(
        output / f"validation_regional_metrics_seed{seed}.csv", index=False
    )
    return history, {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "official_tabpfn": provenance,
        "context_rows": len(context),
        "fit_pois": len(fit_ids),
        "calibration_pois": len(calibration_ids),
        "evaluation_pois": len(evaluation_ids),
        "best_validation_loss": best_validation_loss,
        "best_validation_clean_macro_f1": best_clean_macro_f1,
        "model_selection_metric": "calibration_clean_macro_f1",
        "regional_decision_rule": REGIONAL_DECISION_RULE,
        "classification_threshold": float(cfg["classification_threshold"]),
        "unfrozen_blocks_initial": unfrozen_blocks,
        "unfrozen_blocks_final": model.unfrozen_block_count,
        "fine_tuning_started_epoch": fine_tuning_started_epoch,
        "run_label": cfg.get("run_label", "full_model"),
        "ablation_kind": cfg.get("ablation_kind", "reference"),
        "ablation_changes": cfg.get("ablation_changes", []),
        "architecture_flags": model.architecture_flags,
        "attack_intent_diagnostics": attack_intent_diagnostics(
            train_attack_intent,
            clean_train.loc[clean_train["POI_ID"].isin(fit_ids)],
        ),
        "model_selection_split": "validation_calibration_ids",
        "model_selection_attack_sample_seed": seed,
        "evaluation_ids_used_for_model_selection": 0,
        "train_batches_per_epoch": int(
            np.ceil(int(cfg["train_pairs_per_epoch"]) / int(cfg["query_batch_size"]))
        ),
        "validation_batches_per_epoch": int(
            np.ceil(int(cfg["validation_pairs_per_epoch"]) / int(cfg["query_batch_size"]))
        ),
        "official_forwards_per_epoch": 2
        * (
            int(np.ceil(int(cfg["train_pairs_per_epoch"]) / int(cfg["query_batch_size"])))
            + int(np.ceil(int(cfg["validation_pairs_per_epoch"]) / int(cfg["query_batch_size"])))
        ),
    }


@torch.no_grad()
def _validation_loss(
    model: ProposedTabPFNDefense,
    attacks: pd.DataFrame,
    codebook: POIFeatureCodebook,
    regions: tuple[str, ...],
    context_encoded: torch.Tensor,
    context_labels: torch.Tensor,
    weights: DefenseLossWeights,
    batch_size: int,
) -> float:
    frame = attacks
    model.eval()
    total = 0.0
    batches = 0
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        clean_query = codebook.encode(feature_matrix(batch, "_clean"), device=context_encoded.device)
        attack_query = codebook.encode(feature_matrix(batch, "_adv"), device=context_encoded.device)
        query_labels = torch.as_tensor(
            region_indices(batch["region"], regions),
            dtype=torch.long,
            device=context_encoded.device,
        )
        mask = torch.as_tensor(changed_mask(batch), device=context_encoded.device)
        clean_output = model(torch.cat((context_encoded, clean_query)), context_labels)
        attack_output = model(torch.cat((context_encoded, attack_query)), context_labels)
        loss, _ = paired_defense_loss(
            clean_output,
            attack_output,
            query_region_indices=query_labels,
            clean_encoded_query=clean_query,
            changed_feature_mask=mask,
            weights=weights,
        )
        total += float(loss)
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def _regional_validation_metrics(
    model: ProposedTabPFNDefense,
    frame: pd.DataFrame,
    codebook: POIFeatureCodebook,
    regions: tuple[str, ...],
    context_encoded: torch.Tensor,
    context_labels: torch.Tensor,
    batch_size: int,
    *,
    condition: str,
    classification_threshold: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate deployed hard-projected regional predictions on calibration only."""
    if condition not in {"clean", "attack"}:
        raise ValueError("Validation condition must be clean or attack")
    suffix = "" if condition == "clean" else "_adv"
    labels = region_indices(frame["region"], regions)
    probability_parts = []
    model.eval()
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        query = codebook.encode(feature_matrix(batch, suffix), device=context_encoded.device)
        combined = torch.cat((context_encoded, query), dim=0)
        latent = model(combined, context_labels)
        projection = model.adapter.project_for_deployment(combined, latent.structural)
        deployed = model.forward_projected_features(projection.repaired_features, context_labels)
        probability_parts.append(torch.sigmoid(deployed.region_logits).cpu().numpy())
    probabilities = np.concatenate(probability_parts, axis=0)
    targets = np.eye(len(regions), dtype=np.int64)[labels]
    predictions = _ovr_binary_predictions(probabilities, classification_threshold)
    rows: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        truth = targets[:, index]
        score = probabilities[:, index]
        predicted = predictions[:, index]
        rows.append(
            {
                "condition": condition,
                "region": region,
                "f1": f1_score(truth, predicted, zero_division=0),
                "precision": precision_score(truth, predicted, zero_division=0),
                "recall": recall_score(truth, predicted, zero_division=0),
                "accuracy": accuracy_score(truth, predicted),
                "roc_auc": roc_auc_score(truth, score) if len(np.unique(truth)) == 2 else np.nan,
                "average_precision": average_precision_score(truth, score) if truth.any() else np.nan,
                "positive_rows": int(truth.sum()),
                "rows": len(truth),
                "prediction_path": "hard_train_support_projected_own_ovr_heads",
                "decision_rule": "independent_ovr_probability_gte_threshold",
                "classification_threshold": classification_threshold,
            }
        )
    return float(np.mean([row["f1"] for row in rows])), rows


def _ovr_binary_predictions(
    probabilities: np.ndarray, classification_threshold: float
) -> np.ndarray:
    """Apply the frozen independent 17-head decision rule used by primary metrics."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("OvR probabilities must be a finite two-dimensional matrix")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must be strictly between zero and one")
    return (values >= classification_threshold).astype(np.int64)


def _maybe_start_staged_fine_tuning(
    model: ProposedTabPFNDefense,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    history: list[dict[str, float]],
    epoch: int,
    started_epoch: int | None,
) -> int | None:
    setting = cfg.get("fine_tuning", {"mode": "disabled"})
    if setting.get("mode", "disabled") != "staged" or started_epoch is not None:
        return started_epoch
    warmup = int(setting["warmup_epochs"])
    window = int(setting["stability_window"])
    if epoch < warmup or len(history) < window:
        return None
    values = [row["validation_clean_macro_f1"] for row in history[-window:]]
    if float(np.std(values)) > float(setting["max_clean_macro_f1_std"]):
        return None
    count = int(setting["unfreeze_last_blocks"])
    model.unfreeze_last_blocks(count)
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("base_model.")
    ]
    if not base_parameters:
        raise RuntimeError("Staged fine-tuning selected no official block parameters")
    optimizer.add_param_group(
        {"params": base_parameters, "lr": float(cfg["optimizer"]["base_learning_rate"])}
    )
    return epoch + 1


def _seed_everything(seed: int) -> None:
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


def _require_current_data_preflight(cfg: dict[str, Any]) -> None:
    path = Path(cfg["preflight_output"]) / "preflight.json"
    if not path.is_file():
        raise RuntimeError(
            f"Defense training requires the train/validation integrity preflight first: {path}"
        )
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or report.get("training_ready") is not True:
        raise RuntimeError(f"Defense data preflight is not training-ready: {report.get('status')}")
    if report.get("test_rows_read") != 0:
        raise RuntimeError("Defense data preflight accessed locked test rows")
    attack_report = report.get("attack", {})
    inventory = attack_report.get("complete_validated_inventory", {})
    expected_rows = 27_200 * 32 * 3
    if (
        inventory.get("status") != "PASS"
        or inventory.get("rows") != expected_rows
        or inventory.get("expected_rows") != expected_rows
        or inventory.get("clean_development_pois") != 27_200
        or inventory.get("expected_clean_development_pois") != 27_200
        or inventory.get("clean_pois_by_split") != {"train": 23_800, "validation": 3_400}
        or inventory.get("expected_clean_pois_by_split") != {
            "train": 23_800, "validation": 3_400
        }
        or inventory.get("groups") != 192
        or inventory.get("expected_groups") != 192
        or inventory.get("generation_seed_count") != 3
        or inventory.get("attack_condition_count") != 32
        or inventory.get("unique_complete_poi_membership") is not True
        or inventory.get("split_region_manifest_truth") is not True
    ):
        raise RuntimeError("Defense preflight lacks the exact full attack inventory")
    lineage = attack_report.get("producer_lineage", {})
    if (
        lineage.get("registered_shards") != 192
        or lineage.get("status") != "generated_pending_fresh_independent_review"
        or attack_report.get("stable_snapshot_before_after") is not True
    ):
        raise RuntimeError("Defense preflight lacks immutable producer completion lineage")
    attack_path = Path(cfg["attack_table"])
    data_dir = Path(cfg["data_dir"])
    generation_config = Path(cfg["attack_generation_config"])
    surrogate_artifact_dir = Path(cfg["surrogate_artifact_dir"])
    if (
        Path(report.get("attack_table", "")).resolve() != attack_path.resolve()
        or Path(report.get("data_dir", "")).resolve() != data_dir.resolve()
        or Path(report.get("attack_generation_config", "")).resolve()
        != generation_config.resolve()
        or Path(report.get("surrogate_artifact_dir", "")).resolve()
        != surrogate_artifact_dir.resolve()
    ):
        raise RuntimeError("Defense preflight input paths differ from launch configuration")
    if attack_report.get("sha256") != _sha256(attack_path):
        raise RuntimeError("Generated attack table changed after defense data preflight")
    source_hashes = report.get("implementation_sha256", {})
    module_dir = Path(__file__).resolve().parent
    current_sources = {
        name: _sha256(module_dir / name)
        for name in ("preflight.py", "data.py", "schema.py")
    }
    if source_hashes != current_sources:
        raise RuntimeError("Defense integrity source map changed after preflight")
    dependencies = report.get("dependency_sha256", {})
    current_dependencies = {
        f"src/poi/{name}": _sha256(module_dir.parent / name)
        for name in ("adversarial.py", "data.py", "geospatial.py")
    }
    if dependencies != current_dependencies:
        raise RuntimeError("Defense preflight dependency map changed after preflight")
    ure_source = module_dir.parent / "ure" / "preprocessing.py"
    if report.get("ure_preprocessing_sha256") != _sha256(ure_source):
        raise RuntimeError("URE preprocessing source changed after defense preflight")
    clean_hashes = report.get("clean", {}).get("source_sha256", {})
    for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json"):
        if clean_hashes.get(name) != _sha256(data_dir / name):
            raise RuntimeError(f"Clean source changed after defense preflight: {name}")
    if lineage.get("local_generation_config_sha256") != _sha256(generation_config):
        raise RuntimeError("Attack-generation config changed after defense preflight")
    try:
        current_lineage = _validate_producer_lineage(
            attack_path,
            generation_config,
            data_dir,
            surrogate_artifact_dir=surrogate_artifact_dir,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Current producer lineage validation failed: {error}") from error
    if current_lineage.get("report") != lineage:
        raise RuntimeError("Current producer lineage bytes differ from defense preflight")
    if current_lineage.get("attack_table_sha256") != attack_report.get("sha256"):
        raise RuntimeError("Current producer metadata differs from the validated attack table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/08_proposed_tabpfn_defense.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
