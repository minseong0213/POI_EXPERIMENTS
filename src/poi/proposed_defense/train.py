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

from .data import (
    changed_mask,
    feature_matrix,
    load_stage08_data,
    one_attack_per_poi,
    split_validation_ids,
    stratified_context,
)
from .losses import DefenseLossWeights, paired_defense_loss
from .model import ProposedTabPFNDefense, load_official_tabpfn25
from .predictor import build_checkpoint_payload
from .schema import POIFeatureCodebook, region_indices


def run(config_path: str | Path, output_override: str | Path | None = None) -> None:
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
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
    codebook = POIFeatureCodebook.fit(feature_matrix(clean_train))
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
        "valid_attack_rows": len(attacks),
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
    model = ProposedTabPFNDefense(
        base,
        codebook,
        regions,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        numeric_scales=tuple(float(v) for v in cfg["model"]["numeric_scales"]),
        activation_checkpointing=bool(cfg["model"]["activation_checkpointing"]),
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
    train_attacks = attacks.loc[
        attacks["split"].eq("train") & attacks["POI_ID"].isin(fit_ids)
    ].reset_index(drop=True)
    validation_attacks = attacks.loc[
        attacks["split"].eq("validation")
        & attacks["POI_ID"].isin(set(evaluation_ids))
    ].reset_index(drop=True)
    if set(calibration_ids).intersection(fit_ids) or set(evaluation_ids).intersection(fit_ids):
        raise RuntimeError("Validation POIs entered stage-08 fitting")
    weights = DefenseLossWeights(**cfg["loss_weights"])
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    checkpoint_path = output / "checkpoints" / f"proposed_tabpfn25_seed{seed}.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["mixed_precision"]))
    for epoch in range(int(cfg["epochs"])):
        epoch_attacks = one_attack_per_poi(train_attacks, seed=seed + epoch)
        epoch_attacks = epoch_attacks.sample(frac=1.0, random_state=seed + epoch).reset_index(drop=True)
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
                [parameter for group in parameter_groups for parameter in group["params"]],
                float(cfg["optimizer"]["gradient_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for name, value in terms.items():
                totals[name] += float(value.detach())
            batches += 1
        validation_loss = _validation_loss(
            model,
            validation_attacks,
            codebook,
            regions,
            context_encoded,
            context_labels,
            weights,
            int(cfg["query_batch_size"]),
            seed + epoch,
        )
        row = {
            "seed": float(seed),
            "epoch": float(epoch + 1),
            **{name: total / max(batches, 1) for name, total in totals.items()},
            "validation_loss": validation_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            payload = build_checkpoint_payload(
                model,
                context_encoded=context_encoded,
                context_region_indices=context_labels,
                official_provenance=provenance,
                training_metadata={
                    "seed": seed,
                    "epoch": epoch + 1,
                    "best_validation_loss": validation_loss,
                    "context_poi_ids": context["POI_ID"].tolist(),
                    "fit_poi_count": len(fit_ids),
                    "calibration_ids_path": str(output / "calibration_ids.json"),
                    "calibration_poi_count": len(calibration_ids),
                    "evaluation_poi_count": len(evaluation_ids),
                    "test_rows_read": 0,
                },
            )
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
    return history, {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "official_tabpfn": provenance,
        "context_rows": len(context),
        "fit_pois": len(fit_ids),
        "calibration_pois": len(calibration_ids),
        "evaluation_pois": len(evaluation_ids),
        "best_validation_loss": best_validation,
        "unfrozen_blocks": unfrozen_blocks,
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
    seed: int,
) -> float:
    frame = one_attack_per_poi(attacks, seed=seed)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/08_proposed_tabpfn_defense.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
