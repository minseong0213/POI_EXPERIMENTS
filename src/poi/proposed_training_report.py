"""Summarize terminal Stage-08 training evidence without running model inference.

The report reads only the trainer's saved train/calibration artifacts.  It verifies all
three checkpoint payloads on CPU, plots epoch curves, and exports the complete
3-seed x 20-epoch x 2-condition x 17-region validation table.  It makes no Stage-10
downstream or locked-test claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SEEDS = (42, 202, 340)
EPOCHS = tuple(range(1, 21))
CONDITIONS = ("clean", "attack")
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
CHECKPOINT_FORMAT = "poi-proposed-tabpfn25-v3"
OFFICIAL_CHECKPOINT_SHA256 = (
    "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"
)
STAGE08_CONFIG_SHA256 = (
    "aea6dc052a9657a27630f7bb2e5ba2b75eafa85f1bdec498e3b604d7a83fd694"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required artifact absent: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_inventory(stage08: Path, regions: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        metadata_path = stage08 / f"seed_{seed}_metadata.json"
        metadata = _load_json(metadata_path)
        checkpoint = stage08 / f"checkpoints/proposed_tabpfn25_seed{seed}.pt"
        _require(checkpoint.is_file(), f"checkpoint absent for seed {seed}")
        checkpoint_sha = sha256(checkpoint)
        _require(metadata.get("seed") == seed, f"seed metadata identity differs: {seed}")
        _require(metadata.get("checkpoint_sha256") == checkpoint_sha, f"checkpoint hash differs: {seed}")
        _require(metadata.get("model_selection_metric") == "calibration_clean_macro_f1", "selection metric differs")
        _require(metadata.get("evaluation_ids_used_for_model_selection") == 0, "evaluation IDs entered model selection")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _require(payload.get("format") == CHECKPOINT_FORMAT, f"checkpoint format differs: {seed}")
        _require(tuple(payload.get("regions", ())) == regions, f"checkpoint regions differ: {seed}")
        official = payload.get("official_tabpfn", {})
        _require(official.get("checkpoint_sha256") == OFFICIAL_CHECKPOINT_SHA256, "official checkpoint digest differs")
        training = payload.get("training_metadata", {})
        _require(training.get("seed") == seed, f"checkpoint training seed differs: {seed}")
        _require(training.get("model_selection_split") == "validation_calibration_ids", "selection split differs")
        _require(training.get("evaluation_ids_used_for_model_selection") == 0, "checkpoint selected on evaluation IDs")
        selected_epoch = int(training.get("epoch", 0))
        _require(selected_epoch in EPOCHS, f"selected epoch outside 1..20: {seed}")
        state = payload.get("trainable_state")
        _require(isinstance(state, dict) and state, f"empty trainable state: {seed}")
        tensor_count = 0
        parameter_count = 0
        for name, tensor in state.items():
            _require(torch.is_tensor(tensor), f"non-tensor checkpoint state: {seed}/{name}")
            if tensor.is_floating_point() or tensor.is_complex():
                _require(bool(torch.isfinite(tensor).all()), f"non-finite checkpoint state: {seed}/{name}")
            tensor_count += 1
            parameter_count += int(tensor.numel())
        rows.append({
            "seed": seed,
            "selected_epoch": selected_epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_format": payload["format"],
            "official_checkpoint_sha256": official["checkpoint_sha256"],
            "state_tensors": tensor_count,
            "state_parameters": parameter_count,
            "best_validation_loss": float(metadata["best_validation_loss"]),
            "best_validation_clean_macro_f1": float(metadata["best_validation_clean_macro_f1"]),
            "fine_tuning_started_epoch": metadata.get("fine_tuning_started_epoch"),
            "unfrozen_blocks_final": int(metadata["unfrozen_blocks_final"]),
            "test_rows_read": 0,
        })
    return pd.DataFrame(rows)


def _load_evidence(
    stage08: Path, stage08_status: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status = _load_json(stage08_status)
    _require(status.get("stage") == "08", "Stage08 status identity differs")
    _require(status.get("exit_code") == 0, "Stage08 did not exit successfully")
    _require(
        isinstance(status.get("started_unix"), int)
        and isinstance(status.get("ended_unix"), int)
        and status["ended_unix"] >= status["started_unix"],
        "Stage08 status timestamps are invalid",
    )
    metadata = _load_json(stage08 / "metadata.json")
    _require(metadata.get("status") == "trained_not_independently_reviewed", "Stage08 is not terminal")
    _require(metadata.get("test_rows_read") == 0, "Stage08 metadata records test access")
    _require(
        metadata.get("config_sha256") == STAGE08_CONFIG_SHA256,
        "Stage08 metadata is not bound to the launched revision6 config",
    )
    _require((stage08 / "config.yaml").is_file(), "terminal Stage08 config copy is absent")
    _require(tuple(metadata.get("seeds", ())) == SEEDS, "Stage08 seed set differs")
    regions = tuple(metadata.get("regions", ()))
    _require(len(regions) == 17 and len(set(regions)) == 17, "Stage08 must contain 17 unique regions")
    _require(metadata.get("clean_train_rows") == 23_800, "clean-train row count differs")
    _require(metadata.get("calibration_pois") == 1_693, "calibration count differs")
    _require(metadata.get("evaluation_pois") == 1_707, "evaluation count differs")

    partition = _load_json(stage08 / "calibration_ids.json")
    calibration = tuple(map(str, partition.get("calibration_ids", ())))
    evaluation = tuple(map(str, partition.get("evaluation_ids", ())))
    _require(len(calibration) == 1_693 and len(evaluation) == 1_707, "partition cardinality differs")
    _require(not set(calibration).intersection(evaluation) and partition.get("overlap") == 0, "partition overlap")

    history = pd.read_csv(stage08 / "training_history.csv")
    required_history = {
        "seed", "epoch", "loss", "clean_region", "attack_region", "consistency",
        "feature_detection", "reconstruction", "attack_detection", "validation_loss",
        "validation_clean_macro_f1", "unfrozen_blocks",
    }
    _require(required_history.issubset(history.columns), "training history columns are incomplete")
    _require(len(history) == len(SEEDS) * len(EPOCHS), "training history must be 3 x 20")
    _require(not history.duplicated(["seed", "epoch"]).any(), "duplicate seed/epoch history rows")
    for seed in SEEDS:
        observed = tuple(sorted(history.loc[history.seed.astype(int).eq(seed), "epoch"].astype(int)))
        _require(observed == EPOCHS, f"epoch grid differs for seed {seed}")
    numeric = history[list(required_history - {"seed", "epoch"})].to_numpy(dtype=float)
    _require(np.isfinite(numeric).all(), "non-finite training history")

    regional_parts = []
    for seed in SEEDS:
        path = stage08 / f"validation_regional_metrics_seed{seed}.csv"
        frame = pd.read_csv(path)
        _require(len(frame) == len(EPOCHS) * len(CONDITIONS) * len(regions), f"regional row count differs: {seed}")
        _require(set(frame["condition"].astype(str)) == set(CONDITIONS), f"regional conditions differ: {seed}")
        _require(set(frame["region"].astype(str)) == set(regions), f"regional labels differ: {seed}")
        _require(set(frame["epoch"].astype(int)) == set(EPOCHS), f"regional epochs differ: {seed}")
        _require(set(frame["seed"].astype(int)) == {seed}, f"regional seed differs: {seed}")
        counts = frame.groupby(["epoch", "condition"], observed=True).size()
        _require((counts == 17).all() and len(counts) == 40, f"regional grid differs: {seed}")
        _require(set(frame.loc[frame.condition.eq("clean"), "rows"].astype(int)) == {1_693}, "clean validation denominator differs")
        _require(set(frame.loc[frame.condition.eq("attack"), "rows"].astype(int)) == {1_024}, "attack validation denominator differs")
        regional_parts.append(frame)
    regional = pd.concat(regional_parts, ignore_index=True)
    inventory = _checkpoint_inventory(stage08, regions)
    selected_epochs = inventory.set_index("seed")["selected_epoch"].to_dict()
    selected = regional.loc[
        regional.apply(lambda row: int(row["epoch"]) == selected_epochs[int(row["seed"])], axis=1)
    ].copy()
    _require(len(selected) == len(SEEDS) * len(CONDITIONS) * len(regions), "selected regional grid differs")
    return metadata, history, regional, inventory


def _plot_losses(history: pd.DataFrame, figures: Path) -> None:
    columns = (
        "loss", "clean_region", "attack_region", "consistency", "feature_detection",
        "reconstruction", "attack_detection", "validation_loss",
    )
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for axis, name in zip(axes.flat, columns):
        for seed, group in history.groupby(history.seed.astype(int), sort=True):
            axis.plot(group.epoch, group[name], marker="o", markersize=2, label=str(seed))
        axis.set(title=name.replace("_", " "), xlabel="Epoch", ylabel="Loss")
    axes.flat[0].legend(title="Seed")
    _save(figure, figures, "training_loss_curves")


def _plot_clean_selection(history: pd.DataFrame, inventory: pd.DataFrame, figures: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    selected = inventory.set_index("seed")["selected_epoch"].to_dict()
    for seed, group in history.groupby(history.seed.astype(int), sort=True):
        axis.plot(group.epoch, group.validation_clean_macro_f1, marker="o", markersize=3, label=str(seed))
        row = group.loc[group.epoch.astype(int).eq(selected[int(seed)])].iloc[0]
        axis.scatter([row.epoch], [row.validation_clean_macro_f1], s=55, zorder=4)
    axis.set(title="Calibration-clean Macro F1 and selected epochs", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    axis.legend(title="Seed")
    _save(figure, figures, "validation_clean_macro_f1")


def _plot_selected_regions(selected: pd.DataFrame, regions: tuple[str, ...], figures: Path) -> None:
    aggregate = selected.groupby(["condition", "region"], observed=True)[list(METRICS)].mean().reset_index()
    figure, axes = plt.subplots(len(METRICS), 1, figsize=(16, 13), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        matrix = aggregate.pivot(index="condition", columns="region", values=metric).loc[list(CONDITIONS), list(regions)]
        image = axis.imshow(matrix.to_numpy(float), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_yticks(range(len(CONDITIONS)), labels=CONDITIONS)
        axis.set_xticks(range(len(regions)), labels=regions, rotation=45, ha="right")
        axis.set_title(f"Selected-checkpoint regional {metric.replace('_', ' ')} (mean of 3 seeds)")
        figure.colorbar(image, ax=axis, fraction=.015, pad=.01)
    figure.savefig(figures / "selected_checkpoint_regional_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(stage08: str | Path, stage08_status: str | Path, output: str | Path) -> None:
    stage08 = Path(stage08)
    stage08_status = Path(stage08_status)
    output = Path(output)
    _require(stage08.is_dir(), f"Stage08 artifact directory absent: {stage08}")
    _require(not output.exists(), f"refusing to overwrite report output: {output}")
    metadata, history, regional, inventory = _load_evidence(stage08, stage08_status)
    regions = tuple(metadata["regions"])
    selected_epochs = inventory.set_index("seed")["selected_epoch"].to_dict()
    selected = regional.loc[
        regional.apply(lambda row: int(row["epoch"]) == selected_epochs[int(row["seed"])], axis=1)
    ].copy()
    summary = selected.groupby(["condition", "region"], observed=True)[
        [*METRICS, "average_precision"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(map(str, value)).rstrip("_") if isinstance(value, tuple) else str(value)
        for value in summary.columns
    ]

    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    history.to_csv(tables / "training_history.csv", index=False)
    regional.to_csv(tables / "regional_validation_all_epochs.csv", index=False)
    inventory.to_csv(tables / "checkpoint_inventory.csv", index=False)
    selected.to_csv(tables / "selected_checkpoint_regional_metrics.csv", index=False)
    summary.to_csv(tables / "selected_checkpoint_regional_summary.csv", index=False)
    _plot_losses(history, figures)
    _plot_clean_selection(history, inventory, figures)
    _plot_selected_regions(selected, regions, figures)

    source_hashes = {
        str(path): sha256(path)
        for path in sorted(stage08.rglob("*"))
        if path.is_file()
    }
    report_metadata = {
        "status": "generated_pending_independent_review",
        "scope": "Stage08 terminal train/calibration evidence only",
        "model_seeds": list(SEEDS),
        "epochs_per_seed": len(EPOCHS),
        "conditions": list(CONDITIONS),
        "regions": list(regions),
        "regional_rows": len(regional),
        "selected_regional_rows": len(selected),
        "checkpoint_cpu_payloads_loaded": len(inventory),
        "five_requested_metrics": list(METRICS),
        "attack_validation_semantics": (
            "fixed per-seed balanced primary-eligible calibration attack sample of 1024 rows; "
            "not 32-condition downstream evaluation"
        ),
        "clean_validation_semantics": "1693 calibration POIs used for checkpoint selection",
        "test_rows_read": 0,
        "downstream_claim": False,
        "stage08_status": {
            "path": str(stage08_status),
            "sha256": sha256(stage08_status),
        },
        "input_sha256": source_hashes,
    }
    (output / "metadata.json").write_text(json.dumps(report_metadata, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# Proposed structural defense training report\n\n"
        "This report summarizes the terminal Stage 08 training and calibration artifacts for "
        "seeds 42, 202, and 340. It verifies and loads each selected checkpoint payload on CPU, "
        "preserves all 2,040 regional epoch rows, and reports the selected-epoch 102-row "
        "seed/condition/region grid.\n\n"
        "The clean rows are the 1,693-POI calibration partition used for checkpoint selection. "
        "The attack rows are the fixed 1,024-row balanced primary-eligible calibration sample "
        "used during training; they are not a 32-condition Stage 10 evaluation. These outputs do "
        "not compare the proposed detector with URE-RF, do not establish a downstream advantage, "
        "and do not read the locked test split. Independent review is pending.\n"
    )
    marker = {
        "status": "GENERATED_PENDING_INDEPENDENT_REVIEW",
        "metadata_sha256": sha256(output / "metadata.json"),
        "checkpoint_inventory_sha256": sha256(tables / "checkpoint_inventory.csv"),
        "regional_validation_sha256": sha256(tables / "regional_validation_all_epochs.csv"),
        "selected_regional_sha256": sha256(tables / "selected_checkpoint_regional_metrics.csv"),
        "test_rows_read": 0,
    }
    (output / "_GENERATED.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage08", required=True)
    parser.add_argument("--stage08-status", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.stage08, args.stage08_status, args.output)


if __name__ == "__main__":
    main()
