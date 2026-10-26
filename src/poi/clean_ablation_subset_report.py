"""Audit and summarize the completed TabPFN 2.5 clean feature-ablation subset.

This is an assembly/reporting command.  It never fits a model, never reads attack
predictions, and asks ``load_clean_splits`` to materialize validation features only.
Prediction shards are processed one at a time so the 1.39M rows are not concatenated.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import load_clean_splits


MODEL = "tabpfn_v2_5"
CONDITION = "clean"
SEEDS = (42, 202, 340)
METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc")
EXPECTED_FEATURE_SETS = (
    "full",
    "no_coordinates",
    "coordinates_only",
    "drop_x_coord",
    "drop_y_coord",
    "drop_lclass",
    "drop_mclass",
    "drop_sclass",
)
SOURCE_ENTRIES = (
    "AGENTS.md",
    ".codex/agents/poi-experiment-runner.toml",
    "docs/experiments/independent_review_protocol.md",
    "docs/experiments/07_backbone_ablation.md",
    "docs/experiments/2026-09-15_full_robustness_experiment_plan.md",
    "configs/stages/07_clean_feature_ablation.yaml",
    "src/poi/feature_ablation.py",
    "src/poi/data.py",
    "src/poi/selected_explain.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values) -> str:
    """Hash a string sequence with unambiguous lengths and order."""
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def git_blob(repo: Path, commit: str, name: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{name}"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ValueError(
            f"Cannot read {name!r} at source commit {commit}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def ast_function_hash(source: bytes, names: tuple[str, ...]) -> str:
    """Hash selected function ASTs to compare loader logic despite unrelated edits."""
    tree = ast.parse(source.decode("utf-8"))
    found = {node.name: ast.dump(node, include_attributes=False)
             for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    if set(found) != set(names):
        raise ValueError(f"Missing loader functions: {sorted(set(names) - set(found))}")
    return sha256_bytes(json.dumps(found, sort_keys=True).encode("utf-8"))


def marker_fingerprint(cfg: dict, clean_hash: str, manifest_hash: str,
                       feature_code_hash: str, factory_hash: str) -> str:
    payload = {
        "config": cfg,
        "clean_sha256": clean_hash,
        "manifest_sha256": manifest_hash,
        "attack_sha256": None,
        "code_sha256": feature_code_hash,
        "factory_sha256": factory_hash,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def load_executed_contract(repo: Path, source_tar: Path, deployed_commit_path: Path,
                           clean_hash: str, manifest_hash: str,
                           observed_fingerprint: str) -> tuple[dict, dict, pd.DataFrame, dict]:
    commit = deployed_commit_path.read_text().strip()
    if commit != "0eee797940c1bc42eb3860287f149404075d9c0a":
        raise ValueError(f"Unexpected deployed source commit: {commit}")

    rows = []
    contents: dict[str, bytes] = {}
    with tarfile.open(source_tar, "r") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        missing = set(SOURCE_ENTRIES) - set(members)
        if missing:
            raise ValueError(f"Source archive is missing required entries: {sorted(missing)}")
        for name in SOURCE_ENTRIES:
            extracted = archive.extractfile(members[name])
            if extracted is None:
                raise ValueError(f"Could not extract source archive entry: {name}")
            archived = extracted.read()
            committed = git_blob(repo, commit, name)
            archive_hash = sha256_bytes(archived)
            commit_hash = sha256_bytes(committed)
            if archive_hash != commit_hash:
                raise ValueError(f"Source archive differs from commit for {name}")
            contents[name] = archived
            rows.append({
                "source_entry": name,
                "bytes": len(archived),
                "archive_sha256": archive_hash,
                "commit_sha256": commit_hash,
                "matches_deployed_commit": True,
            })

    cfg_path = "configs/stages/07_clean_feature_ablation.yaml"
    archived_cfg = yaml.safe_load(contents[cfg_path])
    feature_sets = archived_cfg.get("feature_sets", {})
    if tuple(feature_sets) != EXPECTED_FEATURE_SETS:
        raise ValueError(
            f"Executed feature-set order/content keys differ: {tuple(feature_sets)}"
        )
    if tuple(archived_cfg.get("seeds", ())) != SEEDS:
        raise ValueError("Executed seeds differ from the requested frozen seeds")
    if archived_cfg.get("threshold") != 0.5:
        raise ValueError("Executed threshold differs from 0.5")
    if MODEL not in archived_cfg.get("model_ids", ()):
        raise ValueError(f"Executed config did not request {MODEL}")

    feature_code_hash = sha256_bytes(contents["src/poi/feature_ablation.py"])
    factory_hash = sha256_bytes(contents["src/poi/selected_explain.py"])
    run_id = source_tar.parent.name
    candidates = []
    base = json.loads(json.dumps(archived_cfg))
    candidates.append(("archived_yaml_without_environment_overrides", base, {}))
    remote = json.loads(json.dumps(archived_cfg))
    remote_data = "/workspace/data/poi-34k"
    remote_result = f"/workspace/results/poi/{run_id}"
    remote["data_dir"] = remote_data
    remote["output"] = str(Path(remote_result) / "reports" / "07_clean_feature_ablation")
    candidates.append((
        "orchestrator_environment_reconstruction",
        remote,
        {"DATA_DIR": remote_data, "RESULT_DIR": remote_result},
    ))
    matches = []
    for label, candidate, overrides in candidates:
        fingerprint = marker_fingerprint(
            candidate, clean_hash, manifest_hash, feature_code_hash, factory_hash
        )
        if fingerprint == observed_fingerprint:
            matches.append((label, candidate, overrides, fingerprint))
    if len(matches) != 1:
        raise ValueError(
            "Marker fingerprint did not uniquely identify the archived/effective executed config; "
            f"matches={len(matches)}"
        )
    label, effective_cfg, overrides, fingerprint = matches[0]
    reconstruction = {
        "method": label,
        "environment_overrides": overrides,
        "observed_marker_fingerprint": observed_fingerprint,
        "recomputed_marker_fingerprint": fingerprint,
        "unique_candidate_match": True,
        "note": (
            "Environment values are reconstructed from the run ID and accepted only because "
            "they uniquely reproduce the common marker fingerprint."
        ),
    }
    return archived_cfg, effective_cfg, pd.DataFrame(rows), {
        "commit": commit,
        "contents": contents,
        "reconstruction": reconstruction,
    }


def expected_paths(input_root: Path, feature_sets: dict[str, list[str]], regions: list[str]):
    expected = []
    for ablation in feature_sets:
        for seed in SEEDS:
            for region in regions:
                stem = f"{MODEL}_{ablation}_{seed}_{region}"
                expected.append({
                    "ablation": ablation,
                    "seed": seed,
                    "region": region,
                    "marker": input_root / "work" / f"{stem}.json",
                    "prediction": input_root / "predictions" / f"{stem}_{CONDITION}.parquet",
                })
    return expected


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def inspect_png(path: Path) -> dict:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (None, None))
        result = {
            "path": str(path),
            "format": image.format,
            "width_px": image.width,
            "height_px": image.height,
            "dpi_x": None if dpi[0] is None else float(dpi[0]),
            "dpi_y": None if dpi[1] is None else float(dpi[1]),
            "sha256": sha256_file(path),
        }
    if result["format"] != "PNG":
        raise ValueError(f"Figure is not PNG: {path}")
    if result["dpi_x"] is None or result["dpi_y"] is None:
        raise ValueError(f"Figure has no embedded DPI metadata: {path}")
    if min(result["dpi_x"], result["dpi_y"]) < 299:
        raise ValueError(f"Figure is below 300dpi (allowing PNG rounding): {path}")
    return result


def annotated_heatmap(values: pd.DataFrame, path: Path, title: str,
                      label: str, diverging: bool = False) -> None:
    matrix = values.to_numpy(float)
    fig, axis = plt.subplots(figsize=(19, 7.5))
    if diverging:
        limit = max(float(np.abs(matrix).max()), 1e-12)
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(values.columns)), values.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(values.index)), values.index)
    axis.set_xlabel("OvR region")
    axis.set_ylabel("Feature set")
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.9)
    colorbar.set_label(label)
    threshold = np.nanmedian(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if diverging:
                color = "white" if abs(value) > 0.55 * max(np.abs(matrix).max(), 1e-12) else "black"
            else:
                color = "white" if value < threshold else "black"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                      fontsize=6.5, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def point_comparison(macro_by_seed: pd.DataFrame, summary: pd.DataFrame,
                     path: Path, deltas: bool = False) -> None:
    feature_order = list(EXPECTED_FEATURE_SETS)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(24, 8), sharey=True)
    colors = {42: "#1b9e77", 202: "#d95f02", 340: "#7570b3"}
    y = np.arange(len(feature_order))
    for axis, metric in zip(axes, METRICS):
        value_col = f"delta_{metric}_vs_full" if deltas else metric
        mean_col = f"{metric}_delta_vs_full_mean" if deltas else f"{metric}_mean"
        sd_col = f"{metric}_delta_vs_full_sd" if deltas else f"{metric}_sd"
        for seed in SEEDS:
            part = macro_by_seed.loc[macro_by_seed.seed.eq(seed)].set_index("ablation").loc[feature_order]
            axis.scatter(part[value_col], y, s=28, alpha=0.8, color=colors[seed], label=str(seed))
        ordered = summary.set_index("ablation").loc[feature_order]
        axis.errorbar(ordered[mean_col], y, xerr=ordered[sd_col], fmt="D", markersize=5,
                      color="black", ecolor="black", capsize=3, label="mean +/- sample SD")
        if deltas:
            axis.axvline(0, color="gray", linewidth=1, linestyle="--")
            axis.set_xlabel(f"Delta macro {metric}\n(feature set - full)")
        else:
            axis.set_xlabel(f"Macro {metric}")
            axis.set_xlim(-0.02, 1.02)
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(y, feature_order)
        axis.invert_yaxis()
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    title = ("Paired full-feature differences by model seed" if deltas else
             "Clean validation macro metrics by model seed")
    fig.suptitle(title + " | error bars are sample SD, not confidence intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def build_report(summary: pd.DataFrame, metadata: dict) -> str:
    display = summary[[
        "ablation", "f1_mean", "f1_sd", "f1_delta_vs_full_mean",
        "roc_auc_mean", "roc_auc_sd", "roc_auc_delta_vs_full_mean",
    ]].copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    table = display.to_markdown(index=False)
    return f"""# TabPFN 2.5 clean feature-ablation interim report

## Status and scope

`SUBSET_GENERATED_PENDING_INDEPENDENT_REVIEW`. This report covers only the completed
TabPFN 2.5 clean-validation subset. The full feature-ablation stage is **NOT COMPLETE**:
TabPFN v3 belongs to the ongoing GPU A work and is not read or summarized here. No locked-test
features or predictions and no attack predictions were materialized.

The source grid contains {metadata['feature_set_count']} feature sets x
{metadata['seed_count']} model seeds x {metadata['region_count']} OvR regions =
{metadata['stratum_count']} marker/prediction pairs and {metadata['prediction_rows']:,}
prediction rows. Each shard contains the same {metadata['validation_pois']:,} validation POIs in
the same order, with truth reconstructed from the validation region labels.

## Method

The effective executed configuration was reconstructed from the archived YAML in `source.tar` and
the run-specific environment paths; the reconstruction is accepted because it uniquely reproduces
the fingerprint shared by all {metadata['marker_count']} markers. Required archived sources match
deployed commit `{metadata['deployed_commit']}` byte-for-byte. Every marker and prediction file was
hashed before use and again after aggregation. Each marker-declared prediction SHA-256 matched.

F1, precision, recall, accuracy, and ROC-AUC were recomputed from each prediction shard using the
stored binary truth, score, and threshold-0.5 decision. Region macro values are computed separately
inside each model seed; the final mean and sample SD are then computed across the three seed macros.
Full-feature differences are paired on the exact region and seed before aggregation.

## Selected exact results

{table}

All five metrics, region/seed values, marker comparison errors, and paired full-feature differences
are in `tables/`. CSVs retain round-trippable numeric precision. Figure error bars are sample SD over
three model seeds; **SD is not a confidence interval**.

## Provenance limitations and remaining gaps

- Provenance is **PARTIAL**. GPU A did not persist per-fit checkpoint hashes, runtime attestations,
  training-matrix hashes, or reload/probe attestations in these {metadata['marker_count']} legacy
  markers. This CPU report does not invent them retroactively and does not refit the models.
- The original plan's 2,000-draw paired-bootstrap 95% confidence intervals are **NOT RUN** in this
  bounded interim report. The reported variability is seed sample SD only.
- Adversarial ablation, robust F1, TabPFN v3, the full selected-backbone conclusion, locked test,
  and independent review are **NOT RUN** for this artifact.
- Prior historical test-coordinate exposure remains a study-level limitation. This command made a
  validation-only `load_clean_splits(..., splits=('validation',))` call and materialized zero test
  feature rows.

Fresh independent review must inspect the archived source/config, all source hashes, prediction
identity checks, exact metric tables, and PNG metadata before any completion claim.
"""


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    repo = args.repo.resolve()
    input_root = args.input_root.resolve()
    source_tar = args.source_tar.resolve()
    deployed_commit_path = args.deployed_commit.resolve()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report artifact: {output}")
    for path in (input_root, source_tar, deployed_commit_path, data_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    clean_path = data_dir / "poi_data_region.csv"
    manifest_path = data_dir / "sample_manifest.csv"
    initial_fixed_hashes = {
        str(source_tar): sha256_file(source_tar),
        str(deployed_commit_path): sha256_file(deployed_commit_path),
        str(clean_path): sha256_file(clean_path),
        str(manifest_path): sha256_file(manifest_path),
    }

    validation, validation_manifest = load_clean_splits(
        data_dir, splits=("validation",)
    )
    if len(validation) != 3400 or validation.region.nunique() != 17:
        raise ValueError("Frozen validation cardinality differs from 3,400 POIs / 17 regions")
    if set(validation.split) != {"validation"} or set(validation_manifest.split) != {"validation"}:
        raise ValueError("Validation-only loader returned a different split")
    if validation.POI_ID.duplicated().any():
        raise ValueError("Validation POI identity is not unique")
    regions = sorted(validation.region.unique())
    poi_order = validation.POI_ID.astype(str).to_numpy()
    poi_order_hash = sequence_sha256(poi_order)
    validation_region = validation.region.astype(str).to_numpy()

    marker_paths = sorted((input_root / "work").glob(f"{MODEL}_*.json"))
    prediction_paths = sorted((input_root / "predictions").glob(f"{MODEL}_*.parquet"))
    if not marker_paths:
        raise ValueError("No TabPFN 2.5 markers found")
    marker_fingerprints = {json.loads(path.read_text())["fingerprint"] for path in marker_paths}
    if len(marker_fingerprints) != 1:
        raise ValueError(f"Markers do not have one common fingerprint: {len(marker_fingerprints)}")
    observed_fingerprint = next(iter(marker_fingerprints))

    archived_cfg, effective_cfg, source_rows, source_info = load_executed_contract(
        repo, source_tar, deployed_commit_path,
        initial_fixed_hashes[str(clean_path)], initial_fixed_hashes[str(manifest_path)],
        observed_fingerprint,
    )
    feature_sets = effective_cfg["feature_sets"]
    expected = expected_paths(input_root, feature_sets, regions)
    expected_markers = {item["marker"] for item in expected}
    expected_predictions = {item["prediction"] for item in expected}
    if set(marker_paths) != expected_markers:
        missing = sorted(str(path) for path in expected_markers - set(marker_paths))
        extra = sorted(str(path) for path in set(marker_paths) - expected_markers)
        raise ValueError(f"Marker grid mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if set(prediction_paths) != expected_predictions:
        missing = sorted(str(path) for path in expected_predictions - set(prediction_paths))
        extra = sorted(str(path) for path in set(prediction_paths) - expected_predictions)
        raise ValueError(f"Prediction grid mismatch; missing={missing[:5]}, extra={extra[:5]}")

    required_columns = {
        "model", "ablation", "seed", "region", "attack_condition", "features",
        "POI_ID", "y_true", "y_score", "y_pred",
    }
    metric_rows = []
    hash_rows = []
    identity_rows = []
    initial_input_hashes: dict[str, str] = {}
    max_marker_error = 0.0
    for index, item in enumerate(expected, start=1):
        ablation, seed, region = item["ablation"], item["seed"], item["region"]
        marker_path, prediction_path = item["marker"], item["prediction"]
        marker_hash = sha256_file(marker_path)
        prediction_hash = sha256_file(prediction_path)
        initial_input_hashes[str(marker_path)] = marker_hash
        initial_input_hashes[str(prediction_path)] = prediction_hash
        marker = json.loads(marker_path.read_text())
        if marker.get("fingerprint") != observed_fingerprint:
            raise ValueError(f"Marker fingerprint differs: {marker_path}")
        if len(marker.get("metrics", ())) != 1 or len(marker.get("files", {})) != 1:
            raise ValueError(f"Marker must bind exactly one clean prediction: {marker_path}")
        relative_prediction = str(prediction_path.relative_to(input_root))
        if marker["files"] != {relative_prediction: prediction_hash}:
            raise ValueError(f"Marker prediction hash/path mismatch: {marker_path}")
        marker_metric = marker["metrics"][0]
        expected_features = "|".join(feature_sets[ablation])
        identity = {
            "model": MODEL,
            "ablation": ablation,
            "seed": seed,
            "region": region,
            "attack_condition": CONDITION,
            "features": expected_features,
        }
        if any(marker_metric.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Marker identity/features differ from executed config: {marker_path}")

        # Sequential shard processing: this frame is released at the next iteration.
        frame = pd.read_parquet(prediction_path)
        if set(frame.columns) != required_columns:
            raise ValueError(
                f"Prediction schema differs for {prediction_path}: "
                f"missing={sorted(required_columns-set(frame.columns))}, "
                f"extra={sorted(set(frame.columns)-required_columns)}"
            )
        if len(frame) != len(validation) or frame.POI_ID.duplicated().any():
            raise ValueError(f"Prediction row/POI cardinality differs: {prediction_path}")
        for key, value in identity.items():
            if frame[key].nunique(dropna=False) != 1 or frame[key].iloc[0] != value:
                raise ValueError(f"Prediction {key} differs from grid/config: {prediction_path}")
        file_poi = frame.POI_ID.astype(str).to_numpy()
        if not np.array_equal(file_poi, poi_order):
            raise ValueError(f"Prediction POI identity/order differs: {prediction_path}")
        expected_truth = (validation_region == region).astype(np.int64)
        y_true = pd.to_numeric(frame.y_true, errors="raise").to_numpy(np.int64)
        if not np.array_equal(y_true, expected_truth):
            raise ValueError(f"Prediction truth differs from validation label: {prediction_path}")
        y_score = pd.to_numeric(frame.y_score, errors="raise").to_numpy(np.float64)
        y_pred = pd.to_numeric(frame.y_pred, errors="raise").to_numpy(np.int64)
        if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
            raise ValueError(f"Prediction score outside [0,1]: {prediction_path}")
        if not np.isin(y_pred, (0, 1)).all() or not np.array_equal(
                y_pred, (y_score >= effective_cfg["threshold"]).astype(np.int64)):
            raise ValueError(f"Stored prediction differs from thresholded score: {prediction_path}")

        recomputed = binary_metrics(y_true, y_score, y_pred)
        row = {**identity, "rows": len(frame), "positive_truth": int(y_true.sum()),
               "poi_order_sha256": sequence_sha256(file_poi),
               "truth_sha256": sequence_sha256(y_true)}
        for metric in METRICS:
            marker_value = float(marker_metric[metric])
            error = abs(recomputed[metric] - marker_value)
            max_marker_error = max(max_marker_error, error)
            row[metric] = recomputed[metric]
            row[f"marker_{metric}"] = marker_value
            row[f"marker_abs_error_{metric}"] = error
        metric_rows.append(row)
        hash_rows.extend((
            {
                "kind": "marker", "ablation": ablation, "seed": seed, "region": region,
                "path": str(marker_path), "bytes": marker_path.stat().st_size,
                "sha256": marker_hash, "marker_declared_sha256": "",
                "marker_hash_match": True,
            },
            {
                "kind": "prediction", "ablation": ablation, "seed": seed, "region": region,
                "path": str(prediction_path), "bytes": prediction_path.stat().st_size,
                "sha256": prediction_hash, "marker_declared_sha256": marker["files"][relative_prediction],
                "marker_hash_match": prediction_hash == marker["files"][relative_prediction],
            },
        ))
        if ablation == "full" and seed == SEEDS[0]:
            identity_rows.append({
                "region": region,
                "validation_rows": len(frame),
                "positive_truth": int(y_true.sum()),
                "poi_order_sha256": sequence_sha256(file_poi),
                "truth_sha256": sequence_sha256(y_true),
            })
        del frame, file_poi, expected_truth, y_true, y_score, y_pred
        if index % 51 == 0:
            print(json.dumps({"validated_strata": index, "expected_strata": len(expected)}),
                  flush=True)

    if max_marker_error > 1e-12:
        raise ValueError(f"Marker metric mismatch exceeds tolerance: {max_marker_error}")
    metrics = pd.DataFrame(metric_rows)
    if len(metrics) != 408 or int(metrics.rows.sum()) != 1_387_200:
        raise ValueError("Expected 408 strata and 1,387,200 prediction rows")
    if set(metrics.poi_order_sha256) != {poi_order_hash}:
        raise ValueError("Common validation POI-order hash check failed")

    full = metrics.loc[metrics.ablation.eq("full"), ["seed", "region", *METRICS]].copy()
    full = full.rename(columns={metric: f"full_{metric}" for metric in METRICS})
    paired = metrics.merge(full, on=["seed", "region"], validate="many_to_one")
    for metric in METRICS:
        paired[f"delta_{metric}_vs_full"] = paired[metric] - paired[f"full_{metric}"]
    paired_columns = [
        "model", "ablation", "seed", "region", "features", "rows", "positive_truth",
        *METRICS,
        *(f"full_{metric}" for metric in METRICS),
        *(f"delta_{metric}_vs_full" for metric in METRICS),
    ]
    paired = paired[paired_columns]
    macro_by_seed = paired.groupby(["model", "ablation", "seed"], sort=False)[
        [*METRICS, *(f"delta_{metric}_vs_full" for metric in METRICS)]
    ].mean().reset_index()
    macro_by_seed["region_count"] = len(regions)
    summary_rows = []
    for ablation in EXPECTED_FEATURE_SETS:
        part = macro_by_seed.loc[macro_by_seed.ablation.eq(ablation)]
        if tuple(sorted(part.seed)) != SEEDS:
            raise ValueError(f"Seed macro coverage differs for {ablation}")
        row = {"model": MODEL, "ablation": ablation, "seed_count": len(part),
               "region_count_per_seed": len(regions)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_sd"] = float(part[metric].std(ddof=1))
            delta = part[f"delta_{metric}_vs_full"]
            row[f"{metric}_delta_vs_full_mean"] = float(delta.mean())
            row[f"{metric}_delta_vs_full_sd"] = float(delta.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Re-hash every source input after all reads to catch concurrent replacement.
    final_input_hashes = {path: sha256_file(Path(path)) for path in initial_input_hashes}
    changed_inputs = [path for path, digest in initial_input_hashes.items()
                      if final_input_hashes[path] != digest]
    if changed_inputs:
        raise RuntimeError(f"Marker/prediction inputs changed during aggregation: {changed_inputs[:5]}")
    final_fixed_hashes = {path: sha256_file(Path(path)) for path in initial_fixed_hashes}
    if final_fixed_hashes != initial_fixed_hashes:
        raise RuntimeError("Source archive, deployed commit marker, or clean data changed during run")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    tables = staging / "tables"
    figures = staging / "figures"
    provenance = staging / "provenance"
    for directory in (tables, figures, provenance):
        directory.mkdir()

    metric_output_columns = [
        "model", "ablation", "seed", "region", "attack_condition", "features", "rows",
        "positive_truth", "poi_order_sha256", "truth_sha256",
        *METRICS,
        *(f"marker_{metric}" for metric in METRICS),
        *(f"marker_abs_error_{metric}" for metric in METRICS),
    ]
    write_csv(metrics[metric_output_columns], tables / "metrics_by_region_seed.csv")
    write_csv(paired, tables / "paired_full_differences_by_region_seed.csv")
    write_csv(macro_by_seed, tables / "region_macro_by_seed.csv")
    write_csv(summary, tables / "seed_mean_sd_summary.csv")
    write_csv(pd.DataFrame(identity_rows), tables / "validation_identity_by_region.csv")
    write_csv(pd.DataFrame(hash_rows), provenance / "input_source_hashes.csv")
    write_csv(source_rows, provenance / "source_archive_commit_matches.csv")
    (staging / "executed_config_archived.yaml").write_text(
        yaml.safe_dump(archived_cfg, sort_keys=False), encoding="utf-8"
    )
    (staging / "executed_config_effective_reconstruction.yaml").write_text(
        yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8"
    )
    (provenance / "execution_config_reconstruction.json").write_text(
        json.dumps(source_info["reconstruction"], indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(Path(__file__), provenance / Path(__file__).name)

    f1_region = metrics.groupby(["ablation", "region"], sort=False).f1.mean().unstack("region")
    f1_region = f1_region.loc[list(EXPECTED_FEATURE_SETS), regions]
    f1_delta = paired.groupby(["ablation", "region"], sort=False).delta_f1_vs_full.mean().unstack("region")
    f1_delta = f1_delta.loc[list(EXPECTED_FEATURE_SETS), regions]
    annotated_heatmap(
        f1_region, figures / "f1_region_seed_mean_heatmap.png",
        "TabPFN 2.5 clean validation F1 | regional mean over three model seeds", "F1",
    )
    annotated_heatmap(
        f1_delta, figures / "f1_paired_delta_vs_full_region_heatmap.png",
        "TabPFN 2.5 clean validation F1 difference | feature set - full | mean over seeds",
        "Paired F1 difference", diverging=True,
    )
    point_comparison(
        macro_by_seed, summary, figures / "macro_metric_point_comparisons.png", deltas=False
    )
    point_comparison(
        macro_by_seed, summary, figures / "macro_paired_delta_point_comparisons.png", deltas=True
    )
    figure_rows = []
    for path in sorted(figures.glob("*.png")):
        figure_row = inspect_png(path)
        # Persist a stable artifact-relative path, never the temporary staging directory.
        figure_row["path"] = str(path.relative_to(staging))
        figure_rows.append(figure_row)
    if list(staging.rglob("*.svg")):
        raise ValueError("SVG output is prohibited")
    write_csv(pd.DataFrame(figure_rows), tables / "figure_inventory.csv")

    loader_functions = (
        "sha256", "_verify_files", "_read_manifest", "_validate_feature_frame",
        "load_clean_splits",
    )
    current_data_path = repo / "src" / "poi" / "data.py"
    current_data_bytes = current_data_path.read_bytes()
    archived_data_bytes = source_info["contents"]["src/poi/data.py"]
    loader_ast_current = ast_function_hash(current_data_bytes, loader_functions)
    loader_ast_archived = ast_function_hash(archived_data_bytes, loader_functions)
    if loader_ast_current != loader_ast_archived:
        raise ValueError("Current validation loader logic differs from archived executed loader logic")

    metadata = {
        "status": "SUBSET_GENERATED_PENDING_INDEPENDENT_REVIEW",
        "scope": "TabPFN 2.5 clean feature-ablation subset only",
        "full_ablation_stage_status": "NOT_COMPLETE",
        "independent_review_status": "NOT_RUN",
        "provenance_status": "PARTIAL",
        "provenance_reason": (
            "All 408 legacy markers bind prediction hashes but contain zero per-fit checkpoint, "
            "runtime, train-matrix, reload, or probe attestations. No retrospective attestation added."
        ),
        "per_fit_checkpoint_attestations": 0,
        "per_fit_runtime_attestations": 0,
        "deployed_commit": source_info["commit"],
        "source_tar": str(source_tar),
        "source_tar_sha256": initial_fixed_hashes[str(source_tar)],
        "source_archive_entries_checked_against_commit": len(source_rows),
        "source_archive_all_checked_entries_match_commit": bool(
            source_rows.matches_deployed_commit.all()
        ),
        "executed_config_source": (
            f"{source_tar}:configs/stages/07_clean_feature_ablation.yaml"
        ),
        "execution_config_reconstruction": source_info["reconstruction"],
        "model": MODEL,
        "condition": CONDITION,
        "feature_sets": feature_sets,
        "feature_set_count": len(feature_sets),
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "regions": regions,
        "region_count": len(regions),
        "threshold": effective_cfg["threshold"],
        "stratum_count": len(metrics),
        "marker_count": len(marker_paths),
        "prediction_file_count": len(prediction_paths),
        "prediction_rows": int(metrics.rows.sum()),
        "rows_per_prediction_file": int(metrics.rows.min()),
        "validation_pois": len(validation),
        "validation_poi_order_sha256": poi_order_hash,
        "common_validation_poi_order": True,
        "validation_truth_matches_dataset_for_all_files": True,
        "marker_prediction_hash_matches": int(
            pd.DataFrame(hash_rows).loc[lambda frame: frame.kind.eq("prediction"),
                                        "marker_hash_match"].sum()
        ),
        "input_files_hashed_before_and_after": len(initial_input_hashes),
        "input_files_changed_during_run": 0,
        "maximum_marker_metric_absolute_error": max_marker_error,
        "marker_metric_tolerance": 1e-12,
        "aggregation": (
            "equal macro mean over 17 OvR regions separately per seed; mean and sample SD "
            "over three seed macros; paired differences matched on region and seed"
        ),
        "seed_variability": "sample SD (ddof=1), not a confidence interval",
        "paired_bootstrap_confidence_intervals": "NOT_RUN",
        "test_split_requested_from_loader": False,
        "test_feature_rows_materialized": 0,
        "attack_predictions_read": False,
        "validation_loader_call": "load_clean_splits(data_dir, splits=('validation',))",
        "current_data_source_sha256": sha256_file(current_data_path),
        "archived_data_source_sha256": sha256_bytes(archived_data_bytes),
        "validation_loader_ast_sha256": loader_ast_current,
        "current_and_archived_validation_loader_ast_match": True,
        "source_data_sha256": {
            "poi_data_region.csv": initial_fixed_hashes[str(clean_path)],
            "sample_manifest.csv": initial_fixed_hashes[str(manifest_path)],
        },
        "figure_count": len(figure_rows),
        "figures": figure_rows,
        "svg_count": 0,
        "cpu_aggregation_threads_requested": 1,
        "runtime_seconds": time.time() - started,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "Pillow")
        },
        "limitations": [
            "GPU A legacy fits lack per-fit checkpoint/runtime/train-matrix/probe attestations.",
            "No retroactive fit attestation or model refit is created.",
            "Seed SD is not a confidence interval; the planned 2000-draw paired bootstrap is not run.",
            "TabPFN v3 and adversarial ablation remain outside this interim artifact.",
            "Historical test-coordinate exposure is not erased; this command materialized validation only.",
        ],
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "report.md").write_text(build_report(summary, metadata), encoding="utf-8")
    pending = {
        "status": metadata["status"],
        "full_ablation_stage_status": metadata["full_ablation_stage_status"],
        "independent_review_status": metadata["independent_review_status"],
        "provenance_status": metadata["provenance_status"],
        "artifact": str(output),
    }
    (staging / "_GENERATED_PENDING_REVIEW.json").write_text(
        json.dumps(pending, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("artifacts/poi-remediation-20261001/reports/07_clean_feature_ablation"),
    )
    parser.add_argument(
        "--source-tar", type=Path,
        default=Path(
            "artifacts/poi-remediation-20261001/runtime/"
            "poi-clean-ablation-20261001-003/source.tar"
        ),
    )
    parser.add_argument(
        "--deployed-commit", type=Path,
        default=Path(
            "artifacts/poi-remediation-20261001/runtime/"
            "poi-clean-ablation-20261001-003/deployed_commit"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/poi_34k_seed42"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "artifacts/poi-remediation-20261001/reports/"
            "07_clean_ablation_tabpfn25_interim"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
