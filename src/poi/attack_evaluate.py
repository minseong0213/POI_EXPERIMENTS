"""Evaluate the clean-validation winner on every constrained attack condition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
import yaml

from .data import CATEGORICAL, FEATURES, NUMERIC, sha256

METRICS = ["f1", "precision", "recall", "accuracy", "roc_auc", "average_precision"]
EXPECTED_MODELS = {"decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
                   "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_clean(data_dir):
    data_dir = Path(data_dir)
    report = json.loads((data_dir / "report.json").read_text())
    for name in ["poi_data_region.csv", "sample_manifest.csv"]:
        if sha256(data_dir / name) != report["outputs"][name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    clean = pd.read_csv(data_dir / "poi_data_region.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(data_dir / "sample_manifest.csv", dtype=str, keep_default_na=False)
    clean = clean.set_index("POI_ID").loc[manifest.POI_ID].reset_index()
    clean["split"] = manifest.split.to_numpy()
    return clean, manifest


def _attack_frame(rows, clean_order):
    rows = rows.set_index("POI_ID").loc[clean_order].reset_index()
    frame = rows[["POI_ID", "region", "split"]].copy()
    for feature in FEATURES:
        values = rows[f"{feature}_adv"]
        if feature in CATEGORICAL:
            values = pd.to_numeric(values, errors="raise").astype("float64").astype(str)
        frame[feature] = values.to_numpy()
    return frame


def _matrix(frame):
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model input")
    return values


def _metrics(truth, score, threshold):
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {"f1": f1_score(truth, prediction, zero_division=0),
            "precision": precision_score(truth, prediction, zero_division=0),
            "recall": recall_score(truth, prediction, zero_division=0),
            "accuracy": accuracy_score(truth, prediction),
            "roc_auc": roc_auc_score(truth, score),
            "average_precision": average_precision_score(truth, score),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _atomic_frame(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _condition_frames(clean_validation, attacks):
    frames = {"clean": clean_validation.reset_index(drop=True)}
    order = clean_validation.POI_ID.tolist()
    for condition, rows in attacks.groupby("attack_condition", sort=True):
        frames[condition] = _attack_frame(rows, order)
    return frames


def _tree_scores(model_id, cfg, train, frames, regions, seed):
    from .benchmark import estimator, preprocessor
    params = yaml.safe_load(Path(cfg["tree_config"]).read_text())["models"][model_id]
    transform = preprocessor()
    train_x = transform.fit_transform(train)
    eval_x = {name: transform.transform(frame) for name, frame in frames.items()}
    result = {}
    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        positive, negative = truth.sum(), len(truth) - truth.sum()
        weights = np.where(truth == 1, len(truth) / (2 * positive), len(truth) / (2 * negative))
        model = estimator(model_id, params, seed)
        model.fit(train_x, truth, sample_weight=weights)
        result[region] = {name: model.predict_proba(values)[:, 1]
                          for name, values in eval_x.items()}
    return result


def _tabpfn_context(model_id, cfg, train, seed):
    if not os.environ.get("TABPFN_TOKEN"):
        token_path = Path(os.environ.get("TABPFN_TOKEN_FILE", "/root/.config/poi/tabpfn_token"))
        if token_path.is_file():
            os.environ["TABPFN_TOKEN"] = token_path.read_text().strip()
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN or TABPFN_TOKEN_FILE is required")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    import torch
    versions = {"tabpfn_v2_5": ModelVersion.V2_5, "tabpfn_v2_6": ModelVersion.V2_6,
                "tabpfn_v3": ModelVersion.V3}
    model = TabPFNClassifier.create_default_for_version(
        versions[model_id], device=cfg["device"], n_estimators=cfg["tabpfn_n_estimators"],
        random_state=seed, categorical_features_indices=[2, 3, 4], show_progress_bar=True,
        memory_saving_mode="auto")
    return model, _matrix(train), torch


def _tabpfn_frame_scores(model, train_x, torch, cfg, train, frame, regions):
    frame_x = _matrix(frame)
    result = {}
    for region_start in range(0, len(regions), cfg["tabpfn_region_batch_size"]):
        region_batch = regions[region_start:region_start + cfg["tabpfn_region_batch_size"]]
        labels = [train.region.eq(region).astype(int).to_numpy() for region in region_batch]
        probabilities = model.predict_proba_batched(
            [train_x] * len(region_batch), labels, [frame_x] * len(region_batch))
        for region_index, region in enumerate(region_batch):
            result[region] = np.asarray(probabilities[region_index])[:, 1]
        torch.cuda.empty_cache()
    return result


def _select_model(path):
    ranking = pd.read_csv(path)
    if set(ranking.model) != EXPECTED_MODELS or ranking.model.duplicated().any():
        raise ValueError("Complete eight-model clean ranking is required")
    if ranking["mean"].isna().any():
        raise ValueError("Ranking has missing clean F1")
    return ranking.sort_values("mean", ascending=False).iloc[0].model


def _plot(metrics, predictions, tables, figures, representative):
    summary = metrics.groupby("condition")[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, value)) for value in summary.columns]
    clean = summary.loc[summary.condition.eq("clean")].iloc[0]
    for metric in METRICS:
        summary[f"{metric}_drop_from_clean"] = clean[f"{metric}_mean"] - summary[f"{metric}_mean"]
    summary.to_csv(tables / "attack_performance_summary.csv", index=False)
    region = metrics.groupby(["condition", "region"]).f1.mean().reset_index()
    region.pivot(index="condition", columns="region", values="f1").to_csv(
        tables / "f1_by_condition_region.csv")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(region.pivot(index="condition", columns="region", values="f1"),
                vmin=0, vmax=1, cmap="viridis", ax=ax)
    ax.set_title("OvR F1 by attack condition and region")
    fig.savefig(figures / "f1_heatmap_all_attacks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    selected = summary.loc[summary.condition.isin(["clean", representative])]
    long = selected.melt(id_vars="condition", value_vars=[f"{m}_mean" for m in METRICS],
                         var_name="metric", value_name="value")
    long["metric"] = long.metric.str.removesuffix("_mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="metric", y="value", hue="condition", ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Clean vs representative attack performance")
    fig.savefig(figures / "clean_vs_representative_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pred = predictions.loc[predictions.seed.eq(predictions.seed.min()) &
                            predictions.condition.isin(["clean", representative])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for condition, group in pred.groupby("condition"):
        fpr, tpr, _ = roc_curve(group.y_true, group.y_score)
        precision, recall, _ = precision_recall_curve(group.y_true, group.y_score)
        axes[0].plot(fpr, tpr, label=f"{condition} AUC={roc_auc_score(group.y_true, group.y_score):.4f}")
        axes[1].plot(recall, precision,
                     label=f"{condition} AP={average_precision_score(group.y_true, group.y_score):.4f}")
    representative_rows = pred.loc[pred.condition.eq(representative)]
    matrix = confusion_matrix(representative_rows.y_true, representative_rows.y_pred, labels=[0, 1])
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    sns.heatmap(matrix, annot=True, fmt=".4f", vmin=0, vmax=1, ax=axes[2])
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR")
    axes[2].set(xlabel="Predicted", ylabel="Actual", title=f"Confusion — {representative}")
    axes[0].legend(); axes[1].legend()
    fig.savefig(figures / "representative_confusion_pr_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def run(config_path, output_override=None):
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    output = Path(output_override or cfg["output"])
    if (output / "_SUCCESS.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed {output}")
    ranking_path = os.environ.get("SELECTION_RANKING", cfg["selection_ranking"])
    model_id = _select_model(ranking_path)
    clean, manifest = _load_clean(os.environ.get("DATA_DIR", cfg["data_dir"]))
    train = clean.loc[clean.split.eq("train")].reset_index(drop=True)
    validation = clean.loc[clean.split.eq("validation")].reset_index(drop=True)
    attacks_path = Path(os.environ.get("ATTACKS_FILE", cfg["attacks_file"]))
    attacks = pd.read_parquet(attacks_path)
    attacks = attacks.loc[attacks.split.eq("validation")].reset_index(drop=True)
    if attacks.attack_condition.nunique() != 32 or (~attacks.constraints_valid.astype(bool)).any():
        raise ValueError("Expected 32 fully valid validation attack conditions")
    frames = _condition_frames(validation, attacks)
    regions = sorted(manifest.region.unique())
    work = output / "work" / "shards"
    work.mkdir(parents=True, exist_ok=True)
    all_metrics, all_predictions = [], []
    for seed in cfg["seeds"]:
        if model_id.startswith("tabpfn_"):
            model, train_x, torch = _tabpfn_context(model_id, cfg, train, seed)
            for condition, frame in frames.items():
                safe_condition = condition.replace("/", "_")
                metric_path = work / f"seed{seed}-{safe_condition}.csv"
                prediction_path = work / f"seed{seed}-{safe_condition}.parquet"
                if metric_path.exists() and prediction_path.exists():
                    all_metrics.append(pd.read_csv(metric_path))
                    all_predictions.append(pd.read_parquet(prediction_path))
                    print(f"resumed attack evaluation seed={seed} condition={condition}", flush=True)
                    continue
                if metric_path.exists() or prediction_path.exists():
                    raise RuntimeError(f"Incomplete condition shard seed={seed} condition={condition}")
                condition_scores = _tabpfn_frame_scores(
                    model, train_x, torch, cfg, train, frame, regions)
                metric_rows, prediction_rows = [], []
                for region in regions:
                    truth = frame.region.eq(region).astype(int).to_numpy()
                    score = condition_scores[region]
                    metric_rows.append({"model": model_id, "seed": seed, "region": region,
                                        "condition": condition, "split": "validation",
                                        **_metrics(truth, score, cfg["threshold"])})
                    prediction_rows.append(pd.DataFrame({
                        "POI_ID": frame.POI_ID, "model": model_id, "seed": seed,
                        "region": region, "condition": condition, "split": "validation",
                        "y_true": truth, "y_score": score,
                        "y_pred": (score >= cfg["threshold"]).astype(int),
                    }))
                condition_metrics = pd.DataFrame(metric_rows)
                condition_predictions = pd.concat(prediction_rows, ignore_index=True)
                _atomic_frame(condition_metrics, metric_path)
                _atomic_frame(condition_predictions, prediction_path)
                all_metrics.append(condition_metrics); all_predictions.append(condition_predictions)
                print(f"completed attack evaluation seed={seed} condition={condition}", flush=True)
            continue
        metric_path, prediction_path = work / f"seed{seed}.csv", work / f"seed{seed}.parquet"
        if metric_path.exists() and prediction_path.exists():
            all_metrics.append(pd.read_csv(metric_path)); all_predictions.append(pd.read_parquet(prediction_path))
            continue
        if metric_path.exists() or prediction_path.exists():
            raise RuntimeError(f"Incomplete seed shard {seed}")
        scores = _tree_scores(model_id, cfg, train, frames, regions, seed)
        metric_rows, prediction_rows = [], []
        for region in regions:
            for condition, frame in frames.items():
                truth = frame.region.eq(region).astype(int).to_numpy()
                score = scores[region][condition]
                metric_rows.append({"model": model_id, "seed": seed, "region": region,
                                    "condition": condition, "split": "validation",
                                    **_metrics(truth, score, cfg["threshold"])})
                prediction_rows.append(pd.DataFrame({
                    "POI_ID": frame.POI_ID, "model": model_id, "seed": seed,
                    "region": region, "condition": condition, "split": "validation",
                    "y_true": truth, "y_score": score,
                    "y_pred": (score >= cfg["threshold"]).astype(int),
                }))
        seed_metrics = pd.DataFrame(metric_rows)
        seed_predictions = pd.concat(prediction_rows, ignore_index=True)
        _atomic_frame(seed_metrics, metric_path); _atomic_frame(seed_predictions, prediction_path)
        all_metrics.append(seed_metrics); all_predictions.append(seed_predictions)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    expected_metrics = len(cfg["seeds"]) * 17 * 33
    expected_predictions = expected_metrics * len(validation)
    if len(metrics) != expected_metrics or len(predictions) != expected_predictions:
        raise RuntimeError("Attack evaluation cardinality is incomplete")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    metrics.to_csv(tables / "metrics_by_region.csv", index=False)
    predictions.to_parquet(tables / "all_predictions.parquet", index=False)
    summary = _plot(metrics, predictions, tables, figures, cfg["representative_attack"])
    metadata = {"status": "complete", "model": model_id, "scope": "validation only",
                "test_used": False, "seeds": cfg["seeds"], "regions": 17,
                "attack_conditions": 32, "metric_rows": len(metrics),
                "prediction_rows": len(predictions), "selection_ranking_sha256": _hash(ranking_path),
                "attacks_sha256": _hash(attacks_path), "python": platform.python_version(),
                "seconds": time.time() - started}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(
        {"summary": summary.to_dict(orient="records")}, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    worst = summary.sort_values("f1_mean").iloc[0]
    (output / "report.md").write_text(
        f"# 선택 모델의 적대적 공격 성능\n\nclean validation 1위 `{model_id}`를 clean train으로 "
        f"학습하고 32개 검증 공격 전체를 평가했다. test split은 사용하지 않았다.\n\n"
        f"Macro OvR F1이 가장 낮은 조건은 `{worst.condition}`이며 {worst.f1_mean:.6f}이다. "
        "전체 F1·precision·recall·accuracy·ROC-AUC·AP와 clean 대비 감소량은 tables에, "
        "confusion matrix·PR·ROC 및 전 조건 heatmap은 300dpi PNG로 저장했다.\n")
    (output / "_SUCCESS.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/06_attack_evaluation.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
