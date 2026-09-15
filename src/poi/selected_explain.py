"""Explain the clean-validation winner on clean and representative attack data."""
import argparse
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
import shap
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.multitest import multipletests
import yaml

from .attack_evaluate import EXPECTED_MODELS, _attack_frame, _load_clean
from .data import CATEGORICAL, FEATURES, NUMERIC


def _save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _matrix(frame, features=FEATURES):
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Explanation input contains non-finite values")
    return values


def _positions(frame, region, per_class, seed):
    positive = frame.index[frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    negative = frame.index[~frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    return np.concatenate([positive, negative])


def _beeswarm(records, figures, name, title):
    order = records.POI_ID.drop_duplicates().tolist()
    values = records.pivot(index="POI_ID", columns="feature", values="feature_value").loc[order, FEATURES]
    shap_values = records.pivot(index="POI_ID", columns="feature", values="shap_value").loc[order, FEATURES]
    base = records.groupby("POI_ID").base_value.first().loc[order].to_numpy()
    shap.summary_plot(shap_values.to_numpy(), values.to_numpy(), feature_names=FEATURES,
                      plot_type="dot", color_bar=True, color_bar_label="Feature value",
                      show=False, max_display=len(FEATURES))
    figure = plt.gcf(); axis = plt.gca()
    axis.axvline(0, color="#444444", linewidth=.8, zorder=0)
    axis.set_xlabel("SHAP value"); figure.suptitle(title)
    _save(figure, figures, name)
    return values.to_numpy(), shap_values.to_numpy(), base


def _odds_ratios(train, regions):
    numeric = train[NUMERIC].apply(pd.to_numeric, errors="raise").astype(float)
    numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    categorical = pd.get_dummies(train[CATEGORICAL], prefix=CATEGORICAL,
                                 drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([numeric, categorical], axis=1), has_constant="add")
    rows, failures = [], []
    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        try:
            fitted = sm.GLM(truth, design, family=sm.families.Binomial()).fit(
                maxiter=200, cov_type="HC3")
            for name, coefficient, standard_error, p_value in zip(
                    design.columns, fitted.params, fitted.bse, fitted.pvalues):
                identifiable = bool(np.isfinite(standard_error) and np.isfinite(p_value))
                rows.append({"region": region, "feature": name, "identifiable": identifiable,
                             "log_odds": coefficient,
                             "odds_ratio": np.exp(np.clip(coefficient, -50, 50)),
                             "ci_low": np.exp(np.clip(coefficient - 1.96 * standard_error, -50, 50)),
                             "ci_high": np.exp(np.clip(coefficient + 1.96 * standard_error, -50, 50)),
                             "p_value": p_value})
        except Exception as error:
            failures.append({"region": region, "error": repr(error)})
    result = pd.DataFrame(rows)
    if len(result):
        result["q_value"] = multipletests(result.p_value.fillna(1), method="fdr_bh")[1]
    return result, pd.DataFrame(failures)


class ModelFactory:
    def __init__(self, model_id, cfg, train):
        self.model_id, self.cfg, self.train = model_id, cfg, train
        self.tabpfn_runtime = {}

    def fit(self, features, truth):
        if self.model_id.startswith("tabpfn_"):
            if not os.environ.get("TABPFN_TOKEN"):
                token_path = Path(os.environ.get("TABPFN_TOKEN_FILE", "/root/.config/poi/tabpfn_token"))
                if token_path.is_file():
                    os.environ["TABPFN_TOKEN"] = token_path.read_text().strip()
            if not os.environ.get("TABPFN_TOKEN"):
                raise RuntimeError("TABPFN_TOKEN or TABPFN_TOKEN_FILE is required")
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            import tabpfn
            import torch
            versions = {"tabpfn_v2_5": ModelVersion.V2_5, "tabpfn_v2_6": ModelVersion.V2_6,
                        "tabpfn_v3": ModelVersion.V3}
            categorical = [index for index, feature in enumerate(features) if feature in CATEGORICAL]
            model = TabPFNClassifier.create_default_for_version(
                versions[self.model_id], device=self.cfg["device"],
                n_estimators=self.cfg["n_estimators"], random_state=self.cfg["seed"],
                categorical_features_indices=categorical, show_progress_bar=False,
                memory_saving_mode="auto")
            model.fit(_matrix(self.train, features), truth)
            self.tabpfn_runtime = {"tabpfn": tabpfn.__version__, "torch": torch.__version__,
                                   "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(0)}
            return lambda values: model.predict_proba(np.asarray(values, dtype=np.float32))[:, 1]
        from .ablation import transformer
        from .benchmark import estimator
        params = yaml.safe_load(Path(self.cfg["tree_config"]).read_text())["models"][self.model_id]
        transform = transformer(features)
        train_x = transform.fit_transform(self.train)
        positive, negative = truth.sum(), len(truth) - truth.sum()
        weights = np.where(truth == 1, len(truth) / (2 * positive), len(truth) / (2 * negative))
        model = estimator(self.model_id, params, self.cfg["seed"])
        model.fit(train_x, truth, sample_weight=weights)
        def predict(values):
            frame = pd.DataFrame(np.asarray(values), columns=features)
            for feature in NUMERIC:
                if feature in frame:
                    frame[feature] = pd.to_numeric(frame[feature], errors="raise")
            for feature in CATEGORICAL:
                if feature in frame:
                    frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float).astype(str)
            return model.predict_proba(transform.transform(frame))[:, 1]
        return predict


def _explain(predict, background, values, features, seed):
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation",
                              feature_names=features, seed=seed)
    return explainer(values, max_evals=2 * len(features) + 1, batch_size="auto", silent=True)


def _lime_arrays(train_values):
    encoded = train_values.copy()
    categorical_names, reverse = {}, {}
    for index in range(len(NUMERIC), len(FEATURES)):
        observed = np.unique(train_values[:, index])
        mapping = {value: ordinal for ordinal, value in enumerate(observed)}
        encoded[:, index] = np.array([mapping[value] for value in train_values[:, index]])
        categorical_names[index] = [str(value) for value in observed]
        reverse[index] = observed
    return encoded, categorical_names, reverse


def run(config_path, output_override=None):
    started = time.time(); cfg = yaml.safe_load(Path(config_path).read_text())
    output = Path(output_override or cfg["output"])
    if (output / "_SUCCESS.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed {output}")
    ranking = pd.read_csv(cfg["selection_ranking"])
    if set(ranking.model) != EXPECTED_MODELS or ranking.model.duplicated().any():
        raise ValueError("Complete eight-model clean ranking is required")
    model_id = ranking.sort_values("mean", ascending=False).iloc[0].model
    clean, manifest = _load_clean(os.environ.get("DATA_DIR", cfg["data_dir"]))
    train = clean.loc[clean.split.eq("train")].reset_index(drop=True)
    validation = clean.loc[clean.split.eq("validation")].reset_index(drop=True)
    attacks = pd.read_parquet(cfg["attacks_file"])
    attack_rows = attacks.loc[attacks.split.eq("validation") &
                              attacks.attack_condition.eq(cfg["representative_attack"])]
    attack = _attack_frame(attack_rows, validation.POI_ID.tolist())
    evaluations = {"clean": validation, "adversarial": attack}
    regions = sorted(manifest.region.unique())
    tables, figures, html, shards = (output / "tables", output / "figures", output / "html",
                                     output / "work" / "shap")
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(exist_ok=True)
    html.mkdir(exist_ok=True); shards.mkdir(parents=True, exist_ok=True)
    train_values = _matrix(train)
    background_positions = np.linspace(0, len(train) - 1, min(cfg["background_rows"], len(train)), dtype=int)
    background = train_values[background_positions]
    factory = ModelFactory(model_id, cfg, train)
    detailed = []
    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        predict = factory.fit(FEATURES, truth)
        positions = _positions(validation, region, cfg["explain_per_class"], cfg["seed"])
        for condition, frame in evaluations.items():
            shard = shards / f"{region}-{condition}.csv"
            if shard.exists():
                part = pd.read_csv(shard)
                if len(part) != len(positions) * len(FEATURES):
                    raise RuntimeError(f"Incomplete SHAP shard {shard}")
                detailed.append(part); continue
            values = _matrix(frame)[positions]
            explanation = _explain(predict, background, values, FEATURES, cfg["seed"])
            rows = []
            for row_index, position in enumerate(positions):
                for feature_index, feature in enumerate(FEATURES):
                    rows.append({"model": model_id, "region": region, "condition": condition,
                                 "POI_ID": frame.POI_ID.iloc[position],
                                 "true_region": frame.region.iloc[position], "feature": feature,
                                 "feature_value": values[row_index, feature_index],
                                 "shap_value": explanation.values[row_index, feature_index],
                                 "base_value": explanation.base_values[row_index]})
            part = pd.DataFrame(rows); temporary = shard.with_suffix(".csv.tmp")
            part.to_csv(temporary, index=False); os.replace(temporary, shard)
            detailed.append(part)
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    detailed = pd.concat(detailed, ignore_index=True)
    expected = len(regions) * 2 * cfg["explain_per_class"] * 2 * len(FEATURES)
    if len(detailed) != expected or set(detailed.model) != {model_id}:
        raise RuntimeError("SHAP result cardinality is incomplete")
    detailed.to_csv(tables / "shap_values.csv", index=False)
    grouped = (detailed.assign(abs_shap=detailed.shap_value.abs())
               .groupby(["region", "condition", "feature"], as_index=False)
               .abs_shap.mean().rename(columns={"abs_shap": "mean_abs_shap"}))
    grouped.to_csv(tables / "shap_grouped_importance.csv", index=False)
    summary = grouped.groupby(["condition", "feature"], as_index=False).mean_abs_shap.mean()
    summary.to_csv(tables / "shap_global_summary.csv", index=False)
    representative_arrays = {}
    for region in regions:
        for condition in evaluations:
            subset = detailed.loc[detailed.region.eq(region) & detailed.condition.eq(condition)]
            arrays = _beeswarm(subset, figures, f"shap_beeswarm_{region.lower()}_{condition}",
                               f"{model_id} SHAP — {region} vs others — {condition}")
            if region == cfg["representative_region"]:
                representative_arrays[condition] = arrays
    values, shap_values, base = representative_arrays["clean"]
    for feature in NUMERIC:
        shap.dependence_plot(FEATURES.index(feature), shap_values, values,
                             feature_names=FEATURES, show=False, interaction_index=None)
        _save(plt.gcf(), figures, f"shap_dependence_{feature.lower()}")
    explanation = shap.Explanation(values=shap_values[0], base_values=float(base[0]),
                                   data=values[0], feature_names=FEATURES)
    shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
    _save(plt.gcf(), figures, "shap_waterfall_representative")
    shap.save_html(str(html / "shap_force_representative.html"),
                   shap.force_plot(float(base[0]), shap_values[0], values[0], feature_names=FEATURES))

    representative_truth = train.region.eq(cfg["representative_region"]).astype(int).to_numpy()
    representative_predict = factory.fit(FEATURES, representative_truth)
    encoded_train, categorical_names, reverse = _lime_arrays(train_values)
    sample = encoded_train[np.linspace(0, len(encoded_train) - 1,
                                       min(cfg["lime_background_rows"], len(encoded_train)), dtype=int)]
    encoded_value = values[0].copy()
    for index, observed in reverse.items():
        encoded_value[index] = int(np.where(observed == encoded_value[index])[0][0])
    def lime_predict(matrix):
        decoded = np.asarray(matrix, dtype=np.float32).copy()
        for index, observed in reverse.items():
            ordinal = np.rint(decoded[:, index]).astype(int).clip(0, len(observed) - 1)
            decoded[:, index] = observed[ordinal]
        probability = representative_predict(decoded)
        return np.column_stack([1 - probability, probability])
    lime = LimeTabularExplainer(sample, feature_names=FEATURES,
        class_names=["other_region", cfg["representative_region"]],
        categorical_features=list(reverse), categorical_names=categorical_names,
        mode="classification", random_state=cfg["seed"], discretize_continuous=True)
    lime_result = lime.explain_instance(encoded_value, lime_predict, num_features=len(FEATURES),
                                        num_samples=cfg["lime_samples"])
    lime_result.save_to_file(str(html / "lime_representative.html"))
    pd.DataFrame(lime_result.as_list(label=1), columns=["rule", "weight"]).to_csv(
        tables / "lime_representative.csv", index=False)
    _save(lime_result.as_pyplot_figure(label=1), figures, "lime_representative")

    top_feature = summary.groupby("feature").mean_abs_shap.mean().sort_values(ascending=False).index[0]
    reduced = [feature for feature in FEATURES if feature != top_feature]
    reduced_predict = factory.fit(reduced, representative_truth)
    reduced_train = _matrix(train, reduced); reduced_values = _matrix(validation, reduced)[
        _positions(validation, cfg["representative_region"], cfg["explain_per_class"], cfg["seed"])]
    reduced_explanation = _explain(reduced_predict, reduced_train[background_positions], reduced_values,
                                   reduced, cfg["seed"])
    pd.DataFrame({"feature": reduced,
                  "mean_abs_shap": np.abs(reduced_explanation.values).mean(axis=0),
                  "removed_feature": top_feature}).sort_values("mean_abs_shap", ascending=False).to_csv(
        tables / "shap_top_feature_removed.csv", index=False)
    shap.summary_plot(reduced_explanation.values, reduced_values, feature_names=reduced,
                      plot_type="dot", color_bar=True, color_bar_label="Feature value",
                      show=False, max_display=len(reduced))
    plt.gca().axvline(0, color="#444444", linewidth=.8, zorder=0)
    _save(plt.gcf(), figures, "shap_summary_top_feature_removed")

    odds, failures = _odds_ratios(train, regions)
    odds.to_csv(tables / "odds_ratios.csv", index=False); failures.to_csv(
        tables / "odds_ratio_failures.csv", index=False)
    representative = odds.loc[odds.region.eq(cfg["representative_region"]) &
                              odds.feature.ne("const") & odds.identifiable].copy()
    representative["magnitude"] = representative.log_odds.abs()
    representative = representative.nlargest(15, "magnitude").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(representative.odds_ratio, range(len(representative)),
                xerr=[representative.odds_ratio - representative.ci_low,
                      representative.ci_high - representative.odds_ratio], fmt="o")
    ax.set_yticks(range(len(representative)), representative.feature)
    ax.axvline(1, color="black", linewidth=.8); ax.set_xscale("log")
    ax.set_title(f"Odds ratios — {cfg['representative_region']} vs others")
    _save(fig, figures, "odds_ratio_forest_representative")
    metadata = {"status": "complete", "model": model_id, "scope": "validation",
                "test_used": False, "representative_attack": cfg["representative_attack"],
                "seed": cfg["seed"], "regions": len(regions), "shap_algorithm": "permutation",
                "shap_beeswarm_pngs": len(list(figures.glob("shap_beeswarm_*.png"))),
                "background_rows": len(background), "top_feature": top_feature,
                "top_feature_removed": True, "runtime": factory.tabpfn_runtime,
                "python": platform.python_version(), "seconds": time.time() - started}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(
        {"global_shap": summary.to_dict(orient="records"), "top_feature": top_feature,
         "odds_failures": failures.to_dict(orient="records")}, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "report.md").write_text(
        f"# 07. 선택 모델 SHAP·LIME·odds ratio\n\nclean validation 1위 `{model_id}`의 17개 "
        f"OvR 모델을 clean과 `{cfg['representative_attack']}`에서 설명했다. SHAP은 5개 원본 "
        "피처의 음·양 방향, 점 분포와 피처값 색상을 함께 표시하는 beeswarm dot plot이며 "
        "17지역×2조건 34개를 300dpi PNG로 저장했다.\n\n"
        f"전체 조건 평균 |SHAP| 1위 `{top_feature}`를 제거한 설명도 재계산했다. LIME, "
        "좌표 dependence, waterfall/force와 별도 다변량 Binomial GLM의 odds ratio·95% CI·"
        "p-value·FDR q-value는 tables, figures, html에 기록했다. test는 사용하지 않았다.\n")
    (output / "_SUCCESS.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selected_explain.yaml")
    parser.add_argument("--output")
    args = parser.parse_args(); run(args.config, args.output)


if __name__ == "__main__":
    main()
