"""Explicit model × region × attack explanations with resumable, hashed evidence.

Explanations are descriptive: LIME neighbourhoods and marginal SHAP masking can
leave the observed categorical manifold. Neither is evidence of valid attacks.
"""
import argparse
import hashlib
import json
import os
import importlib.metadata
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
import shap
import yaml
from lime.lime_tabular import LimeTabularExplainer

from .data import FEATURES, NUMERIC, CATEGORICAL, sha256
from .selected_explain import ModelFactory, _matrix, _positions, _explain, _beeswarm, _save, _lime_arrays


def aligned_condition(attacks, validation, condition, attack_seed):
    rows = attacks.loc[attacks.attack_condition.eq(condition)].copy()
    seed_column = "generation_seed" if "generation_seed" in rows else "seed"
    if seed_column not in rows:
        raise ValueError("Attack generation seed is required")
    rows = rows.loc[rows[seed_column].eq(attack_seed)]
    if rows.POI_ID.duplicated().any() or set(rows.POI_ID) != set(validation.POI_ID):
        raise ValueError(f"Incomplete/duplicate paired attacks: {condition}, seed={attack_seed}")
    rows = rows.set_index("POI_ID").loc[validation.POI_ID].reset_index()
    if not np.array_equal(rows.region.to_numpy(), validation.region.to_numpy()):
        raise ValueError("Attack/clean region mismatch")
    for feature in FEATURES:
        if f"{feature}_adv" in rows:
            rows[feature] = rows[f"{feature}_adv"]
        elif feature not in rows:
            raise ValueError(f"Attack feature absent: {feature}")
    for feature in CATEGORICAL:
        rows[feature] = pd.to_numeric(rows[feature], errors="raise").astype(float).astype(str)
    return rows


def primary_mask(frame):
    for column in ("primary_evaluation_eligible", "constraints_valid", "label_preserved"):
        if column not in frame or not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any():
            raise ValueError(f"Regenerated attack requires non-null boolean {column}")
    mask = frame.primary_evaluation_eligible.to_numpy()
    if np.any(mask & ~(frame.constraints_valid & frame.label_preserved).to_numpy()):
        raise ValueError("Primary mask includes invalid attacks")
    if "victim_model" not in frame or frame.victim_model.fillna("").eq("").any():
        raise ValueError("Attack victim provenance is incomplete")
    return mask


def encode_lime_row(value, reverse):
    result = value.copy()
    for index, observed in reverse.items():
        found = np.flatnonzero(observed == value[index])
        if len(found) != 1:
            raise ValueError("LIME input category absent from training vocabulary")
        result[index] = found[0]
    return result


def make_lime(train_values, cfg, region, predict, features=FEATURES, matrix_dtype=np.float32):
    encoded, names, reverse = _lime_arrays(train_values, features)
    positions = np.linspace(0, len(encoded)-1, min(cfg["lime_background_rows"], len(encoded)), dtype=int)
    explainer = LimeTabularExplainer(encoded[positions], feature_names=features,
        class_names=["other_region", region], categorical_features=list(reverse),
        categorical_names=names, mode="classification", random_state=cfg["seed"],
        discretize_continuous=True)
    def probability(values):
        decoded = np.asarray(values, dtype=matrix_dtype).copy()
        for index, observed in reverse.items():
            ordinals = np.rint(decoded[:, index]).astype(int).clip(0, len(observed)-1)
            decoded[:, index] = observed[ordinals]
        positive = predict(decoded)
        return np.column_stack([1-positive, positive])
    return explainer, probability, reverse


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    from .data import load_clean_splits
    features = cfg.get("features", FEATURES)
    if not features or len(set(features)) != len(features) or not set(features) <= set(FEATURES):
        raise ValueError("features must be a nonempty unique subset of the permitted five features")
    if cfg["model_id"] == "proposed_tabpfn" and list(features) != list(FEATURES):
        raise ValueError("Proposed checkpoint requires its fixed five-feature architecture")
    explanation_dtype = (
        np.float32
        if cfg["model_id"] == "proposed_tabpfn" or cfg["model_id"].startswith("tabpfn_")
        else np.float64
    )
    started = time.time()
    output = Path(cfg["output"])
    if (output / "_GENERATED.json").exists():
        raise FileExistsError(output)
    # Select before materializing features. Test is not requested by this runner.
    clean, manifest = load_clean_splits(cfg["data_dir"], splits=("train", "validation"))
    train = clean.loc[clean.split.eq("train")].reset_index(drop=True)
    validation = clean.loc[clean.split.eq("validation")].reset_index(drop=True)
    conditions = cfg["attack_conditions"]
    attack_path = Path(cfg["attacks_file"]) if conditions else None
    attacks = (pd.read_parquet(attack_path, filters=[("split", "==", "validation")])
               if attack_path is not None else pd.DataFrame())
    if conditions == "all":
        conditions = sorted(attacks.attack_condition.unique())
    evaluations = {"clean": validation}
    for condition in conditions:
        evaluations[condition] = aligned_condition(attacks, validation, condition, cfg["attack_seed"])
    # Use identical explainable POIs across clean and every requested attack condition.
    eligible = np.ones(len(validation), dtype=bool)
    input_coverage = []
    for condition, frame in evaluations.items():
        if condition != "clean":
            valid = primary_mask(frame)
            eligible &= valid
            input_coverage.append(dict(condition=condition, attempted_rows=len(frame),
                primary_valid_rows=int(valid.sum()), primary_valid_coverage=float(valid.mean())))
    evaluations = {condition:frame.loc[eligible].reset_index(drop=True)
                   for condition,frame in evaluations.items()}
    validation = evaluations["clean"]
    source_files = [Path(__file__), Path(__file__).with_name("selected_explain.py"),
                    Path(__file__).with_name("data.py"), Path(__file__).with_name("benchmark.py"),
                    Path(__file__).with_name("ablation.py"), Path(cfg["tree_config"])]
    dependency_hashes = {str(path):sha256(path) for path in source_files}
    fingerprint = hashlib.sha256(json.dumps({"config":cfg, "dependencies":dependency_hashes, "attacks":sha256(attack_path) if attack_path else None,
        "clean":sha256(Path(cfg["data_dir"])/"poi_data_region.csv"),
        "manifest":sha256(Path(cfg["data_dir"])/"sample_manifest.csv"),
        "code":sha256(Path(__file__)),
        "factory_code":sha256(Path(__file__).with_name("selected_explain.py")),
        "checkpoint":sha256(Path(cfg["checkpoint"])) if cfg.get("checkpoint") else None
        }, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    identity = output / "input_identity.json"
    if identity.exists() and json.loads(identity.read_text())["sha256"] != fingerprint:
        raise ValueError("Resume inputs/config/code changed; use a new output path")
    identity.write_text(json.dumps({"sha256":fingerprint}, indent=2)+"\n")
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for path in source_files:
        shutil.copy2(path, provenance/path.name)
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "shap", "lime", "matplotlib", "tabpfn",
                 "xgboost", "lightgbm", "catboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (provenance/"runtime.json").write_text(json.dumps({"packages":packages,
        "source_hashes":dependency_hashes, "captured_before_model_fitting":True}, indent=2)+"\n")
    for child in ("tables", "figures", "html", "work"):
        (output/child).mkdir(exist_ok=True)
    pd.DataFrame(input_coverage).to_csv(output/"tables"/"attack_eligibility.csv", index=False)
    regions = sorted(train.region.unique())
    train_values = _matrix(train, features, dtype=explanation_dtype)
    background = train_values[np.random.default_rng(cfg["seed"]).choice(len(train),
        min(cfg["background_rows"], len(train)), replace=False)]
    factory = ModelFactory(cfg["model_id"], cfg, train)
    proposed = None
    if cfg["model_id"] == "proposed_tabpfn":
        from .proposed_defense import load_proposed_predictor
        proposed = load_proposed_predictor(cfg["checkpoint"], device=cfg["device"])
    coverage = []
    for region in regions:
        positions = _positions(validation, region, cfg["explain_per_class"], cfg["seed"])
        lime_count = min(cfg.get("lime_per_class", 1), cfg["explain_per_class"])
        lime_indices = set(range(lime_count)) | set(range(cfg["explain_per_class"],
            cfg["explain_per_class"] + lime_count))
        if proposed is None:
            predict = factory.fit(features, train.region.eq(region).astype(int).to_numpy())
            fit_record = dict(factory.last_fit_attestation)
            fit_record.update(model_id=cfg["model_id"], region=region, seed=cfg["seed"])
            (provenance/f"fit_attestation_{region}.json").write_text(
                json.dumps(fit_record, indent=2, default=str)+"\n")
        else:
            region_index = proposed.regions.index(region)
            predict = lambda values, index=region_index: proposed.predict_region_proba(values)[:, index]
        lime, lime_predict, reverse = make_lime(
            train_values, cfg, region, predict, features, matrix_dtype=explanation_dtype
        )
        for condition, frame in evaluations.items():
            stem = f"{region.lower()}_{condition}"
            marker = output/"work"/f"{stem}.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                if saved["input_identity"] != fingerprint or any(
                    not (output/name).is_file() or sha256(output/name) != checksum
                    for name, checksum in saved["files"].items()):
                    raise ValueError(f"Corrupt explanation shard: {stem}")
                coverage.append(saved["coverage"])
                continue
            values = _matrix(frame, features, dtype=explanation_dtype)[positions]
            explanation = _explain(predict, background, values, features, cfg["seed"])
            shap_rows, lime_rows, files = [], [], []
            def remember(relative):
                files.append(relative)
                return output/relative
            for row_index, position in enumerate(positions):
                poi_id = str(frame.POI_ID.iloc[position])
                for feature_index, feature in enumerate(features):
                    shap_rows.append({"model":cfg["model_id"], "region":region, "condition":condition,
                        "POI_ID":poi_id, "true_region":frame.region.iloc[position], "feature":feature,
                        "feature_value":values[row_index, feature_index],
                        "shap_value":explanation.values[row_index, feature_index],
                        "base_value":explanation.base_values[row_index]})
                if row_index not in lime_indices:
                    continue
                result = lime.explain_instance(encode_lime_row(values[row_index], reverse),
                    lime_predict, num_features=len(features), num_samples=cfg["lime_samples"], labels=(1,))
                for rule, weight in result.as_list(label=1):
                    lime_rows.append({"model":cfg["model_id"], "region":region, "condition":condition,
                        "POI_ID":poi_id, "rule":rule, "weight":weight,
                        "local_fidelity_r2":result.score, "true_region":frame.region.iloc[position]})
                local_stem = f"lime_{stem}_{row_index}"
                result.save_to_file(str(remember(f"html/{local_stem}.html")))
                lime_figure = result.as_pyplot_figure(label=1)
                lime_figure.suptitle(f"{cfg['model_id']} | {region} | {condition} | LIME R²={result.score:.3f}")
                _save(lime_figure, output/"figures", local_stem)
                remember(f"figures/{local_stem}.png")
            part = pd.DataFrame(shap_rows)
            part.to_csv(remember(f"tables/shap_{stem}.csv"), index=False)
            pd.DataFrame(lime_rows).to_csv(remember(f"tables/lime_{stem}.csv"), index=False)
            _beeswarm(part, output/"figures", f"shap_beeswarm_{stem}",
                f"{cfg['model_id']} | {region} vs others | {condition}", features=features)
            remember(f"figures/shap_beeswarm_{stem}.png")
            for feature in (name for name in features if name in NUMERIC):
                import matplotlib.pyplot as plt
                shap.dependence_plot(features.index(feature), explanation.values, values,
                    feature_names=features, show=False, interaction_index=None)
                _save(plt.gcf(), output/"figures", f"shap_dependence_{stem}_{feature}")
                remember(f"figures/shap_dependence_{stem}_{feature}.png")
            shap.save_html(str(remember(f"html/shap_force_{stem}.html")), shap.force_plot(
                float(explanation.base_values[0]), explanation.values[0], values[0], feature_names=features))
            entry = {"model":cfg["model_id"], "region":region, "condition":condition,
                "shap_pois":len(positions), "lime_pois":len(lime_indices), "seed":cfg["seed"],
                "attack_seed":None if condition == "clean" else cfg["attack_seed"]}
            coverage.append(entry)
            marker.write_text(json.dumps({"input_identity":fingerprint, "coverage":entry,
                "files":{name:sha256(output/name) for name in files}}, indent=2)+"\n")
            print(json.dumps(entry), flush=True)
    pd.DataFrame(coverage).to_csv(output/"tables"/"coverage.csv", index=False)
    metadata = {"status":"GENERATED_PENDING_INDEPENDENT_REVIEW", "config":cfg,
        "groups":len(coverage), "expected_groups":len(regions)*len(evaluations),
        "common_primary_valid_pois":len(validation),
        "explanation_matrix_dtype":np.dtype(explanation_dtype).name,
        "tree_inference_representation":(
            "float64 SHAP/LIME arrays mapped to exact clean-train categorical strings"
            if explanation_dtype == np.float64 else "existing float32 TabPFN matrix contract"
        ),
        "test_used":False, "seconds":time.time()-started, "input_identity":fingerprint,
        "limitations":["Local explanations use a small balanced sample; not population SHAP distributions.",
            "Marginal SHAP and LIME can construct unobserved category tuples.",
            "Categorical colour values are codes, not an ordinal scale.",
            "Explanations alone do not validate attacks or establish causal effects."]}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (output/"report.md").write_text("# 지역·공격 조건별 SHAP 및 LIME\n\n"
        f"모델 `{cfg['model_id']}`, {len(regions)}지역 × {len(evaluations)}조건. "
        "그룹별 SHAP dot, 좌표 dependence, force HTML 및 동일 표본 LIME을 저장했다. "
        "LIME의 local fidelity R²와 실제 설명 POI를 CSV에 기록했다. 독립 검수 대기.\n\n"
        "작은 균형 표본의 국소 설명이며 전체 모집단 중요도로 해석하지 않는다. "
        "범주형 색상은 크기 관계가 없는 코드이다. LIME/SHAP의 주변 표본은 관측되지 않은 범주 "
        "조합을 포함할 수 있으며 공격 유효성의 근거로 사용하지 않는다.\n")
    (output/"_GENERATED.json").write_text(json.dumps(metadata, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
