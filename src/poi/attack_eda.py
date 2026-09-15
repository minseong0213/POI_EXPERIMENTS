"""Create reports 01-04 from the newly generated constrained attacks.

The visual comparison uses one predeclared representative condition (CAA high),
while the tabular statistics cover every generated attack condition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t, ttest_rel, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.contingency_tables import SquareTable
from statsmodels.stats.multitest import multipletests

from .data import CATEGORICAL, FEATURES, NUMERIC, sha256
from .eda import descriptive, distributions, embeddings, mkdir_stage, statistics


def _attack_sha256(path):
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
    if not clean.POI_ID.is_unique or not manifest.POI_ID.is_unique:
        raise ValueError("Clean data or manifest has duplicate POI_ID")
    clean = clean.set_index("POI_ID").loc[manifest.POI_ID].reset_index()
    if clean.region.tolist() != manifest.region.tolist():
        raise ValueError("Clean labels differ from manifest")
    clean["split"] = manifest.split.to_numpy()
    return clean, manifest


def _attack_frame(rows):
    result = rows[["POI_ID", "region", "split"]].copy()
    for feature in FEATURES:
        values = rows[f"{feature}_adv"]
        if feature in CATEGORICAL:
            # Parquet keeps generated category codes as integral floats, while
            # the source CSV loads them as strings. Canonicalize both to the
            # source representation before paired comparisons/encoding.
            values = pd.to_numeric(values, errors="raise").astype("float64").astype(str)
        result[feature] = values.to_numpy()
    return result


def _aligned(clean, rows):
    by_id = rows.set_index("POI_ID")
    missing = set(clean.POI_ID) - set(by_id.index)
    if missing:
        raise ValueError(f"Attack condition lacks {len(missing)} clean POIs")
    result = _attack_frame(by_id.loc[clean.POI_ID].reset_index())
    if result.region.tolist() != clean.region.tolist():
        raise ValueError("Attack labels differ from clean labels")
    return result


def _fisher_ci(correlation, n, alpha=.05):
    if n <= 3 or abs(correlation) >= 1:
        return correlation, correlation
    z = np.arctanh(correlation)
    delta = 1.959963984540054 / np.sqrt(n - 3)
    return float(np.tanh(z - delta)), float(np.tanh(z + delta))


def _all_condition_descriptive(clean, attacks, stage):
    rows = []
    change_rows = []
    for condition, group in attacks.groupby("attack_condition", sort=True):
        adv = _aligned(clean, group)
        for region, part in adv.groupby("region", sort=True):
            for feature in NUMERIC:
                values = pd.to_numeric(part[feature], errors="raise")
                rows.append({
                    "attack_condition": condition, "region": region, "feature": feature,
                    "n": len(values), "mean": values.mean(), "std": values.std(),
                    "median": values.median(), "q1": values.quantile(.25),
                    "q3": values.quantile(.75), "min": values.min(), "max": values.max(),
                })
        for feature in FEATURES:
            changed = clean[feature].astype(str).ne(adv[feature].astype(str))
            change_rows.append({"attack_condition": condition, "feature": feature,
                                "changed_count": int(changed.sum()),
                                "changed_percent": float(changed.mean() * 100)})
    pd.DataFrame(rows).to_csv(stage / "tables" / "table_one_numeric_all_attacks.csv", index=False)
    pd.DataFrame(change_rows).to_csv(stage / "tables" / "paired_change_rates_all_attacks.csv", index=False)
    quality_columns = [
        "attack_method", "attack_condition", "numeric_budget_m", "categorical_budget_l0",
        "distance_m", "categorical_l0", "constraints_valid", "attack_success_source",
        "clean_source_correct", "query_count",
    ]
    quality = attacks[quality_columns].groupby(
        ["attack_method", "attack_condition", "numeric_budget_m", "categorical_budget_l0"],
        dropna=False,
    ).agg(
        rows=("constraints_valid", "size"),
        valid_rate=("constraints_valid", "mean"),
        source_attack_success_rate=("attack_success_source", "mean"),
        clean_source_accuracy=("clean_source_correct", "mean"),
        mean_distance_m=("distance_m", "mean"),
        mean_categorical_l0=("categorical_l0", "mean"),
        mean_queries=("query_count", "mean"),
    ).reset_index()
    quality.to_csv(stage / "tables" / "attack_condition_summary.csv", index=False)


def _all_condition_statistics(clean, attacks, stage, seed):
    transformer = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    clean_features = clean[FEATURES].copy()
    clean_features[NUMERIC] = clean_features[NUMERIC].apply(pd.to_numeric, errors="raise")
    clean_matrix = transformer.fit_transform(clean_features)
    components = min(10, clean_matrix.shape[1] - 1)
    projector = PCA(n_components=components, random_state=seed).fit(clean_matrix)
    clean_pca = projector.transform(clean_matrix)
    test_rows, cca_rows = [], []
    for condition, group in attacks.groupby("attack_condition", sort=True):
        adv = _aligned(clean, group)
        for feature in NUMERIC:
            x = pd.to_numeric(clean[feature], errors="raise").to_numpy()
            y = pd.to_numeric(adv[feature], errors="raise").to_numpy()
            diff = y - x
            if np.allclose(diff, 0):
                t_stat = w_stat = 0.0
                t_p = w_p = 1.0
            else:
                paired = ttest_rel(y, x)
                signed = wilcoxon(diff)
                t_stat, t_p = float(paired.statistic), float(paired.pvalue)
                w_stat, w_p = float(signed.statistic), float(signed.pvalue)
            standard_error = diff.std(ddof=1) / np.sqrt(len(diff))
            margin = t.ppf(.975, len(diff) - 1) * standard_error
            standardized = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0.0
            test_rows.extend([
                {"attack_condition": condition, "family": "paired_numeric", "feature": feature,
                 "test": "paired_t", "statistic": t_stat, "p_value": t_p,
                 "effect": diff.mean(), "ci_low": diff.mean() - margin,
                 "ci_high": diff.mean() + margin, "standardized_effect": standardized},
                {"attack_condition": condition, "family": "paired_numeric", "feature": feature,
                 "test": "wilcoxon", "statistic": w_stat, "p_value": w_p,
                 "effect": np.median(diff), "ci_low": np.nan, "ci_high": np.nan,
                 "standardized_effect": np.nan},
            ])
        for feature in CATEGORICAL:
            before, after = clean[feature].astype(str), adv[feature].astype(str)
            if before.equals(after):
                statistic, p_value = 0.0, 1.0
            else:
                categories = sorted(set(before) | set(after))
                table = pd.crosstab(before, after).reindex(index=categories, columns=categories, fill_value=0)
                result = SquareTable(table, shift_zeros=False).symmetry()
                statistic, p_value = float(result.statistic), float(result.pvalue)
            test_rows.append({"attack_condition": condition, "family": "paired_categorical",
                              "feature": feature, "test": "bowker_symmetry",
                              "statistic": statistic, "p_value": p_value,
                              "effect": float(before.ne(after).mean()), "ci_low": np.nan,
                              "ci_high": np.nan, "standardized_effect": np.nan})

        adv_features = adv[FEATURES].copy()
        adv_features[NUMERIC] = adv_features[NUMERIC].apply(pd.to_numeric, errors="raise")
        adv_pca = projector.transform(transformer.transform(adv_features))
        cca = CCA(n_components=min(5, components), max_iter=2000)
        clean_c, adv_c = cca.fit_transform(clean_pca, adv_pca)
        for index in range(clean_c.shape[1]):
            result = pearsonr(clean_c[:, index], adv_c[:, index])
            correlation = float(result.statistic)
            low, high = _fisher_ci(correlation, len(clean))
            cca_rows.append({"attack_condition": condition, "component": index + 1,
                             "canonical_correlation": correlation, "ci_low": low,
                             "ci_high": high, "p_value": float(result.pvalue),
                             "n": len(clean), "preprocess_fit": "clean_train"})
    tests = pd.DataFrame(test_rows)
    tests["q_value"] = multipletests(tests.p_value.fillna(1), method="fdr_bh")[1]
    tests.to_csv(stage / "tables" / "statistical_tests_all_attacks.csv", index=False)
    pd.DataFrame(cca_rows).to_csv(stage / "tables" / "cca_all_attacks.csv", index=False)


def run(data_dir, attacks_file, output_root, representative="caa_high", seed=42):
    started = time.time()
    clean_all, _ = _load_clean(data_dir)
    clean = clean_all.loc[clean_all.split.eq("train")].reset_index(drop=True)
    attacks_path = Path(attacks_file)
    attacks = pd.read_parquet(attacks_path)
    attacks = attacks.loc[attacks.split.eq("train")].reset_index(drop=True)
    if set(attacks.attack_condition.unique()) == set():
        raise ValueError("No train attacks found")
    if (~attacks.constraints_valid.astype(bool)).any() or (~attacks.label_preserved.astype(bool)).any():
        raise ValueError("Invalid constrained attack rows found")
    representative_rows = attacks.loc[attacks.attack_condition.eq(representative)]
    attack = _aligned(clean, representative_rows)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stages = [mkdir_stage(root, 2, "descriptive"), mkdir_stage(root, 3, "distribution"),
              mkdir_stage(root, 4, "embedding_clustering"), mkdir_stage(root, 5, "statistics")]
    meta = {
        "analysis_scope": "train split only", "paired_unit": "POI_ID",
        "scope_rows_per_condition": len(clean), "attack_conditions": int(attacks.attack_condition.nunique()),
        "representative_attack": representative, "seed": seed,
        "dataset_sha256": {
            "poi_data_region.csv": sha256(Path(data_dir) / "poi_data_region.csv"),
            "sample_manifest.csv": sha256(Path(data_dir) / "sample_manifest.csv"),
            "attacks.parquet": _attack_sha256(attacks_path),
        },
        "test_used": False, "started_unix": started,
    }
    descriptive(clean, attack, stages[0], meta)
    _all_condition_descriptive(clean, attacks, stages[0])
    distributions(clean, attack, stages[1], meta)
    embeddings(clean, attack, stages[2], meta)
    statistics(clean, attack, stages[3], meta)
    _all_condition_statistics(clean, attacks, stages[3], seed)
    for stage in stages:
        report = stage / "report.md"
        report.write_text(report.read_text() + (
            f"\n## 공격 범위\n\n시각화는 사전 지정 대표 조건 `{representative}`를 사용했다. "
            f"전체 {attacks.attack_condition.nunique()}개 공격 조건의 수치 결과는 CSV에 기록했다.\n"
        ))
        metadata = json.loads((stage / "metadata.json").read_text())
        metadata.update({"attack_conditions": int(attacks.attack_condition.nunique()),
                         "representative_attack": representative,
                         "finished_unix": time.time(), "test_used": False})
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"status": "complete", "stages": [str(path) for path in stages],
              "attack_conditions": int(attacks.attack_condition.nunique()),
              "representative_attack": representative, "test_used": False,
              "seconds": time.time() - started}
    (root / "_SUCCESS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/poi_34k_seed42")
    parser.add_argument("--attacks-file", default=(
        "artifacts/poi-adversarial-20260915-001/reports/01_attack_generation/tables/attacks.parquet"))
    parser.add_argument("--output-root", default="artifacts/poi-adversarial-20260915-001/reports")
    parser.add_argument("--representative", default="caa_high")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.attacks_file, args.output_root, args.representative, args.seed)


if __name__ == "__main__":
    main()
