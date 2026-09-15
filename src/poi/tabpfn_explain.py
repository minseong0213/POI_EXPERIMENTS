"""Stage 07 model-agnostic SHAP/LIME and inferential odds ratios for TabPFN v2.5."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.multitest import multipletests
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import tabpfn
import torch
import yaml

from .data import CATEGORICAL, FEATURES, NUMERIC, load_dataset, sha256


def numeric_matrix(frame, features=FEATURES):
    matrix = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('TabPFN explanation input contains non-finite values')
    return matrix


def model_for(features, cfg):
    categorical = [index for index, feature in enumerate(features) if feature in CATEGORICAL]
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, device=cfg['device'], n_estimators=cfg['n_estimators'],
        random_state=cfg['seed'], categorical_features_indices=categorical,
        show_progress_bar=False, memory_saving_mode='auto')


def representative_positions(frame, region, per_class, seed):
    positive = frame.index[frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    negative = frame.index[~frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    return np.concatenate([positive, negative])


def save(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def shap_beeswarm(records, features, figures, name, title):
    """Save a SHAP dot summary with direction and feature-value colour."""
    order = records.POI_ID.drop_duplicates().tolist()
    feature_values = (records.pivot(index='POI_ID', columns='feature', values='feature_value')
                      .loc[order, features].to_numpy())
    shap_values = (records.pivot(index='POI_ID', columns='feature', values='shap_value')
                   .loc[order, features].to_numpy())
    base_values = records.groupby('POI_ID').base_value.first().loc[order].to_numpy()
    shap.summary_plot(
        shap_values, feature_values, feature_names=features, plot_type='dot',
        color_bar=True, color_bar_label='Feature value', show=False,
        max_display=len(features))
    figure = plt.gcf()
    axis = plt.gca()
    axis.axvline(0, color='#444444', linewidth=.8, zorder=0)
    axis.set_xlabel('SHAP value')
    figure.suptitle(title)
    save(figure, figures, name)
    return feature_values, shap_values, base_values


def explain_values(model, background, values, features, seed):
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(
        lambda matrix: model.predict_proba(np.asarray(matrix, dtype=np.float32))[:, 1],
        masker, algorithm='permutation', feature_names=features, seed=seed)
    return explainer(values, max_evals=2 * len(features) + 1,
                     batch_size='auto', silent=True)


def odds_ratios(train, regions):
    numeric = train[NUMERIC].apply(pd.to_numeric, errors='raise').astype(float)
    numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    categorical = pd.get_dummies(train[CATEGORICAL], prefix=CATEGORICAL,
                                 drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([numeric, categorical], axis=1), has_constant='add')
    rows, failures = [], []
    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        try:
            fitted = sm.GLM(truth, design, family=sm.families.Binomial()).fit(
                maxiter=200, cov_type='HC3')
            for name, coefficient, standard_error, p_value in zip(
                    design.columns, fitted.params, fitted.bse, fitted.pvalues):
                identifiable = bool(np.isfinite(standard_error) and np.isfinite(p_value))
                rows.append({
                    'region': region, 'feature': name, 'identifiable': identifiable,
                    'log_odds': coefficient,
                    'odds_ratio': np.exp(np.clip(coefficient, -50, 50)),
                    'ci_low': np.exp(np.clip(coefficient - 1.96 * standard_error, -50, 50)),
                    'ci_high': np.exp(np.clip(coefficient + 1.96 * standard_error, -50, 50)),
                    'p_value': p_value,
                })
        except Exception as error:
            failures.append({'region': region, 'error': repr(error)})
    result = pd.DataFrame(rows)
    if len(result):
        result['q_value'] = multipletests(result.p_value.fillna(1), method='fdr_bh')[1]
    return result, pd.DataFrame(failures)


def lime_inputs(x_train):
    encoded = x_train.copy()
    categorical_names, reverse = {}, {}
    for index in range(len(NUMERIC), len(FEATURES)):
        observed = np.unique(x_train[:, index])
        mapping = {value: ordinal for ordinal, value in enumerate(observed)}
        encoded[:, index] = np.array([mapping[value] for value in x_train[:, index]])
        categorical_names[index] = [str(value) for value in observed]
        reverse[index] = observed
    return encoded, categorical_names, reverse


def run(config_path, output_override=None):
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_override or cfg['output'])
    if (out / '_SUCCESS.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed {out}')
    if out.exists() and not cfg.get('resume_partial', False):
        raise FileExistsError(f'Refusing to resume {out}')
    if not os.environ.get('TABPFN_TOKEN'):
        token_path = Path(os.environ.get('TABPFN_TOKEN_FILE', '/root/.config/poi/tabpfn_token'))
        if token_path.is_file():
            os.environ['TABPFN_TOKEN'] = token_path.read_text().strip()
    if not os.environ.get('TABPFN_TOKEN'):
        raise RuntimeError('TABPFN_TOKEN or TABPFN_TOKEN_FILE is required')

    tables, figures, html, shards = (out / 'tables', out / 'figures', out / 'html',
                                     out / 'work' / 'shap')
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    html.mkdir(exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    data_dir = os.environ.get('DATA_DIR', cfg['data_dir'])
    frames, manifest = load_dataset(data_dir)
    train = frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    evaluations = {
        'clean': frames['clean'].loc[frames['clean'].split.eq('validation')].reset_index(drop=True),
        'adversarial': frames['attack'].loc[frames['attack'].split.eq('validation')].reset_index(drop=True),
    }
    regions = sorted(manifest.region.unique())
    x_train = numeric_matrix(train)
    background_positions = np.linspace(
        0, len(train) - 1, min(cfg['background_rows'], len(train)), dtype=int)
    background = x_train[background_positions]
    detailed = []

    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        model = model_for(FEATURES, cfg)
        model.fit(x_train, truth)
        positions = representative_positions(
            evaluations['clean'], region, cfg['explain_per_class'], cfg['seed'])
        for condition, frame in evaluations.items():
            shard = shards / f'{region}-{condition}.csv'
            if shard.exists():
                part = pd.read_csv(shard)
                expected = len(positions) * len(FEATURES)
                if len(part) != expected:
                    raise RuntimeError(f'Incomplete SHAP checkpoint: {shard}')
                detailed.append(part)
                print(f'resumed SHAP {region} {condition}', flush=True)
                continue
            values = numeric_matrix(frame)[positions]
            explanation = explain_values(model, background, values, FEATURES, cfg['seed'])
            records = []
            for row_index, position in enumerate(positions):
                for feature_index, feature in enumerate(FEATURES):
                    records.append({
                        'model': 'tabpfn_v2_5', 'region': region, 'condition': condition,
                        'POI_ID': frame.POI_ID.iloc[position], 'true_region': frame.region.iloc[position],
                        'feature': feature, 'feature_value': values[row_index, feature_index],
                        'shap_value': explanation.values[row_index, feature_index],
                        'base_value': explanation.base_values[row_index],
                    })
            part = pd.DataFrame(records)
            temporary = shard.with_suffix('.csv.tmp')
            part.to_csv(temporary, index=False)
            os.replace(temporary, shard)
            detailed.append(part)
            print(f'completed SHAP {region} {condition}', flush=True)
        torch.cuda.empty_cache()

    detailed = pd.concat(detailed, ignore_index=True)
    expected_rows = len(regions) * 2 * cfg['explain_per_class'] * 2 * len(FEATURES)
    if len(detailed) != expected_rows or set(detailed.model) != {'tabpfn_v2_5'}:
        raise RuntimeError('Incomplete TabPFN SHAP cardinality')
    detailed.to_csv(tables / 'shap_values.csv', index=False)
    grouped = (detailed.assign(abs_shap=detailed.shap_value.abs())
               .groupby(['region', 'condition', 'feature'], as_index=False)
               .abs_shap.mean().rename(columns={'abs_shap': 'mean_abs_shap'}))
    grouped.to_csv(tables / 'shap_grouped_importance.csv', index=False)
    summary = (grouped.groupby(['condition', 'feature'], as_index=False)
               .mean_abs_shap.mean())
    summary.to_csv(tables / 'shap_global_summary.csv', index=False)

    representative_arrays = {}
    for region in regions:
        for condition in evaluations:
            subset = detailed.loc[
                detailed.region.eq(region) & detailed.condition.eq(condition)]
            arrays = shap_beeswarm(
                subset, FEATURES, figures,
                f'shap_beeswarm_{region.lower()}_{condition}',
                f'TabPFN SHAP — {region} vs others — {condition}')
            if region == cfg['representative_region']:
                representative_arrays[condition] = arrays

    values, shap_values, base_values = representative_arrays['clean']
    for feature in NUMERIC:
        shap.dependence_plot(FEATURES.index(feature), shap_values, values,
                             feature_names=FEATURES, show=False, interaction_index=None)
        save(plt.gcf(), figures, f'shap_dependence_{feature.lower()}')
    explanation = shap.Explanation(
        values=shap_values[0], base_values=float(base_values[0]), data=values[0],
        feature_names=FEATURES)
    shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
    save(plt.gcf(), figures, 'shap_waterfall_representative')
    force = shap.force_plot(float(base_values[0]), shap_values[0], values[0],
                            feature_names=FEATURES)
    shap.save_html(str(html / 'shap_force_representative.html'), force)

    representative_truth = train.region.eq(cfg['representative_region']).astype(int).to_numpy()
    representative_model = model_for(FEATURES, cfg)
    representative_model.fit(x_train, representative_truth)
    encoded_train, categorical_names, reverse = lime_inputs(x_train)
    sample = encoded_train[np.linspace(
        0, len(encoded_train) - 1, min(cfg['lime_background_rows'], len(encoded_train)),
        dtype=int)]
    encoded_value = values[0].copy()
    for index, observed in reverse.items():
        encoded_value[index] = int(np.where(observed == encoded_value[index])[0][0])

    def lime_predict(matrix):
        decoded = np.asarray(matrix, dtype=np.float32).copy()
        for index, observed in reverse.items():
            ordinals = np.rint(decoded[:, index]).astype(int).clip(0, len(observed) - 1)
            decoded[:, index] = observed[ordinals]
        return representative_model.predict_proba(decoded)

    lime = LimeTabularExplainer(
        sample, feature_names=FEATURES,
        class_names=['other_region', cfg['representative_region']],
        categorical_features=list(reverse), categorical_names=categorical_names,
        mode='classification', random_state=cfg['seed'], discretize_continuous=True)
    lime_result = lime.explain_instance(
        encoded_value, lime_predict, num_features=len(FEATURES),
        num_samples=cfg['lime_samples'])
    lime_result.save_to_file(str(html / 'lime_representative.html'))
    pd.DataFrame(lime_result.as_list(label=1), columns=['rule', 'weight']).to_csv(
        tables / 'lime_representative.csv', index=False)
    save(lime_result.as_pyplot_figure(label=1), figures, 'lime_representative')

    top_feature = (summary.groupby('feature').mean_abs_shap.mean()
                   .sort_values(ascending=False).index[0])
    reduced_features = [feature for feature in FEATURES if feature != top_feature]
    reduced_train = numeric_matrix(train, reduced_features)
    reduced_model = model_for(reduced_features, cfg)
    reduced_model.fit(reduced_train, representative_truth)
    reduced_values = numeric_matrix(evaluations['clean'], reduced_features)[
        representative_positions(evaluations['clean'], cfg['representative_region'],
                                 cfg['explain_per_class'], cfg['seed'])]
    reduced_background = reduced_train[background_positions]
    reduced_explanation = explain_values(
        reduced_model, reduced_background, reduced_values, reduced_features, cfg['seed'])
    reduced_summary = pd.DataFrame({
        'feature': reduced_features,
        'mean_abs_shap': np.abs(reduced_explanation.values).mean(axis=0),
        'removed_feature': top_feature,
    }).sort_values('mean_abs_shap', ascending=False)
    reduced_summary.to_csv(tables / 'shap_top_feature_removed.csv', index=False)
    shap.summary_plot(
        reduced_explanation.values, reduced_values, feature_names=reduced_features,
        plot_type='dot', color_bar=True, color_bar_label='Feature value',
        show=False, max_display=len(reduced_features))
    plt.gca().axvline(0, color='#444444', linewidth=.8, zorder=0)
    plt.gcf().suptitle(f'SHAP after removing {top_feature}')
    save(plt.gcf(), figures, 'shap_summary_top_feature_removed')

    odds, odds_failures = odds_ratios(train, regions)
    odds.to_csv(tables / 'odds_ratios.csv', index=False)
    odds_failures.to_csv(tables / 'odds_ratio_failures.csv', index=False)
    representative_odds = odds.loc[
        odds.region.eq(cfg['representative_region']) & odds.feature.ne('const') &
        odds.identifiable].copy()
    representative_odds['magnitude'] = representative_odds.log_odds.abs()
    representative_odds = representative_odds.nlargest(15, 'magnitude').sort_values('odds_ratio')
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(representative_odds.odds_ratio, range(len(representative_odds)),
                xerr=[representative_odds.odds_ratio - representative_odds.ci_low,
                      representative_odds.ci_high - representative_odds.odds_ratio], fmt='o')
    ax.set_yticks(range(len(representative_odds)), representative_odds.feature)
    ax.axvline(1, color='black', linewidth=.8)
    ax.set_xscale('log')
    ax.set_title(f"Odds ratios — {cfg['representative_region']} vs others")
    save(fig, figures, 'odds_ratio_forest_representative')

    metadata = {
        'status': 'complete', 'model': 'tabpfn_v2_5', 'scope': 'validation',
        'seed': cfg['seed'], 'regions': len(regions),
        'shap_algorithm': 'permutation', 'background_rows': len(background),
        'explained_rows_per_model_condition': cfg['explain_per_class'] * 2,
        'top_feature': top_feature, 'top_feature_removed': True,
        'odds_ratio_failed_regions': odds_failures.to_dict(orient='records'),
        'python': platform.python_version(), 'tabpfn': tabpfn.__version__,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'data_sha256': {name: sha256(Path(data_dir) / name) for name in
                        ['poi_data_region.csv', 'poi_adversarial_data_final.csv',
                         'sample_manifest.csv']},
        'seconds': time.time() - started,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'metrics.json').write_text(json.dumps({
        'global_shap': summary.to_dict(orient='records'),
        'top_feature': top_feature,
        'odds_failures': odds_failures.to_dict(orient='records'),
    }, indent=2) + '\n')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'report.md').write_text(f'''# 07. TabPFN v2.5 SHAP, LIME, odds ratio

최종 선택 모델 `tabpfn_v2_5`의 17개 OvR 모델을 validation clean/adversarial에서
모델 비종속 permutation SHAP으로 해석했다. 전역 mean |SHAP| 1위 피처는
`{top_feature}`이다. 17개 지역의 clean/attack SHAP beeswarm은 SHAP 값 0을 기준으로
양·음 방향을 표시하고 특성값이 높으면 빨강, 낮으면 파랑인 dot plot으로 저장했다.
대표 `{cfg['representative_region']} vs others`의 좌표 dependence, waterfall/force와
LIME 설명은 표·PNG 300dpi·HTML로 저장했다.

가장 중요한 `{top_feature}`를 제거한 뒤 SHAP을 다시 계산했고, 제거 성능은 6단계
ablation에서 확인할 수 있다. Odds ratio, 95% CI, p-value와 FDR q-value는 예측 모델
중요도가 아니라 train split에 적합한 별도 다변량 Binomial GLM의 추론 결과다.
''')
    (out / '_SUCCESS.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/legacy/13_explainability_tabpfn25.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
