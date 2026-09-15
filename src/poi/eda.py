"""Run the staged paired EDA (reports 01-04) and save reproducible artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, spearmanr, t, ttest_rel, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.contingency_tables import SquareTable
from statsmodels.stats.multitest import multipletests
import umap

from .data import CATEGORICAL, NUMERIC, load_dataset, sha256

FEATURES = NUMERIC + CATEGORICAL
CONDITION_COLORS = {'clean': '#2671B8', 'adversarial': '#D24B40'}


def mkdir_stage(root, number, name):
    stage = root / f'{number:02d}_{name}'
    if stage.exists():
        raise FileExistsError(f'Refusing to overwrite {stage}')
    for child in ['tables', 'figures']:
        (stage / child).mkdir(parents=True)
    return stage


def save_figure(fig, stage, name):
    fig.savefig(stage / 'figures' / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def bh_frame(rows):
    result = pd.DataFrame(rows)
    if len(result):
        result['q_value'] = multipletests(result.p_value, method='fdr_bh')[1]
    return result


def cramers_v(table):
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(np.sqrt((chi2 / n) / denom)), float(p), float(chi2)


def matrix_for(clean, attack, fit_on='joint'):
    transformer = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
    clean_x = clean[FEATURES].copy()
    attack_x = attack[FEATURES].copy()
    for frame in [clean_x, attack_x]:
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
    fit = pd.concat([clean_x, attack_x], ignore_index=True) if fit_on == 'joint' else clean_x
    transformer.fit(fit)
    return transformer.transform(clean_x), transformer.transform(attack_x), transformer


def metadata(data_dir, scope_rows, seed, started):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn',
                'statsmodels', 'umap-learn']
    return {
        'analysis_scope': 'train split only', 'paired_unit': 'POI_ID', 'scope_rows_per_condition': scope_rows,
        'seed': seed, 'git_commit': commit, 'git_dirty': dirty, 'python': platform.python_version(),
        'dataset_sha256': {name: sha256(data_dir / name) for name in
                           ['poi_data_region.csv', 'poi_adversarial_data_final.csv', 'sample_manifest.csv']},
        'versions': {name: importlib.metadata.version(name) for name in packages},
        'started_unix': started, 'finished_unix': time.time(),
    }


def write_stage(stage, title, body, meta, metrics):
    figures = sorted(path.name for path in (stage / 'figures').glob('*.png'))
    tables = sorted(path.name for path in (stage / 'tables').glob('*.csv'))
    report = [f'# {title}', '', body, '', '## 생성 파일', '']
    report += [f'- `figures/{name}`' for name in figures]
    report += [f'- `tables/{name}`' for name in tables]
    report += ['', '## 실행 범위', '', f"- 분석 데이터: train split의 paired POI {meta['scope_rows_per_condition']:,}개",
               f"- Seed: {meta['seed']}", '- 모든 그림을 PNG 300 dpi로 생성함',
               '- 상세 수치는 CSV와 metrics.json에 저장함', '']
    (stage / 'report.md').write_text('\n'.join(report))
    (stage / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (stage / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


def descriptive(clean, attack, stage, meta):
    combined = pd.concat([clean.assign(condition='clean'), attack.assign(condition='adversarial')])
    numeric_rows = []
    for (condition, region), group in combined.groupby(['condition', 'region']):
        for feature in NUMERIC:
            values = pd.to_numeric(group[feature], errors='coerce')
            numeric_rows.append({'condition': condition, 'region': region, 'feature': feature,
                                 'n': int(values.notna().sum()), 'missing': int(values.isna().sum()),
                                 'mean': values.mean(), 'std': values.std(), 'median': values.median(),
                                 'q1': values.quantile(.25), 'q3': values.quantile(.75),
                                 'min': values.min(), 'max': values.max()})
    numeric = pd.DataFrame(numeric_rows)
    numeric.to_csv(stage / 'tables' / 'table_one_numeric_by_region.csv', index=False)
    categorical_rows = []
    for (condition, region), group in combined.groupby(['condition', 'region']):
        for feature in CATEGORICAL:
            counts = group[feature].value_counts(dropna=False)
            for value, count in counts.items():
                categorical_rows.append({'condition': condition, 'region': region, 'feature': feature,
                                         'value': value, 'count': count, 'percent': count / len(group) * 100})
    pd.DataFrame(categorical_rows).to_csv(stage / 'tables' / 'table_one_categorical_by_region.csv', index=False)
    change = pd.DataFrame({'feature': FEATURES,
                           'changed_count': [int(clean[c].ne(attack[c]).sum()) for c in FEATURES]})
    change['changed_percent'] = change.changed_count / len(clean) * 100
    change.to_csv(stage / 'tables' / 'paired_change_rates.csv', index=False)
    balance = clean.groupby(['split', 'region']).size().rename('count').reset_index()
    balance.to_csv(stage / 'tables' / 'class_balance.csv', index=False)
    missing = combined[FEATURES].eq('').groupby(combined.condition).sum().T
    missing.to_csv(stage / 'tables' / 'missing_counts.csv')

    fig, ax = plt.subplots(figsize=(12, 5))
    train_balance = balance.loc[balance.split.eq('train')]
    sns.barplot(data=train_balance, x='region', y='count', ax=ax, color='#2671B8')
    ax.tick_params(axis='x', rotation=55); ax.set_title('Train POI count by region')
    save_figure(fig, stage, 'class_balance_train')
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=change, x='feature', y='changed_percent', ax=ax, color='#D24B40')
    ax.tick_params(axis='x', rotation=25); ax.set_ylabel('Changed paired POIs (%)')
    ax.set_title('Feature changes: clean to adversarial')
    save_figure(fig, stage, 'paired_change_rate')
    metrics = {'paired_rows': len(clean), 'regions': clean.region.nunique(),
               'change_rate_percent': dict(zip(change.feature, change.changed_percent))}
    body = ('Table One 형식의 지역·조건별 기술통계와 범주 빈도를 생성했다. '
            '분석 단위는 POI_ID이며 clean/adversarial은 짝지어진 관측이다. '
            f"좌표 변경률은 X {metrics['change_rate_percent']['X_COORD']:.1f}%, "
            f"Y {metrics['change_rate_percent']['Y_COORD']:.1f}%였다.")
    write_stage(stage, '01. 기술통계와 Table One', body, meta, metrics)


def distributions(clean, attack, stage, meta):
    long = []
    for condition, frame in [('clean', clean), ('adversarial', attack)]:
        part = frame[['POI_ID', 'region'] + NUMERIC].copy()
        for col in NUMERIC:
            part[col] = pd.to_numeric(part[col], errors='raise')
        long.append(part.melt(id_vars=['POI_ID', 'region'], value_vars=NUMERIC,
                              var_name='feature', value_name='value').assign(condition=condition))
    long = pd.concat(long, ignore_index=True)
    summary = long.groupby(['region', 'condition', 'feature']).value.agg(['count', 'mean', 'std', 'median']).reset_index()
    summary.to_csv(stage / 'tables' / 'distribution_summary.csv', index=False)
    delta = clean[['POI_ID', 'region']].copy()
    for col in NUMERIC:
        delta[f'{col}_delta'] = pd.to_numeric(attack[col]) - pd.to_numeric(clean[col])
    delta.to_csv(stage / 'tables' / 'paired_numeric_deltas.csv', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, feature in zip(axes, NUMERIC):
        sns.boxplot(data=long.loc[long.feature.eq(feature)], x='region', y='value', hue='condition',
                    palette=CONDITION_COLORS, showfliers=False, ax=ax)
        ax.tick_params(axis='x', rotation=55); ax.set_title(f'{feature} by region and condition')
    fig.tight_layout(); save_figure(fig, stage, 'boxplot_region_condition')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, feature in enumerate(NUMERIC):
        sns.histplot(data=long.loc[long.feature.eq(feature)], x='value', hue='condition',
                     palette=CONDITION_COLORS, bins=60, stat='density', common_norm=False, ax=axes[row, 0])
        sns.histplot(delta[f'{feature}_delta'], bins=60, color='#7A5195', ax=axes[row, 1])
        axes[row, 0].set_title(f'{feature} distribution'); axes[row, 1].set_title(f'{feature} paired delta')
    fig.tight_layout(); save_figure(fig, stage, 'histograms_and_deltas')
    sample_ids = clean.groupby('region', group_keys=False).sample(n=30, random_state=meta['seed']).POI_ID
    sample_clean = clean.set_index('POI_ID').loc[sample_ids]
    sample_attack = attack.set_index('POI_ID').loc[sample_ids]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (condition, frame) in zip(axes, [('clean', clean), ('adversarial', attack)]):
        ax.hexbin(pd.to_numeric(frame.X_COORD), pd.to_numeric(frame.Y_COORD), gridsize=60, mincnt=1, cmap='viridis')
        ax.set_title(f'{condition}: coordinate density'); ax.set_xlabel('X_COORD'); ax.set_ylabel('Y_COORD')
    save_figure(fig, stage, 'coordinate_hexbin')
    fig, ax = plt.subplots(figsize=(9, 8))
    for poi_id in sample_ids:
        ax.plot([float(sample_clean.loc[poi_id, 'X_COORD']), float(sample_attack.loc[poi_id, 'X_COORD'])],
                [float(sample_clean.loc[poi_id, 'Y_COORD']), float(sample_attack.loc[poi_id, 'Y_COORD'])],
                color='#999999', alpha=.25, linewidth=.5)
    ax.scatter(pd.to_numeric(sample_clean.X_COORD), pd.to_numeric(sample_clean.Y_COORD), s=8, label='clean')
    ax.scatter(pd.to_numeric(sample_attack.X_COORD), pd.to_numeric(sample_attack.Y_COORD), s=8, label='adversarial')
    ax.legend(); ax.set_title('Paired coordinate shifts (30 POIs per region)')
    save_figure(fig, stage, 'paired_coordinate_scatter')
    metrics = {'paired_rows': len(clean), 'delta': {col: {'mean': float(delta[f'{col}_delta'].mean()),
               'std': float(delta[f'{col}_delta'].std()), 'max_abs': float(delta[f'{col}_delta'].abs().max())}
               for col in NUMERIC}}
    body = ('지역별 box plot, 공통 bin histogram, 좌표 밀도와 paired 이동 그림을 생성했다. '
            '그래프는 같은 축과 조건 색상을 사용하며 산점도 이동선은 지역당 30개 고정 표본이다.')
    write_stage(stage, '02. 분포 시각화', body, meta, metrics)


def embeddings(clean, attack, stage, meta):
    sample_ids = clean.groupby('region', group_keys=False).sample(n=100, random_state=meta['seed']).POI_ID
    c = clean.set_index('POI_ID').loc[sample_ids].reset_index()
    a = attack.set_index('POI_ID').loc[sample_ids].reset_index()
    clean_x, attack_x, _ = matrix_for(c, a, fit_on='joint')
    joint = np.vstack([clean_x, attack_x])
    pca_dims = min(20, joint.shape[1] - 1)
    reduced = PCA(n_components=pca_dims, random_state=meta['seed']).fit_transform(joint)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca',
                max_iter=1000, random_state=meta['seed']).fit_transform(reduced)
    embedding = pd.concat([c[['POI_ID', 'region']].assign(condition='clean'),
                           a[['POI_ID', 'region']].assign(condition='adversarial')], ignore_index=True)
    embedding[['tsne_1', 'tsne_2']] = tsne

    clean_fit_x, attack_transform_x, _ = matrix_for(c, a, fit_on='clean')
    reducer = umap.UMAP(n_neighbors=15, min_dist=.1, metric='euclidean', random_state=meta['seed'],
                        transform_seed=meta['seed'])
    clean_umap = reducer.fit_transform(clean_fit_x)
    attack_umap = reducer.transform(attack_transform_x)
    embedding[['umap_1', 'umap_2']] = np.vstack([clean_umap, attack_umap])
    embedding.to_csv(stage / 'tables' / 'embedding_coordinates.csv', index=False)
    for method in ['tsne', 'umap']:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, condition in zip(axes, ['clean', 'adversarial']):
            part = embedding.loc[embedding.condition.eq(condition)]
            sns.scatterplot(data=part, x=f'{method}_1', y=f'{method}_2', hue='region',
                            s=12, alpha=.75, legend=False, ax=ax)
            ax.set_title(f'{method.upper()} — {condition}')
        save_figure(fig, stage, f'{method}_region_condition')

    full_clean_x, full_attack_x, _ = matrix_for(clean, attack, fit_on='joint')
    centroid_rows, centroids = [], []
    for condition, frame, values in [('clean', clean, full_clean_x), ('adversarial', attack, full_attack_x)]:
        for region in sorted(frame.region.unique()):
            centroid_rows.append({'label': f'{region}:{condition}', 'region': region, 'condition': condition})
            centroids.append(values[frame.region.eq(region).to_numpy()].mean(axis=0))
    centroid_table = pd.DataFrame(centroid_rows)
    centroid_table.to_csv(stage / 'tables' / 'dendrogram_centroids.csv', index=False)
    linked = linkage(np.asarray(centroids), method='ward')
    fig, ax = plt.subplots(figsize=(15, 7))
    dendrogram(linked, labels=centroid_table.label.tolist(), leaf_rotation=90, ax=ax)
    ax.set_title('Region-condition centroid dendrogram (Ward linkage)')
    save_figure(fig, stage, 'dendrogram_region_condition')
    paired = embedding.pivot(index='POI_ID', columns='condition', values=['tsne_1', 'tsne_2', 'umap_1', 'umap_2'])
    metrics = {'sampled_paired_rows': len(c), 'tsne': {'perplexity': 30, 'pca_dimensions': pca_dims},
               'umap': {'n_neighbors': 15, 'min_dist': .1},
               'mean_umap_shift': float(np.sqrt((paired[('umap_1','clean')]-paired[('umap_1','adversarial')])**2 +
                                                  (paired[('umap_2','clean')]-paired[('umap_2','adversarial')])**2).mean())}
    body = ('지역당 100개 paired POI를 사용해 joint t-SNE와 clean-fit UMAP을 생성했다. '
            'Dendrogram은 개별 23,800행이 아니라 34개 지역×조건 centroid를 Ward 방식으로 군집화했다. '
            '임베딩은 탐색 시각화이며 분류 성능 근거로 사용하지 않는다.')
    write_stage(stage, '03. t-SNE, UMAP, dendrogram', body, meta, metrics)


def statistics(clean, attack, stage, meta):
    tests = []
    for feature in NUMERIC:
        x, y = pd.to_numeric(clean[feature]), pd.to_numeric(attack[feature])
        diff = y - x
        t_result = ttest_rel(y, x)
        w_result = wilcoxon(diff)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        margin = t.ppf(.975, len(diff) - 1) * se
        tests += [
            {'family': 'paired_numeric', 'condition': 'all', 'feature': feature, 'test': 'paired_t',
             'statistic': t_result.statistic, 'p_value': t_result.pvalue, 'effect': diff.mean(),
             'ci_low': diff.mean() - margin, 'ci_high': diff.mean() + margin,
             'standardized_effect': diff.mean() / diff.std(ddof=1)},
            {'family': 'paired_numeric', 'condition': 'all', 'feature': feature, 'test': 'wilcoxon',
             'statistic': w_result.statistic, 'p_value': w_result.pvalue, 'effect': diff.median(),
             'ci_low': np.nan, 'ci_high': np.nan, 'standardized_effect': np.nan},
        ]
    for feature in CATEGORICAL:
        categories = sorted(set(clean[feature]) | set(attack[feature]))
        table = pd.crosstab(clean[feature], attack[feature]).reindex(index=categories, columns=categories, fill_value=0)
        result = SquareTable(table, shift_zeros=False).symmetry()
        tests.append({'family': 'paired_categorical', 'condition': 'all', 'feature': feature,
                      'test': 'bowker_symmetry', 'statistic': result.statistic, 'p_value': result.pvalue,
                      'effect': clean[feature].ne(attack[feature]).mean(), 'ci_low': np.nan,
                      'ci_high': np.nan, 'standardized_effect': np.nan})
    for condition, frame in [('clean', clean), ('adversarial', attack)]:
        for feature in CATEGORICAL:
            table = pd.crosstab(frame.region, frame[feature])
            effect, p, statistic = cramers_v(table)
            tests.append({'family': 'region_association', 'condition': condition, 'feature': feature,
                          'test': 'chi_square', 'statistic': statistic, 'p_value': p, 'effect': effect,
                          'ci_low': np.nan, 'ci_high': np.nan, 'standardized_effect': np.nan})
    tests = bh_frame(tests)
    tests.to_csv(stage / 'tables' / 'statistical_tests.csv', index=False)

    corr_rows = []
    for condition, frame in [('clean', clean), ('adversarial', attack)]:
        for left in NUMERIC:
            for right in NUMERIC:
                rho, p = spearmanr(pd.to_numeric(frame[left]), pd.to_numeric(frame[right]))
                corr_rows.append({'condition': condition, 'left': left, 'right': right,
                                  'method': 'spearman', 'value': rho, 'p_value': p})
    corr = pd.DataFrame(corr_rows); corr.to_csv(stage / 'tables' / 'correlations.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, condition in zip(axes, ['clean', 'adversarial']):
        matrix = corr.loc[corr.condition.eq(condition)].pivot(index='left', columns='right', values='value')
        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, cmap='vlag', ax=ax)
        ax.set_title(f'Spearman correlation — {condition}')
    save_figure(fig, stage, 'correlation_heatmap')

    clean_x, attack_x, _ = matrix_for(clean, attack, fit_on='joint')
    joint = np.vstack([clean_x, attack_x])
    pca_dims = min(10, joint.shape[1] - 1)
    projector = PCA(n_components=pca_dims, random_state=meta['seed']).fit(joint)
    clean_pca, attack_pca = projector.transform(clean_x), projector.transform(attack_x)
    cca = CCA(n_components=min(5, pca_dims), max_iter=1000)
    clean_c, attack_c = cca.fit_transform(clean_pca, attack_pca)
    canonical = [float(np.corrcoef(clean_c[:, i], attack_c[:, i])[0, 1]) for i in range(clean_c.shape[1])]
    pd.DataFrame({'component': range(1, len(canonical)+1), 'canonical_correlation': canonical}).to_csv(
        stage / 'tables' / 'cca_correlations.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=list(range(1, len(canonical)+1)), y=canonical, color='#2671B8', ax=ax)
    ax.set_ylim(0, 1); ax.set_xlabel('Canonical component'); ax.set_ylabel('Correlation')
    ax.set_title('Paired clean/adversarial canonical correlations')
    save_figure(fig, stage, 'cca_correlations')
    numeric_tests = tests.loc[tests.test.eq('paired_t')]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(numeric_tests.effect, numeric_tests.feature,
                xerr=[numeric_tests.effect - numeric_tests.ci_low, numeric_tests.ci_high - numeric_tests.effect],
                fmt='o', color='#7A5195'); ax.axvline(0, color='black', linewidth=.8)
    ax.set_title('Paired adversarial-clean mean difference (95% CI)')
    save_figure(fig, stage, 'paired_t_effects')
    metrics = {'tests': len(tests), 'fdr_significant_q_0_05': int(tests.q_value.lt(.05).sum()),
               'canonical_correlations': canonical, 'cca_pca_dimensions': pca_dims}
    body = ('좌표의 paired t/Wilcoxon, 범주 코드의 Bowker symmetry, 지역 연관 chi-square와 '
            "Cramér's V를 계산하고 전체 검정에 BH-FDR을 적용했다. CCA는 joint one-hot/표준화 "
            '공간을 공통 PCA로 축소한 뒤 paired clean/adversarial 블록에 적용했다.')
    write_stage(stage, '04. CCA, correlation, paired 통계 검정', body, meta, metrics)


def run(data_dir, output_root, seed=42):
    started = time.time()
    frames, manifest = load_dataset(data_dir, verify_hashes=True)
    clean = frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    attack = frames['attack'].loc[frames['attack'].split.eq('train')].reset_index(drop=True)
    if clean.POI_ID.tolist() != attack.POI_ID.tolist():
        raise ValueError('Paired POI order mismatch')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    meta = metadata(Path(data_dir), len(clean), seed, started)
    stages = [mkdir_stage(root, 1, 'descriptive'), mkdir_stage(root, 2, 'distribution'),
              mkdir_stage(root, 3, 'embedding'), mkdir_stage(root, 4, 'statistics')]
    descriptive(clean, attack, stages[0], meta)
    distributions(clean, attack, stages[1], meta)
    embeddings(clean, attack, stages[2], meta)
    statistics(clean, attack, stages[3], meta)
    manifest_out = {'status': 'complete', 'stages': [str(path) for path in stages],
                    'seconds': time.time() - started}
    (root / '_SUCCESS.json').write_text(json.dumps(manifest_out, indent=2) + '\n')
    print(json.dumps(manifest_out, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/poi_34k_seed42')
    parser.add_argument('--output-root', default='artifacts/eda-001/reports')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output_root, args.seed)


if __name__ == '__main__':
    main()
