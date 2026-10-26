"""Region/condition paired analysis with held-out CCA on the training split.

CCA directions and all transforms use a fit subset. Inference permutes the held-out
paired rows with directions fixed. The inference subset is part of train, never test.
"""
import argparse
import json
from pathlib import Path
import shutil
import importlib.metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t, wilcoxon, spearmanr, pearsonr, chi2
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
import yaml

from .data import NUMERIC, CATEGORICAL, FEATURES, sha256


def paired_tests(clean, attack):
    rows = []
    for feature in NUMERIC:
        x = pd.to_numeric(clean[feature]).to_numpy(float)
        y = pd.to_numeric(attack[feature]).to_numpy(float)
        delta = y-x
        sd = delta.std(ddof=1)
        if sd == 0:
            statistic, p_value = (0., 1.) if delta.mean() == 0 else (np.nan, np.nan)
        else:
            statistic, p_value = ttest_rel(y, x)
        margin = t.ppf(.975, len(delta)-1)*sd/np.sqrt(len(delta))
        rows.append(dict(feature=feature, test='paired_t', n=len(delta), statistic=statistic,
            p_value=p_value, effect=delta.mean(), ci_low=delta.mean()-margin,
            ci_high=delta.mean()+margin, assumption='paired POIs; mean-difference t interval'))
        stat, p = (0., 1.) if np.all(delta == 0) else wilcoxon(delta)
        rows.append(dict(feature=feature, test='wilcoxon', n=len(delta), statistic=stat,
            p_value=p, effect=np.median(delta), assumption='symmetric paired differences'))
    for feature in CATEGORICAL:
        x, y = clean[feature].astype(str).to_numpy(), attack[feature].astype(str).to_numpy()
        categories = sorted(set(x)|set(y))
        table = pd.crosstab(pd.Series(x), pd.Series(y)).reindex(
            index=categories, columns=categories, fill_value=0).to_numpy()
        upper = np.triu_indices(len(categories), 1)
        a, b = table[upper], table.T[upper]
        nonzero = (a+b)>0
        statistic = float(np.sum((a[nonzero]-b[nonzero])**2/(a[nonzero]+b[nonzero])))
        df = int(nonzero.sum())
        # Asymptotic Bowker may be unreliable with sparse off-diagonal cells.
        rows.append(dict(feature=feature, test='bowker_symmetry', n=len(x), statistic=statistic,
            df=df, p_value=float(chi2.sf(statistic, df)) if df else 1., effect=float(np.mean(x!=y)),
            sparse_pairs=int(np.sum((a+b)[nonzero]<5)), assumption='paired categorical; asymptotic chi-square'))
    return rows


def heldout_cca(clean, attack, seed=42, permutations=499):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))
    fit_ids, infer_ids = np.split(order, [len(order)//2])
    transform = ColumnTransformer([('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    # Numeric scaling/category vocabulary learned only from the clean fit subset.
    cx = transform.fit_transform(clean.iloc[fit_ids][FEATURES])
    ax = transform.transform(attack.iloc[fit_ids][FEATURES])
    dims = min(8, cx.shape[1], len(fit_ids)-2)
    pca = PCA(n_components=dims, random_state=seed).fit(cx)
    left, right = pca.transform(cx), pca.transform(ax)
    valid = (left.std(axis=0)>1e-9)&(right.std(axis=0)>1e-9)
    if not valid.any():
        return [dict(component=None, status='UNIDENTIFIABLE', fit_rows=len(fit_ids), inference_rows=len(infer_ids))]
    components = min(3, int(valid.sum()))
    cca = CCA(n_components=components, max_iter=2000).fit(left[:,valid], right[:,valid])
    u, v = cca.transform(pca.transform(transform.transform(clean.iloc[infer_ids][FEATURES]))[:,valid],
                         pca.transform(transform.transform(attack.iloc[infer_ids][FEATURES]))[:,valid])
    u = u-u.mean(0); v=v-v.mean(0)
    denom = np.sqrt((u*u).sum(0)*(v*v).sum(0))
    observed = np.divide((u*v).sum(0), denom, out=np.full(components, np.nan), where=denom>0)
    exceed = np.zeros(components, dtype=int)
    for _ in range(permutations):
        null = np.divide((u*v[rng.permutation(len(v))]).sum(0), denom,
            out=np.full(components, np.nan), where=denom>0)
        exceed += np.abs(null) >= np.abs(observed)
    return [dict(component=index+1, correlation=float(value),
        p_value=float((exceed[index]+1)/(permutations+1)) if np.isfinite(value) else np.nan,
        fit_rows=len(fit_ids), inference_rows=len(infer_ids), permutations=permutations,
        status='ESTIMATED' if np.isfinite(value) else 'UNIDENTIFIABLE') for index,value in enumerate(observed)]


def add_fdr(frame):
    frame['q_value']=np.nan
    finite = np.isfinite(frame.p_value)
    if finite.any():
        frame.loc[finite, 'q_value'] = multipletests(frame.loc[finite,'p_value'], method='fdr_bh')[1]
    return frame


def visualization_axes(clean, attacks_file, conditions, generation_seed):
    """Choose common axes from train and every plotted condition, never evaluation/test."""
    from .expanded_explain import aligned_condition, primary_mask
    limits={feature:[float(clean[feature].astype(float).min()),
                     float(clean[feature].astype(float).max())] for feature in NUMERIC}
    density={feature:0. for feature in NUMERIC}; delta={feature:0. for feature in NUMERIC}
    columns=['POI_ID','region','attack_condition','generation_seed','victim_model',
             'primary_evaluation_eligible','constraints_valid','label_preserved']
    columns += [f'{feature}_adv' for feature in FEATURES]
    for condition in conditions:
        rows=pd.read_parquet(attacks_file, columns=columns, filters=[('split','==','train'),
            ('generation_seed','==',generation_seed),('attack_condition','==',condition)])
        attack=aligned_condition(rows,clean,condition,generation_seed)
        valid=primary_mask(attack)
        if not valid.any():
            raise ValueError(f'No primary eligible visualization rows: {condition}')
        for feature in NUMERIC:
            left=clean.loc[valid,feature].to_numpy(float)
            right=attack.loc[valid,feature].to_numpy(float)
            bins=np.linspace(*limits[feature],61)
            if (right < bins[0]).any() or (right > bins[-1]).any():
                raise ValueError('Primary visualization coordinates exceed clean-train bounds')
            density[feature]=max(density[feature],float(np.histogram(left,bins=bins,density=True)[0].max()),
                                 float(np.histogram(right,bins=bins,density=True)[0].max()))
            delta[feature]=max(delta[feature],float(np.abs(right-left).max()))
    return dict(numeric_axis_limits=limits, histogram_density_max={k:1.05*v for k,v in density.items()},
                numeric_delta_axis_limits={k:[-1.05*max(v,1e-9),1.05*max(v,1e-9)] for k,v in delta.items()},
                axis_reference_rows=len(clean), axis_policy='all plotted train conditions at declared generation seed')


def run(config_path):
    from .data import load_clean_splits
    from .expanded_explain import aligned_condition, primary_mask
    from .eda import descriptive, distributions, embeddings
    cfg = yaml.safe_load(Path(config_path).read_text())
    output=Path(cfg['output'])
    if output.exists():
        raise FileExistsError(output)
    clean, _ = load_clean_splits(cfg['data_dir'], splits=('train',))
    clean=clean.reset_index(drop=True)
    for feature in CATEGORICAL:
        clean[feature]=pd.to_numeric(clean[feature], errors='raise').astype(float).astype(str)
    import pyarrow.dataset as ds
    dataset=ds.dataset(cfg['attacks_file'], format='parquet')
    condition_values=dataset.to_table(columns=['attack_condition'],
        filter=ds.field('split') == 'train').column('attack_condition')
    conditions=sorted(condition_values.unique().to_pylist())
    del condition_values
    if len(clean) != 23800 or clean.region.nunique() != 17 or len(conditions) != 32:
        raise ValueError('Expanded analysis requires the full frozen development design')
    if sorted(cfg['attack_seeds']) != [42, 202, 340]:
        raise ValueError('Expanded analysis requires all three generation seeds')
    output.mkdir(parents=True)
    (output/'tables').mkdir(); (output/'figures').mkdir()
    provenance=output/'provenance'; provenance.mkdir()
    source_paths=[Path(__file__), Path(__file__).with_name('eda.py'),
                  Path(__file__).with_name('data.py'), Path(__file__).with_name('expanded_explain.py'),
                  Path(config_path)]
    source_hashes={str(path):sha256(path) for path in source_paths}
    for path in source_paths:
        shutil.copyfile(path, provenance/path.name)
    input_hash=sha256(Path(cfg['attacks_file']))
    clean_hashes={name:sha256(Path(cfg['data_dir'])/name)
                  for name in ('poi_data_region.csv','sample_manifest.csv')}
    (provenance/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    axes=visualization_axes(clean,cfg['attacks_file'],conditions,cfg['visualization_attack_seed'])
    (provenance/'common_visualization_axes.json').write_text(json.dumps(axes,indent=2)+'\n')
    tests, cca_rows, correlation_rows, summaries, coverage = [], [], [], [], []
    for attack_seed in cfg['attack_seeds']:
        for condition in conditions:
            attacks=pd.read_parquet(cfg['attacks_file'], filters=[('split','==','train'),
                ('generation_seed','==',attack_seed), ('attack_condition','==',condition)])
            attack=aligned_condition(attacks, clean, condition, attack_seed)
            del attacks
            eligible=primary_mask(attack)
            eligible_clean=clean.loc[eligible].reset_index(drop=True)
            eligible_attack=attack.loc[eligible].reset_index(drop=True)
            for region in sorted(clean.region.unique()):
                selected=eligible_clean.region.eq(region)
                c=eligible_clean.loc[selected].reset_index(drop=True)
                a=eligible_attack.loc[selected].reset_index(drop=True)
                identity=dict(region=region, attack_condition=condition, generation_seed=attack_seed)
                tests.extend([dict(**identity, **row) for row in paired_tests(c,a)])
                cca_rows.extend([dict(**identity, **row) for row in heldout_cca(c,a,
                    seed=cfg['seed'], permutations=cfg['cca_permutations'])])
                for label, frame in [('clean',c),('adversarial',a)]:
                    for method,function in [('spearman',spearmanr),('pearson',pearsonr)]:
                        rho,p=function(frame[NUMERIC[0]].astype(float), frame[NUMERIC[1]].astype(float))
                        correlation_rows.append(dict(**identity, condition=label, method=method, correlation=rho,p_value=p))
                    for feature in NUMERIC:
                        values=frame[feature].astype(float)
                        summaries.append(dict(**identity, condition=label,feature=feature,n=len(values),
                            mean=values.mean(),std=values.std(),median=values.median(),
                            q25=values.quantile(.25),q75=values.quantile(.75),min=values.min(),max=values.max()))
                attempted=int(clean.region.eq(region).sum())
                coverage.append(dict(**identity, paired_rows=len(c), attempted_rows=attempted,
                    quarantined_rows=attempted-len(c), primary_valid_coverage=len(c)/attempted))
            if attack_seed == cfg['visualization_attack_seed']:
                root=output/f'{condition}_seed{attack_seed}'
                meta={'seed':cfg['seed'], 'analysis_scope':'train only', 'generation_seed':attack_seed,
                      'scope_rows_per_condition':len(eligible_clean),
                      'attack_condition':condition,'quarantined_pairs':int((~eligible).sum()),**axes}
                for name, function in [('02_descriptive',descriptive), ('03_distributions',distributions),
                                       ('04_embeddings',embeddings)]:
                    stage=root/name
                    (stage/'tables').mkdir(parents=True); (stage/'figures').mkdir()
                    function(eligible_clean,eligible_attack,stage,meta)
            print(json.dumps({'condition':condition,'generation_seed':attack_seed,'regions':17}),flush=True)
            # Persist partial tables after every condition so interrupted work is visible.
            for name, rows in [('paired_tests',tests),('heldout_cca',cca_rows),('correlations',correlation_rows)]:
                add_fdr(pd.DataFrame(rows)).to_csv(output/'tables'/f'{name}.csv',index=False)
            pd.DataFrame(summaries).to_csv(output/'tables'/'numeric_descriptive.csv',index=False)
            pd.DataFrame(coverage).to_csv(output/'tables'/'coverage.csv',index=False)
    for name, rows, metric in [('cca',cca_rows,'correlation'),('correlation',correlation_rows,'correlation')]:
        table=pd.DataFrame(rows).pivot_table(index='region',columns='attack_condition',values=metric,aggfunc='mean')
        fig,ax=plt.subplots(figsize=(15,7))
        im=ax.imshow(table.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set_xticks(range(len(table.columns)),table.columns,rotation=90)
        ax.set_yticks(range(len(table.index)),table.index)
        ax.set_title(f'{name}: mean across components / seeds (see individual CSV)')
        fig.colorbar(im,ax=ax)
        fig.savefig(output/'figures'/f'{name}_region_condition.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    if input_hash != sha256(Path(cfg['attacks_file'])) or any(
            sha256(Path(path)) != digest for path,digest in source_hashes.items()):
        raise RuntimeError('Analysis inputs/source changed during generation')
    if any(sha256(Path(cfg['data_dir'])/name) != digest for name,digest in clean_hashes.items()):
        raise RuntimeError('Clean analysis input changed during generation')
    metadata={'status':'GENERATED_PENDING_INDEPENDENT_REVIEW','test_used':False,'config':cfg,
        'groups':len(coverage),'expected_groups':len(clean.region.unique())*len(conditions)*len(cfg['attack_seeds']),
        'input_sha256':input_hash,'source_sha256':source_hashes,
        'clean_input_sha256':clean_hashes,
        'visualization_axes':axes,
        'runtime':{name:importlib.metadata.version(name) for name in
                   ('numpy','pandas','scipy','scikit-learn','statsmodels','umap-learn','matplotlib')},
        'limitations':['CCA inference conditions on fit-subset directions; train-only exploratory analysis.',
            'FDR families span all region/condition/seed rows within each CSV.',
            'Bowker asymptotic p-values can be unreliable for sparse pairs; sparse counts reported.',
            'Joint t-SNE is descriptive, not a predictive evaluation; embeddings plotted for one declared generation seed.']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'report.md').write_text('# 지역·공격 조건별 분석\n\n'
        '모든 지역·공격 조건·생성 seed의 paired 검정과 held-out CCA, 좌표 상관관계를 CSV로 기록했다. '
        'CCA는 train 내부를 적합·추론 표본으로 나누어 고정된 방향의 상관관계를 permutation으로 검정한다. '
        '다중 검정은 테이블별 전체 비교를 대상으로 BH FDR을 적용했다.\n\n'
        '선언된 시각화 seed에 대해 조건마다 Table One, box/histogram/scatter, t-SNE/UMAP/dendrogram을 생성했다. '
        'Test를 사용하지 않았으며 독립 검수 대기 상태이다.\n')
    (output/'_GENERATED.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(parser.parse_args().config)


if __name__=='__main__':
    main()
