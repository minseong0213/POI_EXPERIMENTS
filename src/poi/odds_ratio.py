"""Exploratory logistic-reference associations, never neural/tree parameter odds ratios."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import time
import platform
import importlib.metadata
import shutil

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.optimize import linprog
from statsmodels.stats.multitest import multipletests
import yaml

from .data import NUMERIC, CATEGORICAL, load_clean_splits, sha256


def design_matrix(train, frame):
    """Fit scaling, category vocabulary, reference levels and aliases on clean train only."""
    center = train[NUMERIC].astype(float).mean()
    scale = train[NUMERIC].astype(float).std(ddof=0).replace(0, 1)
    def encode(data):
        result = (data[NUMERIC].astype(float)-center)/scale
        for feature in CATEGORICAL:
            vocabulary = sorted(pd.to_numeric(train[feature]).unique())
            values = pd.to_numeric(data[feature])
            if not values.isin(vocabulary).all():
                raise ValueError(f'Unknown category for logistic reference: {feature}')
            for value in vocabulary[1:]:
                result[f'{feature}={value:g}'] = values.eq(value).astype(float)
        return result.reset_index(drop=True)
    fitted = encode(train)
    centered = fitted.to_numpy()-fitted.to_numpy().mean(axis=0)
    _, triangular, pivots = qr(centered, mode='economic', pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = max(centered.shape)*np.finfo(float).eps*(diagonal.max() if len(diagonal) else 0)
    rank = int((diagonal > tolerance).sum())
    retained = sorted(pivots[:rank])
    columns = fitted.columns[retained].tolist()
    excluded = [name for name in fitted if name not in columns]
    result = sm.add_constant(encode(frame)[columns], has_constant='add')
    return result, {'mean':center.to_dict(),'scale':scale.to_dict(),'retained':columns,
        'excluded_aliased_columns':excluded,'category_reference':'lowest observed clean-train code',
        'numeric_unit':'one clean-train standard deviation'}


def separation_check(design, truth):
    """Detect a nonzero separating direction with bounded coefficients via linear programming."""
    signed = (2*np.asarray(truth, float)-1)[:,None]*np.asarray(design,float)
    result = linprog(-signed.mean(axis=0), A_ub=-signed, b_ub=np.zeros(len(signed)),
                     bounds=[(-1,1)]*signed.shape[1], method='highs')
    if not result.success:
        return 'unresolved', str(result.message)
    margin = signed@result.x
    if margin.min() < -1e-7:
        return 'unresolved', 'solver constraint tolerance exceeded'
    if margin.mean() > 1e-7:
        return 'separation_or_quasiseparation', 'finite joint ordinary MLE does not exist'
    return 'no_separating_direction_detected', None


def fit_associations(design, truth, groups=None):
    truth = np.asarray(truth, int)
    if set(np.unique(truth)) != {0,1}:
        raise ValueError('Both binary outcomes are required')
    status, reason = separation_check(design, truth)
    fitted = None
    warning_messages = []
    if reason is None:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = sm.GLM(truth, design, family=sm.families.Binomial())
                fitted = model.fit(maxiter=200, cov_type='cluster' if groups is not None else 'HC0',
                                   cov_kwds={'groups':np.asarray(groups)} if groups is not None else None)
            warning_messages = [str(item.message) for item in caught]
            if not fitted.converged or any('separation' in message.lower() for message in warning_messages):
                reason = 'nonconvergence or separation warning'
        except Exception as error:
            reason = f'{type(error).__name__}: {error}'
    rows=[]
    for column in design:
        coefficient = standard_error = p_value = np.nan
        if fitted is not None and reason is None:
            coefficient, standard_error, p_value = (float(fitted.params[column]),
                float(fitted.bse[column]), float(fitted.pvalues[column]))
        identifiable = bool(reason is None and np.isfinite([coefficient,standard_error,p_value]).all())
        log_bounds = np.array([coefficient,coefficient-1.96*standard_error,coefficient+1.96*standard_error])
        # Never clip divergent coefficients into apparently finite odds ratios.
        estimable_ratio = identifiable and np.isfinite(log_bounds).all() and (np.abs(log_bounds)<700).all()
        ratios=np.exp(log_bounds) if estimable_ratio else [np.nan]*3
        row_reason = reason or ('' if identifiable else 'nonfinite coefficient, standard error or p-value')
        fit_outcome = ('separation' if status == 'separation_or_quasiseparation' else
                       'failed' if row_reason else 'fitted')
        rows.append({'feature':column,'identifiable':identifiable,'odds_ratio_representable':bool(estimable_ratio),
            'log_odds':coefficient,'standard_error':standard_error,'odds_ratio':ratios[0],
            'ci_low':ratios[1],'ci_high':ratios[2],'p_value':p_value,
            'fit_status':fit_outcome,'separation_status':status,'failure_reason':row_reason,
            'identifiable_semantics':'finite joint-fit estimate available; not individual coefficient identifiability',
            'covariance':'POI-cluster robust' if groups is not None else 'HC0',
            'rows':len(truth),'positive_rows':int(truth.sum())})
    return rows,warning_messages


def run(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());output=Path(cfg['output'])
    if output.exists():raise FileExistsError(output)
    started = time.time()
    source_files = [Path(__file__), Path(config_path), Path(__file__).with_name('data.py')]
    if cfg['mode'] == 'attack_detection':
        source_files += [Path(__file__).with_name('expanded_explain.py'),
                         Path(__file__).with_name('selected_explain.py')]
    input_files = [Path(cfg['data_dir'])/'poi_data_region.csv',
                   Path(cfg['data_dir'])/'sample_manifest.csv']
    if cfg.get('attacks_file'):
        input_files.append(Path(cfg['attacks_file']))
    before_hashes = {str(path): sha256(path) for path in source_files + input_files}
    output.mkdir(parents=True)
    provenance = output/'provenance'; provenance.mkdir()
    for index, path in enumerate(source_files):
        shutil.copyfile(path, provenance/f'{index}_{path.name}')
    execution = dict(started_unix=started, status='RUNNING',
        source_and_input_sha256_before=before_hashes,
        python=platform.python_version(),
        runtime={name:importlib.metadata.version(name) for name in
                 ['numpy','pandas','scipy','statsmodels','pyarrow','matplotlib']})
    (provenance/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    train,_=load_clean_splits(cfg['data_dir'],splits=('train',))
    train=train.reset_index(drop=True)
    for feature in NUMERIC+CATEGORICAL:train[feature]=pd.to_numeric(train[feature])
    (output/'tables').mkdir();rows=[];diagnostics=[]
    if cfg['mode']=='region':
        design,encoding=design_matrix(train,train)
        for region in sorted(train.region.unique()):
            result,notes=fit_associations(design,train.region.eq(region).astype(int))
            rows.extend([dict(region=region,condition='clean',**entry) for entry in result])
            diagnostics.append(dict(region=region,warnings=notes))
    elif cfg['mode']=='attack_detection':
        import pyarrow.dataset as ds
        from .expanded_explain import aligned_condition,primary_mask
        dataset=ds.dataset(cfg['attacks_file'],format='parquet')
        conditions=sorted(dataset.to_table(columns=['attack_condition'],filter=ds.field('split')=='train')
                          .column('attack_condition').unique().to_pylist())
        for seed in cfg['attack_seeds']:
            for condition in conditions:
                attacks=pd.read_parquet(cfg['attacks_file'],filters=[('split','==','train'),
                    ('generation_seed','==',seed),('attack_condition','==',condition)])
                attack=aligned_condition(attacks,train,condition,seed);valid=primary_mask(attack)
                clean_part=train.loc[valid].reset_index(drop=True)
                attack_part=attack.loc[valid].reset_index(drop=True)
                paired=pd.concat([clean_part,attack_part],ignore_index=True)
                design,encoding=design_matrix(train,paired)
                truth=np.r_[np.zeros(len(clean_part),int),np.ones(len(attack_part),int)]
                result,notes=fit_associations(design,truth,paired.POI_ID)
                rows.extend([dict(condition=condition,generation_seed=seed,**entry) for entry in result])
                diagnostics.append(dict(condition=condition,generation_seed=seed,warnings=notes,
                    eligible_pairs=int(valid.sum()),quarantined_pairs=int((~valid).sum())))
                print(json.dumps(diagnostics[-1]),flush=True)
    else:raise ValueError('mode must be region or attack_detection')
    result=pd.DataFrame(rows);result['q_value']=np.nan
    valid=result.p_value.notna() & result.feature.ne('const')
    if valid.any():result.loc[valid,'q_value']=multipletests(result.loc[valid,'p_value'],method='fdr_bh')[1]
    result.to_csv(output/'tables'/'odds_ratios.csv',index=False)
    figures=output/'figures';figures.mkdir()
    group_columns=['region'] if cfg['mode']=='region' else ['generation_seed','condition']
    for key,part in result.groupby(group_columns,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        label='_'.join(map(str,key))
        usable=part.loc[part.odds_ratio_representable & part.feature.ne('const')]
        fig,ax=plt.subplots(figsize=(9,max(4,.3*len(usable)+1)))
        if len(usable):
            ax.errorbar(usable.odds_ratio,range(len(usable)),
                xerr=np.vstack([usable.odds_ratio-usable.ci_low,usable.ci_high-usable.odds_ratio]),
                fmt='o',capsize=2)
            ax.set_xscale('log');ax.set_yticks(range(len(usable)),usable.feature)
            ax.axvline(1,color='grey',linestyle='--');ax.set_xlabel('Odds ratio, robust 95% Wald CI')
        else:
            ax.text(.5,.5,'No finite joint ordinary-MLE odds-ratio estimates',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
        ax.set_title(f'Logistic reference: {label} | excluded coefficients: {len(part)-len(usable)}')
        fig.tight_layout();fig.savefig(figures/f'odds_ratio_{label}.png',dpi=300,bbox_inches='tight');plt.close(fig)
    (output/'report.md').write_text('# Logistic reference odds ratios\n\n'
        'These are exploratory logistic-reference associations, not neural or tree-model parameter odds ratios. '
        'The attack-detection outcome is generated attack intent, not an existing detector decision. '
        'Train-only scaling and category alias removal are recorded in metadata. '
        'Perfect/quasi separation is checked by linear programming; fits without a finite joint MLE have missing OR/CI/p-values. The identifiable flag means finite joint-fit estimate availability, not a proof about each coefficient separately. '
        'Attack pairs use POI-cluster robust covariance; regional models use HC0 (unadjusted heteroskedasticity-robust sandwich). '
        'Spatial dependence between different POIs and causal effects are not addressed. '
        'Exploratory BH q-values use available nonintercept p-values in this output; formal FDR control under their dependence is not established. Independent review pending.\n')
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    metadata={'status':'generated_pending_independent_review','mode':cfg['mode'],'test_used':False,
        'fit_split':'train','encoding':encoding,'diagnostics':diagnostics,
        'input_sha256':{'clean':sha256(Path(cfg['data_dir'])/'poi_data_region.csv'),
                        'manifest':sha256(Path(cfg['data_dir'])/'sample_manifest.csv')},
        'code_sha256':sha256(Path(__file__)),
        'limitations':['Logistic reference associations, not coefficients of TabPFN or tree detectors.',
        'Attack outcome is generated attack intent, not a detector decision or natural attack prevalence.',
        'Conditional exploratory associations; not causal and not held-out model performance.',
        'POI clusters account for clean/attack pairing, not spatial dependence between different POIs.',
        'Exploratory BH family covers available nonintercept p-values; formal dependence conditions are unverified.',
        'Dependent categorical columns are removed using clean-train rank; retained contrasts depend on reference coding.']}
    if cfg.get('attacks_file'):metadata['input_sha256']['attacks']=sha256(Path(cfg['attacks_file']))
    if any(sha256(Path(path)) != digest for path,digest in before_hashes.items()):
        raise RuntimeError('Source/config/input changed during odds-ratio execution')
    execution.update(status='GENERATED_PENDING_INDEPENDENT_REVIEW',
                     ended_unix=time.time(),wall_seconds=time.time()-started,
                     source_and_input_unchanged=True,scientific_completion=False)
    (provenance/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    metadata['execution'] = execution
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'_GENERATED.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
