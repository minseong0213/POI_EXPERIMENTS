"""Stage 07: SHAP, LIME, feature removal, and inferential odds ratios."""
import argparse
import json
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.multitest import multipletests
import yaml

from .ablation import transformer
from .data import CATEGORICAL, NUMERIC, load_dataset

warnings.filterwarnings('ignore', message='X does not have valid feature names')


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def group_name(transformed_name):
    for feature in NUMERIC+CATEGORICAL:
        if transformed_name == f'numeric__{feature}' or transformed_name.startswith(f'categorical__{feature}_'):
            return feature
    return transformed_name


def dense(value):
    return value.toarray() if hasattr(value,'toarray') else np.asarray(value)


def shap_array(explainer, x):
    values=explainer.shap_values(x)
    if isinstance(values,list): values=values[-1]
    values=np.asarray(values)
    if values.ndim==3: values=values[:,:,1]
    return values


def fit_models(x_train, train, regions, params, seed):
    models={}
    for region in regions:
        y=train.region.eq(region).astype(int).to_numpy();pos=y.sum();neg=len(y)-pos
        weight=np.where(y==1,len(y)/(2*pos),len(y)/(2*neg))
        model=LGBMClassifier(random_state=seed,verbosity=-1,**params);model.fit(x_train,y,sample_weight=weight)
        models[region]=model
    return models


def run(config_path):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    tables=out/'tables';figures=out/'figures';html=out/'html';tables.mkdir(parents=True);figures.mkdir();html.mkdir()
    frames,manifest=load_dataset(cfg['data_dir']);train=frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    clean=frames['clean'].loc[frames['clean'].split.eq('validation')].reset_index(drop=True)
    attack=frames['attack'].loc[frames['attack'].split.eq('validation')].reset_index(drop=True)
    regions=sorted(manifest.region.unique());features=NUMERIC+CATEGORICAL
    prep=transformer(features);x_train=prep.fit_transform(train);x_clean=prep.transform(clean);x_attack=prep.transform(attack)
    names=prep.get_feature_names_out().tolist();groups=[group_name(n) for n in names]
    models=fit_models(x_train,train,regions,cfg['model_params'],cfg['seed'])
    sample_ids=clean.groupby('region',group_keys=False).sample(n=cfg['explain_per_region'],random_state=cfg['seed']).POI_ID
    positions=clean.index[clean.POI_ID.isin(sample_ids)].to_numpy();importance=[]
    seoul_values=None;seoul_x=None;seoul_explainer=None
    for region,model in models.items():
        explainer=shap.TreeExplainer(model)
        for condition,matrix in [('clean',x_clean),('adversarial',x_attack)]:
            x=dense(matrix[positions]);values=shap_array(explainer,x)
            for feature in features:
                cols=[i for i,g in enumerate(groups) if g==feature]
                importance.append({'region':region,'condition':condition,'feature':feature,
                                   'mean_abs_shap':float(np.abs(values[:,cols]).sum(axis=1).mean())})
            if region==cfg['representative_region'] and condition=='clean':
                seoul_values,seoul_x,seoul_explainer=values,x,explainer
    importance=pd.DataFrame(importance);importance.to_csv(tables/'shap_grouped_importance.csv',index=False)
    summary=importance.groupby(['condition','feature']).mean_abs_shap.mean().reset_index()
    summary.to_csv(tables/'shap_global_summary.csv',index=False)
    shap.summary_plot(seoul_values,seoul_x,feature_names=names,plot_type='dot',
                      color_bar=True,color_bar_label='Feature value',show=False,max_display=15)
    fig=plt.gcf();plt.gca().axvline(0,color='#444444',linewidth=.8,zorder=0)
    plt.gca().set_xlabel('SHAP value')
    fig.suptitle(f"SHAP — {cfg['representative_region']} vs others — clean")
    save(fig,figures,'shap_beeswarm_representative_clean')
    for feature in NUMERIC:
        index=names.index(f'numeric__{feature}')
        shap.dependence_plot(index,seoul_values,seoul_x,feature_names=names,show=False,interaction_index=None)
        save(plt.gcf(),figures,f'shap_dependence_{feature.lower()}')
    base=seoul_explainer.expected_value
    if isinstance(base,(list,np.ndarray)):base=np.asarray(base).reshape(-1)[-1]
    explanation=shap.Explanation(values=seoul_values[0],base_values=float(base),data=seoul_x[0],feature_names=names)
    shap.plots.waterfall(explanation,max_display=15,show=False);save(plt.gcf(),figures,'shap_waterfall_representative')
    force=shap.force_plot(float(base),seoul_values[0],seoul_x[0],feature_names=names)
    shap.save_html(str(html/'shap_force_representative.html'),force)

    train_dense=dense(x_train);clean_dense=dense(x_clean)
    lime_train=train_dense[np.linspace(0,len(train_dense)-1,min(5000,len(train_dense)),dtype=int)]
    lime=LimeTabularExplainer(lime_train,feature_names=names,class_names=['other_region',cfg['representative_region']],
                              mode='classification',random_state=cfg['seed'],discretize_continuous=True)
    lime_exp=lime.explain_instance(clean_dense[positions[0]],models[cfg['representative_region']].predict_proba,
                                   num_features=min(15,len(names)),num_samples=cfg['lime_samples'])
    lime_exp.save_to_file(str(html/'lime_representative.html'))
    pd.DataFrame(lime_exp.as_list(label=1),columns=['rule','weight']).to_csv(tables/'lime_representative.csv',index=False)
    save(lime_exp.as_pyplot_figure(label=1),figures,'lime_representative')

    # Recompute explanations after removing the dominant coordinate group.
    reduced_features=CATEGORICAL;reduced_prep=transformer(reduced_features)
    reduced_train=reduced_prep.fit_transform(train);reduced_clean=reduced_prep.transform(clean)
    reduced_model=fit_models(reduced_train,train,[cfg['representative_region']],cfg['model_params'],cfg['seed'])[cfg['representative_region']]
    reduced_names=reduced_prep.get_feature_names_out().tolist();reduced_values=shap_array(shap.TreeExplainer(reduced_model),dense(reduced_clean[positions]))
    shap.summary_plot(reduced_values,dense(reduced_clean[positions]),feature_names=reduced_names,
                      plot_type='dot',color_bar=True,color_bar_label='Feature value',show=False,max_display=15)
    plt.gca().axvline(0,color='#444444',linewidth=.8,zorder=0)
    save(plt.gcf(),figures,'shap_summary_no_coordinates')
    pd.DataFrame({'feature':reduced_names,'mean_abs_shap':np.abs(reduced_values).mean(axis=0)}).sort_values(
        'mean_abs_shap',ascending=False).to_csv(tables/'shap_no_coordinates.csv',index=False)

    # Multivariable GLM odds ratios. Robust HC3 standard errors; one reference dummy per category.
    odds_prep=transformer(features);odds_x=dense(odds_prep.fit_transform(train));odds_names=odds_prep.get_feature_names_out().tolist()
    keep=np.ones(len(odds_names),dtype=bool)
    for cat in CATEGORICAL:
        indices=[i for i,n in enumerate(odds_names) if n.startswith(f'categorical__{cat}_')]
        if indices:keep[indices[0]]=False
    odds_x=sm.add_constant(odds_x[:,keep],has_constant='add');odds_names=['intercept']+[n for i,n in enumerate(odds_names) if keep[i]]
    odds=[];failed=[]
    for region in regions:
        y=train.region.eq(region).astype(int).to_numpy()
        try:
            fit=sm.GLM(y,odds_x,family=sm.families.Binomial()).fit(maxiter=200,cov_type='HC3')
            for name,coef,se,pvalue in zip(odds_names,fit.params,fit.bse,fit.pvalues):
                identifiable=bool(np.isfinite(se) and np.isfinite(pvalue))
                odds.append({'region':region,'feature':name,'identifiable':identifiable,
                             'log_odds':coef,'odds_ratio':np.exp(np.clip(coef,-50,50)),
                             'ci_low':np.exp(np.clip(coef-1.96*se,-50,50)),
                             'ci_high':np.exp(np.clip(coef+1.96*se,-50,50)),'p_value':pvalue})
        except Exception as error:
            failed.append({'region':region,'error':repr(error)})
    odds=pd.DataFrame(odds)
    if len(odds):odds['q_value']=multipletests(odds.p_value.fillna(1),method='fdr_bh')[1]
    odds.to_csv(tables/'odds_ratios.csv',index=False);pd.DataFrame(failed).to_csv(tables/'odds_ratio_failures.csv',index=False)
    non_identifiable=odds.loc[~odds.identifiable,['region','feature']].to_dict(orient='records')
    representative=odds.loc[(odds.region.eq(cfg['representative_region'])) & odds.feature.ne('intercept') & odds.identifiable].copy()
    representative['magnitude']=representative.log_odds.abs();representative=representative.nlargest(15,'magnitude').sort_values('odds_ratio')
    fig,ax=plt.subplots(figsize=(9,7));ax.errorbar(representative.odds_ratio,range(len(representative)),
        xerr=[representative.odds_ratio-representative.ci_low,representative.ci_high-representative.odds_ratio],fmt='o')
    ax.set_yticks(range(len(representative)),representative.feature);ax.axvline(1,color='black',linewidth=.8);ax.set_xscale('log')
    ax.set_title(f"Odds ratios — {cfg['representative_region']} vs others");save(fig,figures,'odds_ratio_forest_representative')

    metadata={'status':'complete','model':'lightgbm provisional','seed':cfg['seed'],'regions':len(regions),
              'explained_pois':len(positions),'odds_ratio_failed_regions':failed,
              'non_identifiable_odds_coefficients':non_identifiable,'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'metrics.json').write_text(json.dumps({'global_shap':summary.to_dict(orient='records'),'odds_failures':failed},indent=2)+'\n')
    top=summary.groupby('feature').mean_abs_shap.mean().sort_values(ascending=False)
    (out/'report.md').write_text(f'''# 07. LightGBM SHAP, LIME, odds ratio

잠정 최고 LightGBM의 17개 OvR 모델에 대해 validation clean/adversarial SHAP을 계산했다.
그룹 기준 가장 중요한 피처는 `{top.index[0]}`이며 mean |SHAP|은 {top.iloc[0]:.4f}이다.
대표 상세 그림은 {cfg['representative_region']} vs others로 만들고, 좌표 제거 후 SHAP도 재계산했다.

LIME HTML/정적 그림, SHAP summary/dependence/waterfall/force, 전체 지역 odds ratio와
95% CI/p-value/FDR q-value를 저장했다. Odds ratio는 예측 모델 중요도가 아니라 별도
다변량 Binomial GLM의 해석 결과다. 완전분리로 표준오차 또는 p-value가 식별되지 않은
계수는 `identifiable=false`로 표시하며 해석에서 제외한다. 수렴 실패는 failures CSV에 남긴다.
''')
    (out.parent/'_SUCCESS_07.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/legacy/13_explainability_tree.yaml');run(p.parse_args().config)
if __name__=='__main__':main()
