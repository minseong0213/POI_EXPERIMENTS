"""Stage 08: paired normal=0 versus adversarial=1 detection."""
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
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import precision_recall_curve, roc_curve
from statsmodels.stats.multitest import multipletests
import yaml

from .ablation import transformer
from .benchmark import METRICS, binary_metrics, estimator
from .data import CATEGORICAL, NUMERIC, load_dataset

warnings.filterwarnings('ignore', message='X does not have valid feature names')


def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(directory/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def run(config_path):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    tables=out/'tables';figures=out/'figures';tables.mkdir(parents=True);figures.mkdir()
    frames,manifest=load_dataset(cfg['data_dir']);features=NUMERIC+CATEGORICAL
    def combined(split):
        clean=frames['clean'].loc[frames['clean'].split.eq(split)].copy();clean['attack_label']=0;clean['condition']='clean'
        attack=frames['attack'].loc[frames['attack'].split.eq(split)].copy();attack['attack_label']=1;attack['condition']='adversarial'
        return pd.concat([clean,attack],ignore_index=True)
    train,validation=combined('train'),combined('validation')
    prep=transformer(features);x_train=prep.fit_transform(train);x_val=prep.transform(validation)
    y_train=train.attack_label.to_numpy();rows=[];predictions=[]
    for seed in cfg['seeds']:
        for model_name,params in cfg['models'].items():
            model=estimator(model_name,params,seed);fit_started=time.time();model.fit(x_train,y_train);fit_seconds=time.time()-fit_started
            score=model.predict_proba(x_val)[:,1];overall=binary_metrics(validation.attack_label.to_numpy(),score,cfg['threshold'])
            rows.append({'scope':'overall','region':'ALL','model':model_name,'seed':seed,'fit_seconds':fit_seconds,**overall})
            for region,indices in validation.groupby('region').groups.items():
                idx=np.asarray(list(indices));result=binary_metrics(validation.attack_label.iloc[idx].to_numpy(),score[idx],cfg['threshold'])
                rows.append({'scope':'region','region':region,'model':model_name,'seed':seed,'fit_seconds':fit_seconds,**result})
            predictions.append(pd.DataFrame({'POI_ID':validation.POI_ID,'region':validation.region,'condition':validation.condition,
                'y_true':validation.attack_label,'y_score':score,'y_pred':(score>=cfg['threshold']).astype(int),
                'model':model_name,'seed':seed,'split':'validation'}))
            print('completed',model_name,seed,flush=True)
    metrics=pd.DataFrame(rows);pred=pd.concat(predictions,ignore_index=True)
    metrics.to_csv(tables/'attack_detection_metrics.csv',index=False);pred.to_parquet(tables/'attack_detection_predictions.parquet',index=False)
    overall=metrics.loc[metrics.scope.eq('overall')];summary=overall.groupby('model')[METRICS].agg(['mean','std'])
    summary.to_csv(tables/'attack_detection_summary.csv')
    long=overall.melt(id_vars=['model','seed'],value_vars=METRICS,var_name='metric',value_name='value')
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for metric,ax in zip(METRICS,axes.flat):
        sns.barplot(data=long.loc[long.metric.eq(metric)],x='model',y='value',errorbar='sd',color='#2671B8',ax=ax)
        ax.tick_params(axis='x',rotation=35);ax.set_ylim(0,1);ax.set_title(metric)
    save(fig,figures,'attack_detection_metrics')
    seed=cfg['seeds'][0];selected=pred.loc[pred.seed.eq(seed)]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for model,group in selected.groupby('model'):
        fpr,tpr,_=roc_curve(group.y_true,group.y_score);precision,recall,_=precision_recall_curve(group.y_true,group.y_score)
        axes[0].plot(fpr,tpr,label=model);axes[1].plot(recall,precision,label=model)
    axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set_title('Attack detector ROC');axes[1].set_title('Attack detector PR')
    for ax in axes:ax.legend(fontsize=7)
    save(fig,figures,'attack_detection_pr_roc')
    region_f1=metrics.loc[metrics.scope.eq('region')].pivot_table(index='region',columns='model',values='f1',aggfunc='mean')
    region_f1.to_csv(tables/'attack_detection_region_f1.csv')
    fig,ax=plt.subplots(figsize=(10,8));sns.heatmap(region_f1,annot=True,fmt='.2f',vmin=0,vmax=1,cmap='viridis',ax=ax)
    ax.set_title('Attack detection F1 by region');save(fig,figures,'attack_detection_region_heatmap')
    fig,axes=plt.subplots(1,len(cfg['models']),figsize=(4*len(cfg['models']),4))
    for ax,(model,group) in zip(axes,overall.groupby('model')):
        matrix=np.array([[group.tn.sum(),group.fp.sum()],[group.fn.sum(),group.tp.sum()]],dtype=float);matrix/=matrix.sum(axis=1,keepdims=True)
        sns.heatmap(matrix,annot=True,fmt='.3f',vmin=0,vmax=1,cbar=False,ax=ax);ax.set_title(model);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    save(fig,figures,'attack_detection_confusion')

    # Cluster-robust logistic odds ratios account for paired clean/adversarial rows per POI.
    x=np.asarray(x_train.toarray() if hasattr(x_train,'toarray') else x_train);names=prep.get_feature_names_out().tolist()
    keep=np.ones(len(names),dtype=bool)
    for cat in CATEGORICAL:
        ids=[i for i,n in enumerate(names) if n.startswith(f'categorical__{cat}_')]
        if ids:keep[ids[0]]=False
    x=sm.add_constant(x[:,keep],has_constant='add');names=['intercept']+[n for i,n in enumerate(names) if keep[i]]
    odds=[];failure=None
    try:
        fit=sm.GLM(y_train,x,family=sm.families.Binomial()).fit(maxiter=200,cov_type='cluster',cov_kwds={'groups':train.POI_ID})
        for name,coef,se,p in zip(names,fit.params,fit.bse,fit.pvalues):
            ok=bool(np.isfinite(se) and np.isfinite(p));odds.append({'feature':name,'identifiable':ok,'log_odds':coef,
                'odds_ratio':np.exp(np.clip(coef,-50,50)),'ci_low':np.exp(np.clip(coef-1.96*se,-50,50)),
                'ci_high':np.exp(np.clip(coef+1.96*se,-50,50)),'p_value':p})
    except Exception as error:failure=repr(error)
    odds=pd.DataFrame(odds)
    if len(odds):odds['q_value']=multipletests(odds.p_value.fillna(1),method='fdr_bh')[1]
    odds.to_csv(tables/'attack_detection_odds_ratios.csv',index=False)
    if len(odds):
        shown=odds.loc[odds.identifiable & odds.feature.ne('intercept')].assign(magnitude=lambda d:d.log_odds.abs()).nlargest(15,'magnitude').sort_values('odds_ratio')
        fig,ax=plt.subplots(figsize=(9,7));ax.errorbar(shown.odds_ratio,range(len(shown)),xerr=[shown.odds_ratio-shown.ci_low,shown.ci_high-shown.odds_ratio],fmt='o')
        ax.set_yticks(range(len(shown)),shown.feature);ax.axvline(1,color='black',linewidth=.8);ax.set_xscale('log');ax.set_title('Attack detection odds ratios')
        save(fig,figures,'attack_detection_odds_ratio')
    ranking=overall.groupby('model').f1.mean().sort_values(ascending=False);best=ranking.index[0]
    metadata={'status':'complete','models':list(cfg['models']),'seeds':cfg['seeds'],'train_rows':len(train),
              'validation_rows':len(validation),'best_validation_f1_model':best,'odds_failure':failure,'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps({'f1_ranking':ranking.to_dict()},indent=2)+'\n')
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'report.md').write_text(f'''# 08. 공격 탐지

normal=0, adversarial=1로 두고 paired POI split을 유지해 5개 모델 × 3 seeds를 평가했다.
Validation F1 1위는 `{best}`이며 평균 F1은 {ranking.iloc[0]:.4f}다. 전체·지역별 지표,
PR/ROC, confusion matrix, POI별 score 원자료를 저장했다.

Odds ratio는 POI_ID cluster-robust 표준오차를 사용한 별도 logistic GLM이다. 식별되지
않는 계수는 `identifiable=false`, 전체 실패는 metadata의 `odds_failure`로 기록한다.
테스트 split은 사용하지 않았다.

식별 가능한 계수 수는 {int(odds.identifiable.sum()) if len(odds) else 0}/{len(odds)}다.
완전분리된 계수의 OR/CI/p-value는 해석하지 않는다.
''')
    (out.parent/'_SUCCESS_08.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/attack_detection.yaml');run(p.parse_args().config)
if __name__=='__main__':main()
