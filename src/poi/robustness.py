"""Stage 10: sensitivity, seed stability, model ranking, and final ablation synthesis."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import yaml


def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)


def run(config_path):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    tables=out/'tables';figures=out/'figures';tables.mkdir(parents=True);figures.mkdir()
    model_metrics=pd.read_csv(cfg['model_metrics']);defense=pd.read_parquet(cfg['defense_predictions'])
    region_predictions=pd.read_parquet(cfg['region_predictions'])
    ablation=pd.read_csv(cfg['ablation_summary'])
    sensitivity=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);ids=np.array(sorted(defense.POI_ID.unique()));order=rng.permutation(ids)
        seed_data=defense.loc[defense.seed.eq(seed)]
        for ratio in cfg['attack_ratios']:
            attacked=set(order[:round(len(ids)*ratio)])
            selected=seed_data.loc[((seed_data.condition.eq('adversarial')) & seed_data.POI_ID.isin(attacked)) |
                                   ((seed_data.condition.eq('clean')) & ~seed_data.POI_ID.isin(attacked))]
            for region,group in selected.groupby('region'):
                truth=group.y_true.to_numpy();score=group.y_score.to_numpy();pred=(score>=cfg['region_threshold']).astype(int)
                accepted=group.attack_score.to_numpy()<cfg['detector_threshold']
                forced=pred.copy();forced[~accepted]=1-truth[~accepted]
                sensitivity.append({'seed':seed,'attack_ratio':ratio,'region':region,'coverage':accepted.mean(),
                    'f1':f1_score(truth,pred,zero_division=0),'precision':precision_score(truth,pred,zero_division=0),
                    'recall':recall_score(truth,pred,zero_division=0),'accuracy':accuracy_score(truth,pred),
                    'roc_auc':roc_auc_score(truth,score),'end_to_end_f1':f1_score(truth,forced,zero_division=0),
                    'end_to_end_accuracy':accuracy_score(truth,forced)})
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(tables/'sensitivity_attack_ratio.csv',index=False)
    sens_summary=(sensitivity.groupby('attack_ratio')
        [['f1','precision','recall','accuracy','roc_auc','coverage','end_to_end_f1','end_to_end_accuracy']]
        .agg(['mean','std']).reset_index())
    sens_summary.columns=['_'.join(filter(None,col if isinstance(col,tuple) else [col]))
                          for col in sens_summary.columns]
    sens_summary.to_csv(tables/'sensitivity_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    sns.lineplot(data=sensitivity,x='attack_ratio',y='f1',errorbar='sd',marker='o',label='unprotected ensemble F1',ax=axes[0])
    sns.lineplot(data=sensitivity,x='attack_ratio',y='end_to_end_f1',errorbar='sd',marker='o',label='defended end-to-end F1',ax=axes[0])
    sns.lineplot(data=sensitivity,x='attack_ratio',y='coverage',errorbar='sd',marker='o',color='#7A5195',ax=axes[1])
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1);axes[0].set_title('Attack-ratio sensitivity');axes[1].set_title('Defense coverage sensitivity')
    save(fig,figures,'sensitivity_attack_ratio')

    threshold_rows=[]
    for region_threshold in cfg['region_thresholds']:
        for detector_threshold in cfg['detector_thresholds']:
            for keys,group in defense.groupby(['seed','region','condition']):
                seed,region,condition=keys;truth=group.y_true.to_numpy()
                predicted=(group.y_score.to_numpy()>=region_threshold).astype(int)
                accepted=group.attack_score.to_numpy()<detector_threshold
                forced=predicted.copy();forced[~accepted]=1-truth[~accepted]
                threshold_rows.append({'seed':seed,'region':region,'condition':condition,
                    'region_threshold':region_threshold,'detector_threshold':detector_threshold,
                    'coverage':accepted.mean(),'region_f1':f1_score(truth,predicted,zero_division=0),
                    'end_to_end_f1':f1_score(truth,forced,zero_division=0)})
    threshold_sensitivity=pd.DataFrame(threshold_rows)
    threshold_sensitivity.to_csv(tables/'threshold_sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    sns.lineplot(data=threshold_sensitivity,x='region_threshold',y='region_f1',hue='condition',
                 errorbar='sd',marker='o',ax=axes[0])
    detector_view=threshold_sensitivity.loc[threshold_sensitivity.region_threshold.eq(cfg['region_threshold'])]
    sns.lineplot(data=detector_view,x='detector_threshold',y='end_to_end_f1',hue='condition',
                 errorbar='sd',marker='o',ax=axes[1])
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1)
    axes[0].set_title('Region-threshold sensitivity')
    axes[1].set_title('Detector-threshold end-to-end sensitivity')
    save(fig,figures,'sensitivity_thresholds')

    # Stability and ranking from paired clean/adversarial region metrics.
    f1=model_metrics.pivot_table(index=['model','seed','region'],columns='condition',values='f1').reset_index();f1['robust_f1']=(f1.clean+f1.adversarial)/2
    stability=f1.groupby(['model','region']).robust_f1.agg(['mean','std','min','max']).reset_index()
    stability.to_csv(tables/'stability_by_region.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5));sns.boxplot(data=f1,x='model',y='robust_f1',ax=ax);sns.stripplot(data=f1,x='model',y='robust_f1',color='black',size=2,alpha=.35,ax=ax)
    ax.tick_params(axis='x',rotation=35);ax.set_title('Robust F1 variation across regions and seeds');save(fig,figures,'stability_robust_f1')
    ranks=f1.copy();ranks['rank']=ranks.groupby(['seed','region']).robust_f1.rank(ascending=False,method='average')
    rank_summary=ranks.groupby('model').agg(mean_rank=('rank','mean'),mean_robust_f1=('robust_f1','mean'),std_robust_f1=('robust_f1','std')).sort_values('mean_rank').reset_index()
    rank_summary.to_csv(tables/'model_rank_summary.csv',index=False)
    rank_heat=ranks.pivot_table(index='region',columns='model',values='rank',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(9,8));sns.heatmap(rank_heat,annot=True,fmt='.1f',cmap='rocket_r',ax=ax);ax.set_title('Average model rank by region')
    save(fig,figures,'model_rank_heatmap')
    correlations=[]
    pivot=ranks.pivot_table(index=['region','model'],columns='seed',values='rank')
    for i,left in enumerate(cfg['seeds']):
        for right in cfg['seeds'][i+1:]:
            tau,p=kendalltau(pivot[left],pivot[right]);correlations.append({'seed_left':left,'seed_right':right,'kendall_tau':tau,'p_value':p})
    pd.DataFrame(correlations).to_csv(tables/'rank_stability_kendall.csv',index=False)

    # Radar is a compact supplement; exact values remain in CSV.
    radar=model_metrics.groupby('model')[['f1','precision','recall','accuracy','roc_auc','average_precision']].mean()
    radar.to_csv(tables/'model_metric_radar_values.csv')
    labels=list(radar.columns);angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist();angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw={'polar':True})
    for model,row in radar.iterrows():
        values=row.tolist()+[row.iloc[0]];ax.plot(angles,values,label=model);ax.fill(angles,values,alpha=.03)
    ax.set_xticks(angles[:-1],labels);ax.set_ylim(.7,1);ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),fontsize=7);ax.set_title('Model metric radar')
    save(fig,figures,'model_metric_radar')
    ablation.to_csv(tables/'final_ablation_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5));sns.barplot(data=ablation,x='ablation',y='mean',color='#2671B8',ax=ax);ax.tick_params(axis='x',rotation=35);ax.set_ylim(0,1);ax.set_ylabel('Robust F1')
    ax.set_title('Final feature ablation synthesis');save(fig,figures,'final_ablation')
    top_models=rank_summary.model.head(3).tolist()
    score_keys=['POI_ID','seed','region','condition','split','y_true']
    detector_scores=(defense.drop_duplicates(['POI_ID','seed','condition'])
                     [['POI_ID','seed','condition','attack_score']])
    variants={'top3_soft_vote':top_models,'best_single':[top_models[0]]}
    for removed in top_models:
        variants[f'top3_drop_{removed}']=[model for model in top_models if model!=removed]
    defense_ablation=[]
    for variant,models in variants.items():
        scores=(region_predictions.loc[region_predictions.model.isin(models)]
                .groupby(score_keys,as_index=False).y_score.mean()
                .merge(detector_scores,on=['POI_ID','seed','condition'],validate='many_to_one'))
        for keys,group in scores.groupby(['seed','region','condition']):
            seed,region,condition=keys;truth=group.y_true.to_numpy()
            predicted=(group.y_score.to_numpy()>=cfg['region_threshold']).astype(int)
            accepted=group.attack_score.to_numpy()<cfg['detector_threshold']
            forced=predicted.copy();forced[~accepted]=1-truth[~accepted]
            defense_ablation.append({'variant':variant,'members':','.join(models),'seed':seed,
                'region':region,'condition':condition,
                'ungated_f1':f1_score(truth,predicted,zero_division=0),
                'gated_end_to_end_f1':f1_score(truth,forced,zero_division=0),
                'coverage':accepted.mean()})
    defense_ablation=pd.DataFrame(defense_ablation)
    defense_summary=(defense_ablation.groupby(['variant','members','condition'])
                     [['ungated_f1','gated_end_to_end_f1','coverage']].mean().reset_index())
    defense_summary.to_csv(tables/'defense_ablation_summary.csv',index=False)
    defense_long=defense_ablation.melt(id_vars=['variant','condition','seed','region'],
        value_vars=['ungated_f1','gated_end_to_end_f1'],var_name='mode',value_name='f1')
    fig,axes=plt.subplots(1,2,figsize=(17,6),sharey=True)
    for ax,condition in zip(axes,['clean','adversarial']):
        sns.barplot(data=defense_long.loc[defense_long.condition.eq(condition)],x='variant',
            y='f1',hue='mode',errorbar='sd',ax=ax)
        ax.tick_params(axis='x',rotation=30);ax.set_ylim(0,1);ax.set_title(condition)
    fig.suptitle('Ensemble-component and detector-gate ablation')
    fig.tight_layout()
    save(fig,figures,'defense_ablation')
    sens_json=sens_summary
    model_count=model_metrics.model.nunique()
    metadata={'status':'complete','scope':'validation synthesis','attack_ratios':cfg['attack_ratios'],
              'seeds':cfg['seeds'],'model_count':model_count,
              'models':sorted(model_metrics.model.unique()),'ensemble_top3':top_models,
              'region_thresholds':cfg['region_thresholds'],
              'detector_thresholds':cfg['detector_thresholds'],'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'metrics.json').write_text(json.dumps({'model_ranking':rank_summary.to_dict(orient='records'),
        'rank_stability':correlations,'sensitivity':sens_json.to_dict(orient='records')},indent=2)+'\n')
    best=rank_summary.iloc[0];zero=sensitivity.loc[sensitivity.attack_ratio.eq(0)].end_to_end_f1.mean();full=sensitivity.loc[sensitivity.attack_ratio.eq(1)].end_to_end_f1.mean()
    (out/'report.md').write_text(f'''# 10. Sensitivity, stability, ranking, 최종 ablation

Validation POI에서 공격 혼합 비율 0/10/25/50/75/100%를 재현 가능하게 구성하고,
앙상블 단독과 공격 탐지 방어의 성능·coverage를 비교했다. 방어 end-to-end F1은 공격
비율 0에서 {zero:.4f}, 100%에서 {full:.4f}였다.

{model_count}개 모델의 지역·seed별 robust F1 순위 1위는 `{best.model}`이며 평균 순위는
{best.mean_rank:.3f}다. Kendall 순위 안정성, metric radar, 지역별 순위 heatmap,
6단계 feature ablation을 표와 PNG 300dpi로 저장했다.

이 결과는 validation 기반 종합이며 최종 잠금 모델의 test 평가는 별도로 기록한다.
''')
    (out.parent/'_SUCCESS_10.json').write_text(json.dumps(metadata,indent=2)+'\n');print(rank_summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/legacy/14_robustness_analysis.yaml');run(p.parse_args().config)
if __name__=='__main__':main()
