"""Stage 09: validation-only soft-voting ensemble and attack-detector defense."""
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import yaml

from .benchmark import binary_metrics


def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)


def end_to_end_metrics(y_true,y_pred,accepted):
    forced=y_pred.copy();forced[~accepted]=1-y_true[~accepted]
    return {'precision':precision_score(y_true,forced,zero_division=0),'recall':recall_score(y_true,forced,zero_division=0),
            'f1':f1_score(y_true,forced,zero_division=0),'accuracy':accuracy_score(y_true,forced)}


def run(config_path):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    tables=out/'tables';figures=out/'figures';tables.mkdir(parents=True);figures.mkdir()
    region_pred=pd.read_parquet(cfg['region_predictions']);detect=pd.read_parquet(cfg['detector_predictions'])
    region_pred=region_pred.loc[region_pred.model.isin(cfg['ensemble_models'])]
    if set(region_pred.model.unique()) != set(cfg['ensemble_models']):
        raise ValueError('One or more configured ensemble models have no predictions')
    ensemble=region_pred.groupby(['POI_ID','seed','region','condition','split','y_true'],as_index=False).y_score.mean()
    ensemble['y_pred']=(ensemble.y_score>=cfg['region_threshold']).astype(int)
    detector=detect.loc[detect.model.eq(cfg['detector_model']),['POI_ID','seed','condition','y_score']].rename(columns={'y_score':'attack_score'})
    merged=ensemble.merge(detector,on=['POI_ID','seed','condition'],validate='many_to_one')
    merged['accepted']=merged.attack_score<cfg['detector_threshold']
    rows=[]
    for keys,group in merged.groupby(['seed','region','condition']):
        seed,region,condition=keys;truth=group.y_true.to_numpy();pred=group.y_pred.to_numpy();accepted=group.accepted.to_numpy()
        baseline=binary_metrics(truth,group.y_score.to_numpy(),cfg['region_threshold'])
        if accepted.any() and len(set(truth[accepted]))==2:
            accepted_metrics={'accepted_'+k:v for k,v in binary_metrics(truth[accepted],group.y_score.to_numpy()[accepted],cfg['region_threshold']).items() if k in ['precision','recall','f1','accuracy']}
        else:accepted_metrics={f'accepted_{k}':np.nan for k in ['precision','recall','f1','accuracy']}
        rows.append({'seed':seed,'region':region,'condition':condition,'coverage':accepted.mean(),
                     'rejection_rate':1-accepted.mean(),**{f'baseline_{k}':v for k,v in baseline.items() if k in ['precision','recall','f1','accuracy']},
                     **accepted_metrics,**{f'end_to_end_{k}':v for k,v in end_to_end_metrics(truth,pred,accepted).items()}})
    metrics=pd.DataFrame(rows);metrics.to_csv(tables/'defense_metrics_by_region.csv',index=False)
    merged.to_parquet(tables/'defense_predictions.parquet',index=False)
    summary=metrics.groupby('condition').mean(numeric_only=True).drop(columns='seed').reset_index();summary.to_csv(tables/'defense_summary.csv',index=False)
    heat=metrics.pivot_table(index='region',columns='condition',values='end_to_end_f1',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(7,8));sns.heatmap(heat,annot=True,fmt='.2f',vmin=0,vmax=1,cmap='viridis',ax=ax);ax.set_title('Defense end-to-end F1')
    save(fig,figures,'defense_region_heatmap')
    long=metrics.melt(id_vars=['seed','region','condition'],value_vars=['baseline_f1','accepted_f1','end_to_end_f1'],var_name='metric',value_name='value')
    fig,ax=plt.subplots(figsize=(9,5));sns.barplot(data=long,x='metric',y='value',hue='condition',errorbar='sd',ax=ax);ax.set_ylim(0,1);ax.set_title('Ensemble and detector-gated region F1')
    save(fig,figures,'defense_f1_summary')
    detector_summary=merged.drop_duplicates(['POI_ID','seed','condition']).groupby(['seed','condition']).accepted.mean().reset_index()
    fig,ax=plt.subplots(figsize=(7,5));sns.barplot(data=detector_summary,x='condition',y='accepted',errorbar='sd',ax=ax);ax.set_ylim(0,1);ax.set_ylabel('Acceptance / coverage')
    ax.set_title('Detector gate coverage');save(fig,figures,'defense_coverage')

    # Multiclass risk-coverage curve from the 17 OvR scores.
    risks=[]
    base=merged.loc[merged.seed.eq(cfg['seeds'][0])]
    for condition,condition_data in base.groupby('condition'):
        wide=condition_data.pivot(index='POI_ID',columns='region',values='y_score')
        truth=condition_data.loc[condition_data.y_true.eq(1)].drop_duplicates('POI_ID').set_index('POI_ID').loc[wide.index,'region']
        predicted=wide.idxmax(axis=1);attack_score=condition_data.drop_duplicates('POI_ID').set_index('POI_ID').loc[wide.index,'attack_score']
        for threshold in np.linspace(0,1,41):
            accepted=attack_score<threshold
            risks.append({'condition':condition,'threshold':threshold,'coverage':accepted.mean(),
                          'selective_risk':1-accuracy_score(truth[accepted],predicted[accepted]) if accepted.any() else np.nan})
    risks=pd.DataFrame(risks);risks.to_csv(tables/'risk_coverage.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5));sns.lineplot(data=risks,x='coverage',y='selective_risk',hue='condition',marker='o',ax=ax)
    ax.set_title('Selective risk–coverage');save(fig,figures,'risk_coverage')
    outcomes=merged.drop_duplicates(['POI_ID','seed','condition']).groupby(['condition','accepted']).size().rename('count').reset_index()
    outcomes.to_csv(tables/'gate_outcomes.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5));sns.barplot(data=outcomes,x='condition',y='count',hue='accepted',ax=ax);ax.set_title('Gate outcomes: accepted vs rejected')
    save(fig,figures,'gate_outcomes')
    metadata={'status':'complete','ensemble_models':cfg['ensemble_models'],'detector_model':cfg['detector_model'],
              'seeds':cfg['seeds'],'rows':len(merged),'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps({'summary':summary.to_dict(orient='records')},indent=2)+'\n')
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    adv=summary.loc[summary.condition.eq('adversarial')].iloc[0]
    members='·'.join(cfg['ensemble_models'])
    (out/'report.md').write_text(f'''# 09. 앙상블과 탐지 후 지역분류 방어

validation 상위 3개 모델 {members}의 확률로 soft-voting ensemble을 만들고, 8단계
Random Forest 공격 탐지기가 정상으로 통과시킨 입력만 지역분류했다. test split은 사용하지 않았다.

Adversarial coverage는 {adv.coverage:.4f}, accepted-only F1은 {adv.accepted_f1:.4f},
거부를 실패로 포함한 end-to-end F1은 {adv.end_to_end_f1:.4f}다. 높은 공격 거부율만으로
성능이 좋아 보이지 않도록 baseline, coverage, accepted-only, end-to-end를 모두 기록했다.

결과는 `tables/defense_*`, risk-coverage와 gate 결과를 포함한 PNG 300dpi 그림에 저장했다.
''')
    (out.parent/'_SUCCESS_09.json').write_text(json.dumps(metadata,indent=2)+'\n');print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/legacy/10_filtered_region_classification.yaml');run(p.parse_args().config)
if __name__=='__main__':main()
