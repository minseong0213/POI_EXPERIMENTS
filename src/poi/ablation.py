"""Stage 06: feature-group ablation for the provisional best tree model."""
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
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .benchmark import binary_metrics, estimator
from .data import CATEGORICAL, NUMERIC, load_dataset


def transformer(features):
    groups = []
    numeric = [x for x in NUMERIC if x in features]
    categorical = [x for x in CATEGORICAL if x in features]
    if numeric:
        groups.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                            ('scale', StandardScaler())]), numeric))
    if categorical:
        groups.append(('categorical', OneHotEncoder(handle_unknown='ignore'), categorical))
    return ColumnTransformer(groups)


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(directory/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def run(config_path):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    tables=out/'tables';figures=out/'figures';tables.mkdir(parents=True);figures.mkdir()
    frames,manifest=load_dataset(cfg['data_dir'])
    train=frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    validations={c:frames[k].loc[frames[k].split.eq('validation')].reset_index(drop=True)
                 for c,k in [('clean','clean'),('adversarial','attack')]}
    regions=sorted(manifest.region.unique());rows=[]
    for ablation,features in cfg['feature_sets'].items():
        prep=transformer(features);x_train=prep.fit_transform(train)
        x_val={c:prep.transform(f) for c,f in validations.items()}
        for seed in cfg['seeds']:
            for region in regions:
                y=train.region.eq(region).astype(int).to_numpy();pos=y.sum();neg=len(y)-pos
                weights=np.where(y==1,len(y)/(2*pos),len(y)/(2*neg))
                model=estimator('lightgbm',cfg['model_params'],seed);model.fit(x_train,y,sample_weight=weights)
                for condition,frame in validations.items():
                    truth=frame.region.eq(region).astype(int).to_numpy();score=model.predict_proba(x_val[condition])[:,1]
                    rows.append({'ablation':ablation,'features':'|'.join(features),'seed':seed,'region':region,
                                 'condition':condition,**binary_metrics(truth,score,cfg['threshold'])})
        print('completed',ablation,flush=True)
    result=pd.DataFrame(rows);result.to_csv(tables/'ablation_metrics.csv',index=False)
    pivot=result.pivot_table(index=['ablation','seed','region'],columns='condition',values='f1').reset_index()
    pivot['robust_f1']=(pivot.clean+pivot.adversarial)/2
    summary=pivot.groupby('ablation').robust_f1.agg(['mean','std']).sort_values('mean',ascending=False).reset_index()
    baseline=float(summary.loc[summary.ablation.eq('full'),'mean'].iloc[0]);summary['delta_from_full']=summary['mean']-baseline
    summary.to_csv(tables/'ablation_summary.csv',index=False)
    heat=pivot.pivot_table(index='region',columns='ablation',values='robust_f1',aggfunc='mean')
    heat.to_csv(tables/'ablation_by_region.csv')
    fig,ax=plt.subplots(figsize=(11,7));sns.heatmap(heat,annot=True,fmt='.2f',vmin=.5,vmax=1,cmap='viridis',ax=ax)
    ax.set_title('LightGBM robust F1 feature ablation');save(fig,figures,'ablation_heatmap')
    fig,ax=plt.subplots(figsize=(9,5));sns.barplot(data=summary,x='ablation',y='delta_from_full',color='#7A5195',ax=ax)
    ax.axhline(0,color='black',linewidth=.8);ax.tick_params(axis='x',rotation=35);ax.set_ylabel('Robust F1 delta')
    save(fig,figures,'ablation_delta')
    metadata={'status':'complete','model':'lightgbm provisional','scope':'validation','seeds':cfg['seeds'],
              'regions':len(regions),'fits':len(cfg['feature_sets'])*len(cfg['seeds'])*len(regions),
              'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps({'summary':summary.to_dict(orient='records')},indent=2)+'\n')
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    best=summary.iloc[0]
    (out/'report.md').write_text(f'''# 06. 잠정 LightGBM 피처 ablation

5단계 트리 비교의 validation 1위 LightGBM에 대해 {len(cfg['feature_sets'])}개 피처 구성을
17개 OvR × 3 seeds로 평가했다. 총 {metadata['fits']}회 학습이며 테스트셋은 사용하지 않았다.

가장 높은 구성은 `{best.ablation}`이고 robust F1은 {best['mean']:.4f}이다.
TabPFN 비교 전의 잠정 결과이므로 최종 모델 고정이나 테스트 공개에 사용하지 않는다.

표는 `tables/ablation_metrics.csv`, 요약은 `ablation_summary.csv`, 그림은
`figures/ablation_heatmap.*`, `ablation_delta.*`에 저장했다.
''')
    (out.parent/'_SUCCESS_06.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ablation.yaml');run(p.parse_args().config)
if __name__=='__main__':main()
