"""Refit each feature subset on train and retain paired validation predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .data import FEATURES, sha256
from .expanded_explain import aligned_condition
from .selected_explain import ModelFactory, _matrix


def run(config_path):
    from .data import load_clean_splits
    cfg=yaml.safe_load(Path(config_path).read_text())
    cfg['data_dir']=os.environ.get('DATA_DIR',cfg['data_dir'])
    if os.environ.get('RESULT_DIR'):
        cfg['output']=str(Path(os.environ['RESULT_DIR'])/'reports'/'07_clean_feature_ablation')
    out=Path(cfg['output'])
    if (out/'_GENERATED.json').exists() or (out.exists() and not cfg.get('resume_partial',False)):
        raise FileExistsError(out)
    clean,_=load_clean_splits(cfg['data_dir'],splits=('train','validation'))
    train=clean.loc[clean.split.eq('train')].reset_index(drop=True)
    validation=clean.loc[clean.split.eq('validation')].reset_index(drop=True)
    attacks=None
    if cfg.get('attacks_file'):
        attacks=pd.read_parquet(cfg['attacks_file'],filters=[('split','==','validation')])
    (out/'tables').mkdir(parents=True,exist_ok=True)
    (out/'predictions').mkdir(exist_ok=True); (out/'work').mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,
        clean_sha256=sha256(Path(cfg['data_dir'])/'poi_data_region.csv'),
        manifest_sha256=sha256(Path(cfg['data_dir'])/'sample_manifest.csv'),
        attack_sha256=sha256(Path(cfg['attacks_file'])) if attacks is not None else None,
        code_sha256=sha256(Path(__file__)),
        factory_sha256=sha256(Path(__file__).with_name('selected_explain.py'))),sort_keys=True).encode()).hexdigest()
    identity_path=out/'input_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['sha256']!=fingerprint:
        raise ValueError('Ablation resume inputs changed; use a new output path')
    identity_path.write_text(json.dumps(dict(sha256=fingerprint))+'\n')
    rows=[]
    for model_id in cfg['model_ids']:
        for seed in cfg['seeds']:
            evaluations={'clean':validation}
            if attacks is not None:
                for condition in sorted(attacks.attack_condition.unique()):
                    evaluations[condition]=aligned_condition(attacks,validation,condition,seed)
            for ablation, features in cfg['feature_sets'].items():
                if not features or not set(features).issubset(FEATURES):
                    raise ValueError(f'Invalid features: {features}')
                factory=ModelFactory(model_id,dict(cfg,seed=seed),train)
                for region in sorted(train.region.unique()):
                    marker=out/'work'/f'{model_id}_{ablation}_{seed}_{region}.json'
                    if marker.exists():
                        saved=json.loads(marker.read_text())
                        if saved['fingerprint']!=fingerprint or any(
                                not (out/name).is_file() or sha256(out/name)!=digest
                                for name,digest in saved['files'].items()):
                            raise ValueError(f'Invalid resume shard: {marker}')
                        rows.extend(saved['metrics'])
                        continue
                    predict=factory.fit(features,train.region.eq(region).astype(int).to_numpy())
                    shard_rows,files=[],{}
                    for condition,frame in evaluations.items():
                        y=frame.region.eq(region).astype(int).to_numpy()
                        score=predict(_matrix(frame,features)); prediction=score>=cfg['threshold']
                        identity=dict(model=model_id,ablation=ablation,seed=seed,region=region,
                            attack_condition=condition,features='|'.join(features))
                        metrics=dict(f1=f1_score(y,prediction,zero_division=0),
                            precision=precision_score(y,prediction,zero_division=0),
                            recall=recall_score(y,prediction,zero_division=0),
                            accuracy=accuracy_score(y,prediction),roc_auc=roc_auc_score(y,score))
                        shard_rows.append(dict(**identity,**metrics))
                        prediction_path=out/'predictions'/f'{model_id}_{ablation}_{seed}_{region}_{condition}.parquet'
                        pd.DataFrame(dict(**{key:[value]*len(frame) for key,value in identity.items()},
                            POI_ID=frame.POI_ID.to_numpy(),y_true=y,y_score=score,
                            y_pred=prediction.astype(int))).to_parquet(prediction_path,index=False)
                        files[str(prediction_path.relative_to(out))]=sha256(prediction_path)
                    temporary=marker.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(fingerprint=fingerprint,metrics=shard_rows,files=files)))
                    os.replace(temporary,marker)
                    rows.extend(shard_rows)
                    pd.DataFrame(rows).to_csv(out/'tables'/'metrics.csv',index=False)
                    print(json.dumps(dict(model=model_id,seed=seed,ablation=ablation,region=region)),flush=True)
    summary=pd.DataFrame(rows).groupby(['model','ablation','attack_condition'])[
        ['f1','precision','recall','accuracy','roc_auc']].mean().reset_index()
    summary.to_csv(out/'tables'/'summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'tables'/'metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir(exist_ok=True)
    for model_id in cfg['model_ids']:
        part=summary.loc[summary.model.eq(model_id)&summary.attack_condition.eq('clean')]
        fig,axes=plt.subplots(1,5,figsize=(22,5))
        for axis,metric in zip(axes,['f1','precision','recall','accuracy','roc_auc']):
            axis.plot(part[metric],range(len(part)),marker='o')
            axis.set_yticks(range(len(part)),part.ablation)
            axis.set_title(metric); axis.grid(alpha=.2)
        fig.suptitle(f'{model_id}: refit feature ablation / validation / mean over regions and seeds')
        fig.tight_layout()
        fig.savefig(out/'figures'/f'{model_id}_clean_metrics.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    metadata=dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW',config=cfg,test_used=False,
        metric_rows=len(rows),clean_sha256=sha256(Path(cfg['data_dir'])/'poi_data_region.csv'),
        attack_sha256=sha256(Path(cfg['attacks_file'])) if attacks is not None else None,
        limitations=['Attack generation seed is matched to model seed; no full seed cross-product.',
                    'Paired bootstrap and SHAP/LIME after feature removal are separate stages.'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'report.md').write_text('# 선택 모델 피처 제거 ablation\n\n'
        f"모델 {', '.join(cfg['model_ids'])}; {len(cfg['feature_sets'])}개 피처 조합; "
        f"{len(cfg['seeds'])}개 seed. 각 조합을 train에서 다시 적합하고 validation을 평가했다. "
        '각 POI의 예측과 5개 지표를 저장했다. 독립 검수 대기.\n')
    (out/'_GENERATED.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(parser.parse_args().config)


if __name__=='__main__':
    main()
