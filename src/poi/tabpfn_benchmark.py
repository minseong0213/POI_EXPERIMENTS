"""GPU stage 06 clean runner for TabPFN v2.5, v2.6 and v3."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import tabpfn
import torch
import yaml

from .data import CATEGORICAL, NUMERIC, load_dataset, sha256

VERSION_MAP={'v2.5':ModelVersion.V2_5,'v2.6':ModelVersion.V2_6,'v3':ModelVersion.V3}


def numeric_matrix(frame):
    matrix=frame[NUMERIC+CATEGORICAL].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():raise ValueError('TabPFN input contains non-finite values')
    return matrix


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def metrics(y,score,threshold):
    pred=(score>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
            'f1':f1_score(y,pred,zero_division=0),'accuracy':accuracy_score(y,pred),
            'roc_auc':roc_auc_score(y,score),'average_precision':average_precision_score(y,score),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def atomic_frame(frame,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    if path.suffix=='.csv':frame.to_csv(temporary,index=False)
    else:frame.to_parquet(temporary,index=False)
    os.replace(temporary,path)


def run(config_path,output_override=None):
    started=time.time();cfg=yaml.safe_load(Path(config_path).read_text());out=Path(output_override or cfg['output'])
    if (out/'_SUCCESS.json').exists():raise FileExistsError(f'Refusing to overwrite completed {out}')
    if out.exists() and not cfg.get('resume_partial',False):raise FileExistsError(f'Refusing to resume {out}')
    data_dir=os.environ.get('DATA_DIR',cfg['data_dir'])
    if not os.environ.get('TABPFN_TOKEN'):
        token_path=Path(os.environ.get('TABPFN_TOKEN_FILE','/root/.config/poi/tabpfn_token'))
        if token_path.is_file():os.environ['TABPFN_TOKEN']=token_path.read_text().strip()
    if not os.environ.get('TABPFN_TOKEN'):raise RuntimeError('TABPFN_TOKEN or TABPFN_TOKEN_FILE is required')
    out.mkdir(parents=True,exist_ok=True);work=out/'work';metric_shards=work/'metrics';prediction_shards=work/'predictions';timing_shards=work/'timing'
    metric_shards.mkdir(parents=True,exist_ok=True);prediction_shards.mkdir(exist_ok=True);timing_shards.mkdir(exist_ok=True)
    if cfg.get('conditions',['clean']) != ['clean']:
        raise ValueError('Clean model selection accepts only conditions: [clean]')
    frames,manifest=load_dataset(data_dir);train=frames['clean'].loc[frames['clean'].split.eq('train')].reset_index(drop=True)
    clean=frames['clean'].loc[frames['clean'].split.eq('validation')].reset_index(drop=True)
    x_train=numeric_matrix(train);x_test=numeric_matrix(clean)
    regions=sorted(manifest.region.unique());all_metrics=[];all_predictions=[];timings=[]
    for version in cfg['versions']:
        for seed in cfg['seeds']:
            model=None
            for start in range(0,len(regions),cfg['region_batch_size']):
                batch=regions[start:start+cfg['region_batch_size']]
                stem=f'{version.replace(".","_")}-seed{seed}-regions{start:02d}'
                metric_path=metric_shards/f'{stem}.csv';prediction_path=prediction_shards/f'{stem}.parquet';timing_path=timing_shards/f'{stem}.csv'
                existing=[metric_path.exists(),prediction_path.exists(),timing_path.exists()]
                if any(existing) and not all(existing):raise RuntimeError(f'Incomplete checkpoint shard: {stem}')
                if all(existing):
                    all_metrics.extend(pd.read_csv(metric_path).to_dict(orient='records'))
                    all_predictions.append(pd.read_parquet(prediction_path))
                    timings.extend(pd.read_csv(timing_path).to_dict(orient='records'))
                    print(f'resumed TabPFN {version} seed={seed} regions={batch}',flush=True)
                    continue
                if model is None:
                    model=TabPFNClassifier.create_default_for_version(VERSION_MAP[version],device=cfg['device'],
                        n_estimators=cfg['n_estimators'],random_state=seed,categorical_features_indices=[2,3,4],
                        show_progress_bar=True,memory_saving_mode='auto')
                labels=[train.region.eq(region).astype(int).to_numpy() for region in batch]
                tick=time.time();prob=None
                for attempt in range(1,cfg.get('batch_retries',1)+1):
                    try:
                        prob=model.predict_proba_batched([x_train]*len(batch),labels,[x_test]*len(batch))
                        break
                    except Exception as error:
                        if attempt>=cfg.get('batch_retries',1):raise
                        print(f'retrying TabPFN batch after {type(error).__name__} attempt={attempt}',flush=True)
                        torch.cuda.empty_cache();time.sleep(min(10*attempt,30))
                elapsed=time.time()-tick;batch_metrics=[];batch_predictions=[];batch_timings=[]
                for batch_index,region in enumerate(batch):
                    score=np.asarray(prob[batch_index])[:,1]
                    for condition,offset,frame in [('clean',0,clean)]:
                        values=score[offset:offset+len(frame)];truth=frame.region.eq(region).astype(int).to_numpy()
                        model_id=f'tabpfn_{version.replace(".","_")}'
                        batch_metrics.append({'model':model_id, 'checkpoint_version':version,
                            'seed':seed,'region':region,'condition':condition,**metrics(truth,values,cfg['threshold'])})
                        batch_predictions.append(pd.DataFrame({'POI_ID':frame.POI_ID,'model':model_id,
                            'seed':seed,'region':region,'condition':condition,'split':'validation','y_true':truth,
                            'y_score':values,'y_pred':(values>=cfg['threshold']).astype(int)}))
                    batch_timings.append({'version':version,'seed':seed,'region':region,'batch_seconds':elapsed,'batch_size':len(batch)})
                batch_predictions=pd.concat(batch_predictions,ignore_index=True)
                atomic_frame(pd.DataFrame(batch_metrics),metric_path);atomic_frame(batch_predictions,prediction_path);atomic_frame(pd.DataFrame(batch_timings),timing_path)
                all_metrics.extend(batch_metrics);all_predictions.append(batch_predictions);timings.extend(batch_timings)
                print(f'completed TabPFN {version} seed={seed} regions={batch}',flush=True)
                torch.cuda.empty_cache()
    result=pd.DataFrame(all_metrics);pred=pd.concat(all_predictions,ignore_index=True)
    if len(result)!=len(cfg['versions'])*len(cfg['seeds'])*len(regions):raise RuntimeError('Incomplete TabPFN metric cardinality')
    if len(pred)!=len(cfg['versions'])*len(cfg['seeds'])*len(regions)*len(clean):raise RuntimeError('Incomplete TabPFN prediction cardinality')
    result.to_csv(out/'metrics_by_region.csv',index=False);pred.to_parquet(out/'all_predictions.parquet',index=False)
    pd.DataFrame(timings).to_csv(out/'timing.csv',index=False)
    cache=Path.home()/'.cache'/'tabpfn';checkpoints=[]
    if cache.exists():
        for path in cache.rglob('*.ckpt'):checkpoints.append({'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha256(path)})
    metadata={'status':'complete','versions':cfg['versions'],'seeds':cfg['seeds'],'n_estimators':cfg['n_estimators'],
              'region_batch_size':cfg['region_batch_size'],'batch_retries':cfg.get('batch_retries',1),
              'resumable_shards':len(list(metric_shards.glob('*.csv'))),'fits':len(cfg['versions'])*len(cfg['seeds'])*len(regions),
              'python':platform.python_version(),'tabpfn':tabpfn.__version__,'torch':torch.__version__,
              'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'checkpoints':checkpoints,
              'data_sha256':{name:sha256(Path(data_dir)/name) for name in ['poi_data_region.csv','sample_manifest.csv']},
              'seconds':time.time()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'_SUCCESS.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stages/06_region_models_tabpfn.yaml');p.add_argument('--output')
    args=p.parse_args();run(args.config,args.output)
if __name__=='__main__':main()
