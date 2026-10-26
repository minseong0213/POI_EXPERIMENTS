"""Analyze frozen URE detector predictions, separately from downstream regional performance."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
from .data import sha256

METRICS=['f1','precision','recall','accuracy','roc_auc']


def score_pair(clean, attack, threshold):
    if clean.POI_ID.duplicated().any() or attack.POI_ID.duplicated().any():
        raise ValueError('Duplicate paired detector identity')
    if set(clean.POI_ID)!=set(attack.POI_ID):
        raise ValueError('Detector clean/attack pairs differ')
    if not len(clean):
        return dict(pois=0, **{metric:np.nan for metric in METRICS},
                    clean_retention=np.nan,attack_rejection=np.nan,tn=0,fp=0,fn=0,tp=0)
    values=np.r_[clean.attack_score,attack.attack_score]
    truth=np.r_[np.zeros(len(clean),int),np.ones(len(attack),int)]
    detected=values>threshold
    tn,fp,fn,tp=confusion_matrix(truth,detected,labels=[0,1]).ravel()
    return dict(pois=len(clean),f1=f1_score(truth,detected,zero_division=0),
                precision=precision_score(truth,detected,zero_division=0),
                recall=recall_score(truth,detected,zero_division=0),accuracy=accuracy_score(truth,detected),
                roc_auc=roc_auc_score(truth,values),clean_retention=float(np.mean(clean.attack_score<=threshold)),
                attack_rejection=float(np.mean(attack.attack_score>threshold)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def run(source, output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    pred_path=source/'tables/ure_rf_validation_predictions.parquet'
    source_paths=[pred_path,source/'calibration_ids.json',source/'seed_metadata.json',source/'metadata.json',
                  *sorted((source/'tables').glob('ure_training_seed*.csv')),Path(__file__)]
    hashes={str(p):sha256(p) for p in source_paths}
    frame=pd.read_parquet(pred_path)
    if not frame.split.eq('validation').all() or set(frame.seed)!={42,202,340}:
        raise ValueError('Unexpected split or model seeds')
    values=frame.attack_score.to_numpy(float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid detector probability')
    ids=json.loads((source/'calibration_ids.json').read_text())
    calibration=set(ids['calibration_ids']);evaluation=set(ids['evaluation_ids'])
    if calibration&evaluation or len(calibration|evaluation)!=3400:raise ValueError('Partition overlap/incomplete')
    expected=frame.POI_ID.map(lambda value:'calibration' if value in calibration else 'evaluation' if value in evaluation else None)
    if not expected.eq(frame.partition).all():raise ValueError('Partition disagrees with saved IDs')
    conditions=sorted(frame.loc[frame.condition.eq('attack'),'attack_condition'].unique())
    regions=sorted(frame.region.unique())
    if len(conditions)!=32 or len(regions)!=17:raise ValueError('Incomplete detector design')
    tables,figures,provenance=(output/n for n in ['tables','figures','provenance'])
    for directory in [tables,figures,provenance]:directory.mkdir(parents=True)
    shutil.copyfile(__file__,provenance/'detector_report.py')
    rows=[]; calibration_rows=[]; curve_pairs={}
    stored_thresholds={int(x['seed']):x['threshold'] for x in json.loads((source/'seed_metadata.json').read_text())}
    for seed in [42,202,340]:
        selected=frame.loc[frame.seed.eq(seed)]
        clean_cal=selected.loc[selected.condition.eq('clean')&selected.partition.eq('calibration')]
        threshold=float(stored_thresholds[seed])
        calibrated=float(np.quantile(clean_cal.attack_score,.95,method='higher'))
        if threshold!=calibrated or not selected.threshold.eq(threshold).all():
            raise ValueError('Stored detector threshold differs from clean calibration quantile')
        if not selected.accepted.eq(selected.attack_score<=threshold).all():raise ValueError('Acceptance mismatch')
        calibration_rows.append(dict(seed=seed,threshold=threshold,calibration_pois=len(clean_cal),
                                      calibration_clean_retention=float(np.mean(clean_cal.attack_score<=threshold))))
        clean=selected.loc[selected.condition.eq('clean')&selected.partition.eq('evaluation')]
        if set(clean.POI_ID)!=evaluation or clean.POI_ID.duplicated().any():raise ValueError('Incomplete clean evaluation')
        if not clean.attack_label.eq(0).all():raise ValueError('Invalid clean attack label')
        attacks=selected.loc[selected.condition.eq('attack')&selected.partition.eq('evaluation')]
        if set(attacks[['generation_seed','attack_condition']].itertuples(index=False,name=None))!={
            (g,c) for g in [42,202,340] for c in conditions}:raise ValueError('Attack factorial grid missing')
        for (generation,condition),attack in attacks.groupby(['generation_seed','attack_condition']):
            if set(attack.POI_ID)!=evaluation or attack.POI_ID.duplicated().any():raise ValueError('Incomplete attack evaluation')
            if not attack.attack_label.eq(1).all():raise ValueError('Invalid attack intent label')
            masks=attack.primary_evaluation_eligible
            if masks.isna().any() or not masks.isin([True,False,0,1]).all():raise ValueError('Invalid primary mask')
            for cohort in ['all_intent','primary_valid']:
                active=attack if cohort=='all_intent' else attack.loc[masks.astype(bool)]
                matched=clean.loc[clean.POI_ID.isin(active.POI_ID)]
                if seed==42 and generation==42 and cohort=='primary_valid':
                    curve_pairs[condition]=(matched,active,threshold)
                for region in ['ALL',*regions]:
                    c=matched if region=='ALL' else matched.loc[matched.region.eq(region)]
                    a=active if region=='ALL' else active.loc[active.region.eq(region)]
                    rows.append(dict(model_seed=seed,generation_seed=int(generation),attack_condition=condition,
                                     source_region=region,cohort=cohort,threshold=threshold,
                                     **score_pair(c,a,threshold)))
    result=pd.DataFrame(rows);result.to_csv(tables/'detector_metrics.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(tables/'calibration_thresholds.csv',index=False)
    regional=result.loc[result.source_region.ne('ALL')]
    macro=regional.groupby(['model_seed','generation_seed','attack_condition','cohort'])[METRICS+['clean_retention','attack_rejection']].mean().reset_index()
    macro.to_csv(tables/'region_macro_metrics.csv',index=False)
    for metric in METRICS:
        grid=regional.loc[regional.cohort.eq('primary_valid')].pivot_table(index='source_region',columns='attack_condition',values=metric,aggfunc='mean')
        fig,ax=plt.subplots(figsize=(16,8));sns.heatmap(grid,vmin=0,vmax=1,cmap='viridis',ax=ax)
        ax.set_title(f'URE-RF attack detection {metric}: mean over 3×3 seeds; 50% paired attack prevalence')
        fig.tight_layout();fig.savefig(figures/f'detector_{metric}_by_region_condition.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17,7))
    for condition,(clean,attack,threshold) in curve_pairs.items():
        if not len(clean):continue
        y=np.r_[np.zeros(len(clean)),np.ones(len(attack))];prob=np.r_[clean.attack_score,attack.attack_score]
        fpr,tpr,_=roc_curve(y,prob);precision,recall,_=precision_recall_curve(y,prob)
        axes[0].plot(fpr,tpr,label=condition,alpha=.75);axes[1].plot(recall,precision,label=condition,alpha=.75)
    axes[0].plot([0,1],[0,1],'k--');axes[1].axhline(.5,color='black',ls='--')
    axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Attack detector ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Attack detector PR (50% attack prevalence)')
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7)
    fig.suptitle('URE-RF | primary-valid paired evaluation | model and generation seed42')
    fig.tight_layout();fig.savefig(figures/'detector_roc_pr_seed42.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for path in sorted((source/'tables').glob('ure_training_seed*.csv')):
        history=pd.read_csv(path);label=f'seed {int(history.seed.iloc[0])}'
        axes[0].plot(history.epoch,history.clean_region_loss,label=label)
        axes[1].plot(history.epoch,history.adversarial_region_loss,label=label)
    for ax,title in zip(axes,['URE clean training loss','URE inner-adversarial training loss']):
        ax.set(xlabel='Epoch',ylabel='Loss',title=title);ax.legend()
    fig.tight_layout();fig.savefig(figures/'ure_training_curves.png',dpi=300);plt.close(fig)
    if any(sha256(p)!=h for p,h in hashes.items()):raise RuntimeError('Frozen detector inputs changed')
    metadata=dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW',source_sha256=hashes,
        evaluation_pois=len(evaluation),calibration_pois=len(calibration),metric_rows=len(result),
        comparison='attack detection only; not downstream region classification',attack_prevalence=.5,
        threshold='saved 95% clean-retention quantile from calibration only',test_features_read=0,
        limitations=['Conditional evaluation partition; earlier exploratory whole-validation use remains disclosed.',
                     'Point metrics only; proposed-model comparison and paired CI belong to Stage10.',
                     'Unchanged or unsuccessful attack attempts retain attack-intent label1.',
                     'Full original URE equivalence is not claimed; see producer method deviations.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'report.md').write_text('# URE-RF detector evaluation\n\n'
        f'{len(evaluation)} evaluation POIs, disjoint from {len(calibration)} calibration POIs. '
        'Three model seeds × three attack generation seeds × 32 conditions are evaluated globally and '
        'within 17 source regions, for both all-intent and primary-valid paired cohorts. '
        'Every condition uses equal clean/attack counts (50% prevalence). Thresholds come only from '
        'saved clean-calibration 95% quantiles. F1, precision, recall, accuracy, ROC-AUC, retention and '
        'attack rejection are saved with confusion counts.\n\n'
        'These are attack-detector metrics, not regional-classifier results. No proposed-model '
        'superiority or defense improvement is inferred. Independent review is pending.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.output)
