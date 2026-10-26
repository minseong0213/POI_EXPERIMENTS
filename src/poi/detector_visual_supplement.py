"""AP and regional detector curves from frozen URE validation predictions."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve
from .data import sha256


def run(source, analysis, output):
    source, analysis, output = map(Path, (source, analysis, output))
    if output.exists():
        raise FileExistsError(output)
    pred_path = source/'tables/ure_rf_validation_predictions.parquet'
    metrics_path = analysis/'tables/detector_metrics.csv'
    paths = [pred_path, metrics_path, source/'calibration_ids.json', Path(__file__)]
    hashes = {str(p):sha256(p) for p in paths}
    pred = pd.read_parquet(pred_path)
    ids = json.loads((source/'calibration_ids.json').read_text())
    evaluation = set(ids['evaluation_ids'])
    if evaluation & set(ids['calibration_ids']):
        raise ValueError('Calibration overlaps evaluation')
    pred = pred.loc[pred.partition.eq('evaluation')].copy()
    if not pred.split.eq('validation').all() or set(pred.POI_ID) != evaluation:
        raise ValueError('Evaluation split or identity mismatch')
    if not np.isfinite(pred.attack_score).all() or not pred.attack_score.between(0,1).all():
        raise ValueError('Invalid scores')
    metrics = pd.read_csv(metrics_path)
    regions = sorted(pred.region.unique())
    rows = []; curves = {}
    for seed, part in pred.groupby('seed'):
        clean = part.loc[part.condition.eq('clean')]
        if clean.POI_ID.duplicated().any() or set(clean.POI_ID) != evaluation:
            raise ValueError('Incomplete clean evaluation')
        attacks = part.loc[part.condition.eq('attack')]
        for (generation, condition), attack in attacks.groupby(['generation_seed','attack_condition']):
            if attack.POI_ID.duplicated().any() or set(attack.POI_ID) != evaluation:
                raise ValueError('Incomplete paired attack evaluation')
            mask = attack.primary_evaluation_eligible
            if mask.isna().any() or not mask.isin([True,False,0,1]).all():
                raise ValueError('Invalid eligibility')
            for cohort in ['all_intent','primary_valid']:
                active = attack if cohort == 'all_intent' else attack.loc[mask.astype(bool)]
                paired = clean.loc[clean.POI_ID.isin(active.POI_ID)]
                for region in ['ALL',*regions]:
                    c = paired if region == 'ALL' else paired.loc[paired.region.eq(region)]
                    a = active if region == 'ALL' else active.loc[active.region.eq(region)]
                    if set(c.POI_ID) != set(a.POI_ID):
                        raise ValueError('Regional pairs differ')
                    truth = np.r_[np.zeros(len(c),int),np.ones(len(a),int)]
                    scores = np.r_[c.attack_score,a.attack_score]
                    ap = average_precision_score(truth,scores) if len(c) else np.nan
                    rows.append(dict(model_seed=int(seed),generation_seed=int(generation),
                        attack_condition=condition,source_region=region,cohort=cohort,
                        pois=len(c),average_precision=ap,attack_prevalence=.5 if len(c) else np.nan))
                    if seed == 42 and generation == 42 and cohort == 'primary_valid' and len(c):
                        curves[(region,condition)] = (truth,scores)
    ap = pd.DataFrame(rows)
    keys = ['model_seed','generation_seed','attack_condition','source_region','cohort']
    merged = metrics.merge(ap,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('','_ap'))
    if len(ap) != 10368 or not merged._merge.eq('both').all() or not merged.pois.eq(merged.pois_ap).all():
        raise ValueError('AP factorial grid differs from frozen metric report')
    for name in ['tables','figures','provenance']:
        (output/name).mkdir(parents=True)
    ap.to_csv(output/'tables/average_precision.csv',index=False)
    counts = metrics.loc[metrics.model_seed.eq(42)&metrics.generation_seed.eq(42)&metrics.cohort.eq('primary_valid')]
    for region in ['ALL',*regions]:
        fig, axes = plt.subplots(1,3,figsize=(20,10),gridspec_kw={'width_ratios':[1,1,1.1]})
        for (selected,condition),(truth,scores) in sorted(curves.items()):
            if selected != region:
                continue
            fpr,tpr,_ = roc_curve(truth,scores)
            precision,recall,_ = precision_recall_curve(truth,scores)
            axes[0].plot(fpr,tpr,alpha=.7,label=condition)
            axes[1].plot(recall,precision,alpha=.7,label=condition)
        axes[0].plot([0,1],[0,1],'k--')
        axes[1].axhline(.5,color='black',ls='--')
        axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='PR: 50% paired attack prevalence')
        view = counts.loc[counts.source_region.eq(region)].sort_values('attack_condition').set_index('attack_condition')
        grid = view[['tn','fp','fn','tp']].div(view.pois.replace(0,np.nan),axis=0)
        # Each clean/attack class denominator is the number of paired POIs.
        sns.heatmap(grid,vmin=0,vmax=1,cmap='viridis',annot=True,fmt='.2f',ax=axes[2],cbar=False)
        axes[2].set(title='True-class normalized confusion by condition',xlabel='TN / FP / FN / TP',ylabel='Attack condition')
        axes[0].legend(fontsize=5,loc='lower right',ncol=2)
        fig.suptitle(f'URE-RF attack detection | {region} | primary-valid | model/generation seed42')
        fig.tight_layout();fig.savefig(output/'figures'/f'detector_{region}.png',dpi=300);plt.close(fig)
    shutil.copyfile(__file__,output/'provenance/detector_visual_supplement.py')
    if any(sha256(Path(p)) != h for p,h in hashes.items()):
        raise RuntimeError('Inputs/source changed')
    metadata = dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW',scientific_completion=False,
        input_source_sha256=hashes,ap_rows=len(ap),png_count=18,test_features_read=0,
        interpretation='Attack intent detection, not downstream regional classification',
        limitations=['Curves represent model/generation seed42; AP table covers all 3x3 seeds.',
                     '50% attack prevalence is a paired experimental design, not natural prevalence.',
                     'No new detector training, confidence interval, or superiority claim.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'report.md').write_text('# URE-RF AP and regional curves\n\n'
        'AP covers all three model seeds, three generation seeds, 32 conditions, overall and 17 source regions, '
        'for all-intent and primary-valid paired cohorts. Every condition has matched clean/attack POIs. '
        'Eighteen PNGs show seed42 ROC/PR and class-normalized confusion for each condition. '
        'Counts are inherited from the frozen detector report and each class is divided by paired POI count. '
        'These are detection metrics only; full Stage09 and downstream regional performance are not complete. '
        'Independent review pending.\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['source','analysis','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.source,a.analysis,a.output)
