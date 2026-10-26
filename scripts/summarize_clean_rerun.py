"""Summarize macro metrics across seeds, keeping region variability distinct."""
import argparse
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ['f1','precision','recall','accuracy','roc_auc','average_precision']

def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_csv(source)
    keys = ['model','condition','seed','region']
    if frame.duplicated(keys).any() or frame[METRICS].isna().any().any():
        raise ValueError('Duplicate strata or undefined metrics')
    groups = frame.groupby(['model','condition','seed'])
    if not groups.region.nunique().eq(17).all():
        raise ValueError('Every macro metric requires exactly 17 regional tasks')
    macro = groups[METRICS].mean().reset_index()
    if not macro.groupby(['model','condition']).seed.nunique().eq(3).all():
        raise ValueError('Three model seeds are required')
    summary = macro.groupby(['model','condition'])[METRICS].agg(['mean','std'])
    summary.columns = [f'{metric}_{stat}' for metric,stat in summary.columns]
    summary = summary.reset_index()
    output.mkdir(parents=True)
    macro.to_csv(output/'macro_by_seed.csv',index=False)
    summary.to_csv(output/'macro_seed_summary.csv',index=False)
    fig,axes=plt.subplots(1,5,figsize=(17,5))
    for ax,metric in zip(axes,METRICS[:5]):
        labels=summary.model+' / '+summary.condition
        ax.errorbar(summary[f'{metric}_mean'],range(len(summary)),
                    xerr=summary[f'{metric}_std'],fmt='o',capsize=3)
        ax.set_yticks(range(len(summary)),labels if metric=='f1' else ['']*len(summary))
        ax.set_xlabel(metric);ax.grid(alpha=.2)
    fig.suptitle('17-region macro metrics: mean ± SD across 3 model seeds (not a CI)')
    fig.tight_layout();fig.savefig(output/'seed_macro_metrics.png',dpi=300,bbox_inches='tight');plt.close(fig)
    (output/'metadata.json').write_text(json.dumps({'status':'generated_pending_independent_review',
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'macro_rows':len(macro),'summary_rows':len(summary),
        'definition':'Equal mean across 17 regions within each seed, then mean and sample SD across 3 seeds.',
        'limitations':['Seed SD is not a confidence interval.','Historical metrics_summary.csv std pools region-seed strata and is not seed uncertainty.','Shared validation and model selection imply conditional descriptive results.']},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metrics',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.metrics,args.output)
