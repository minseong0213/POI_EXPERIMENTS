"""Row-normalized binary OvR confusion matrices supplement the frozen clean comparison."""
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
from sklearn.metrics import confusion_matrix
from .data import sha256


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    path = source/'tables/all_predictions.parquet'
    hashes = {str(path): sha256(path), str(source/'metadata.json'):sha256(source/'metadata.json'),
              str(Path(__file__)):sha256(__file__)}
    frame = pd.read_parquet(path)
    meta = json.loads((source/'metadata.json').read_text())
    models = meta['models']; regions = sorted(frame.region.unique()); seeds = meta['seeds']
    if len(models) != 8 or len(regions) != 17 or sorted(seeds) != [42, 202, 340]:
        raise ValueError('Expected full eight-model/17-region/three-seed comparison')
    if not frame.split.eq('validation').all() or not frame.condition.eq('clean').all():
        raise ValueError('Only clean validation predictions are permitted')
    if frame.duplicated(['model','seed','region','POI_ID']).any():
        raise ValueError('Duplicate prediction identity')
    groups = frame.groupby(['model','seed','region'])
    if len(groups) != 408 or not groups.size().eq(3400).all():
        raise ValueError('Incomplete comparison grid')
    tables, figures, provenance = (output/n for n in ['tables','figures','provenance'])
    for directory in (tables,figures,provenance): directory.mkdir(parents=True)
    shutil.copyfile(__file__, provenance/'normalized_confusion.py')
    rows, matrices = [], {}
    for (model,seed,region), group in groups:
        counts = confusion_matrix(group.y_true,group.y_pred,labels=[0,1])
        support = counts.sum(axis=1)
        if (support == 0).any(): raise ValueError('Undefined normalization: absent truth class')
        normalized = counts/support[:,None]
        matrices[(model,seed,region)] = normalized
        for truth in [0,1]:
            for pred in [0,1]:
                rows.append(dict(model=model,seed=int(seed),region=region,y_true=truth,y_pred=pred,
                                 count=int(counts[truth,pred]),true_class_support=int(support[truth]),
                                 normalized_by_true_class=float(normalized[truth,pred])))
    pd.DataFrame(rows).to_csv(tables/'normalized_confusion_all_seeds.csv',index=False)
    for region in regions:
        fig,axes = plt.subplots(2,4,figsize=(18,8))
        for ax,model in zip(axes.flat,models):
            sns.heatmap(matrices[(model,42,region)],vmin=0,vmax=1,annot=True,fmt='.2%',
                        cmap='Blues',cbar=False,ax=ax,xticklabels=['other','target'],yticklabels=['other','target'])
            ax.set(title=model,xlabel='Predicted',ylabel='True')
        fig.suptitle(f'{region} vs other regions | clean validation | row normalized | seed 42')
        fig.tight_layout();fig.savefig(figures/f'normalized_confusion_{region.lower()}.png',dpi=300)
        plt.close(fig)
    if any(sha256(p)!=h for p,h in hashes.items()): raise RuntimeError('Source changed')
    (output/'metadata.json').write_text(json.dumps(dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW',
        input_sha256=hashes,normalized_cells=len(rows),model_region_seed_groups=408,
        normalization='counts divided by true class support within each binary OvR task',
        figures_seed=42,table_seeds=seeds,test_features_read=0),indent=2)+'\n')
    (output/'report.md').write_text('# Normalized regional confusion matrices\n\n'
        '408 binary OvR matrices (1,632 cells) preserve all three model seeds. Each true-class row '
        'sums to one; counts and denominators remain in the CSV. The 17 PNG panels show all eight '
        'models for seed42. These are separate binary tasks, not a single multiclass matrix. '
        'Independent review is pending.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.output)
