"""Binary one-vs-rest evaluation for paired clean and attack inputs."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import features

METRICS = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'average_precision']


def evaluate_ovr(models, frame, threshold=0.5):
    """Evaluate one binary model per target region on the same POI rows."""
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    matrix = features(frame)
    per_region = {}
    predictions = []
    for target_region, model in sorted(models.items()):
        y_true = frame.region.eq(target_region).astype(int).to_numpy()
        if set(y_true) != {0, 1}:
            raise ValueError(f'Evaluation requires both classes for {target_region}')
        if model.classes_.tolist() != [0, 1]:
            raise ValueError(f'Unexpected model classes for {target_region}')
        y_score = model.predict_proba(matrix)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        per_region[target_region] = {
            'rows': len(frame), 'positives': int(y_true.sum()), 'negatives': int((1 - y_true).sum()),
            'threshold': threshold,
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_score)),
            'average_precision': float(average_precision_score(y_true, y_score)),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        }
        predictions.append(pd.DataFrame({
            'POI_ID': frame.POI_ID, 'target_region': target_region, 'y_true': y_true,
            'y_score': y_score, 'y_pred': y_pred,
        }))
    summary = {
        metric: float(np.mean([result[metric] for result in per_region.values()]))
        for metric in METRICS
    }
    summary['regions'] = len(per_region)
    summary['source_rows'] = len(frame)
    return {'summary': summary, 'per_region': per_region}, pd.concat(predictions, ignore_index=True)
