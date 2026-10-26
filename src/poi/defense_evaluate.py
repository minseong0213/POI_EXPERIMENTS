"""Paired evaluation primitives for detectors followed by one frozen classifier.

A rejected row is always retained in the denominator of coverage and of
accepted-and-correct fraction. Conditional classification metrics describe only
accepted rows. Full-population repaired metrics require a prediction for every POI.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .data import NUMERIC, CATEGORICAL, FEATURES

METRICS=('f1','precision','recall','accuracy','roc_auc')


def binary_metrics(truth, score, threshold=.5, *, threshold_operator="gte"):
    truth=np.asarray(truth,dtype=int); score=np.asarray(score,dtype=float)
    if truth.shape!=score.shape or truth.ndim!=1:
        raise ValueError('Truth and score must be matching one-dimensional arrays')
    if not np.isin(truth,[0,1]).all() or not np.isfinite(score).all() or ((score<0)|(score>1)).any():
        raise ValueError('Binary labels and finite [0,1] scores required')
    if threshold_operator not in {"gte", "gt"}:
        raise ValueError("threshold_operator must be 'gte' or 'gt'")
    if not len(truth):
        return dict(n=0,positives=0,f1=0.,precision=0.,recall=0.,accuracy=np.nan,
                    roc_auc=np.nan,zero_division_policy=0,empty_population=True)
    prediction=score>=threshold if threshold_operator=="gte" else score>threshold
    return dict(n=len(truth),positives=int(truth.sum()),
        f1=f1_score(truth,prediction,zero_division=0),
        precision=precision_score(truth,prediction,zero_division=0),
        recall=recall_score(truth,prediction,zero_division=0),
        accuracy=accuracy_score(truth,prediction),
        roc_auc=roc_auc_score(truth,score) if len(np.unique(truth))==2 else np.nan,
        zero_division_policy=0,empty_population=False)


def select_detection_threshold(clean_calibration_scores, clean_retention=.95):
    """Calibrate once using a declared clean calibration partition, not test."""
    values=np.asarray(clean_calibration_scores,dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0<clean_retention<=1:
        raise ValueError('Valid nonempty calibration scores and retention are required')
    # Tied scores are accepted; empirical retention may exceed the target.
    return float(np.quantile(values,clean_retention,method='higher'))


def paired_defense_metrics(truth, unmodified_score, repaired_score, detector_score,
                           detection_threshold, classification_threshold=.5):
    truth=np.asarray(truth); original=np.asarray(unmodified_score); repaired=np.asarray(repaired_score)
    detection=np.asarray(detector_score)
    if any(array.shape!=truth.shape for array in [original,repaired,detection]):
        raise ValueError('Every original POI needs aligned detector and classifier scores')
    if (not np.isfinite(detection).all() or ((detection<0)|(detection>1)).any()
            or not np.isfinite(original).all() or not np.isfinite(repaired).all()
            or ((original<0)|(original>1)).any() or ((repaired<0)|(repaired>1)).any()):
        raise ValueError('Finite [0,1] detector and classifier probabilities required')
    accepted=detection<=detection_threshold
    # The repair branch applies only to flagged rows; accepted predictions stay fixed.
    final=np.where(accepted,original,repaired)
    full=binary_metrics(truth,final,classification_threshold)
    baseline=binary_metrics(truth,original,classification_threshold)
    conditional=binary_metrics(truth[accepted],original[accepted],classification_threshold)
    coverage=float(accepted.mean()) if len(accepted) else np.nan
    correct=((original>=classification_threshold)==truth)&accepted
    return dict(coverage=coverage,accepted=int(accepted.sum()),rejected=int((~accepted).sum()),
        total=len(truth),accepted_and_correct_fraction=(float(correct.mean()) if len(truth) else np.nan),
        empty_population=not len(truth),zero_division_policy=0,
        unfiltered=baseline,accepted_only=conditional,repaired_all=full)


class SharedNearestTrainRepair:
    """Non-oracle comparator: project flagged rows to a clean TRAIN feature vector.

    It uses neither paired clean validation rows nor region labels. Numeric squared
    distance is in train-standard-deviation units; a differing category contributes
    one to squared distance. It is a common repair baseline, not URE reconstruction.
    """
    def fit(self, clean_train):
        self.reference=clean_train[FEATURES].copy().reset_index(drop=True)
        self.transform=ColumnTransformer([
            ('numeric',StandardScaler(),NUMERIC),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],
            transformer_weights={'numeric':1.,'categorical':1/np.sqrt(2)})
        values=self.transform.fit_transform(self.reference)
        self.nearest=NearestNeighbors(n_neighbors=1).fit(values)
        return self

    def repair(self, frame):
        indices=self.nearest.kneighbors(self.transform.transform(frame[FEATURES]),return_distance=False)[:,0]
        return self.reference.iloc[indices].reset_index(drop=True)


def flattened_metrics(values, identity):
    return [dict(identity,scope=scope,coverage=values['coverage'],total=values['total'],
        accepted=values['accepted'],rejected=values['rejected'],
        accepted_and_correct_fraction=values['accepted_and_correct_fraction'],**values[scope])
        for scope in ['unfiltered','accepted_only','repaired_all']]
