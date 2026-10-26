import numpy as np
import pandas as pd
import pytest
from poi.detector_report import score_pair


def frame(values):
    return pd.DataFrame({'POI_ID': ['a','b'], 'attack_score': values})


def test_saved_retention_threshold_not_half_is_used_and_ties_are_accepted():
    result=score_pair(frame([.2,.6]),frame([.6,.9]),.6)
    assert (result['tn'],result['fp'],result['fn'],result['tp']) == (2,0,1,1)
    assert result['clean_retention']==1 and result['attack_rejection']==.5
    assert result['f1']==pytest.approx(2/3)


def test_unpaired_cohort_rejected_and_empty_metrics_not_fabricated():
    with pytest.raises(ValueError,match='pairs differ'):
        score_pair(frame([.2,.3]),frame([.8,.9]).iloc[:1],.5)
    empty=frame([.2,.3]).iloc[:0]
    result=score_pair(empty,empty,.5)
    assert result['pois']==0 and np.isnan(result['f1']) and np.isnan(result['roc_auc'])
