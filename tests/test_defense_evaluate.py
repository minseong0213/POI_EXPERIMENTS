import numpy as np
import pytest
from poi.defense_evaluate import binary_metrics,paired_defense_metrics,select_detection_threshold


def test_rejected_cases_remain_in_coverage_and_full_metrics():
    values=paired_defense_metrics([1,0,1,0],[.9,.1,.2,.8],[.2,.8,.9,.1],[.1,.1,.9,.9],.5)
    assert values['coverage']==.5
    assert values['accepted_only']['f1']==1
    assert values['accepted_and_correct_fraction']==.5
    assert values['unfiltered']['accuracy']==.5
    assert values['repaired_all']['accuracy']==1
    assert values['repaired_all']['n']==4


def test_empty_acceptance_does_not_report_perfect_accuracy():
    values=paired_defense_metrics([1,0],[.9,.1],[.9,.1],[.9,.9],.5)
    assert values['coverage']==0
    assert np.isnan(values['accepted_only']['accuracy'])
    assert values['repaired_all']['n']==2


def test_single_class_auc_is_undefined():
    values=paired_defense_metrics([1,0],[.9,.1],[.9,.1],[.1,.9],.5)
    assert np.isnan(values['accepted_only']['roc_auc'])


def test_misaligned_arrays_rejected():
    with pytest.raises(ValueError,match='aligned'):
        paired_defense_metrics([1,0],[.9],[.9,.1],[.1,.9],.5)


def test_threshold_ties_retain_clean_rows():
    assert select_detection_threshold([.1,.1,.2,.2],.75)==.2


def test_empty_primary_population_is_retained_with_frozen_zero_policy():
    values=paired_defense_metrics([],[],[],[],.5)
    assert values['total']==values['accepted']==values['rejected']==0
    assert np.isnan(values['coverage'])
    assert values['accepted_only']['n']==0
    assert values['accepted_only']['f1']==0
    assert np.isnan(values['accepted_only']['accuracy'])
    assert values['zero_division_policy']==0


def test_detector_strict_threshold_keeps_ties_as_clean():
    values=binary_metrics([0,0,1,1],[.1,.2,.2,.3],.2,threshold_operator='gt')
    assert values['recall']==.5
    assert values['precision']==1
