import numpy as np
import pandas as pd
import pytest

from poi.data import NUMERIC, CATEGORICAL
from poi.expanded_analysis import heldout_cca, paired_tests
from poi.expanded_explain import aligned_condition, encode_lime_row


def fixture_frame(n=80):
    rng=np.random.default_rng(7)
    frame=pd.DataFrame({'POI_ID':[str(i) for i in range(n)],'region':['Seoul']*n})
    for feature in NUMERIC:
        frame[feature]=rng.normal(size=n)
    for feature in CATEGORICAL:
        frame[feature]=rng.integers(0,3,size=n).astype(str)
    return frame


def test_unchanged_pairs_are_null():
    frame=fixture_frame()
    rows=paired_tests(frame,frame.copy())
    assert len(rows)==7
    assert all(row['p_value']==1 for row in rows)
    assert all(row['effect']==0 for row in rows)


def test_cca_uses_disjoint_fit_and_inference_rows():
    frame=fixture_frame()
    rows=heldout_cca(frame,frame.copy(),permutations=39)
    assert len(rows)==3
    assert all(row['fit_rows']==40 and row['inference_rows']==40 for row in rows)
    assert all(row['correlation']>.99 for row in rows)
    assert all(row['p_value']<=.05 for row in rows)


def test_attack_alignment_requires_complete_explicit_seed():
    frame=fixture_frame(8)
    rows=frame.assign(attack_condition='pgd',generation_seed=42).sample(frac=1,random_state=8)
    aligned=aligned_condition(rows,frame,'pgd',42)
    assert aligned.POI_ID.tolist()==frame.POI_ID.tolist()
    with pytest.raises(ValueError,match='Incomplete/duplicate'):
        aligned_condition(pd.concat([rows,rows.iloc[:1]]),frame,'pgd',42)
    with pytest.raises(ValueError,match='Incomplete/duplicate'):
        aligned_condition(rows,frame,'pgd',202)


def test_lime_rejects_unseen_categories():
    with pytest.raises(ValueError,match='absent'):
        encode_lime_row(np.array([1.,2.,99.]),{2:np.array([1.,2.])})


def test_renamed_adversarial_features_are_used_not_clean_values():
    frame=fixture_frame(8)
    rows=frame.assign(attack_condition='pgd',generation_seed=42)
    for feature in NUMERIC+CATEGORICAL:
        rows[f'{feature}_adv']=rows[feature]
    rows['X_COORD_adv']=rows.X_COORD+1
    aligned=aligned_condition(rows,frame,'pgd',42)
    np.testing.assert_allclose(aligned.X_COORD,frame.X_COORD+1)


def test_primary_mask_fails_closed_on_strings_and_missing_victim():
    from poi.expanded_explain import primary_mask
    frame=pd.DataFrame({'primary_evaluation_eligible':[True,False],
        'constraints_valid':[True,False],'label_preserved':[True,False],'victim_model':['source','source']})
    assert primary_mask(frame).tolist()==[True,False]
    with pytest.raises(ValueError,match='non-null boolean'):
        primary_mask(frame.assign(constraints_valid=['True','False']))
    with pytest.raises(ValueError,match='victim'):
        primary_mask(frame.assign(victim_model=['source','']))


def test_lime_reduced_features_preserve_categorical_identity():
    from poi.selected_explain import _lime_arrays
    features = [CATEGORICAL[1], NUMERIC[1], CATEGORICAL[2]]
    values = np.array([[12., 37.5, 900.], [14., 38., 910.]])
    encoded, names, reverse = _lime_arrays(values, features)
    assert set(reverse) == {0, 2}
    np.testing.assert_array_equal(encoded[:, 1], values[:, 1])
    np.testing.assert_array_equal(encoded[:, 0], [0, 1])
    restored = encoded.copy()
    for index, vocabulary in reverse.items():
        restored[:, index] = vocabulary[encoded[:, index].astype(int)]
    np.testing.assert_array_equal(restored, values)
