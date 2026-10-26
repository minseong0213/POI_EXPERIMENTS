import numpy as np
import pandas as pd
import statsmodels.api as sm

from poi.odds_ratio import separation_check, fit_associations, design_matrix
from poi.data import NUMERIC,CATEGORICAL


def test_complete_separation_does_not_emit_finite_odds_ratios():
    x=sm.add_constant(pd.DataFrame({'x':[-3.,-2.,-1.,1.,2.,3.]}))
    y=np.array([0,0,0,1,1,1])
    rows,_=fit_associations(x,y)
    assert all(not row['identifiable'] for row in rows)
    assert all(np.isnan(row['odds_ratio']) for row in rows)


def test_paired_identical_features_have_no_attack_association():
    rng=np.random.default_rng(18)
    raw=rng.normal(size=160)
    x=sm.add_constant(pd.DataFrame({'x':np.r_[raw,raw]}))
    y=np.r_[np.zeros(160,int),np.ones(160,int)]
    rows,_=fit_associations(x,y,np.r_[np.arange(160),np.arange(160)])
    assert separation_check(x,y)[0]=='no_separating_direction_detected'
    assert all(row['identifiable'] for row in rows)
    np.testing.assert_allclose([row['odds_ratio'] for row in rows],1,atol=1e-8)
    assert all(row['covariance']=='POI-cluster robust' for row in rows)


def test_hierarchical_category_aliases_are_removed_without_fitting_evaluation():
    frame=pd.DataFrame({NUMERIC[0]:np.arange(12,dtype=float),NUMERIC[1]:np.arange(12,dtype=float)**2,
        CATEGORICAL[0]:[1,2]*6,CATEGORICAL[1]:[1,2]*6,CATEGORICAL[2]:[1,2,3]*4})
    design,encoding=design_matrix(frame,frame)
    assert np.linalg.matrix_rank(design)==design.shape[1]
    assert len(encoding['excluded_aliased_columns'])==1
    altered=frame.copy();altered[NUMERIC[0]]+=1000
    _,other=design_matrix(frame,altered)
    assert encoding==other


def test_regional_hc0_matches_manual_score_sandwich():
    rng = np.random.default_rng(741)
    x = sm.add_constant(pd.DataFrame({'x': rng.normal(size=200)}))
    y = rng.binomial(1, .5, size=200)
    rows, _ = fit_associations(x, y)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    matrix = x.to_numpy()
    mu = fit.predict()
    bread = np.linalg.inv(matrix.T @ ((mu * (1-mu))[:, None] * matrix))
    scores = matrix * (y-mu)[:, None]
    covariance = bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose([row['standard_error'] for row in rows],
                               np.sqrt(np.diag(covariance)), rtol=1e-8)
    assert all(row['covariance'] == 'HC0' and row['fit_status'] == 'fitted' for row in rows)


def test_solver_exception_is_distinct_from_separation(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('injected solver failure')
    monkeypatch.setattr(sm.GLM, 'fit', fail)
    x = sm.add_constant(pd.DataFrame({'x': [-1., 1., -1., 1.]}))
    rows, _ = fit_associations(x, [0, 0, 1, 1])
    assert all(row['fit_status'] == 'failed' for row in rows)
    assert all(row['separation_status'] == 'no_separating_direction_detected' for row in rows)
    assert all('RuntimeError' in row['failure_reason'] for row in rows)


def test_legacy_explanation_route_does_not_clip_separated_coefficients():
    from poi.selected_explain import _odds_ratios
    frame = pd.DataFrame({NUMERIC[0]: [-3., -2., -1., 1., 2., 3.],
                          NUMERIC[1]: [1., 2., 3., 1., 2., 3.],
                          **{name: [1]*6 for name in CATEGORICAL},
                          'region': ['A']*3+['B']*3})
    rows, failures = _odds_ratios(frame, ['A', 'B'])
    assert rows[['odds_ratio', 'ci_low', 'ci_high', 'p_value', 'q_value']].isna().all().all()
    assert rows.fit_status.eq('separation').all()
    assert rows.covariance.eq('HC0').all()
    assert len(failures) == 2
