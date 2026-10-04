import json

import numpy as np
import pandas as pd
import pytest

from nox.development_experiments import (FUEL, GROUPS, PATTERNS, blend_native,
    candidate_features, daily_stats, fit_candidate, inverse, predict_saved, score_candidate)
from nox.schema import CONDITIONS
from nox.temporal_validation import sha256


PROTOCOL = {'train':['2023-01','2023-12'], 'validation':['2024-01','2024-06'],
            'test':['2024-07','2024-12']}


def fixture_frame():
    rows=[]
    for unit in [1,2]:
        for i,month in enumerate(pd.date_range('2023-01-01','2024-12-01',freq='MS')):
            rows.append({'plant':'분당','unit':unit,'month':month,'target':float(2*unit+i%7),
                **{c:float(1+unit+i%5) for c in CONDITIONS}})
    return pd.DataFrame(rows)


def test_target_transform_and_native_blending_are_explicit():
    a,b=np.log1p([1.]),np.log1p([9.])
    assert blend_native(a,b,.5,'log1p')[0] == pytest.approx(np.sqrt(20)-1)
    assert blend_native(np.array([1.]),np.array([9.]),.5,'ppm')[0] == 5
    np.testing.assert_array_equal(inverse([-2,3],'ppm'),[0,3])
    with pytest.raises(ValueError,match='Nonfinite'):
        inverse([np.inf],'ppm')
    with pytest.raises(ValueError,match='Unknown'):
        inverse([1],'unknown')


def test_fuel_ablation_removes_all_ratios_and_changes_and_operations_closes_dependencies():
    columns=[*FUEL,*PATTERNS,'generation_mwh','thermal_efficiency_pct','utilization_pct',
             'capacity_mw','nox_lag1','month_sin','plant_분당']
    X=pd.DataFrame({c:[1.] for c in columns})
    without=candidate_features(X,'without_fuel')
    assert not set(FUEL)&set(without)
    assert 'generation_mwh' in without
    without=candidate_features(X,'without_current_operations')
    assert not set(GROUPS['without_current_operations'])&set(without)
    assert 'capacity_mw' in without and 'nox_lag1' in without
    assert set(candidate_features(X,'full')) == set(X)


@pytest.mark.parametrize('transform',['log1p','ppm'])
def test_outer_targets_inputs_cannot_change_fitted_models_preprocessor_or_weight(tmp_path,transform):
    df=fixture_frame()
    first=tmp_path/'a';second=tmp_path/'b'
    fit_candidate(df,PROTOCOL,'without_fuel',transform,first)
    assert not (first/'metadata.json').exists()
    assert not list(first.glob('*predictions*'))
    changed=df.copy()
    outer=changed.month.ge('2024-07-01')
    changed.loc[outer,'target']=1e9
    changed.loc[outer,'temperature_c']=1e9
    fit_candidate(changed,PROTOCOL,'without_fuel',transform,second)
    for filename in ['xgb_model.json','lgb_model.txt','preprocessor.json','selection_trials.json']:
        assert sha256(first/filename)==sha256(second/filename)
    assert json.loads((first/'experiment.json').read_text())['weight']==json.loads((second/'experiment.json').read_text())['weight']
    scored=score_candidate(df,PROTOCOL,first)
    assert len(scored)==24 and set(scored.split)=={'validation','test'}
    with pytest.raises(ValueError,match='overwrite'):
        fit_candidate(df,PROTOCOL,'full',transform,first)


def test_saved_raw_model_rejects_corrupt_artifacts(tmp_path):
    from nox.features import build_features
    df=fixture_frame();folder=tmp_path/'model'
    fit_candidate(df,PROTOCOL,'full','ppm',folder)
    predictions=predict_saved(folder,build_features(df))
    assert np.isfinite(predictions['ensemble']).all()
    path=folder/'preprocessor.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='artifact changed'):
        predict_saved(folder,build_features(df))


def test_daily_distribution_preserves_extreme_values_and_zero_days():
    days=pd.DataFrame({'nox_numeric':[0.,10.,20.,1000.]})
    historical=pd.DataFrame({'nox_numeric':[1.,2.,3.]})
    result=daily_stats(days,historical)
    assert result['mean_ppm']==257.5 and result['maximum_ppm']==1000
    assert result['zero_days']==1 and result['days_above_training_daily_q95']==3
    assert result['top5_days_share_of_daily_value_sum']==1
