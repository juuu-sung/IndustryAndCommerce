import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nox.app import create_app
from nox.data import allocate_fuel, build_dataset
from nox.features import build_features, Preprocessor
from nox.history import HistoryError, HistoryStore, import_bundle, model_key
from nox.predict import Predictor
from nox.schema import CONDITIONS, FUELS, KEYS, ROOT
from nox.train import choose_weight, split_masks

RAW = ROOT/'data/unit_fuel_20261001'
BASE_RAW = ROOT/'data/extended_20261001'
ARTIFACT = ROOT/'artifacts/concentration_unit_fuel_20261001'
PROTOCOL = {'train':['2023-01','2025-06'],'validation':['2025-07','2025-12'],'test':['2026-01','2026-08']}


def request_for(row):
    out = {c:row[c] for c in ['plant','unit',*CONDITIONS] if c not in ('wind_sin','wind_cos')}
    out['unit'] = int(row.unit)
    out['month'] = row.month.strftime('%Y-%m')
    out['wind_direction_deg'] = float(np.degrees(np.arctan2(row.wind_sin,row.wind_cos))%360)
    out['fuel_basis'] = row.fuel_basis
    return {k:None if pd.isna(v) else float(v) if isinstance(v,np.floating) else v for k,v in out.items()}


def test_complete_override_retains_targets_and_other_plants():
    old,_ = build_dataset('concentration',BASE_RAW)
    new,audit = build_dataset('concentration',RAW)
    pd.testing.assert_frame_equal(old[KEYS+['target']],new[KEYS+['target']])
    yd = new.plant.eq('영동')
    assert int(yd.sum())==88
    assert audit['fuel_basis_counts']=={'site_generation_share_estimate':880,'provider_reported_unit_month':88}
    pd.testing.assert_frame_equal(old.loc[~yd,list(FUELS.values())],new.loc[~yd,list(FUELS.values())])
    assert {n:int(m.sum()) for n,m in split_masks(new,PROTOCOL).items()}=={'train':636,'validation':124,'test':176}
    source = pd.read_csv(RAW/'fuel_unit.csv').rename(columns={'사업소':'plant','호기':'provider_unit'})
    source['month'] = pd.to_datetime(source['일자'].astype(str),format='%Y%m')
    joined = new.loc[yd].merge(source,on=['plant','month'],suffixes=('','_raw'))
    joined = joined.loc[joined.unit.eq(joined.provider_unit)]
    assert len(joined)==88
    for source_col,model_col in FUELS.items():
        np.testing.assert_allclose(joined[source_col],joined[model_col],rtol=0,atol=0)
    # Zero electrical generation does not erase a provider-reported fuel value.
    zero = new.loc[yd & new.generation_mwh.eq(0)]
    assert (zero[list(FUELS.values())].sum(axis=1)>0).any()


@pytest.mark.parametrize('problem,match',[
    ('duplicate','duplicate'),('partial','incomplete'),('sum','reconcile'),
    ('negative','nonnegative'),('nan','finite'),('unknown','unrepresented'),
    ('missing_source','source_file'),('coal','coal total')])
def test_bad_unit_sources_rejected(tmp_path,problem,match):
    gen = pd.DataFrame({'plant':['영동']*2,'unit':[1,2],
        'month':pd.to_datetime(['2026-01-01']*2),'generation_mwh':[0.,100.]})
    site = pd.DataFrame([{'사업소':'영동','일자':202601,**{c:0. for c in FUELS},'계(석탄)':0.}])
    unit = pd.DataFrame([{'사업소':'영동','호기':u,'일자':202601,
                        **{c:0. for c in FUELS},'계(석탄)':0.,'source_file':'official.csv'} for u in (1,2)])
    if problem=='duplicate': unit = pd.concat([unit,unit.iloc[[0]]])
    elif problem=='partial': unit = unit.iloc[[0]]
    elif problem=='sum': unit.loc[0,'유류']=1
    elif problem=='negative': unit.loc[0,'유류']=-1
    elif problem=='nan': unit.loc[0,'유류']=np.nan
    elif problem=='unknown': unit.loc[0,'호기']=99
    elif problem=='missing_source': unit.loc[0,'source_file']=None
    elif problem=='coal': unit.loc[0,'계(석탄)']=1
    site.to_csv(tmp_path/'fuel_site.csv',index=False)
    unit.to_csv(tmp_path/'fuel_unit.csv',index=False)
    with pytest.raises(ValueError,match=match):
        allocate_fuel(gen,tmp_path)


def test_actual_fuel_history_uses_previous_calendar_month():
    frame,_ = build_dataset('concentration',RAW)
    X = build_features(frame)
    for unit in (1,2):
        selected = frame.loc[(frame.plant=='영동') & (frame.unit==unit) & frame.month.eq('2026-03-01')]
        prior = frame.loc[(frame.plant=='영동') & (frame.unit==unit) & frame.month.eq('2026-02-01')].iloc[0]
        for col in ('oil_kl','pellet_ton'):
            expected = selected.iloc[0][col]/prior[col]-1 if prior[col]>0 else np.nan
            actual = X.loc[selected.index[0],col+'_change']
            assert (pd.isna(expected) and pd.isna(actual)) or actual==pytest.approx(expected)
    changed = frame.copy()
    changed.loc[changed.month.ge('2026-03-01'),'target']=1e9
    rows = frame.loc[frame.month.eq('2026-03-01')]
    pd.testing.assert_frame_equal(build_features(rows,frame),build_features(rows,changed))


@pytest.fixture(scope='module')
def candidate():
    if not (ARTIFACT/'metrics.json').exists():
        pytest.skip('Candidate model has not been trained yet')
    return Predictor(ARTIFACT,history_db=False)


@pytest.mark.parametrize('split',['validation','test'])
def test_saved_candidate_predictions_match_api(candidate,split):
    saved = pd.read_csv(ARTIFACT/(split+'_predictions.csv'),parse_dates=['month'])
    joined = saved[KEYS+['pred_ensemble']].merge(candidate.history,on=KEYS,validate='one_to_one')
    actual = [candidate.predict(request_for(row))['prediction']['value'] for _,row in joined.iterrows()]
    np.testing.assert_allclose(actual,joined.pred_ensemble,rtol=1e-7,atol=1e-8)


def test_candidate_preprocessing_and_weights_have_same_split(candidate):
    frame = candidate.history
    X = build_features(frame)
    masks = split_masks(frame,PROTOCOL)
    pre = Preprocessor().fit(X.loc[masks['train']])
    assert pre.columns==candidate.pre.columns
    assert pre.medians==pytest.approx(candidate.pre.medians)
    inputs = pre.transform(X.loc[masks['validation']])
    weight,_ = choose_weight(frame.loc[masks['validation'],'target'],candidate.xgb.predict(inputs),
                              candidate.lgb.predict(inputs,num_threads=1))
    assert candidate.weight==weight
    assert candidate.metadata['weights_locked_before_test'] is True


def test_candidate_rejects_estimates_and_ambiguous_unit_inputs(candidate):
    row = candidate.history.loc[(candidate.history.plant=='영동') & candidate.history.month.eq('2026-08-01')].iloc[0]
    client = create_app(candidate).test_client()
    good = request_for(row)
    assert client.post('/api/predict',json=good).status_code==200
    for basis in (None,'site_generation_share_estimate','unknown'):
        bad = dict(good)
        if basis is None: bad.pop('fuel_basis')
        else: bad['fuel_basis']=basis
        assert client.post('/api/predict',json=bad).status_code==400


def test_candidate_history_preserves_actual_fuel_and_rejects_old_bundle(candidate,tmp_path):
    store = HistoryStore(tmp_path/'history.sqlite',model_key(ARTIFACT)[0])
    assert store.initialize(ARTIFACT)['row_count']==528
    result = import_bundle(store,RAW,start_month='2025-01',end_month='2026-08',
                           source='verified_actual_unit_fuel',report_dir=tmp_path/'reports')
    assert result['inserted']==440
    frame,_ = store.snapshot()
    assert frame.loc[frame.plant.eq('영동'),'fuel_basis'].eq('provider_reported_unit_month').all()
    row = candidate.history.loc[(candidate.history.plant=='영동') & candidate.history.month.eq('2026-08-01')].iloc[0]
    frozen = candidate.predict(request_for(row))['prediction']['value']
    runtime = Predictor(ARTIFACT,history_db=store.path).predict(request_for(row))['prediction']['value']
    assert runtime==pytest.approx(frozen,rel=1e-7,abs=1e-8)
    before = store.status()['revision']
    with pytest.raises(HistoryError,match='fuel basis'):
        import_bundle(store,BASE_RAW,start_month='2025-01',end_month='2026-08',
                      source='incompatible_estimates',report_dir=tmp_path/'bad',allow_corrections=True)
    assert store.status()['revision']==before
    historical,_ = store.snapshot('2025-01-01T00:00:00+09:00')
    assert historical.month.max()==pd.Timestamp('2024-12-01')
    # Build the incompatible database explicitly so a fresh checkout can verify
    # the model binding without depending on a developer's runtime database.
    mismatched_db = tmp_path / 'mismatched_history.sqlite'
    HistoryStore(mismatched_db).initialize(ROOT / 'artifacts/concentration_20261001')
    with pytest.raises(HistoryError,match='does not match'):
        Predictor(ARTIFACT,history_db=mismatched_db).history_status()
