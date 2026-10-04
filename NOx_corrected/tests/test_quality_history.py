"""QC and transaction/point-in-time checks use temporary synthetic imports only."""
import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from nox.app import create_app
from nox.data import normalize_keys
from nox.history import HistoryError, HistoryStore, import_bundle, model_key, utc_now
from nox.predict import Predictor
from nox.quality import QualityRules, inspect_targets, write_quality_reports
from nox.schema import ROOT

ARTIFACT = ROOT / 'artifacts/concentration'


def daily_row(date='20250101', nox=10, **changes):
    return {'사업소':'분당', '호기':'1호기', '일자':date, 'NOX':nox,
            '산소':15, '유량':10000, '온도':110, **changes}


@pytest.fixture(scope='session')
def seed_db(tmp_path_factory):
    path = tmp_path_factory.mktemp('history_seed') / 'history.sqlite'
    assert HistoryStore(path).initialize(ARTIFACT)['row_count'] == 504
    return path


@pytest.fixture
def store(seed_db, tmp_path):
    path = tmp_path / 'history.sqlite'
    shutil.copyfile(seed_db, path)
    return HistoryStore(path, model_key(ARTIFACT)[0])


def observations(store, months=('2025-01', '2025-02', '2025-03')):
    frame, _ = store.snapshot()
    row = frame.loc[(frame.plant=='분당') & (frame.unit==1)].iloc[-1]
    out = pd.DataFrame([row.copy() for _ in months]).reset_index(drop=True)
    out['month'] = pd.to_datetime([m+'-01' for m in months])
    out['target'] = np.arange(1, len(months)+1) * 10.
    out['quality_status'] = 'valid'
    out['quality_reasons'] = '[]'
    return out


def update(store, frame, key='fixture', **options):
    return store.import_observations(frame, source='synthetic_test_fixture',
        source_files={'fixture':'0'*64}, batch_key=key, **options)


def request_for(month='2025-04'):
    data = json.loads((ROOT / 'examples/request.json').read_text())
    data['month'] = month
    return data


def test_suspicion_flags_keep_values():
    raw = pd.DataFrame([daily_row(nox=6014.68, 산소=21.1, 온도=24.41), daily_row('20250102', 0)])
    before = raw.copy(deep=True)
    daily, monthly, summary = inspect_targets(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert daily.included_in_target.all()
    assert monthly.target.iloc[0] == pytest.approx(3007.34)
    assert {'high_nox','high_oxygen','high_nox_low_temperature'} <= set(json.loads(daily.quality_reasons.iloc[0]))
    assert summary['thresholds_are_review_heuristics'] is True


def test_provider_invalid_and_stack_average():
    raw = pd.DataFrame([daily_row(nox=10, 호기='1A호기'), daily_row(nox=30, 호기='1B호기'),
                        daily_row('20250102', 999, measurement_status='invalid')])
    daily, monthly, _ = inspect_targets(raw)
    assert daily.included_in_target.tolist() == [True, True, False]
    assert monthly.target.iloc[0] == 20
    assert monthly.emission_days.iloc[0] == 1


@pytest.mark.parametrize('changes', [{'NOX':-1}, {'NOX':np.inf}, {'NOX':True}, {'NOX':'bad'},
                                    {'일자':'20250230'}, {'사업소':None}])
def test_malformed_measurements(changes):
    daily, _, summary = inspect_targets(pd.DataFrame([daily_row(**changes)]))
    assert not daily.included_in_target.iloc[0]
    assert summary['structural_error_rows'] == 1


def test_duplicates_unknown_units_and_zero_generation():
    raw = pd.DataFrame([daily_row(), daily_row(), daily_row('20250102', 20, 호기='-'),
                        daily_row('20250103', 30)])
    gen = pd.DataFrame([{'plant':'분당','unit':1,'month':pd.Timestamp('2025-01-01'),
                         'generation_mwh':0.,'thermal_efficiency_pct':0.}])
    daily, monthly, summary = inspect_targets(raw, gen)
    assert summary['structural_error_rows'] == 2
    assert daily.included_in_target.tolist() == [False,False,False,True]
    assert monthly.target.iloc[0] == 30
    assert 'positive_nox_zero_generation' in json.loads(monthly.quality_reasons.iloc[0])


def test_reports_and_strict_codes(tmp_path):
    d,m,s = inspect_targets(pd.DataFrame([daily_row()]))
    write_quality_reports(d,m,s,tmp_path)
    assert json.loads((tmp_path/'summary.json').read_text())['source_rows'] == 1
    assert len(pd.read_csv(tmp_path/'monthly_quality.csv')) == 1
    with pytest.raises(ValueError):
        QualityRules(high_nox=np.inf)
    with pytest.raises(ValueError):
        inspect_targets(pd.DataFrame([daily_row(measurement_status='vendor_code_99')]))
    with pytest.raises(ValueError):
        normalize_keys(pd.DataFrame([{'사업소':'분당','호기':1.5,'월':202501}]))


def test_initialize_idempotent_and_saved_prediction(store):
    assert store.initialize(ARTIFACT)['status'] == 'already_initialized'
    data = json.loads((ROOT/'examples/request.json').read_text())
    assert Predictor(history_db=store.path).predict(data)['prediction'] == Predictor(history_db=False).predict(data)['prediction']


def test_live_update_without_predictor_restart(store):
    predictor = Predictor(history_db=store.path)
    assert predictor.predict(request_for())['input_quality']['observed_previous_months'] == 0
    assert update(store, observations(store))['revision'] == 2
    result = predictor.predict(request_for())
    assert result['input_quality']['observed_previous_months'] == 3
    assert result['input_quality']['history']['revision'] == 2
    assert all(q['source']=='synthetic_test_fixture' for q in result['input_quality']['history_quality'])
    client = create_app(predictor).test_client()
    assert client.get('/api/history/status').json['revision'] == 2
    assert client.post('/api/predict', json=request_for()).status_code == 200


def test_current_future_targets_and_later_imports_do_not_leak(store):
    predictor = Predictor(history_db=store.path)
    update(store, observations(store))
    before = predictor.predict(request_for())['prediction']
    future = observations(store, ('2025-04','2025-05'))
    future['target'] = 1e9
    update(store, future, key='future')
    assert predictor.predict(request_for())['prediction'] == before
    historical = request_for()
    historical['as_of'] = '2025-04-02T00:00:00+09:00'
    assert predictor.predict(historical)['input_quality']['observed_previous_months'] == 0


def test_corrections_require_flag_preserve_prior_version(store):
    frame = observations(store, ('2025-01',))
    update(store, frame)
    first = store.status()['latest_imported_at']
    frame['target'] = 77.
    with pytest.raises(HistoryError):
        update(store, frame, key='correction')
    assert store.status()['revision'] == 2
    update(store, frame, key='correction', allow_corrections=True)
    old,_ = store.snapshot(first)
    new,_ = store.snapshot()
    assert old.loc[old.month==pd.Timestamp('2025-01-01'), 'target'].iloc[0] == 10
    assert new.loc[new.month==pd.Timestamp('2025-01-01'), 'target'].iloc[0] == 77
    with sqlite3.connect(store.path) as con:
        assert con.execute("SELECT count(*) FROM records WHERE month='2025-01'").fetchone()[0] == 2


def test_invalidation_overrides_observed_target(store):
    frame = observations(store)
    update(store, frame)
    frame.loc[0,'target'] = np.nan
    frame.loc[0,'quality_status'] = 'unavailable'
    update(store, frame, key='invalidation', allow_corrections=True)
    assert Predictor(history_db=store.path).predict(request_for())['input_quality']['observed_previous_months'] == 2


def test_dry_run_idempotency_and_revision_conflict(store):
    frame = observations(store)
    assert update(store,frame,dry_run=True)['status'] == 'dry_run'
    assert store.status()['revision'] == 1
    update(store,frame,expected_revision=1)
    assert update(store,frame)['status'] == 'already_imported'
    assert update(store,frame,key='same')['status'] == 'no_changes'
    with pytest.raises(HistoryError):
        update(store,observations(store,('2025-04',)),key='new',expected_revision=1)
    assert store.status()['revision'] == 2


@pytest.mark.parametrize('col,value', [('unit',99),('target',-1),('target',np.inf),('target',True),
                                     ('capacity_mw',0),('thermal_efficiency_pct',101),('humidity_pct',101)])
def test_invalid_batch_rolls_back_all_rows(store,col,value):
    frame = observations(store)
    frame.loc[1,col] = value
    with pytest.raises(HistoryError):
        update(store,frame)
    assert store.status()['revision'] == 1
    assert store.status()['row_count'] == 504


def test_incomplete_future_and_duplicate_rejected(store):
    frame = observations(store)
    with pytest.raises(HistoryError):
        update(store,frame,available_at='2099-01-01T00:00:00+00:00')
    with pytest.raises(HistoryError):
        update(store,frame,available_at='2025-01-15T00:00:00+09:00')
    with pytest.raises(HistoryError):
        update(store,pd.concat([frame,frame]))


def test_wrong_model_identity_is_503(store):
    with sqlite3.connect(store.path) as con:
        con.execute("UPDATE metadata SET value='wrong' WHERE key='model_key'")
    client = create_app(Predictor(history_db=store.path)).test_client()
    for response in (client.get('/api/history/status'),client.get('/api/health'),
                     client.post('/api/predict',json=request_for())):
        assert response.status_code == 503
        assert response.json['error']['code'] == 'history_unavailable'


@pytest.mark.parametrize('as_of', ['2025-01-01',None,'2099-01-01T00:00:00Z','bad'])
def test_invalid_as_of_is_400(store,as_of):
    data = request_for()
    data['as_of'] = as_of
    response = create_app(Predictor(history_db=store.path)).test_client().post('/api/predict',json=data)
    assert response.status_code == 400


def export_fixture(path, nox=10):
    path.mkdir()
    pd.DataFrame([{'사업소':'분당','호기':1,'월':202501,'용량(MW)':100,'발전량(MWh)':1000,
                   '열효율(%)':40,'이용률(%)':30}]).to_csv(path/'generation.csv',index=False)
    pd.DataFrame([{'사업소':'분당','일자':202501,'계(석탄)':0,'유연탄':0,'무연탄':0,
                  '유류':0,'LNG':1000,'고형연료':0,'우드펠릿':0}]).to_csv(path/'fuel_site.csv',index=False)
    pd.DataFrame([daily_row(f'202501{d:02}',nox) for d in range(1,32)]).to_csv(path/'emissions_daily.csv',index=False)
    pd.DataFrame([{'사업소':'분당','호기':'test_station','일자':f'2025-01-{d:02}',
                   '온도':10,'습도':50,'풍향':90,'풍속':2} for d in range(1,32)]).to_csv(path/'weather_daily.csv',index=False)


def test_bundle_import_and_reports(store,tmp_path):
    raw = tmp_path/'export'
    export_fixture(raw)
    options = dict(start_month='2025-01',end_month='2025-01',source='synthetic_test_fixture',report_dir=tmp_path/'reports')
    assert import_bundle(store,raw,**options)['inserted'] == 1
    assert import_bundle(store,raw,**options)['status'] == 'already_imported'
    frame,_ = store.snapshot()
    assert frame.loc[frame.month==pd.Timestamp('2025-01-01'),'emission_coverage'].iloc[0] == 1
    assert (tmp_path/'reports/import_result.json').exists()


def test_malformed_bundle_reported_without_commit(store,tmp_path):
    raw = tmp_path/'export'
    export_fixture(raw,nox=-1)
    with pytest.raises(HistoryError):
        import_bundle(store,raw,start_month='2025-01',end_month='2025-01',
                      source='synthetic_test_fixture',report_dir=tmp_path/'reports')
    assert json.loads((tmp_path/'reports/summary.json').read_text())['structural_error_rows'] == 31
    assert store.status()['revision'] == 1
