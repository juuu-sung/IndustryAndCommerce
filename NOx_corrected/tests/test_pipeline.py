import copy
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nox.app import create_app
from nox.data import build_dataset, monthly_weather, mass_from_intervals, unique
from nox.features import build_features, Preprocessor
from nox.llm import Gemini, explain, validate_analysis
from nox.predict import Predictor
from nox.schema import CONDITIONS, FUELS, ROOT
from nox.train import split_masks, choose_weight


def row(month, target, unit=1, plant='분당'):
    return {'plant':plant, 'unit':unit, 'month':pd.Timestamp(month), 'target':target,
            **{col:1. for col in CONDITIONS}}


def payload(source):
    out = {key:source[key] for key in ['plant','unit', *CONDITIONS] if key not in ('wind_sin','wind_cos')}
    out['unit'] = int(out['unit'])
    out['month'] = source['month'].strftime('%Y-%m')
    out['wind_direction_deg'] = float(np.degrees(np.arctan2(source['wind_sin'],source['wind_cos'])) % 360)
    return {k:None if pd.isna(v) else float(v) if isinstance(v,np.floating) else v for k,v in out.items()}


@pytest.fixture(scope='session', params=['legacy','concentration'])
def predictor(request):
    return Predictor(ROOT / 'artifacts' / request.param, history_db=False)


def test_calendar_lags_shuffled_index_and_group_boundaries():
    df = pd.DataFrame([row('2023-04-01',40),row('2023-01-01',10),row('2023-03-01',30),
                       row('2023-01-01',1000,unit=2),row('2023-02-01',2000,unit=2)],index=[99,3,65,12,51])
    X = build_features(df)
    assert X.loc[0,'nox_lag1']==30
    assert X.loc[0,'nox_roll3']==20  # January and March, no invented February.
    assert X.loc[0,'nox_history_count']==2
    assert pd.isna(X.loc[2,'nox_lag1'])
    assert X.loc[2,'nox_roll3']==10
    assert X.loc[4,'nox_lag1']==1000


def test_current_and_future_targets_cannot_change_features():
    history = pd.DataFrame([row('2023-01-01',10),row('2023-02-01',20),row('2023-03-01',30),row('2023-04-01',40)])
    target_row = history.iloc[[2]]
    original = build_features(target_row,history)
    changed = history.copy()
    changed.loc[changed['month']>=pd.Timestamp('2023-03-01'),'target'] = 1e12
    pd.testing.assert_frame_equal(original,build_features(target_row,changed))


def test_duplicate_history_fails():
    df = pd.DataFrame([row('2023-01-01',10),row('2023-01-01',20)])
    with pytest.raises(ValueError,match='duplicate'):
        build_features(df)


def test_weather_zero_values_and_circular_average(tmp_path):
    frame = pd.DataFrame({'사업소':['분당','분당'],'호기':['-','-'],
        '일자':['2023-01-01','2023-01-02'],'온도':[0.,2.],'습도':[0.,50.],
        '풍향':[359.,1.],'풍속':[0.,2.]})
    frame.to_csv(tmp_path/'weather_daily.csv',index=False)
    monthly, bad = monthly_weather(tmp_path)
    assert monthly.loc[0,'temperature_c']==1
    assert monthly.loc[0,'humidity_pct']==25
    assert monthly.loc[0,'wind_speed_ms']==1
    assert abs(monthly.loc[0,'wind_sin'])<1e-12
    assert monthly.loc[0,'wind_cos']>.999
    assert sum(bad.values())==0


def test_train_only_imputation_is_stable():
    pre = Preprocessor().fit(pd.DataFrame({'a':[1.,3.,np.nan],'constant':[0,0,0],'empty':[np.nan]*3}))
    assert pre.columns==['a']
    assert pre.transform(pd.DataFrame({'a':[np.nan,1e12]})).iloc[0,0]==2
    restored = Preprocessor.from_dict(pre.to_dict())
    pd.testing.assert_frame_equal(pre.transform(pd.DataFrame({'a':[np.nan]})),restored.transform(pd.DataFrame({'a':[np.nan]})))


@pytest.mark.parametrize('mode',['legacy','concentration'])
def test_source_joins_retain_rows_and_calendar_split(mode):
    df, audit = build_dataset(mode)
    assert audit['rows_before_join']==audit['rows_after_join']
    assert len(df)==(455 if mode=='legacy' else 528)
    assert not df.duplicated(['plant','unit','month']).any()
    masks = split_masks(df)
    assert df.loc[masks['train'],'month'].max()<df.loc[masks['validation'],'month'].min()
    assert df.loc[masks['validation'],'month'].max()<df.loc[masks['test'],'month'].min()
    assert 'oil_kl' in build_features(df)
    assert '연료소비량' not in build_features(df)


def test_mass_requires_verified_intervals():
    with pytest.raises(ValueError,match='integrated'):
        mass_from_intervals([10],[1000],volume_is_integrated=False,same_interval=True,mg_per_sm3_per_ppm=2)
    answer = mass_from_intervals([10,20],[1000,2000],volume_is_integrated=True,same_interval=True,mg_per_sm3_per_ppm=2)
    assert answer.sum()==pytest.approx(.1)
    with pytest.raises(ValueError):
        mass_from_intervals([np.nan],[1],volume_is_integrated=True,same_interval=True,mg_per_sm3_per_ppm=2)


def test_serialized_predictions_equal_all_saved_holdout_rows(predictor):
    mode = predictor.metadata['target_mode']
    count = 0
    for split in ['validation','test']:
        saved = pd.read_csv(ROOT/'artifacts'/mode/f'{split}_predictions.csv',parse_dates=['month'])
        joined = saved[['plant','unit','month','pred_ensemble']].merge(predictor.history,on=['plant','unit','month'],validate='one_to_one')
        for _, r in joined.iterrows():
            result = predictor.predict(payload(r))
            assert result['prediction']['value']==pytest.approx(r['pred_ensemble'],rel=1e-6,abs=1e-5)
            count += 1
    assert count==(224 if mode=='legacy' else 252)


def test_saved_preprocessor_and_weight_use_training_and_validation_only(predictor):
    df = predictor.history
    X = build_features(df)
    masks = split_masks(df)
    expected = Preprocessor().fit(X.loc[masks['train']])
    assert predictor.pre.columns==expected.columns
    assert predictor.pre.medians==pytest.approx(expected.medians)
    inputs = predictor.pre.transform(X.loc[masks['validation']])
    weight, _ = choose_weight(df.loc[masks['validation'],'target'],predictor.xgb.predict(inputs),
                              predictor.lgb.predict(inputs,num_threads=1))
    assert weight==predictor.weight


def test_prediction_ignores_same_month_and_future_history(predictor):
    source = predictor.history.loc[(predictor.history['month']==pd.Timestamp('2024-07-01')) & (predictor.history['plant']=='분당')].iloc[0]
    request = payload(source)
    before = predictor.predict(request)['prediction']
    history = predictor.history.copy()
    try:
        predictor.history.loc[predictor.history['month']>=pd.Timestamp('2024-07-01'),'target']=1e12
        assert predictor.predict(request)['prediction']==before
    finally:
        predictor.history = history


def test_api_valid_prediction_and_utilization_above_100(predictor):
    source = predictor.history.loc[predictor.history['plant']=='분당'].iloc[0]
    request = payload(source)
    request['utilization_pct']=109.46
    response = create_app(predictor).test_client().post('/api/predict',json=request)
    assert response.status_code==200
    assert response.json['analysis']['status']=='disabled'
    assert response.json['prediction']['unit']!='kg'


@pytest.mark.parametrize('change',[
    {'generation_mwh':-1}, {'lng_ton':float('inf')}, {'oil_kl':True},
    {'humidity_pct':101}, {'month':'2024-13'}, {'unit':99}, {'unit':True},
    {'capacity_mw':0}, {'NOX_roll3':99}, {'lng_ton':'100'},
])
def test_api_invalid_inputs_are_rejected(predictor,change):
    request = payload(predictor.history.loc[predictor.history['plant']=='분당'].iloc[0])
    request.update(change)
    response = create_app(predictor).test_client().post('/api/predict',json=request)
    assert response.status_code==400


def test_api_bad_json_content_type_size_and_404(predictor):
    client = create_app(predictor).test_client()
    assert client.post('/api/predict',data='{}',content_type='text/plain').status_code==415
    assert client.post('/api/predict',data='{bad',content_type='application/json').status_code==400
    assert client.post('/api/predict',json=[]).status_code==400
    assert client.post('/api/predict',data='x'*40000,content_type='application/json').status_code==413
    assert client.get('/missing').status_code==404


def test_gemini_http_contract_and_numeric_prediction_protection(predictor):
    answer = {'summary':'입력 조건에 대한 추정입니다.','observations':['인과 원인을 확정할 수 없습니다.'],
              'suggested_checks':['실제 계측치를 대조하세요.'],'limitations':['연구용 모델입니다.']}
    requests = []
    class Response(io.BytesIO):
        pass
    def opener(req,timeout):
        requests.append(req)
        assert timeout==12
        assert req.get_header('X-goog-api-key')=='test-key'
        assert 'test-key' not in req.full_url
        body = json.loads(req.data)
        assert body['generationConfig']['responseMimeType']=='application/json'
        assert 'LNG(ton)' in body['contents'][0]['parts'][0]['text']
        envelope = {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(answer,ensure_ascii=False)}]}}]}
        return Response(json.dumps(envelope).encode())
    request = payload(predictor.history.loc[predictor.history['plant']=='분당'].iloc[0])
    model_result = predictor.predict(request)
    client = Gemini('test-key','test-model',opener=opener)
    response = create_app(predictor,enable_gemini=True,llm_client=client).test_client().post('/api/predict',json=request)
    assert response.json['analysis']['status']=='gemini'
    assert response.json['prediction']==model_result['prediction']
    assert set(model_result['warnings'])<=set(response.json['analysis']['content']['limitations'])
    assert len(requests)==1


@pytest.mark.parametrize('failure',['timeout','invalid_json','unknown_fields','wrong_types'])
def test_llm_failure_preserves_prediction(predictor,failure):
    class Fake:
        def generate(self,text):
            if failure=='timeout':raise TimeoutError('secret-value-do-not-expose')
            if failure=='invalid_json':raise json.JSONDecodeError('bad','',0)
            if failure=='unknown_fields':return {'prediction':999,'unit':'kg'}
            return {'summary':123,'observations':[],'suggested_checks':[],'limitations':[]}
    request = payload(predictor.history.loc[predictor.history['plant']=='분당'].iloc[0])
    expected = predictor.predict(request)['prediction']
    response = create_app(predictor,enable_gemini=True,llm_client=Fake()).test_client().post('/api/predict',json=request)
    assert response.status_code==200
    assert response.json['analysis']['status']=='fallback'
    assert response.json['prediction']==expected
    assert 'secret-value-do-not-expose' not in response.get_data(as_text=True)
