import json

import numpy as np
import pandas as pd
import pytest

from nox.collect import normalize_generation, overlap_review, parse_export
from nox.data import build_dataset
from nox.features import Preprocessor
from nox.history import HistoryError, HistoryStore, import_bundle, model_key
from nox.predict import Predictor
from nox.schema import CONDITIONS, ROOT
from nox.train import split_masks, train, validate_split

ARTIFACT = ROOT/'artifacts/concentration_20261001'
RAW = ROOT/'data/extended_20261001'
PROTOCOL = {'train':['2023-01','2025-06'],'validation':['2025-07','2025-12'],'test':['2026-01','2026-08']}


def payload(row):
    out = {c:row[c] for c in ['plant','unit',*CONDITIONS] if c not in ('wind_sin','wind_cos')}
    out['unit']=int(out['unit'])
    out['month']=row['month'].strftime('%Y-%m')
    out['wind_direction_deg']=np.degrees(np.arctan2(row['wind_sin'],row['wind_cos']))%360
    return {k:None if pd.isna(v) else float(v) if isinstance(v,np.floating) else v for k,v in out.items()}


def test_export_formats_headers_and_explicit_mapping():
    data='사업소| 호기 | 일자| 용량(MW)| 발전량(MWh)| 열효율(%)| 이용률(%)\n분당| CG1| 202501| 77.76| 100| 40| 20\n분당| CS1| 202501| 0| 0| 0| 0\n'
    frame=parse_export(data.encode(),'generation')
    mapped,excluded=normalize_generation(frame)
    assert mapped['호기'].tolist()==[1]
    assert excluded[0]['호기']=='CS1'
    frame.loc[0,'호기']='CG99'
    with pytest.raises(ValueError,match='Unverified'):
        normalize_generation(frame)
    with pytest.raises(ValueError):
        parse_export(b'<html>Error</html>','weather')


def test_official_overlap_and_hashes():
    review=overlap_review(ROOT/'data/retrieved/20261001',ROOT/'data/raw')
    assert review['generation_matched_rows']==264
    assert review['fuel_matched_rows']==96
    import hashlib
    manifest=json.loads((RAW/'manifest.json').read_text())
    assert len(manifest['official_exports'])==10
    for item in manifest['files']:
        assert hashlib.sha256((RAW/item['file']).read_bytes()).hexdigest()==item['sha256']
    for item in manifest['official_exports']:
        assert hashlib.sha256((ROOT/'data/retrieved/20261001'/item['file']).read_bytes()).hexdigest()==item['sha256']


@pytest.mark.parametrize('bad',[
    {**PROTOCOL,'train':['2023-01','2025-07']},
    {**PROTOCOL,'test':['2025-01','2026-08']},
    {**PROTOCOL,'test':['2026-08','2026-01']},
    {**PROTOCOL,'test':['2026-13','2026-14']}])
def test_invalid_training_splits_rejected(bad):
    with pytest.raises(ValueError):
        validate_split(bad)


def test_training_scope_and_frozen_overwrite_protection():
    df,_=build_dataset('concentration',RAW)
    masks=split_masks(df,PROTOCOL)
    assert {n:int(m.sum()) for n,m in masks.items()}=={'train':636,'validation':124,'test':176}
    assert not (masks['train'] & masks['test']).any()
    invalid=df.loc[df.reported_thermal_efficiency_pct.notna()]
    assert len(invalid)==3
    assert invalid.thermal_efficiency_pct.isna().all()
    assert invalid.target.notna().all()
    assert invalid.quality_reasons.map(lambda v:'invalid_efficiency' in json.loads(v)).all()
    pre=Preprocessor().fit(pd.DataFrame({'x':[1.,3.,np.nan]}),fit_scope='2023-01 through 2025-06 only')
    assert pre.transform(pd.DataFrame({'x':[np.nan,100000.]})).x.iloc[0]==2
    assert Preprocessor.from_dict(pre.to_dict()).fit_scope=='2023-01 through 2025-06 only'
    with pytest.raises(ValueError,match='overwrite'):
        train('concentration',ARTIFACT,raw_dir=RAW,protocol=PROTOCOL)


@pytest.mark.parametrize('split',['validation','test'])
def test_new_saved_predictions_match_inference(split):
    predictor=Predictor(ARTIFACT,history_db=False)
    saved=pd.read_csv(ARTIFACT/(split+'_predictions.csv'),parse_dates=['month'])
    data=pd.read_csv(ARTIFACT/'history.csv',parse_dates=['month'])
    joined=saved.merge(data,on=['plant','unit','month'],suffixes=('_saved',''),validate='one_to_one')
    actual=[predictor.predict(payload(row))['prediction']['value'] for _,row in joined.iterrows()]
    np.testing.assert_allclose(actual,joined.pred_ensemble,rtol=1e-7,atol=1e-8)
    assert predictor.predict(payload(joined.iloc[0]))['model']['trained_through']=='2025-06'


def test_new_history_bootstrap_does_not_backdate_retrieved_data(tmp_path):
    store=HistoryStore(tmp_path/'history.sqlite',model_key(ARTIFACT)[0])
    assert store.initialize(ARTIFACT)['latest_observed_month']=='2024-12'
    assert store.status()['row_count']==528
    result=import_bundle(store,RAW,start_month='2025-01',end_month='2026-08',
        source='official_test_copy',report_dir=tmp_path/'reports')
    assert result['inserted']==440
    assert store.status()['latest_observed_month']=='2026-08'
    historical,_=store.snapshot('2025-09-01T00:00:00+09:00')
    assert historical.month.max()==pd.Timestamp('2024-12-01')
    df,_=build_dataset('concentration',RAW)
    row=df.loc[(df.plant=='분당') & (df.unit==1) & (df.month==pd.Timestamp('2026-08-01'))].iloc[0].copy()
    request=payload(row);request['month']='2026-09'
    result=Predictor(ARTIFACT,history_db=store.path).predict(request)
    assert result['input_quality']['observed_previous_months']==3
    assert result['model']['trained_through']=='2025-06'


def test_unit_boolean_and_bad_reason_rejected(tmp_path):
    path=tmp_path/'history.sqlite'
    HistoryStore(path).initialize(ROOT/'artifacts/concentration')
    store=HistoryStore(path,model_key(ROOT/'artifacts/concentration')[0])
    frame,_=store.snapshot();frame=frame.iloc[[0]].copy();frame['month']=pd.to_datetime(['2025-01-01'])
    for col,value in [('unit',True),('quality_reasons',None),('quality_reasons','{broken')]:
        bad=frame.copy();bad[col]=value
        with pytest.raises(HistoryError):
            store.import_observations(bad,source='test',source_files={},batch_key='bad')
    assert store.status()['revision']==1
