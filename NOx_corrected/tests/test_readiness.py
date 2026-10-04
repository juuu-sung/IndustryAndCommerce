import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nox.candidate_selection import (select_features, nominate, fit_trial, saved_predict, validate_protocol)
from nox.features import build_features
from nox.gemini_check import check
from nox.operating_patterns import collect
from nox.weather_quality import (recover_weather, monthly_metadata, attach_metadata, POLICY,
                                 QUALITY_FEATURES)


def weather(rows):
    return pd.DataFrame(rows, columns=['사업소', '호기', '일자', '온도', '습도', '풍속', '풍향'])


def test_recovery_preserves_observations_and_rejects_unmapped_station():
    old = weather([['영동', '-', '2024-02-01', 0., np.nan, 0., 0.]])
    new = weather([['영동', '-', '20240201', 20., 50., 1., 90.],
                   ['영동', '-', '20240202', 2., 60., 2., 180.],
                   ['영동', 'new', '20240201', 30., 40., 3., 270.]])
    combined, sources, audit = recover_weather(old, [('fresh', new)])
    assert len(combined) == 2
    assert combined.loc[0, '온도'] == 0 and combined.loc[0, '풍속'] == 0
    assert combined.loc[0, '습도'] == 50
    assert len(audit['preserved_conflicts']) == 3
    assert len(audit['unmapped_stations']) == 1
    assert audit['existing_observations_replaced'] == 0
    gen = pd.DataFrame({'plant': ['영동'], 'month': [pd.Timestamp('2024-02-01')]})
    metadata = monthly_metadata(combined, gen, sources)
    assert metadata.expected_days.iloc[0] == 29
    assert metadata.weather_temperature_coverage.iloc[0] == 2 / 29
    assert metadata.weather_wind_direction_coverage.iloc[0] == 2 / 29


def test_duplicate_weather_and_bad_metadata_fail(tmp_path):
    frame = weather([['영동', '-', '20240101', 1, 50, 1, 0]] * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        recover_weather(frame, [])
    base = weather([['영동', '-', '20240101', 1, 150, 1, 0]])
    recovered, sources, _ = recover_weather(base, [])
    gen = pd.DataFrame({'plant': ['영동', '여수'], 'month': [pd.Timestamp('2024-01-01')] * 2})
    meta = monthly_metadata(recovered, gen, sources)
    assert meta.weather_humidity_missing.eq(1).all()
    assert meta.loc[meta.plant.eq('여수'), 'weather_temperature_missing'].iloc[0] == 1
    meta.to_csv(tmp_path / 'weather_monthly_metadata.csv', index=False)
    attached = attach_metadata(gen, tmp_path, {'weather_quality_policy': POLICY})
    assert len(attached) == 2
    meta.loc[0, 'weather_temperature_coverage'] = 2
    meta.to_csv(tmp_path / 'weather_monthly_metadata.csv', index=False)
    with pytest.raises(ValueError, match='coverage'):
        attach_metadata(gen, tmp_path, {'weather_quality_policy': POLICY})


def test_no_weather_removes_quality_dependents():
    X = pd.DataFrame({**{c: [1.] for c in QUALITY_FEATURES}, 'temperature_c': [3.],
                      'humidity_pct': [70.], 'generation_mwh': [100.]})
    candidate = {'drop_weather': True}
    assert list(select_features(X, candidate, 'weather_quality')) == ['generation_mwh']
    with pytest.raises(ValueError, match='input policy'):
        select_features(X, candidate, 'invalid')


def test_nomination_ignores_outer_results_and_enforces_plant_guard():
    rows = []
    for candidate, predictions in [('full_log_lgb', [12., 12.]), ('other', [10., 12.3])]:
        for plant, pred in zip(['a', 'b'], predictions):
            rows.append({'candidate': candidate, 'input_policy': 'values_only', 'fold': 'A',
                         'split': 'validation', 'plant': plant, 'unit': 1,
                         'month': pd.Timestamp('2024-01-01'), 'target': 10., 'prediction': pred,
                         'previous_month': 8., 'supported': True, 'weather_missing': False, 'high_target': False})
    val = pd.DataFrame(rows)
    outer = val.assign(split='test', target=1e10, prediction=-1e10)
    result = nominate(pd.concat([val, outer], ignore_index=True))
    assert result == nominate(val)
    assert result['nominee']['candidate'] == 'full_log_lgb'
    other = next(r for r in result['validation_comparison'] if r['candidate'] == 'other')
    assert 'plant_mae:b' in other['guard_failures']


@pytest.mark.parametrize('transform', ['log1p', 'ppm'])
def test_trial_is_independent_of_outer_targets_and_reloads(tmp_path, transform):
    from nox.temporal_validation import prepare_data
    from nox.schema import ROOT
    df, _, _ = prepare_data(ROOT / 'data/operations_20261002')
    period = {'train': ['2023-01', '2023-12'], 'validation': ['2024-01', '2024-06'],
              'test': ['2024-07', '2024-12']}
    candidate = {'id': 'full_log_lgb', 'drop_weather': False, 'transform': transform, 'method': 'lightgbm'}
    first, _ = fit_trial(df, period, candidate, 'values_only', 0, tmp_path / 'a')
    changed = df.copy()
    changed.loc[changed.month.ge('2024-07-01'), 'target'] = 1e8
    changed.loc[changed.month.ge('2024-07-01'), 'generation_mwh'] = 1e8
    second, _ = fit_trial(changed, period, candidate, 'values_only', 0, tmp_path / 'b')
    pd.testing.assert_frame_equal(first, second)
    for name in ['lgb_model.txt', 'preprocessor.json', 'weight_trials.json']:
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()
    selected = df.month.between('2024-01-01', '2024-06-01') & df.target.notna()
    predictions = saved_predict(tmp_path / 'a', build_features(df).loc[selected])
    np.testing.assert_allclose(predictions, first.prediction, rtol=0, atol=1e-10)
    (tmp_path / 'a' / 'lgb_model.txt').write_text('corrupt')
    with pytest.raises(ValueError, match='changed'):
        saved_predict(tmp_path / 'a', build_features(df).loc[selected])


def test_current_month_collection_rejected_before_network(tmp_path):
    current = pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m')
    with pytest.raises(ValueError, match='completed'):
        collect(tmp_path, start_month=current, end_month=current)


def test_changed_protocol_is_rejected_before_fitting():
    from nox.schema import ROOT
    protocol = json.loads((ROOT / 'reports/readiness_20261002/protocol.json').read_text())
    validate_protocol(protocol)
    protocol['selection']['primary'] = 'RMSE'
    with pytest.raises(ValueError, match='criteria'):
        validate_protocol(protocol)


def test_gemini_missing_configuration_never_calls(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    result = check(live=True)
    assert result['status'] == 'not_run'
    assert not result['actual_external_call_verified']
