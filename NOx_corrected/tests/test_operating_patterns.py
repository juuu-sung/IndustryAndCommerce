import json

import numpy as np
import pandas as pd
import pytest

from nox.features import build_features
from nox.app import create_app
from nox.llm import prompt
from nox.history import HistoryStore, _payload
from nox.operating_patterns import (FEATURES, HOURS, POLICY, aggregate_unit,
    attach_patterns, digest, fetch, longest_zero_run, read_hourly, validate_operation_history, validate_pattern)
from nox.operations_validation import compare
from nox.predict import Predictor, validate_model_input
from nox.schema import KEYS, ROOT
from nox.temporal_validation import METHODS, prepare_data, request_for


def synthetic(month='2024-02', unit=1, scale=1.):
    dates = pd.date_range(month + '-01', periods=pd.Period(month).days_in_month)
    matrix = np.full((len(dates), 24), 10. * scale)
    matrix[0] = 0.
    frame = pd.DataFrame(matrix, columns=HOURS)
    frame['date'], frame['unit'], frame['source_file'] = dates, unit, 'test.csv'
    frame['총량(KW)'] = frame[HOURS].sum(axis=1)
    return frame


def test_patterns_ignore_scale_row_order_and_other_units():
    a = aggregate_unit(synthetic(), 1, '2024-02')
    frame = pd.concat([synthetic(scale=1000), synthetic(unit=2, scale=999)]).sample(frac=1, random_state=1)
    b = aggregate_unit(frame, 1, '2024-02')
    np.testing.assert_allclose([a[c] for c in FEATURES], [b[c] for c in FEATURES])
    assert a['expected_hours'] == 696
    assert a['gen_zero_positive_transitions'] == 1
    assert a['gen_zero_hour_fraction'] == pytest.approx(1 / 29)
    assert a['gen_longest_zero_run_fraction'] == pytest.approx(24 / 696)


def test_missing_hours_are_not_zeros_and_break_runs():
    assert longest_zero_run([0, 0, np.nan, 0, 0, 0]) == 3
    f = synthetic().drop(index=0)
    row = aggregate_unit(f, 1, '2024-02')
    assert row['missing_hours'] == 24
    assert row['gen_zero_hour_fraction'] == 0
    assert row['gen_hour_coverage'] == pytest.approx(28 / 29)
    f = synthetic().drop(index=[0, 1])
    with pytest.raises(ValueError, match='95%'):
        row = aggregate_unit(f, 1, '2024-02')
        validate_pattern({c: row[c] for c in FEATURES}, '2024-02')


def test_all_zero_generation_is_finite_and_not_a_claim_about_nox():
    f = synthetic(scale=0)
    row = aggregate_unit(f, 1, '2024-02')
    assert row['gen_zero_day_fraction'] == 1
    assert row['gen_longest_zero_run_fraction'] == 1
    assert row['gen_peak_to_positive_mean'] == 0
    validate_pattern({c: row[c] for c in FEATURES}, '2024-02')
    row['gen_positive_hour_cv'] = 3
    with pytest.raises(ValueError, match='All-zero'):
        validate_pattern({c: row[c] for c in FEATURES}, '2024-02')


def test_provider_parser_only_accepts_one_empty_trailing_cell(tmp_path):
    columns = ['발전구분', '호기', '일자', *HOURS, '총량(KW)']
    record = ['영동', '1', '2024-02-01 00:00:00', *(['0'] * 24), '0']
    path = tmp_path / 'hour.csv'
    path.write_text(','.join(columns) + '\n' + ','.join(record) + ',\n', encoding='utf-8-sig')
    assert len(read_hourly(path, '2024-02')) == 1
    path.write_text(','.join(columns) + '\n' + ','.join(record) + ',123\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='column count'):
        read_hourly(path, '2024-02')


@pytest.mark.parametrize('changed', ['content', 'request'])
def test_cached_receipt_does_not_accept_tampered_content_or_another_query(tmp_path, changed):
    path = tmp_path / 'x.csv'
    path.write_text('original')
    receipt = {'url': 'https://example.org/data', 'method': 'POST', 'parameters': {'month': '2024-02'}, 'sha256': digest(path)}
    path.with_name('x.csv.provenance.json').write_text(json.dumps(receipt))
    if changed == 'content':
        path.write_text('altered')
        params = receipt['parameters']
    else:
        params = {'month': '2025-02'}
    with pytest.raises(ValueError, match='Cached source'):
        fetch('x.csv', receipt['url'], params, tmp_path)


@pytest.fixture(scope='module')
def operations():
    return prepare_data(ROOT / 'data/operations_20261002')


def test_source_quality_quarantine_keeps_targets_and_original_inputs(operations):
    df, _, source = operations
    base, _, _ = prepare_data(ROOT / 'data/unit_fuel_20261001')
    pd.testing.assert_frame_equal(df[base.columns], base)
    yd = df.loc[df.plant.eq('영동')]
    assert len(yd) == 72
    assert yd.gen_hour_coverage.notna().sum() == 64
    rejected = yd.loc[yd.operation_pattern_basis.eq('review_required')]
    assert len(rejected) == 8 and rejected[FEATURES].isna().all().all()
    assert rejected.target.notna().all()
    assert source['excluded_future_rows'] == 176
    validate_operation_history(df, POLICY)


def test_current_and_future_nox_cannot_change_generation_features(operations):
    df = operations[0]
    row = df.loc[df.plant.eq('영동') & df.gen_hour_coverage.notna()].tail(1)
    history = df.copy()
    history.loc[history.month >= row.month.iloc[0], 'target'] = 1e12
    a, b = build_features(row, df), build_features(row, history)
    pd.testing.assert_frame_equal(a, b)
    assert a.operation_pattern_available.iloc[0] == 1


def test_api_requires_explicit_unavailable_or_valid_pattern(operations):
    row = operations[0].loc[operations[0].plant.eq('영동') & operations[0].gen_hour_coverage.notna()].iloc[-1]
    known = {('영동', int(row.unit))}
    request = request_for(row)
    result = validate_model_input(request, known, POLICY)
    assert result['gen_hour_coverage'] == 1
    with pytest.raises(ValueError, match='no operation_pattern'):
        validate_model_input(request, known)
    request.pop('operation_pattern')
    with pytest.raises(ValueError, match='requires operation_pattern'):
        validate_model_input(request, known, POLICY)
    request['operation_pattern'] = None
    assert all(np.isnan(validate_model_input(request, known, POLICY)[c]) for c in FEATURES)
    request['operation_pattern'] = {c: -1 for c in FEATURES}
    with pytest.raises(ValueError, match='nonnegative'):
        validate_model_input(request, known, POLICY)


def test_incomplete_history_and_corrupted_quarantine_are_rejected(operations):
    df = operations[0].copy()
    df.loc[df.plant.eq('영동'), 'operation_pattern_basis'] = np.nan
    with pytest.raises(ValueError, match='requires an observed'):
        validate_operation_history(df, POLICY)
    df = operations[0].copy()
    df.loc[df.operation_pattern_basis.eq('review_required'), FEATURES[0]] = 1
    with pytest.raises(ValueError, match='all-null'):
        validate_operation_history(df, POLICY)


def test_comparison_rejects_target_baseline_or_row_changes():
    frame = pd.DataFrame({'fold': ['A'], 'plant': ['영동'], 'unit': [1], 'month': [pd.Timestamp('2024-07-01')],
        'target': [3.], 'supported': [True], 'api_input_accepted': [True], **{'pred_' + m: [2.] for m in METHODS}})
    assert compare(frame, frame)[1][0]['rmse_improvement_pct'] == 0
    for col in ['target', 'pred_last_month']:
        changed = frame.copy()
        changed[col] = 999
        with pytest.raises(AssertionError):
            compare(frame, changed)


def test_saved_candidate_preserves_operations_in_sqlite(operations, tmp_path):
    folder = ROOT / 'reports/operations_validation_20261002/models/C'
    if not (folder / 'metadata.json').exists():
        pytest.skip('Run fixed operation-feature experiment first')
    store = HistoryStore(tmp_path / 'history.sqlite')
    store.initialize(folder)
    frozen, _ = store.snapshot()
    assert set(FEATURES).issubset(frozen.columns)
    assert frozen.gen_hour_coverage.notna().sum() > 0
    df = operations[0]
    imported = df.loc[df.month.eq(pd.Timestamp('2025-01-01')) & ~((df.plant == '여수') & (df.unit == 2))]
    store.import_observations(imported, source='operation test', source_files={'manifest.json': 'test'}, batch_key='operations-test')
    snap, _ = store.snapshot()
    for _, row in imported.loc[imported.plant.eq('영동')].iterrows():
        saved = snap.loc[snap.plant.eq(row.plant) & snap.unit.eq(row.unit) & snap.month.eq(row.month)].iloc[0]
        assert _payload(saved)['operation_source_file'] == _payload(row)['operation_source_file']
        np.testing.assert_allclose(saved[FEATURES].to_numpy(float), row[FEATURES].to_numpy(float), equal_nan=True)
    predictor = Predictor(folder, history_db=store.path)
    future = df.loc[df.plant.eq('영동') & df.month.eq(pd.Timestamp('2025-02-01'))].iloc[0]
    response = predictor.predict(request_for(future))
    assert response['input_quality']['history']['current_revision'] == 2


def test_flask_candidate_and_llm_receive_validated_pattern_semantics(operations):
    folder = ROOT / 'reports/operations_validation_20261002/models/C'
    if not (folder / 'metadata.json').exists():
        pytest.skip('Run fixed operation-feature experiment first')
    df = operations[0]
    row = df.loc[df.plant.eq('영동') & df.month.eq(pd.Timestamp('2025-10-01'))].iloc[0]
    request = request_for(row)
    client = create_app(Predictor(folder, history_db=False)).test_client()
    response = client.post('/api/predict', json=request)
    assert response.status_code == 200
    body = response.get_json()
    assert body['input_quality']['operation_pattern_status'] == 'observed_electrical_generation'
    assert 'electrical_generation_patterns' in prompt(request, body)
    assert '보일러 또는 SCR 정지로 단정하지' in prompt(request, body)
    request['operation_pattern'] = None
    unavailable = client.post('/api/predict', json=request)
    assert unavailable.status_code == 200
    assert unavailable.get_json()['input_quality']['operation_pattern_status'] == 'unavailable_or_review_required'
    request.pop('operation_pattern')
    assert client.post('/api/predict', json=request).status_code == 400
