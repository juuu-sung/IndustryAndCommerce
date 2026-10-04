"""Synthetic provider evidence stays in pytest temporary directories only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nox.target_validation import (template_rule, validate_rule, replay_intervals,
    compare_reference, monthly_targets, verify, load_verified_version)
from nox.temporal_validation import sha256


@pytest.fixture
def rule(tmp_path):
    doc = tmp_path / 'synthetic_evidence.txt'
    doc.write_text('Synthetic fixture: not real provider certification')
    value = template_rule()
    value.update(rule_id='synthetic_rule', confirmation_status='confirmed',
        provider_owner='synthetic_test_owner', confirmed_at='2026-01-01T00:00:00+09:00',
        applicable_from='2023-01-01', applicable_to='2025-12-31',
        nox_basis='synthetic_provider_reported_ppm', minimum_daily_valid_seconds=3600,
        reference_tolerance_ppm=.005001, reference_tolerance_seconds=0.,
        minimum_monthly_day_coverage=.01, status_codebook={'N': 'valid', 'I': 'invalid'},
        stack_map=[{'plant': '분당', 'unit': 1, 'stack_id': '1'}],
        required_case_types=['normal', 'spike'],
        documents=[{'file': str(doc), 'kind': kind, 'sha256': sha256(doc)} for kind in
                   ['daily_definition', 'status_codebook', 'stack_mapping']])
    return value


def slot(date='2025-01-01', *, start=0, end=24, nox=10., status='N', stack='1', valid=None):
    day = pd.Timestamp(date, tz='Asia/Seoul')
    return {'plant': '분당', 'unit': 1, 'stack_id': stack,
        'interval_start_kst': (day + pd.Timedelta(hours=start)).isoformat(),
        'interval_end_kst': (day + pd.Timedelta(hours=end)).isoformat(),
        'nox_ppm_provider': nox, 'valid_seconds': (end-start)*3600 if valid is None else valid,
        'provider_status_code': status}


def reference(daily, cases=('normal', 'spike')):
    ref = daily[['plant', 'stack_id', 'date', 'nox_ppm', 'valid_seconds']].copy()
    ref['case_type'] = [cases[i % len(cases)] for i in range(len(ref))]
    return ref


def test_weighting_preserves_valid_peak_and_zero(rule, tmp_path):
    validate_rule(rule, tmp_path)
    frame = pd.DataFrame([slot(start=0, end=1, nox=100), slot(start=1, end=24, nox=20),
                          slot('2025-01-02', nox=900), slot('2025-01-03', nox=0)])
    before = frame.copy(deep=True)
    normalized, daily = replay_intervals(frame, rule)
    pd.testing.assert_frame_equal(frame, before)
    assert daily.nox_ppm.tolist() == pytest.approx([560/24, 900, 0])
    assert normalized.selected_nox_ppm.max() == 900
    assert monthly_targets(daily, rule).target.iloc[0] == pytest.approx((560/24+900)/3)
    assert compare_reference(daily, reference(daily), rule).matches.all()


def test_invalid_slot_keeps_raw_value_without_adding_to_mean(rule):
    frame = pd.DataFrame([slot(start=0, end=1, nox=10),
                          slot(start=1, end=24, nox=9999, status='I', valid=0)])
    normalized, daily = replay_intervals(frame, rule)
    assert normalized.selected_nox_ppm.iloc[1] == 9999
    assert daily.nox_ppm.iloc[0] == 10
    assert daily.valid_seconds.iloc[0] == 3600


@pytest.mark.parametrize('change,reason', [
    ({'provider_status_code': 'unmapped'}, 'Unmapped'),
    ({'valid_seconds': 90000}, 'valid time'),
    ({'provider_status_code': 'I'}, 'Invalid interval'),
    ({'nox_ppm_provider': -1}, 'negative'),
    ({'unit': 2}, 'confirmed mapping'),
    ({'interval_start_kst': '2025-01-01T00:00:00'}, 'timezone'),
    ({'interval_end_kst': '2025-01-02T01:00:00+09:00'}, 'cross-midnight'),
])
def test_bad_intervals_block_verification(rule, change, reason):
    frame = pd.DataFrame([{**slot(), **change}])
    with pytest.raises(ValueError, match=reason):
        replay_intervals(frame, rule)


def test_unknown_flags_overlap_and_missing_slots_block(rule):
    rule['status_codebook']['U'] = 'unknown'
    with pytest.raises(ValueError, match='unknown'):
        replay_intervals(pd.DataFrame([slot(status='U')]), rule)
    with pytest.raises(ValueError, match='overlapping'):
        replay_intervals(pd.DataFrame([slot(), slot()]), rule)
    with pytest.raises(ValueError, match='Incomplete daily'):
        replay_intervals(pd.DataFrame([slot(end=1)]), rule)


def test_missing_b_stack_cannot_be_averaged_as_complete_unit(rule):
    rule['stack_map'].append({'plant': '분당', 'unit': 1, 'stack_id': '1B'})
    _, daily = replay_intervals(pd.DataFrame([slot(nox=100)]), rule)
    assert monthly_targets(daily, rule).target.isna().all()
    _, daily = replay_intervals(pd.DataFrame([slot(nox=100), slot(nox=20, stack='1B')]), rule)
    assert monthly_targets(daily, rule).target.iloc[0] == 60


def test_provider_reference_mismatch_and_missing_cases(rule):
    _, daily = replay_intervals(pd.DataFrame([slot(), slot('2025-01-02', nox=700)]), rule)
    ref = reference(daily)
    ref.loc[0, 'nox_ppm'] = 99
    assert not compare_reference(daily, ref, rule).matches.all()
    ref.loc[:, 'case_type'] = 'normal'
    with pytest.raises(ValueError, match='required states'):
        compare_reference(daily, ref, rule)


def test_all_invalid_day_and_low_month_coverage_remain_missing(rule):
    _, daily = replay_intervals(pd.DataFrame([slot(status='I', valid=0, nox=np.nan)]), rule)
    assert daily.nox_ppm.isna().all()
    ref = reference(daily, ('calibration',))
    rule['required_case_types'] = ['calibration']
    assert compare_reference(daily, ref, rule).matches.all()
    assert monthly_targets(daily, rule).target.isna().all()
    _, daily = replay_intervals(pd.DataFrame([slot()]), rule)
    rule['minimum_monthly_day_coverage'] = .8
    assert monthly_targets(daily, rule).target.isna().all()


def files_for_verification(tmp_path, rule, dates=('2025-01-01', '2025-01-02')):
    intervals = pd.DataFrame([slot(date, nox=10+i*600) for i, date in enumerate(dates)])
    _, daily = replay_intervals(intervals, rule)
    rule_path, interval_path, ref_path = [tmp_path / name for name in ['rule.json', 'intervals.csv', 'reference.csv']]
    rule_path.write_text(json.dumps(rule))
    intervals.to_csv(interval_path, index=False)
    reference(daily).to_csv(ref_path, index=False)
    return rule_path, interval_path, ref_path


def test_verified_version_checks_hashes_and_refuses_overwrite(tmp_path, rule):
    files = files_for_verification(tmp_path, rule)
    output = tmp_path / 'verified'
    assert verify(*files, output)['status'] == 'verified_research'
    frame, _ = load_verified_version(output)
    assert frame.target.notna().all()
    with pytest.raises(ValueError, match='overwrite'):
        verify(*files, output)
    (output / 'monthly_targets.csv').write_text('changed')
    with pytest.raises(ValueError, match='Frozen target'):
        load_verified_version(output)


def test_pending_missing_evidence_and_wrong_definition_block(tmp_path, rule):
    with pytest.raises(ValueError, match='pending'):
        validate_rule(template_rule(), tmp_path)
    rule['daily_aggregation'] = 'guess_from_daily_oxygen'
    with pytest.raises(ValueError, match='Unsupported'):
        validate_rule(rule, tmp_path)
    rule['documents'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='changed evidence'):
        validate_rule(rule, tmp_path)


def test_replay_mismatch_cannot_be_loaded_for_training(tmp_path, rule):
    files = files_for_verification(tmp_path, rule)
    ref = pd.read_csv(files[2]); ref.loc[0, 'nox_ppm'] = 99; ref.to_csv(files[2], index=False)
    output = tmp_path / 'mismatch'
    assert verify(*files, output)['status'] == 'replay_mismatch'
    assert not (output / 'monthly_targets.csv').exists()
    with pytest.raises(ValueError, match='training blocked'):
        load_verified_version(output)


def raw_fixture(path):
    path.mkdir()
    months = ['202301', '202302', '202401', '202407']
    pd.DataFrame([{'사업소': '분당', '호기': 1, '월': month,
        '용량(MW)': 100, '발전량(MWh)': 1000, '열효율(%)': 40, '이용률(%)': 30}
        for month in months]).to_csv(path / 'generation.csv', index=False)
    pd.DataFrame([{'사업소': '분당', '일자': month, '유연탄': 0, '무연탄': 0,
        '계(석탄)': 0, '유류': 0, 'LNG': 10, '고형연료': 0, '우드펠릿': 0}
        for month in months]).to_csv(path / 'fuel_site.csv', index=False)
    pd.DataFrame([{'사업소': '분당', '호기': '1호기', '일자': month+'01', 'NOX': 10,
        '산소': 10, '유량': 10, '온도': 100} for month in months]).to_csv(path / 'emissions_daily.csv', index=False)
    pd.DataFrame([{'사업소': '분당', '호기': 'fixture', '일자': month[:4]+'-'+month[4:]+'-01',
        '온도': 10, '습도': 50, '풍향': 0, '풍속': 2} for month in months]).to_csv(path / 'weather_daily.csv', index=False)
    (path / 'manifest.json').write_text(json.dumps({'files': [{'file': p.name, 'sha256': sha256(p)}
        for p in path.glob('*.csv')]}))


def test_partial_case_verification_is_not_training_coverage(tmp_path, rule):
    from nox.validated_training import validated_data
    raw = tmp_path / 'raw'; raw_fixture(raw)
    files = files_for_verification(tmp_path, rule, ('2023-01-01', '2024-01-01'))
    version = tmp_path / 'version'; verify(*files, version)
    with pytest.raises(ValueError, match='subset'):
        validated_data(raw, version)


def test_verified_training_reloads_and_freezes_before_same_target_scoring(tmp_path, rule):
    from nox.validated_training import run
    raw = tmp_path / 'raw'; raw_fixture(raw)
    files = files_for_verification(tmp_path, rule, ('2023-01-01', '2023-02-01', '2024-01-01', '2024-07-01'))
    version = tmp_path / 'version'; verify(*files, version)
    output = tmp_path / 'models'
    result = run(raw, version, output, seeds=(0,), folds={'A': {
        'train': ['2023-01', '2023-12'], 'validation': ['2024-01', '2024-06'],
        'test': ['2024-07', '2024-12']}})
    assert result['same_target_and_cohort_all_candidates']
    assert result['saved_reload_max_abs_error'] == 0
    assert result['model_fit_count'] == 5
    assert not result['service_replaced']
    lock = json.loads((output / 'selection_lock.json').read_text())
    assert not lock['outer_predictions_created']
    assert all(sha256(output / name) == expected for name, expected in lock['files'].items())
    assert not (output / 'metadata.json').exists()
