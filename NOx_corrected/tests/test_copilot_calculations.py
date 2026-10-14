import csv
import hashlib
import json
import math
import shutil

import pytest

from nox.copilot.archive_repository import ArchiveRepository, Observation, previous_months
from nox.copilot.calculations import calculate_case, calculate_metrics, main, save_calculation
from nox.copilot.registry import CaseRegistry, RegistrationError
from nox.schema import ROOT
# Reuse the explicit synthetic registry contract; these are not model inference fixtures.
from test_copilot_registry import bundle, registry, CASE, KEY, csv_rows, write_csv


def obs(month, value, status=None):
    return Observation(month, value, status or ('unverified' if value is not None else 'unavailable'))


def window(a=8., b=9., c=10.):
    return tuple(obs(month, value) for month, value in zip(('2025-07', '2025-08', '2025-09'), (a, b, c)))


def test_calendar_metrics_and_signed_errors():
    metrics = calculate_metrics(obs('2025-10', 12), window(), {'xgboost': 14, 'lightgbm': 11})
    assert metrics['previous_month']['delta_ppm'] == 2
    assert metrics['previous_month']['change_percent'] == 20
    assert metrics['previous_3months']['mean_ppm'] == 9
    assert metrics['previous_3months']['current_minus_mean_ppm'] == 3
    assert metrics['previous_3months']['observed_count'] == 3
    assert metrics['prediction_errors']['xgboost']['signed_error_ppm'] == 2
    assert metrics['prediction_errors']['lightgbm']['signed_error_ppm'] == -1
    assert metrics['prediction_errors']['lightgbm']['absolute_error_ppm'] == 1
    assert 'rmse' not in json.dumps(metrics).lower()


def test_missing_previous_month_does_not_use_older_observation():
    metrics = calculate_metrics(obs('2025-10', 12), window(c=None), {'lightgbm': 11})
    previous = metrics['previous_month']
    assert previous['observation']['month'] == '2025-09'
    assert previous['delta_ppm'] is None and previous['change_percent'] is None
    assert previous['delta_reasons'] == ['previous_month_unavailable']
    assert metrics['previous_3months']['mean_ppm'] == 8.5
    assert metrics['previous_3months']['observed_count'] == 2
    assert metrics['previous_3months']['coverage'] == 'partial'


def test_zero_previous_is_included_in_mean_but_percent_is_undefined():
    metrics = calculate_metrics(obs('2025-10', 12), window(c=0), {'lightgbm': 11})
    assert metrics['previous_month']['delta_ppm'] == 12
    assert metrics['previous_month']['change_percent'] is None
    assert metrics['previous_month']['change_percent_reasons'] == ['previous_month_zero']
    assert metrics['previous_3months']['mean_ppm'] == pytest.approx(17 / 3)
    assert metrics['previous_3months']['observed_count'] == 3
    json.dumps(metrics, allow_nan=False)


def test_missing_current_keeps_history_but_disables_errors():
    metrics = calculate_metrics(obs('2025-10', None), window(), {'lightgbm': 11})
    assert metrics['previous_month']['delta_ppm'] is None
    assert metrics['previous_3months']['mean_ppm'] == 9
    assert metrics['previous_3months']['current_minus_mean_ppm'] is None
    error = metrics['prediction_errors']['lightgbm']
    assert error['signed_error_ppm'] is None and error['absolute_error_ppm'] is None
    assert error['status'] == 'unavailable'
    assert error['reasons'] == ['current_observation_unavailable']


def test_missing_window_and_prediction_are_explicit():
    metrics = calculate_metrics(obs('2025-10', 12), window(None, None, None), {'lightgbm': None})
    assert metrics['previous_3months']['mean_ppm'] is None
    assert metrics['previous_3months']['observed_count'] == 0
    assert metrics['previous_3months']['coverage'] == 'unavailable'
    assert metrics['prediction_errors']['lightgbm']['reasons'] == ['prediction_unavailable']


def test_review_values_preserve_quality_and_precision():
    reviewed = Observation('2025-09', 10.123456789, 'review_required', ('high_nox',))
    metrics = calculate_metrics(obs('2025-10', 12), (*window()[:2], reviewed), {'lightgbm': 11})
    assert metrics['previous_month']['observation']['value'] == 10.123456789
    assert metrics['previous_month']['observation']['quality_reasons'] == ['high_nox']
    assert metrics['previous_3months']['review_required_months'] == ['2025-09']
    assert metrics['previous_3months']['observed_count'] == 3


def test_window_crosses_year_boundary():
    assert previous_months('2025-01') == ('2024-10', '2024-11', '2024-12')


@pytest.mark.parametrize('prior', [
    (), window()[:2], window()[::-1],
    (obs('2025-06', 7), *window()[1:]),
    (*window()[:2], obs('2025-10', 12)),
    (*window()[:2], obs('2025-11', 13)),
])
def test_noncalendar_current_or_future_history_is_rejected(prior):
    with pytest.raises(RegistrationError, match='three calendar months'):
        calculate_metrics(obs('2025-10', 12), prior, {'lightgbm': 11})


@pytest.mark.parametrize('value', [True, -1, math.nan, math.inf, '12'])
def test_bad_observation_and_prediction_numbers_are_rejected(value):
    with pytest.raises(RegistrationError):
        obs('2025-10', value)
    with pytest.raises(RegistrationError):
        calculate_metrics(obs('2025-10', 12), window(), {'lightgbm': value})


def test_availability_conflict_and_overflow_are_rejected():
    with pytest.raises(RegistrationError, match='availability'):
        Observation('2025-10', None, 'valid')
    with pytest.raises(RegistrationError, match='immutable tuple'):
        Observation('2025-10', 12, 'valid', 'high_nox')
    with pytest.raises(RegistrationError, match='finite'):
        calculate_metrics(obs('2025-10', 1e308), window(c=1e-308), {'lightgbm': 11})


def test_unknown_prediction_method_is_rejected():
    with pytest.raises(RegistrationError):
        calculate_metrics(obs('2025-10', 12), window(), {'invented': 11})


def test_calculation_is_bound_to_registered_sources_and_idempotent(bundle, registry, tmp_path):
    registered = registry.register(bundle, **CASE)
    original = registry.path_for(KEY).read_bytes()
    result = calculate_case(KEY, registry)
    assert result['provenance']['case_version_id'] == registered['case_version_id']
    assert result['provenance']['history_file_sha256'] == registered['files']['history.csv']['sha256']
    assert result['metrics']['current_observation']['quality_reasons'] == ['high_nox', 'weather_missing']
    assert result['target_verification_status'] == 'pending_provider_confirmation'
    assert result['analysis_mode'] == 'evaluation_replay'
    assert not result['model']['independent_final_evaluation_verified']
    assert calculate_case(KEY, registry)['calculation_id'] == result['calculation_id']
    path, first = save_calculation(KEY, registry, tmp_path / 'results')
    before = path.read_bytes(), path.stat().st_mtime_ns
    same_path, second = save_calculation(KEY, registry, tmp_path / 'results')
    assert same_path == path and first == second
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    assert registry.path_for(KEY).read_bytes() == original


def test_calculation_hash_ignores_machine_path_and_registration_time(bundle, registry, tmp_path):
    registry.register(bundle, **CASE)
    copied = tmp_path / 'copy'
    shutil.copytree(bundle, copied)
    other = CaseRegistry(tmp_path / 'other_cases')
    other.register(copied, **CASE)
    assert calculate_case(KEY, registry)['calculation_id'] == calculate_case(KEY, other)['calculation_id']


def test_source_change_is_blocked_before_any_output(bundle, registry, tmp_path):
    registry.register(bundle, **CASE)
    path = bundle / 'history.csv'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(RegistrationError, match='checksum'):
        save_calculation(KEY, registry, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_case_replacement_during_save_is_blocked(bundle, registry, tmp_path, monkeypatch):
    registry.register(bundle, **CASE)
    verify = registry.verify
    def changed(case_key):
        return {**verify(case_key), 'case_version_id': 'f' * 64}
    monkeypatch.setattr(registry, 'verify', changed)
    with pytest.raises(RegistrationError, match='case changed'):
        save_calculation(KEY, registry, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_captured_bytes_are_used_without_unverified_reopen(bundle, registry, monkeypatch):
    registry.register(bundle, **CASE)
    real_read = registry.read_verified
    def read_then_mutate(case_key):
        manifest, blobs = real_read(case_key)
        rows = csv_rows(bundle / 'history.csv')
        rows[-2]['target'] = '999'
        write_csv(bundle / 'history.csv', rows)
        return manifest, blobs
    monkeypatch.setattr(registry, 'read_verified', read_then_mutate)
    result = calculate_case(KEY, registry)
    assert result['metrics']['previous_month']['observation']['value'] == 10
    with pytest.raises(RegistrationError, match='checksum'):
        calculate_case(KEY, registry)


@pytest.mark.parametrize('kind', ['missing_month', 'missing_target', 'other_unit', 'bad_quality'])
def test_archive_handles_missing_and_unrelated_or_bad_rows(bundle, registry, kind):
    rows = csv_rows(bundle / 'history.csv')
    september = next(row for row in rows if row['month'] == '2025-09-01')
    if kind == 'missing_month':
        rows.remove(september)
    elif kind == 'missing_target':
        september.update(target='', quality_status='unavailable')
    elif kind == 'other_unit':
        rows.append({**september, 'unit': '2', 'target': '999'})
    else:
        september['quality_status'] = 'unknown'
    write_csv(bundle / 'history.csv', rows)
    if kind.startswith('missing'):
        predictions = csv_rows(bundle / 'test_predictions.csv')
        predictions[0]['nox_history_count'] = '2'
        write_csv(bundle / 'test_predictions.csv', predictions)
    registry.register(bundle, **CASE)
    if kind == 'bad_quality':
        with pytest.raises(RegistrationError, match='quality status'):
            calculate_case(KEY, registry)
        return
    result = calculate_case(KEY, registry)['metrics']
    if kind == 'other_unit':
        assert result['previous_3months']['mean_ppm'] == 9
    else:
        assert result['previous_month']['delta_ppm'] is None
        assert result['previous_month']['observation']['missing_reason'] == (
            'missing_calendar_month' if kind == 'missing_month' else 'missing_target')


def test_saved_calculation_tampering_and_unsafe_destination_are_blocked(bundle, registry, tmp_path):
    registry.register(bundle, **CASE)
    path, result = save_calculation(KEY, registry, tmp_path / 'results')
    result['metrics']['previous_month']['delta_ppm'] = 999
    path.write_text(json.dumps(result))
    with pytest.raises(RegistrationError, match='refusing to overwrite'):
        save_calculation(KEY, registry, tmp_path / 'results')
    for output in (bundle / 'derived', registry.directory / 'derived'):
        with pytest.raises(RegistrationError, match='outside'):
            save_calculation(KEY, registry, output)
        assert not output.exists()


def test_cli_success_and_failure_do_not_log_numeric_source_rows(bundle, registry, tmp_path, capsys):
    registry.register(bundle, **CASE)
    args = ['--case-key', KEY, '--registry-dir', str(registry.directory),
            '--output-dir', str(tmp_path / 'results')]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['status'] == 'ok' and output['observed_previous_months'] == 3
    assert 'metrics' not in output and 'prediction_errors' not in output
    (bundle / 'history.csv').write_bytes(b'changed')
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


def test_real_yeongdong_calculation_matches_independent_csv_arithmetic(tmp_path):
    artifact = ROOT / 'reports/temporal_validation_20261001/models/unit_actual/C'
    if not (artifact / 'history.csv').is_file():
        pytest.skip('Private saved evaluation assets are not in the public repository')
    registry = CaseRegistry(tmp_path / 'cases')
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact.iterdir() if path.is_file()}
    key = 'yeongdong_u1_2025_10_c_v1'
    registry.register(artifact, case_key=key, plant='영동', unit=1, month='2025-10')
    result = calculate_case(key, registry)
    with (artifact / 'history.csv').open(encoding='utf-8-sig') as source:
        history = {row['month'][:7]: float(row['target']) for row in csv.DictReader(source)
                   if row['plant'] == '영동' and int(row['unit']) == 1 and row['target']}
    with (artifact / 'test_predictions.csv').open(encoding='utf-8-sig') as source:
        row = next(row for row in csv.DictReader(source) if row['plant'] == '영동'
                   and int(row['unit']) == 1 and row['month'].startswith('2025-10'))
    metrics = result['metrics']
    assert metrics['previous_month']['delta_ppm'] == pytest.approx(history['2025-10'] - history['2025-09'])
    assert metrics['previous_3months']['mean_ppm'] == pytest.approx(sum(history[m] for m in ('2025-07', '2025-08', '2025-09')) / 3)
    for method, error in metrics['prediction_errors'].items():
        expected = float(row['pred_' + method]) - history['2025-10']
        assert error['signed_error_ppm'] == pytest.approx(expected)
        assert error['absolute_error_ppm'] == pytest.approx(abs(expected))
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact.iterdir() if path.is_file()}
