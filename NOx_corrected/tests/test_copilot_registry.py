import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from nox.copilot.registry import CaseRegistry, FILES, MODEL_FILES, RegistrationError, main
from nox.schema import ROOT, TARGETS

KEY = 'demo_u1_2025_10_c_v1'
CASE = {'case_key': KEY, 'plant': 'Demo', 'unit': 1, 'month': '2025-10'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with path.open(encoding='utf-8') as source:
        return list(csv.DictReader(source))


def relock(artifact):
    meta = json.loads((artifact / 'metadata.json').read_text())
    meta['bundle_sha256'] = {name: sha(artifact / name) for name in MODEL_FILES}
    write_json(artifact / 'metadata.json', meta)
    write_json(artifact / 'selection_lock.json', {
        'sha256': {name: sha(artifact / name) for name in ('metadata.json', *MODEL_FILES)},
        'test_predictions_created': False,
    })


@pytest.fixture
def bundle(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    columns = ['feature_a', 'feature_b']
    write_json(artifact / 'xgb_model.json', {'learner': {'feature_names': columns}})
    (artifact / 'lgb_model.txt').write_text('tree\nfeature_names=feature_a feature_b\n')
    write_json(artifact / 'preprocessor.json', {
        'columns': columns, 'medians': {'feature_a': 1., 'feature_b': 2.},
        'fit_scope': '2023-01 through 2024-12 only', 'version': 1,
    })
    write_json(artifact / 'metadata.json', {
        'schema_version': 1, 'target_mode': 'concentration', 'target': TARGETS['concentration'],
        'split': {'train': ['2023-01', '2024-12'], 'validation': ['2025-01', '2025-06'],
                  'test': ['2025-07', '2025-12']},
        'trained_through': '2024-12', 'trained_at': '2026-10-01T12:00:00+00:00',
        'weights_locked_before_test': True, 'known_units': [{'plant': 'Demo', 'unit': 1}],
        'evaluation_mode': 'retrospective monthly research evaluation',
        'selection_scope': 'validation only', 'source_manifest_sha256': 'a' * 64,
        'xgboost_weight': 0., 'lightgbm_weight': 1.,
    })
    values = [
        ('2023-01', 2.), ('2024-12', 3.), ('2025-03', 4.),
        ('2025-04', 5.), ('2025-05', 6.), ('2025-06', 7.),
        ('2025-07', 8.), ('2025-08', 9.), ('2025-09', 10.), ('2025-10', 12.),
    ]
    history = [{
        'plant': 'Demo', 'unit': 1, 'month': month + '-01', 'target': value,
        'quality_status': 'review_required' if month == '2025-10' else 'unverified',
        'quality_reasons': '["high_nox", "weather_missing"]' if month == '2025-10' else '[]',
        'fuel_basis': 'provider_reported_unit_month',
    } for month, value in values]
    write_csv(artifact / 'history.csv', history)
    def prediction(month, value):
        return {
            'plant': 'Demo', 'unit': 1, 'month': month + '-01', 'target': value,
            'supported': True, 'api_input_accepted': True, 'nox_history_count': 3,
            'pred_xgboost': value + 2, 'pred_lightgbm': value - 1, 'pred_ensemble': value - 1,
        }
    write_csv(artifact / 'validation_predictions.csv', [prediction('2025-06', 7.)])
    write_csv(artifact / 'test_predictions.csv', [prediction('2025-10', 12.)])
    write_json(artifact / 'validation_weight_search.json', [
        {'xgboost_weight': 0., 'rmse': 1.}, {'xgboost_weight': 1., 'rmse': 2.},
    ])
    relock(artifact)
    return artifact


@pytest.fixture
def registry(tmp_path):
    return CaseRegistry(tmp_path / 'cases')


def test_registration_retains_research_scope_quality_and_protocol(bundle, registry):
    manifest = registry.register(bundle, **CASE)
    assert len(manifest['files']) == 9
    assert manifest['snapshot']['observed_target'] == 12
    assert manifest['snapshot']['saved_predictions']['lightgbm'] == 11
    assert manifest['snapshot']['quality_status'] == 'review_required'
    assert manifest['snapshot']['quality_reasons'] == ['high_nox', 'weather_missing']
    assert manifest['model']['trained_through'] == '2024-12'
    assert manifest['model']['created_at'].startswith('2026-10-01')
    assert manifest['target_verification_status'] == 'pending_provider_confirmation'
    assert not manifest['source_manifest']['original_manifest_file_verified']
    assert not manifest['model']['historical_deployment_verified']
    assert not manifest['model']['independent_final_evaluation_verified']
    assert manifest['analysis_mode'] == 'evaluation_replay'
    assert len(manifest['case_version_id']) == 64
    assert not Path(manifest['artifact_path']).is_absolute()
    assert registry.verify(KEY) == manifest


def test_repeat_registration_preserves_timestamp_file_and_version(bundle, registry):
    original = registry.register(bundle, **CASE)
    path = registry.path_for(KEY)
    before = path.read_bytes(), path.stat().st_mtime_ns
    repeated = registry.register(bundle, **CASE)
    assert repeated == original
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_versions_depend_on_content_not_machine_path(bundle, registry, tmp_path):
    original = registry.register(bundle, **CASE)
    moved = tmp_path / 'copied'
    shutil.copytree(bundle, moved)
    other = CaseRegistry(tmp_path / 'other_cases').register(moved, **CASE)
    for field in ['case_version_id', 'bundle_version_id']:
        assert original[field] == other[field]
    assert original['model']['model_version_id'] == other['model']['model_version_id']


@pytest.mark.parametrize('name', FILES)
def test_changed_registered_source_is_blocked_without_rebaselining(bundle, registry, name):
    registry.register(bundle, **CASE)
    original_manifest = registry.path_for(KEY).read_bytes()
    source = bundle / name
    source.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(RegistrationError, match='checksum mismatch'):
        registry.verify(KEY)
    with pytest.raises(RegistrationError, match='checksum mismatch'):
        registry.register(bundle, **CASE)
    assert registry.path_for(KEY).read_bytes() == original_manifest


@pytest.mark.parametrize('change', ['observation', 'prediction', 'provider_status', 'version', 'model_period'])
def test_manifest_cannot_relabel_or_change_the_registered_facts(bundle, registry, change):
    manifest = registry.register(bundle, **CASE)
    if change == 'observation':
        manifest['snapshot']['observed_target'] += 1
    elif change == 'prediction':
        manifest['snapshot']['saved_predictions']['lightgbm'] += 1
    elif change == 'provider_status':
        manifest['target_verification_status'] = 'verified'
    elif change == 'version':
        manifest['case_version_id'] = '0' * 64
    else:
        manifest['model']['trained_through'] = '2025-06'
    write_json(registry.path_for(KEY), manifest)
    with pytest.raises(RegistrationError, match='differs from its sources'):
        registry.verify(KEY)


@pytest.mark.parametrize('file', ['history.csv', 'test_predictions.csv', 'validation_predictions.csv'])
def test_duplicate_observation_or_prediction_key_is_rejected(bundle, registry, file):
    path = bundle / file
    rows = csv_rows(path)
    write_csv(path, rows + [rows[-1]])
    with pytest.raises(RegistrationError, match='duplicate plant/unit/month'):
        registry.register(bundle, **CASE)
    assert not registry.path_for(KEY).exists()


def test_prediction_must_match_observation_and_model_cohort(bundle, registry):
    path = bundle / 'test_predictions.csv'
    rows = csv_rows(path)
    rows[0]['target'] = '13'
    write_csv(path, rows)
    with pytest.raises(RegistrationError, match='target differs'):
        registry.register(bundle, **CASE)
    rows[0]['target'] = '12'
    rows[0]['supported'] = 'False'
    write_csv(path, rows)
    with pytest.raises(RegistrationError, match='supported status'):
        registry.register(bundle, **CASE)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-1', 'True'])
def test_nonfinite_negative_or_boolean_predictions_are_rejected(bundle, registry, value):
    path = bundle / 'test_predictions.csv'
    rows = csv_rows(path)
    rows[0]['pred_lightgbm'] = value
    write_csv(path, rows)
    with pytest.raises(RegistrationError):
        registry.register(bundle, **CASE)


def test_rejected_input_or_incorrect_prior_calendar_count_is_not_registered(bundle, registry):
    path = bundle / 'test_predictions.csv'
    rows = csv_rows(path)
    rows[0]['api_input_accepted'] = 'False'
    write_csv(path, rows)
    with pytest.raises(RegistrationError, match='input was rejected'):
        registry.register(bundle, **CASE)
    rows[0]['api_input_accepted'] = 'True'
    rows[0]['nox_history_count'] = '2'
    write_csv(path, rows)
    with pytest.raises(RegistrationError, match='history count differs'):
        registry.register(bundle, **CASE)


@pytest.mark.parametrize('field,value', [
    ('unit', True), ('unit', 1.0), ('unit', 0),
    ('month', '2025-13'), ('month', '2025-10-01'),
    ('month', '2025-06'), ('plant', 'Other'), ('case_key', '../escape'),
])
def test_bad_or_out_of_scope_case_request_is_rejected(bundle, registry, field, value):
    request = {**CASE, field: value}
    with pytest.raises(RegistrationError):
        registry.register(bundle, **request)


@pytest.mark.parametrize('change', ['overlap', 'cutoff', 'timezone', 'weight', 'unlocked', 'schema'])
def test_invalid_temporal_or_selection_metadata_is_rejected(bundle, registry, change):
    path = bundle / 'metadata.json'
    meta = json.loads(path.read_text())
    if change == 'overlap':
        meta['split']['validation'][0] = '2024-12'
    elif change == 'cutoff':
        meta['trained_through'] = '2025-06'
    elif change == 'timezone':
        meta['trained_at'] = '2026-10-01T12:00:00'
    elif change == 'weight':
        meta['xgboost_weight'] = .5
        meta['lightgbm_weight'] = .5
    elif change == 'unlocked':
        meta['weights_locked_before_test'] = False
    else:
        meta['schema_version'] = 1.0
    write_json(path, meta)
    relock(bundle)
    with pytest.raises(RegistrationError):
        registry.register(bundle, **CASE)


@pytest.mark.parametrize('change', ['fit_scope', 'xgb_features', 'lgb_features'])
def test_model_and_preprocessor_schema_are_checked_beyond_hashes(bundle, registry, change):
    if change == 'fit_scope':
        path = bundle / 'preprocessor.json'
        pre = json.loads(path.read_text())
        pre['fit_scope'] = 'all months'
        write_json(path, pre)
    elif change == 'xgb_features':
        write_json(bundle / 'xgb_model.json', {'learner': {'feature_names': ['feature_b', 'feature_a']}})
    else:
        (bundle / 'lgb_model.txt').write_text('feature_names=feature_b feature_a\n')
    relock(bundle)
    with pytest.raises(RegistrationError):
        registry.register(bundle, **CASE)


def test_selection_lock_cannot_reference_another_file(bundle, registry):
    path = bundle / 'selection_lock.json'
    lock = json.loads(path.read_text())
    lock['sha256']['../outside.txt'] = 'a' * 64
    write_json(path, lock)
    with pytest.raises(RegistrationError, match='selection lock'):
        registry.register(bundle, **CASE)


def test_bundle_file_symlink_cannot_redirect_reads(bundle, registry, tmp_path):
    path = bundle / 'history.csv'
    outside = tmp_path / 'outside.csv'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(RegistrationError, match='linked outside'):
        registry.register(bundle, **CASE)


def test_registry_does_not_write_into_frozen_artifact(bundle):
    with pytest.raises(RegistrationError, match='outside the frozen'):
        CaseRegistry(bundle / 'cases').register(bundle, **CASE)
    assert not (bundle / 'cases').exists()


def test_existing_key_cannot_be_reused_for_another_case(bundle, registry):
    registry.register(bundle, **CASE)
    with pytest.raises(RegistrationError, match='already registered'):
        registry.register(bundle, **{**CASE, 'month': '2025-09'})


def test_cli_reports_only_registration_metadata_and_blocks_changes(bundle, registry, capsys):
    assert main(['register', '--artifact-dir', str(bundle), '--registry-dir', str(registry.directory),
                 '--case-key', KEY, '--plant', 'Demo', '--unit', '1', '--month', '2025-10']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['status'] == 'ok' and output['registered_file_count'] == 9
    assert 'observed_target' not in output and 'saved_predictions' not in output
    assert main(['verify', '--registry-dir', str(registry.directory), '--case-key', KEY]) == 0
    capsys.readouterr()
    (bundle / 'history.csv').write_text('changed')
    assert main(['verify', '--registry-dir', str(registry.directory), '--case-key', KEY]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


def test_real_yeongdong_registration_preserves_all_source_files(tmp_path):
    artifact = ROOT / 'reports/temporal_validation_20261001/models/unit_actual/C'
    if not all((artifact / name).is_file() for name in FILES):
        pytest.skip('Frozen local evaluation assets are not published')
    before = {name: sha(artifact / name) for name in FILES}
    registry = CaseRegistry(tmp_path / 'cases')
    result = registry.register(artifact, case_key='yeongdong_u1_2025_10_c_v1',
                               plant='영동', unit=1, month='2025-10')
    assert registry.verify(result['case_key']) == result
    assert result['snapshot']['quality_status'] == 'review_required'
    assert result['snapshot']['observed_previous_months'] == 3
    assert result['protocol']['train'] == ['2023-01', '2024-12']
    assert result['model']['weights'] == {'xgboost': 0., 'lightgbm': 1.}
    assert before == {name: sha(artifact / name) for name in FILES}

