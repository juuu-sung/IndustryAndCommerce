import copy
from concurrent.futures import ThreadPoolExecutor
import json

from bs4 import BeautifulSoup
import pytest

from nox.app import create_app
from nox.copilot.calculations import calculate_case
from nox.copilot.contracts import ReportError, ReportRequest
from nox.copilot.registry import CaseRegistry, _digest
from nox.copilot.report_builder import ReportBuilder, semantic_report
from nox.copilot.renderers import render_html
from nox.copilot.repositories import ReportRepository, ReportService
from nox.copilot.reports import main
from nox.copilot.validators import validate_calculation, validate_report
from nox.copilot.report_sources import load_source_quality
from nox.schema import ROOT, TARGETS
from test_copilot_registry import bundle, registry, CASE, KEY, csv_rows, write_csv


class StubPredictor:
    metadata = {'target': TARGETS['concentration']}

    def history_status(self):
        return {'status': 'test_stub'}

    def predict(self, data):
        return {'prediction': {'value': 7., 'unit': 'ppm'}, 'warnings': [], 'model': {}}


@pytest.fixture
def service(bundle, registry, tmp_path):
    registry.register(bundle, **CASE)
    return ReportService(registry, ReportRepository(tmp_path / 'reports'))


@pytest.fixture
def client(service):
    return create_app(StubPredictor(), copilot_service=service).test_client()


def rehash_calculation(value):
    value['calculation_id'] = _digest({k: v for k, v in value.items() if k not in ('generated_at', 'calculation_id')})


def rehash_report(value):
    value['report_id'] = 'r_' + _digest(semantic_report(value))


def test_report_preserves_numbers_quality_versions_and_unknowns(service):
    report, created = service.create({'case_key': KEY})
    assert created and report['status'] == 'draft' and report['revision'] == 1
    assert report['observed']['value'] == 12
    assert report['observed']['quality_status'] == 'review_required'
    assert report['comparisons']['previous_month']['delta_ppm'] == 2
    assert report['predictions']['xgboost']['signed_error_ppm'] == 2
    assert report['predictions']['lightgbm']['signed_error_ppm'] == -1
    assert report['input_quality']['daily_observation_coverage']['observed_days'] is None
    assert report['input_quality']['feature_imputation']['features'] is None
    assert report['rag_status'] == 'not_enabled' and report['citations'] == []
    assert report['llm_status'] == 'not_used'
    assert report['target_verification_status'] == 'pending_provider_confirmation'
    assert report['provenance']['case_version_id'] == service.registry.verify(KEY)['case_version_id']
    assert validate_report(report) == report
    assert service.get(report['report_id']) == report


def test_repeat_request_retains_id_timestamp_and_file(service):
    first, created = service.create({'case_key': KEY})
    path = service.repository.path_for(first['report_id'])
    before = path.read_bytes(), path.stat().st_mtime_ns
    second, created = service.create({'case_key': KEY})
    assert not created and first == second
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_method_order_is_normalized_and_different_method_set_is_a_new_report(service):
    one, _ = service.create({'case_key': KEY, 'methods': ['lightgbm', 'xgboost']})
    two, created = service.create({'case_key': KEY, 'methods': ['xgboost', 'lightgbm']})
    assert one == two and not created
    subset, created = service.create({'case_key': KEY, 'methods': ['lightgbm']})
    assert created and subset['report_id'] != one['report_id']
    assert list(subset['predictions']) == ['lightgbm']


def test_parallel_saves_create_one_complete_document(service):
    calculation = service.calculation(KEY)
    report = ReportBuilder().build(calculation)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.repository.save, [report, copy.deepcopy(report)]))
    assert sorted(created for _, created in results) == [False, True]
    assert service.repository.read(report['report_id']) == report


def test_builder_owns_its_input_snapshot(service):
    calculation = service.calculation(KEY)
    report = ReportBuilder().build(calculation)
    calculation['metrics']['current_observation']['value'] = 999
    assert report['observed']['value'] == 12
    validate_report(report)


@pytest.mark.parametrize('mutation', ['unit', 'value', 'error', 'quality', 'model', 'extra', 'schema', 'approved', 'citations', 'context'])
def test_report_tampering_is_rejected_even_with_recomputed_content_id(service, mutation):
    report = ReportBuilder().build(service.calculation(KEY))
    if mutation == 'unit':
        report['target_unit'] = 'kg'
    elif mutation == 'value':
        report['observed']['value'] = 999
    elif mutation == 'error':
        report['predictions']['lightgbm']['absolute_error_ppm'] = 0
    elif mutation == 'quality':
        report['observed']['quality_status'] = 'valid'
    elif mutation == 'model':
        report['model']['trained_through'] = '2025-06'
    elif mutation == 'extra':
        report['unrequested'] = True
    elif mutation == 'schema':
        report['schema_version'] = True
    elif mutation == 'approved':
        report['status'] = 'approved'
    elif mutation == 'citations':
        report['citations'] = [{'document_id': 'invented'}]
    else:
        report['context_hash'] = 'a' * 64
    rehash_report(report)
    with pytest.raises(ReportError) as caught:
        validate_report(report)
    assert caught.value.status == 409


@pytest.mark.parametrize('mutation', ['delta', 'mean', 'count', 'bool_error', 'future_month', 'weight', 'provider', 'manifest'])
def test_self_hashed_calculation_still_needs_semantic_consistency(service, mutation):
    c = service.calculation(KEY)
    if mutation == 'delta':
        c['metrics']['previous_month']['delta_ppm'] = 99
    elif mutation == 'mean':
        c['metrics']['previous_3months']['mean_ppm'] = 99
    elif mutation == 'count':
        c['metrics']['previous_3months']['observed_count'] = 2
    elif mutation == 'bool_error':
        c['metrics']['prediction_errors']['lightgbm']['absolute_error_ppm'] = True
    elif mutation == 'future_month':
        c['metrics']['previous_3months']['observations'][0]['month'] = '2025-11'
    elif mutation == 'weight':
        c['model']['weights']['lightgbm'] = 2
    elif mutation == 'provider':
        c['target_verification_status'] = 'verified'
    else:
        c['provenance']['source_manifest']['original_manifest_file_verified'] = True
    rehash_calculation(c)
    with pytest.raises(ReportError):
        validate_calculation(c)


def test_saved_source_change_and_report_file_change_are_not_silently_replaced(service, bundle):
    report, _ = service.create({'case_key': KEY})
    path = service.repository.path_for(report['report_id'])
    original = path.read_bytes()
    (bundle / 'history.csv').write_bytes((bundle / 'history.csv').read_bytes() + b' ')
    with pytest.raises(ReportError) as caught:
        service.get(report['report_id'])
    assert caught.value.code == 'source_conflict'
    assert path.read_bytes() == original
    # Restoring sources does not excuse report corruption.
    (bundle / 'history.csv').write_bytes((bundle / 'history.csv').read_bytes()[:-1])
    changed = json.loads(original)
    changed['observed']['value'] = 999
    path.write_text(json.dumps(changed))
    with pytest.raises(ReportError) as caught:
        service.create({'case_key': KEY})
    assert caught.value.code == 'report_invalid'
    assert json.loads(path.read_text())['observed']['value'] == 999


def test_source_change_during_build_blocks_output(service, bundle, monkeypatch):
    real = ReportBuilder.build
    def changed(self, *args, **kwargs):
        report = real(self, *args, **kwargs)
        (bundle / 'history.csv').write_bytes(b'changed-during-build')
        return report
    monkeypatch.setattr(ReportBuilder, 'build', changed)
    with pytest.raises(ReportError) as caught:
        service.create({'case_key': KEY})
    assert caught.value.code == 'source_conflict'
    assert not service.repository.directory.exists()


def test_storage_cannot_be_in_registered_source(service, bundle):
    unsafe = ReportService(service.registry, ReportRepository(bundle / 'new_report'))
    with pytest.raises(ReportError) as caught:
        unsafe.create({'case_key': KEY})
    assert caught.value.code == 'reports_unavailable'
    assert not (bundle / 'new_report').exists()


def test_symlink_report_does_not_read_untrusted_file(service, tmp_path):
    report, _ = service.create({'case_key': KEY})
    path = service.repository.path_for(report['report_id'])
    other = tmp_path / 'other.json'
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(ReportError):
        service.repository.read(report['report_id'])


def test_html_table_numbers_match_json_and_scope_is_explicit(service):
    report, _ = service.create({'case_key': KEY})
    html = render_html(report)
    soup = BeautifulSoup(html, 'html.parser')
    assert soup.html['lang'] == 'ko'
    assert soup.title.text == 'Demo 1호기 · 2025-10 NOx 분석'
    assert not soup.find_all('script') and not soup.find_all('link')
    tables = soup.find_all('table')
    rows = tables[1].tbody.find_all('tr')
    assert len(rows) == len(report['predictions'])
    for row, value in zip(rows, report['predictions'].values()):
        values = [cell.text for cell in row.find_all('td')]
        assert values == [f"{value['prediction_ppm']:,.2f}", f"{value['signed_error_ppm']:+,.2f}", f"{value['absolute_error_ppm']:,.2f}"]
    visible = soup.get_text()
    assert '해당 월 당시 서비스가 생성한 미래 예측 기록이 아닙니다' in visible
    assert '검토 필요' in visible and '보고서' in visible and '문서 근거: 아직 연결하지 않음' in visible


def test_html_escapes_source_identifiers_and_quality_reasons(bundle, registry):
    marker = '<img src=x onerror=alert(1)>'
    for name in ('history.csv', 'validation_predictions.csv', 'test_predictions.csv'):
        rows = csv_rows(bundle / name)
        for row in rows:
            row['plant'] = marker
            if name == 'history.csv':
                row['quality_reasons'] = json.dumps(['<script>alert(2)</script>'])
        write_csv(bundle / name, rows)
    from test_copilot_registry import relock
    meta_path = bundle / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    meta['known_units'] = [{'plant': marker, 'unit': 1}]
    meta_path.write_text(json.dumps(meta))
    relock(bundle)
    registry.register(bundle, **{**CASE, 'plant': marker})
    report = ReportBuilder().build(calculate_case(KEY, registry))
    html = render_html(report)
    soup = BeautifulSoup(html, 'html.parser')
    assert not soup.find_all('img') and not soup.find_all('script')
    assert marker in soup.title.text and '<script>alert(2)</script>' in soup.get_text()
    assert '&lt;img' in html and '&lt;script' in html


@pytest.mark.parametrize('kind', ['missing_previous', 'zero_previous'])
def test_partial_history_and_zero_are_reported_without_invented_rates(bundle, registry, kind):
    rows = csv_rows(bundle / 'history.csv')
    for row in rows:
        if row['month'] == '2025-09-01':
            row['target'] = '' if kind == 'missing_previous' else '0'
            row['quality_status'] = 'unavailable' if kind == 'missing_previous' else 'review_required'
    write_csv(bundle / 'history.csv', rows)
    if kind == 'missing_previous':
        predictions = csv_rows(bundle / 'test_predictions.csv')
        predictions[0]['nox_history_count'] = '2'
        write_csv(bundle / 'test_predictions.csv', predictions)
    registry.register(bundle, **CASE)
    report = ReportBuilder().build(calculate_case(KEY, registry))
    html = BeautifulSoup(render_html(report), 'html.parser').get_text()
    assert report['comparisons']['previous_month']['change_percent'] is None
    assert '변화율 확인 불가 사유' in html
    if kind == 'missing_previous':
        assert '관측 없음' in html and '2/3개 달' in html
    else:
        assert 'previous_month_zero' in html and '0.00 ppm' in html


def test_api_create_read_html_and_cases(client, service):
    cases = client.get('/api/copilot/cases')
    assert cases.status_code == 200 and cases.json['cases'][0]['status'] == 'available'
    assert 'manifest' not in cases.json['cases'][0] and 'artifact_path' not in cases.get_data(as_text=True)
    response = client.post('/api/reports/drafts', json={'case_key': KEY})
    assert response.status_code == 201 and response.json['created']
    report = response.json['report']
    assert response.headers['Location'] == response.json['links']['json']
    assert client.get(response.json['links']['json']).json == report
    html = client.get(response.json['links']['html'])
    assert html.status_code == 200 and html.mimetype == 'text/html'
    assert 'default-src' in html.headers['Content-Security-Policy']
    assert html.headers['Cache-Control'] == 'no-store'
    assert html.headers['X-Content-Type-Options'] == 'nosniff'
    assert '12.00 ppm' in html.get_data(as_text=True)
    assert str(service.repository.directory) not in html.get_data(as_text=True)
    repeated = client.post('/api/reports/drafts', json={'case_key': KEY})
    assert repeated.status_code == 200 and repeated.json['report'] == report


@pytest.mark.parametrize('data', [
    None, [], {}, {'case_key': True}, {'case_key': '../escape'},
    {'case_key': KEY, 'output_dir': '/tmp/client_controlled'},
    {'case_key': KEY, 'analysis_mode': 'future_forecast'},
    {'case_key': KEY, 'methods': []}, {'case_key': KEY, 'methods': None},
    {'case_key': KEY, 'methods': ['invented']}, {'case_key': KEY, 'methods': ['lightgbm', 'lightgbm']},
    {'case_key': KEY, 'methods': [True]}, {'case_key': KEY, 'methods': 'lightgbm'},
    {'case_key': KEY, 'expected_case_version_id': True},
])
def test_bad_report_requests_are_rejected(client, data):
    response = client.post('/api/reports/drafts', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 400


def test_api_content_type_json_body_size_and_missing_ids(client):
    assert client.post('/api/reports/drafts', data='{}', content_type='text/plain').status_code == 415
    assert client.post('/api/reports/drafts', data='{bad', content_type='application/json').status_code == 400
    assert client.post('/api/reports/drafts', data='x' * 40000, content_type='application/json').status_code == 413
    assert client.post('/api/reports/drafts', json={'case_key': 'not_registered'}).status_code == 404
    assert client.get('/api/reports/not_an_id').status_code == 400
    assert client.get('/api/reports/r_' + 'a' * 64).status_code == 404


@pytest.mark.parametrize('body', [
    '{"case_key":"first","case_key":"second"}',
    '{"case_key":NaN}', '{"case_key":Infinity}',
])
def test_duplicate_keys_and_nonfinite_json_are_rejected(client, body):
    response = client.post('/api/reports/drafts', data=body, content_type='application/json')
    assert response.status_code == 400 and response.json['error']['code'] == 'invalid_json'


def test_report_with_missing_registry_source_is_unavailable(client, service):
    response = client.post('/api/reports/drafts', json={'case_key': KEY})
    service.registry.path_for(KEY).unlink()
    response = client.get(response.json['links']['json'])
    assert response.status_code == 503 and response.json['error']['code'] == 'source_unavailable'


def test_version_conflict_and_source_failure_are_sanitized(client, bundle):
    response = client.post('/api/reports/drafts', json={'case_key': KEY, 'expected_case_version_id': 'a' * 64})
    assert response.status_code == 409 and response.json['error']['code'] == 'case_version_conflict'
    (bundle / 'history.csv').write_bytes(b'changed')
    response = client.post('/api/reports/drafts', json={'case_key': KEY})
    assert response.status_code == 409 and response.json['error']['code'] == 'source_conflict'
    assert str(bundle) not in response.get_data(as_text=True)
    cases = client.get('/api/copilot/cases').json['cases']
    assert cases[0]['status'] == 'blocked'
    (bundle / 'history.csv').unlink()
    response = client.post('/api/reports/drafts', json={'case_key': KEY})
    assert response.status_code == 503 and response.json['error']['code'] == 'source_unavailable'


def test_empty_registry_and_optional_blueprint_do_not_disable_existing_endpoints(tmp_path):
    service = ReportService(CaseRegistry(tmp_path / 'cases'), ReportRepository(tmp_path / 'reports'))
    client = create_app(StubPredictor(), copilot_service=service).test_client()
    assert client.get('/api/copilot/cases').json['cases'] == []
    assert client.get('/api/health').status_code == 200
    assert client.get('/api/history/status').status_code == 200
    disabled = create_app(StubPredictor(), enable_copilot=False).test_client()
    assert disabled.get('/api/copilot/cases').status_code == 404
    assert disabled.get('/api/health').status_code == 200


def test_report_generation_never_calls_prediction_or_llm(service):
    class NoPredict(StubPredictor):
        def predict(self, data):
            raise AssertionError('New inference should not run')
    class NoLLM:
        def generate(self, prompt):
            raise AssertionError('LLM should not run')
    client = create_app(NoPredict(), enable_gemini=True, llm_client=NoLLM(), copilot_service=service).test_client()
    assert client.post('/api/reports/drafts', json={'case_key': KEY}).status_code == 201


def test_html_export_matches_api_and_detects_tampering(service):
    report, _ = service.create({'case_key': KEY})
    path = service.export_html(report['report_id'])
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert path.read_text() == render_html(report)
    assert service.export_html(report['report_id']) == path
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    path.write_bytes(b'<html>tampered</html>')
    with pytest.raises(ReportError) as caught:
        service.export_html(report['report_id'])
    assert caught.value.code == 'report_invalid'
    assert path.read_bytes() == b'<html>tampered</html>'


def test_cli_exports_json_and_html_and_does_not_print_values(service, capsys):
    args = ['--case-key', KEY, '--registry-dir', str(service.registry.directory),
            '--report-dir', str(service.repository.directory)]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['created'] and first['status'] == 'draft'
    assert 'observed' not in first and 'predictions' not in first
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert not second['created'] and first['report_id'] == second['report_id']
    assert main(['--case-key', 'missing', '--registry-dir', str(service.registry.directory)]) == 1
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'case_not_found'


def test_real_report_http_client_and_json_html_alignment(tmp_path):
    artifact = ROOT / 'reports/temporal_validation_20261001/models/unit_actual/C'
    if not (artifact / 'history.csv').is_file():
        pytest.skip('Private evaluation assets are not in the public repository')
    registry = CaseRegistry(tmp_path / 'cases')
    key = 'yeongdong_u1_2025_10_fold_c_v1'
    registry.register(artifact, case_key=key, plant='영동', unit=1, month='2025-10')
    service = ReportService(registry, ReportRepository(tmp_path / 'reports'))
    client = create_app(StubPredictor(), copilot_service=service).test_client()
    response = client.post('/api/reports/drafts', json={'case_key': key})
    assert response.status_code == 201
    report = response.json['report']
    assert report['observed']['value'] == pytest.approx(371.61387096774195)
    assert report['input_quality']['daily_observation_coverage']['observed_days'] == 31
    assert report['input_quality']['daily_observation_coverage']['coverage_ratio'] == 1
    assert report['predictions']['lightgbm']['absolute_error_ppm'] == pytest.approx(287.36634152279663)
    html = client.get(response.json['links']['html']).get_data(as_text=True)
    assert '371.61 ppm' in html and '287.37' in html
    assert '31/31일 (100.0%)' in html
    assert 'review_required' not in BeautifulSoup(html, 'html.parser').get_text()
    assert client.get(response.json['links']['json']).json == report


def test_available_daily_and_weather_coverage_use_verified_rows(bundle, registry, tmp_path):
    rows = csv_rows(bundle / 'history.csv')
    from calendar import monthrange
    for row in rows:
        month = row['month'][:7]
        total = monthrange(int(month[:4]), int(month[5:]))[1]
        row.update(emission_days=str(total), emission_coverage='1.0',
                   temperature_days=str(total - 1), weather_coverage=str((total - 1) / total))
    write_csv(bundle / 'history.csv', rows)
    registry.register(bundle, **CASE)
    service = ReportService(registry, ReportRepository(tmp_path / 'reports'))
    report, _ = service.create({'case_key': KEY})
    coverage = report['input_quality']['daily_observation_coverage']
    assert coverage == {'status':'available','observed_days':31,'calendar_days':31,'coverage_ratio':1.,'reason':None}
    assert report['input_quality']['weather_temperature_coverage']['observed_days'] == 30
    assert service.get(report['report_id']) == report
    empty = ReportBuilder().build(service.calculation(KEY))
    assert empty['context_hash'] != report['context_hash']


@pytest.mark.parametrize('kind', ['days_out_of_bounds', 'ratio_mismatch', 'boolean_days'])
def test_bad_source_coverage_is_blocked_before_report_save(bundle, registry, tmp_path, kind):
    rows = csv_rows(bundle / 'history.csv')
    for row in rows:
        row.update(emission_days='31', emission_coverage='1.0')
        if row['month'] != '2025-10-01':
            row.update(emission_days='', emission_coverage='')
    current = rows[-1]
    if kind == 'days_out_of_bounds':
        current['emission_days'] = '32'
    elif kind == 'ratio_mismatch':
        current['emission_coverage'] = '0.5'
    else:
        current['emission_days'] = 'True'
    write_csv(bundle / 'history.csv', rows)
    registry.register(bundle, **CASE)
    service = ReportService(registry, ReportRepository(tmp_path / 'reports'))
    with pytest.raises(ReportError) as caught:
        service.create({'case_key': KEY})
    assert caught.value.code == 'source_conflict'
    assert not service.repository.directory.exists()
