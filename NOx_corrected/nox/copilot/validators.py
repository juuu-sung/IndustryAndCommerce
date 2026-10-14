"""Check report structure, recalculate metrics and verify source correspondence."""
import math
from .archive_repository import Observation
from .calculations import calculate_metrics
from .contracts import REPORT_VERSION, ReportError, invalid_report, report_id, selected_methods
from .registry import (
    CaseSpec, RegistrationError, _canonical, _digest, _hash, _number, _object,
    _protocol, _text_list, _timestamp,
)
from ..schema import TARGETS
from .report_sources import validate_source_quality

CALCULATION_FIELDS = {
    'schema_version', 'calculation_version', 'implementation_sha256', 'case_key', 'case',
    'analysis_mode', 'unit', 'provenance', 'target', 'target_verification_status', 'model',
    'protocol', 'fuel_basis', 'metrics', 'limitations', 'calculation_id', 'generated_at',
}
OBSERVATION_FIELDS = {
    'month', 'value', 'unit', 'quality_status', 'quality_reasons',
    'observation_row_sha256', 'missing_reason',
}


def _observation(value):
    value = _object(value, 'observation')
    if set(value) != OBSERVATION_FIELDS or value['unit'] != 'ppm':
        raise RegistrationError('Invalid report observation schema or unit')
    _text_list(value['quality_reasons'], 'quality_reasons')
    if value['observation_row_sha256'] is not None:
        _hash(value['observation_row_sha256'], 'observation row hash')
    if value['missing_reason'] not in (None, 'missing_calendar_month', 'missing_target'):
        raise RegistrationError('Invalid observation missing reason')
    if value['value'] is not None and value['missing_reason'] is not None:
        raise RegistrationError('Available observations cannot have missing reasons')
    return Observation(value['month'], value['value'], value['quality_status'],
                       tuple(value['quality_reasons']), value['observation_row_sha256'], value['missing_reason'])


def validate_calculation(calculation):
    try:
        c = _object(calculation, 'calculation')
        if set(c) != CALCULATION_FIELDS or type(c['schema_version']) is not int or c['schema_version'] != 1:
            raise RegistrationError('Unsupported calculation schema')
        if c['calculation_version'] != 'calendar_metrics_v1' or c['analysis_mode'] != 'evaluation_replay' or c['unit'] != 'ppm':
            raise RegistrationError('Unsupported calculation mode')
        case = CaseSpec(c['case_key'], **c['case'])
        if c['target']['name'] != TARGETS['concentration']['name'] or c['target']['unit'] != 'ppm':
            raise RegistrationError('Wrong research target')
        _timestamp(c['generated_at'], 'calculation generated_at')
        _hash(c['implementation_sha256'], 'calculation implementation')
        _text_list(c['limitations'], 'limitations')
        provenance = c['provenance']
        for name in ('case_version_id', 'bundle_version_id', 'model_version_id',
                     'target_definition_sha256', 'history_file_sha256', 'prediction_file_sha256',
                     'observation_row_sha256', 'prediction_row_sha256'):
            _hash(provenance[name], name)
        if provenance['target_definition_sha256'] != _digest(c['target']):
            raise RegistrationError('Target definition mismatch')
        if c['target_verification_status'] != 'pending_provider_confirmation':
            raise RegistrationError('Provider validity is not established by this report')
        split = _protocol(c['protocol'])
        model = c['model']
        if (model['model_version_id'] != provenance['model_version_id']
                or model['trained_through'] != split['train'][1]
                or not split['test'][0] <= case.month <= split['test'][1]
                or model['historical_deployment_verified'] is not False
                or model['independent_final_evaluation_verified'] is not False):
            raise RegistrationError('Wrong replay model or period')
        _timestamp(model['created_at'], 'model created_at')
        weights = model['weights']
        if set(weights) != {'xgboost', 'lightgbm'}:
            raise RegistrationError('Wrong model weight schema')
        xw = _number(weights['xgboost'], 'xgboost weight')
        lw = _number(weights['lightgbm'], 'lightgbm weight')
        if not (0 <= xw <= 1 and 0 <= lw <= 1 and math.isclose(xw + lw, 1, abs_tol=1e-12)):
            raise RegistrationError('Wrong model weights')
        if model['preprocessor_fit_scope'] != f"{split['train'][0]} through {split['train'][1]} only":
            raise RegistrationError('Wrong preprocessor period')
        source = provenance['source_manifest']
        _hash(source['declared_sha256'], 'source manifest hash')
        if source['original_manifest_file_verified'] is not False:
            raise RegistrationError('Original source manifest verification is not established')
        metrics = c['metrics']
        current = _observation(metrics['current_observation'])
        prior = tuple(_observation(value) for value in metrics['previous_3months']['observations'])
        if current.month != case.month or current.row_sha256 != provenance['observation_row_sha256']:
            raise RegistrationError('Current observation mismatch')
        errors = _object(metrics['prediction_errors'], 'errors')
        expected = calculate_metrics(current, prior, {name: value['prediction_ppm'] for name, value in errors.items()})
        if _canonical(metrics) != _canonical(expected):
            raise RegistrationError('Calculated metrics differ from report inputs')
        content = {key: value for key, value in c.items() if key not in ('calculation_id', 'generated_at')}
        if c['calculation_id'] != _digest(content):
            raise RegistrationError('Calculation content hash mismatch')
    except (RegistrationError, KeyError, TypeError, OverflowError, ValueError) as error:
        if isinstance(error, ReportError):
            raise
        raise invalid_report() from error
    return calculation


def validate_report(report, expected_calculation=None):
    from .report_builder import report_content, report_implementation_hash, semantic_report
    try:
        r = _object(report, 'report')
        if type(r['schema_version']) is not int or r['schema_version'] != 1 or r['report_version'] != REPORT_VERSION:
            raise RegistrationError('Unsupported report version')
        if type(r['revision']) is not int or r['revision'] != 1 or r['status'] != 'draft':
            raise RegistrationError('Only unapproved first-revision drafts are implemented')
        report_id(r['report_id'])
        _timestamp(r['report_generated_at'], 'report_generated_at')
        validate_calculation(r['calculation'])
        validate_source_quality(r['source_quality'], r['calculation'])
        if r['implementation_sha256'] != report_implementation_hash():
            raise RegistrationError('Report implementation changed; regenerate a draft')
        methods = selected_methods(r['selected_methods'], r['calculation']['metrics']['prediction_errors'])
        expected = report_content(r['calculation'], methods, r['implementation_sha256'], r['source_quality'])
        if set(r) != set(expected) | {'report_id', 'report_generated_at'}:
            raise RegistrationError('Report contract fields differ')
        if any(_canonical(r[key]) != _canonical(value) for key, value in expected.items()):
            raise RegistrationError('Report view differs from its validated calculation')
        if r['report_id'] != 'r_' + _digest(semantic_report(r)):
            raise RegistrationError('Report content identifier mismatch')
        if expected_calculation is not None:
            validate_calculation(expected_calculation)
            archived = {k: v for k, v in r['calculation'].items() if k != 'generated_at'}
            source = {k: v for k, v in expected_calculation.items() if k != 'generated_at'}
            if _canonical(archived) != _canonical(source):
                raise RegistrationError('Report no longer matches its registered sources')
    except (RegistrationError, KeyError, TypeError, OverflowError, ValueError) as error:
        if isinstance(error, ReportError) and error.status == 409:
            raise
        raise invalid_report() from error
    return report
