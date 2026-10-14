"""Preserve available observation coverage from verified frozen history rows."""
from calendar import monthrange
import math

from .registry import RegistrationError, _csv_integer, _csv_number, _csv_rows, _digest, _number


def coverage(month, row=None, kind='emission'):
    days_in_month = monthrange(int(month[:4]), int(month[5:]))[1]
    days_field = 'emission_days' if kind == 'emission' else 'temperature_days'
    ratio_field = 'emission_coverage' if kind == 'emission' else 'weather_coverage'
    days = None if row is None or not row.get(days_field) else _csv_integer(row[days_field], days_field, 0)
    ratio = None if row is None or not row.get(ratio_field) else _csv_number(row[ratio_field], ratio_field)
    if days is not None and days > days_in_month or ratio is not None and not 0 <= ratio <= 1:
        raise RegistrationError('Observation coverage is outside calendar bounds')
    if days is not None and ratio is not None and not math.isclose(days / days_in_month, ratio, abs_tol=1e-10):
        raise RegistrationError('Observation days and coverage ratio disagree')
    return {'status': 'available' if days is not None and ratio is not None else 'unavailable',
            'observed_days': days, 'calendar_days': days_in_month, 'coverage_ratio': ratio,
            'reason': None if days is not None and ratio is not None else 'not_loaded_or_missing_source_fields'}


def empty_source_quality(calculation):
    observations = [*calculation['metrics']['previous_3months']['observations'],
                    calculation['metrics']['current_observation']]
    return {'history_file_sha256': calculation['provenance']['history_file_sha256'], 'by_month': [
        {'month': o['month'], 'observation_row_sha256': o['observation_row_sha256'],
         'emission': coverage(o['month']), 'weather_temperature': coverage(o['month'], kind='weather')}
        for o in observations]}


def load_source_quality(calculation, registry):
    manifest, blobs = registry.read_verified(calculation['case_key'])
    if manifest['case_version_id'] != calculation['provenance']['case_version_id']:
        raise RegistrationError('Case version changed while report coverage was being read')
    rows = _csv_rows(blobs['history.csv'], 'history.csv', {'plant', 'unit', 'month', 'target'})
    result = empty_source_quality(calculation)
    case = calculation['case']
    for item in result['by_month']:
        row = rows.get((case['plant'], case['unit'], item['month']))
        if (None if row is None else _digest(row)) != item['observation_row_sha256']:
            raise RegistrationError('Report quality row differs from the calculated observation')
        item['emission'] = coverage(item['month'], row)
        item['weather_temperature'] = coverage(item['month'], row, 'weather')
    return result


def validate_source_quality(value, calculation):
    expected = empty_source_quality(calculation)
    if not isinstance(value, dict) or set(value) != set(expected) or value['history_file_sha256'] != expected['history_file_sha256']:
        raise RegistrationError('Report quality source differs')
    if not isinstance(value['by_month'], list) or len(value['by_month']) != len(expected['by_month']):
        raise RegistrationError('Report quality months differ')
    for item, reference in zip(value['by_month'], expected['by_month']):
        if (not isinstance(item, dict) or set(item) != set(reference)
                or item['month'] != reference['month']
                or item['observation_row_sha256'] != reference['observation_row_sha256']):
            raise RegistrationError('Report quality observation correspondence differs')
        for key in ('emission', 'weather_temperature'):
            c = item[key]
            if not isinstance(c, dict) or set(c) != set(reference[key]):
                raise RegistrationError('Report coverage schema differs')
            days, ratio = c['observed_days'], c['coverage_ratio']
            if (type(c['calendar_days']) is not int or c['calendar_days'] != reference[key]['calendar_days']
                    or days is not None and (type(days) is not int or not 0 <= days <= c['calendar_days'])
                    or ratio is not None and not 0 <= _number(ratio, 'coverage ratio') <= 1):
                raise RegistrationError('Report coverage values differ')
            loaded = days is not None and ratio is not None
            if (c['status'] != ('available' if loaded else 'unavailable')
                    or c['reason'] != (None if loaded else 'not_loaded_or_missing_source_fields')
                    or loaded and not math.isclose(days / c['calendar_days'], ratio, abs_tol=1e-10)):
                raise RegistrationError('Report coverage arithmetic or availability differs')
    return value
