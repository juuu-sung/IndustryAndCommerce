"""Deterministic calendar summaries and errors; no LLM, training or new inference."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from ..schema import ROOT
from .archive_repository import ArchiveRepository, Observation, previous_months
from .registry import (
    CaseRegistry, METHODS, RegistrationError, _digest, _json, _number, _sha,
    _timestamp, _write_new,
)

CALCULATION_VERSION = 'calendar_metrics_v1'
DEFAULT_OUTPUT = ROOT / 'data/runtime/copilot/calculations'


def _finite(value, label):
    return _number(value, label)


def calculate_metrics(current, prior, predictions):
    """Calculate using explicit observations in the exact preceding three months.

    The archived adapter supplies one plant/unit only. Review-required and zero
    values remain observed; this calculation does not certify their validity.
    """
    if not isinstance(current, Observation):
        raise RegistrationError('Current input must be an Observation')
    if (not isinstance(prior, (tuple, list)) or len(prior) != 3
            or any(not isinstance(item, Observation) for item in prior)
            or tuple(item.month for item in prior) != previous_months(current.month)):
        raise RegistrationError('History must use exactly the prior three calendar months in order')
    if (not isinstance(predictions, dict) or not predictions
            or any(method not in METHODS for method in predictions)):
        raise RegistrationError('Predictions must name supported saved methods')
    checked = {}
    for method, value in predictions.items():
        if value is not None and _number(value, 'prediction.' + method) < 0:
            raise RegistrationError('Concentration predictions cannot be negative')
        checked[method] = value
    previous = prior[-1]
    missing = []
    if current.value is None:
        missing.append('current_observation_unavailable')
    if previous.value is None:
        missing.append('previous_month_unavailable')
    delta = None if missing else _finite(current.value - previous.value, 'month delta')
    rate_reason = missing + (['previous_month_zero'] if previous.value == 0 else [])
    rate = None if rate_reason else _finite(delta / previous.value * 100, 'month percent change')
    available = [item.value for item in prior if item.value is not None]
    average = _finite(math.fsum(value / len(available) for value in available), 'prior mean') if available else None
    average_delta = (None if current.value is None or average is None
                     else _finite(current.value - average, 'prior mean delta'))
    errors = {}
    for method, value in checked.items():
        reasons = ([] if current.value is not None else ['current_observation_unavailable'])
        if value is None:
            reasons.append('prediction_unavailable')
        error = None if reasons else _finite(value - current.value, 'signed error')
        errors[method] = {
            'prediction_ppm': value, 'signed_error_ppm': error,
            'absolute_error_ppm': None if error is None else abs(error),
            'status': 'unavailable' if reasons else 'available', 'reasons': reasons,
        }
    return {
        'current_observation': current.as_dict(),
        'previous_month': {
            'observation': previous.as_dict(), 'delta_ppm': delta,
            'change_percent': rate, 'delta_status': 'unavailable' if missing else 'available',
            'delta_reasons': missing,
            'change_percent_status': 'unavailable' if rate_reason else 'available',
            'change_percent_reasons': rate_reason,
        },
        'previous_3months': {
            'observations': [item.as_dict() for item in prior],
            'observed_count': len(available), 'expected_count': 3,
            'coverage': 'complete' if len(available) == 3 else 'partial' if available else 'unavailable',
            'mean_ppm': average, 'current_minus_mean_ppm': average_delta,
            'review_required_months': [item.month for item in prior if item.quality_status == 'review_required'],
        },
        'prediction_errors': errors,
    }


def implementation_hash():
    """Bind results to the reader, integrity checks and calculation source bytes."""
    directory = Path(__file__).resolve().parent
    return _digest({name: _sha((directory / name).read_bytes()) for name in
                    ('registry.py', 'archive_repository.py', 'calculations.py')})


def calculate_case(case_key, registry=None):
    manifest, current, prior = ArchiveRepository(registry).load(case_key)
    metrics = calculate_metrics(current, prior, manifest['snapshot']['saved_predictions'])
    if metrics['previous_3months']['observed_count'] != manifest['snapshot']['observed_previous_months']:
        raise RegistrationError('Calculated history coverage differs from registered prediction inputs')
    payload = {
        'schema_version': 1, 'calculation_version': CALCULATION_VERSION,
        'implementation_sha256': implementation_hash(),
        'case_key': manifest['case_key'], 'case': manifest['case'],
        'analysis_mode': manifest['analysis_mode'], 'unit': manifest['target']['unit'],
        'provenance': {
            'case_version_id': manifest['case_version_id'],
            'bundle_version_id': manifest['bundle_version_id'],
            'model_version_id': manifest['model']['model_version_id'],
            'target_definition_sha256': manifest['target_definition_sha256'],
            'history_file_sha256': manifest['files']['history.csv']['sha256'],
            'prediction_file_sha256': manifest['files']['test_predictions.csv']['sha256'],
            'observation_row_sha256': manifest['snapshot']['observation_row_sha256'],
            'prediction_row_sha256': manifest['snapshot']['prediction_row_sha256'],
            'source_manifest': manifest['source_manifest'],
        },
        'target': manifest['target'], 'target_verification_status': manifest['target_verification_status'],
        'model': manifest['model'], 'protocol': manifest['protocol'],
        'fuel_basis': manifest['snapshot']['fuel_basis'], 'metrics': metrics,
        'limitations': list(dict.fromkeys(manifest['limitations'] + [
            '급등·0값·검토 필요 관측을 임의로 삭제하거나 보완하지 않은 계산입니다.',
            '오차는 저장 예측에서 관측을 뺀 값이며, 한 사례의 절대 오차는 RMSE가 아닙니다.',
            '직전 3개월 평균은 이용 가능한 관측의 평균이며 관측 충족 수를 함께 확인해야 합니다.',
        ])),
    }
    payload['calculation_id'] = _digest(payload)
    return {**payload, 'generated_at': datetime.now(timezone.utc).isoformat()}


def _existing_result(path, result):
    existing = _json(path.read_bytes(), 'saved calculation')
    if not isinstance(existing, dict) or set(existing) != set(result):
        raise RegistrationError('Saved calculation schema differs')
    _timestamp(existing['generated_at'], 'generated_at')
    if any(existing[key] != value for key, value in result.items() if key != 'generated_at'):
        raise RegistrationError('Saved calculation content differs; refusing to overwrite')
    return existing


def save_calculation(case_key, registry=None, output_dir=DEFAULT_OUTPUT):
    registry = registry if registry is not None else CaseRegistry()
    output_dir = Path(output_dir).resolve()
    result = calculate_case(case_key, registry)
    # Never write derived output into a registered source bundle or its registry.
    manifest = registry.verify(case_key)
    if manifest['case_version_id'] != result['provenance']['case_version_id']:
        raise RegistrationError('Registered case changed while the calculation was being saved')
    source = (registry.path_for(case_key).parent / manifest['artifact_path']).resolve()
    if output_dir.is_relative_to(source) or output_dir.is_relative_to(registry.directory):
        raise RegistrationError('Calculation output must be outside the frozen source and case registry')
    path = output_dir / (case_key + '_' + result['calculation_id'] + '.json')
    if path.exists():
        return path, _existing_result(path, result)
    try:
        _write_new(path, result)
    except FileExistsError:
        return path, _existing_result(path, result)
    return path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute source-bound NOx calendar summaries and saved errors')
    parser.add_argument('--case-key', required=True)
    parser.add_argument('--registry-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        registry = CaseRegistry(args.registry_dir) if args.registry_dir is not None else CaseRegistry()
        path, result = save_calculation(args.case_key, registry, args.output_dir)
        # Do not print actual row values into generic run logs.
        print(json.dumps({
            'status': 'ok', 'output': str(path), 'case_key': result['case_key'],
            'calculation_id': result['calculation_id'],
            'case_version_id': result['provenance']['case_version_id'],
            'observed_previous_months': result['metrics']['previous_3months']['observed_count'],
            'quality_status': result['metrics']['current_observation']['quality_status'],
        }, ensure_ascii=False, allow_nan=False))
    except (RegistrationError, OSError) as error:
        print(json.dumps({'status': 'blocked', 'error': str(error)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
