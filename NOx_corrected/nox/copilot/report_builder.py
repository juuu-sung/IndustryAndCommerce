"""Build a deterministic research report from source-bound calculation data."""
import copy
from datetime import datetime, timezone
from pathlib import Path

from ..quality import RULE_NOTES
from .contracts import REPORT_VERSION, selected_methods
from .registry import _digest, _sha
from .report_sources import empty_source_quality


def report_implementation_hash():
    directory = Path(__file__).resolve().parent
    files = {name: _sha((directory / name).read_bytes()) for name in (
        'contracts.py', 'report_builder.py', 'validators.py', 'renderers.py',
        'repositories.py', 'routes.py',
        'report_sources.py',
    )}
    files['quality.py'] = _sha((directory.parent / 'quality.py').read_bytes())
    return _digest(files)


def semantic_report(report):
    value = copy.deepcopy(report)
    for field in ('report_id', 'report_generated_at'):
        value.pop(field, None)
    value['calculation'].pop('generated_at', None)
    return value


def report_content(calculation, methods, implementation_sha256, source_quality):
    """Pure report view; unknown source details remain explicitly unavailable."""
    metrics = calculation['metrics']
    observations = [*metrics['previous_3months']['observations'], metrics['current_observation']]
    quality_flags = [{'month': o['month'], 'quality_status': o['quality_status'],
                      'reasons': o['quality_reasons']} for o in observations]
    checks = [{
        'code': 'provider_confirmation', 'source': 'report_scope', 'months': [calculation['case']['month']],
        'message': '관측값의 측정 유효 여부와 일평균·월 집계 기준을 제공기관에 확인하세요.',
    }, {
        'code': 'operating_evidence', 'source': 'report_scope', 'months': [calculation['case']['month']],
        'message': '운전·저감설비 기록과 대조하세요. 이 입력과 예측 오차만으로 급등 원인을 확정할 수 없습니다.',
    }]
    reasons = sorted({reason for o in observations for reason in o['quality_reasons']})
    for reason in reasons:
        checks.append({
            'code': reason, 'source': 'stored_quality_reason',
            'months': [o['month'] for o in observations if reason in o['quality_reasons']],
            'message': RULE_NOTES.get(reason, '원본에 기록된 품질 검토 사유를 제공기관 및 자료와 대조하세요.'),
        })
    if metrics['previous_3months']['observed_count'] < 3:
        checks.append({
            'code': 'incomplete_calendar_history', 'source': 'calculation_rule',
            'months': [o['month'] for o in observations[:-1] if o['value'] is None],
            'message': '직전 3개월 중 누락된 달을 확인하세요. 평균은 이용 가능한 관측만 사용했습니다.',
        })
    coverage = source_quality['by_month'][-1]['emission']
    coverage_unverified = ['daily_observation_coverage'] if coverage['status'] != 'available' else []
    context = {
        'case_version_id': calculation['provenance']['case_version_id'],
        'calculation_id': calculation['calculation_id'], 'methods': list(methods),
        'report_version': REPORT_VERSION, 'report_implementation_sha256': implementation_sha256,
        'analysis_mode': 'evaluation_replay', 'document_bundle_version': None,
        'retrieval_configuration': None,
        'source_quality_sha256': _digest(source_quality),
    }
    return {
        'schema_version': 1, 'report_version': REPORT_VERSION,
        'implementation_sha256': implementation_sha256,
        'revision': 1, 'status': 'draft', 'analysis_mode': 'evaluation_replay',
        'case_key': calculation['case_key'], 'case': calculation['case'],
        'target_definition': calculation['target'], 'target_unit': 'ppm',
        'target_verification_status': calculation['target_verification_status'],
        'selected_methods': list(methods), 'calculation': copy.deepcopy(calculation),
        'source_quality': copy.deepcopy(source_quality),
        'observed': metrics['current_observation'],
        'history': metrics['previous_3months']['observations'],
        'predictions': {method: metrics['prediction_errors'][method] for method in methods},
        'comparisons': {name: metrics[name] for name in ('previous_month', 'previous_3months')},
        'input_quality': {
            'monthly_history_observed_count': metrics['previous_3months']['observed_count'],
            'monthly_history_expected_count': 3, 'observation_flags': quality_flags,
            'daily_observation_coverage': coverage, 'fuel_basis': calculation['fuel_basis'],
            'weather_temperature_coverage': source_quality['by_month'][-1]['weather_temperature'],
            'feature_imputation': {'status': 'not_available_in_frozen_report_contract', 'features': None},
            'outside_training_range': {'status': 'not_available_in_frozen_report_contract', 'features': None},
        },
        'provenance': calculation['provenance'], 'model': calculation['model'],
        'protocol': calculation['protocol'], 'context_hash': _digest(context),
        'checks': checks, 'limitations': list(dict.fromkeys(calculation['limitations'] + [
            '문서 RAG와 LLM 설명은 이 초안에 사용하지 않았습니다.',
            '관측일 충족률은 일자료 존재 비율이며 유효 측정시간·운전시간 충족률이 아닙니다.',
            '특징별 결측 보완·범위 경고는 현재 고정 보고서 계약에서 확인하지 않았습니다.',
        ])),
        'citations': [], 'rag_status': 'not_enabled', 'llm_status': 'not_used',
        'validation': {
            'passed': ['registered_source_integrity', 'case_target_alignment', 'calendar_metrics',
                       'saved_prediction_errors', 'unit_and_protocol', 'content_version'],
            'unverified': ['provider_measurement_validity', 'official_monthly_aggregation',
                           'historical_input_availability',
                           'independent_final_model_evaluation', *coverage_unverified],
        },
    }


class ReportBuilder:
    def build(self, calculation, methods=None, source_quality=None):
        from .validators import validate_calculation, validate_report
        from .report_sources import validate_source_quality
        validate_calculation(calculation)
        source_quality = empty_source_quality(calculation) if source_quality is None else source_quality
        validate_source_quality(source_quality, calculation)
        available = calculation['metrics']['prediction_errors']
        methods = selected_methods(list(available) if methods is None else methods, available)
        report = report_content(calculation, methods, report_implementation_hash(), source_quality)
        report['report_id'] = 'r_' + _digest(semantic_report(report))
        report['report_generated_at'] = datetime.now(timezone.utc).isoformat()
        validate_report(report, expected_calculation=calculation)
        return copy.deepcopy(report)
