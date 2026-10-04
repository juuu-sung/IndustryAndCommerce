"""Evidence-backed daily replay and immutable research target versions.

This checks reproducibility of a documented definition, not legal compliance or
the authenticity of a provider's document. Unknown flags are never guessed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import KEYS, ROOT
from .temporal_validation import sha256, safe_metrics
from .train import write_json

STACK_KEYS = ['plant', 'stack_id', 'date']
INTERVAL_COLUMNS = ['plant', 'unit', 'stack_id', 'interval_start_kst',
    'interval_end_kst', 'nox_ppm_provider', 'nox_ppm_raw',
    'nox_ppm_oxygen_corrected', 'oxygen_pct', 'gas_flow_rate', 'flow_unit',
    'valid_seconds', 'provider_status_code', 'operating_state',
    'scr_operating_state', 'source_available_at']
REFERENCE_COLUMNS = [*STACK_KEYS, 'nox_ppm', 'valid_seconds', 'case_type']
CASES = [('영동', 1, '2025-10'), ('삼천포', 4, '2025-03'),
         ('영동', 1, '2024-10'), ('삼천포', 3, '2025-03')]
RULE_SCHEMA = 'nox_daily_replay_rule_v1'
VERSION_SCHEMA = 'nox_target_version_v1'


def fresh_output(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an evidence/target version: ' + str(output))
    output.mkdir(parents=True)
    return output


def template_rule():
    return {'schema': RULE_SCHEMA, 'rule_id': 'pending_provider_confirmation',
        'confirmation_status': 'pending', 'provider_owner': '', 'confirmed_at': '',
        'documents': [], 'applicable_from': '', 'applicable_to': '',
        'nox_column': 'nox_ppm_provider', 'nox_basis': '',
        'oxygen_handling': 'provider_value_used_without_recorrection',
        'daily_aggregation': 'valid_seconds_weighted',
        'minimum_daily_valid_seconds': None, 'reference_tolerance_ppm': None,
        'reference_tolerance_seconds': None, 'status_codebook': {}, 'stack_map': [],
        'required_case_types': ['normal', 'spike', 'stopped', 'startup', 'calibration'],
        'monthly_aggregation': 'arithmetic_daily_mean_complete_stacks',
        'minimum_monthly_day_coverage': None,
        'monthly_rule_scope': 'research_definition_not_official_mass',
        'note': 'Daily definitions/status/mapping must come from actual provider '
                'evidence. Monthly coverage is a separately declared research '
                'criterion, never a guessed legal threshold or chosen to improve '
                'test error. Unsupported averaging/correction rules require '
                'a separate implementation.'}


def evidence_files(rule, base):
    if rule.get('schema') != RULE_SCHEMA or rule.get('confirmation_status') != 'confirmed':
        raise ValueError('Provider definition remains pending; target/training blocked')
    if not rule.get('provider_owner') or not rule.get('rule_id') or not rule.get('nox_basis'):
        raise ValueError('Provider owner, rule identity and concentration basis required')
    stamp = datetime.fromisoformat(rule.get('confirmed_at', ''))
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Confirmation timestamp must have timezone and not be in the future')
    kinds = {'daily_definition', 'status_codebook', 'stack_mapping'}
    documents = rule.get('documents', [])
    if not kinds <= {d.get('kind') for d in documents}:
        raise ValueError('Evidence documents for definition, codebook and mapping required')
    receipts = []
    for doc in documents:
        path = Path(doc['file'])
        path = path if path.is_absolute() else Path(base) / path
        if not path.is_file() or sha256(path) != doc['sha256']:
            raise ValueError('Missing or changed evidence document: ' + str(path))
        receipts.append({'file': str(path.resolve()), 'sha256': sha256(path), 'kind': doc['kind']})
    return receipts


def validate_rule(rule, base):
    documents = evidence_files(rule, base)
    if rule.get('daily_aggregation') != 'valid_seconds_weighted' or rule.get('oxygen_handling') != 'provider_value_used_without_recorrection':
        raise ValueError('Unsupported daily/correction definition; do not approximate it')
    if rule.get('nox_column') not in {'nox_ppm_provider', 'nox_ppm_raw', 'nox_ppm_oxygen_corrected'}:
        raise ValueError('Unknown concentration column')
    if rule.get('monthly_aggregation') != 'arithmetic_daily_mean_complete_stacks' or rule.get('monthly_rule_scope') != 'research_definition_not_official_mass':
        raise ValueError('Unsupported research monthly target definition')
    for name, low, high in [('minimum_daily_valid_seconds', 0, 86400),
                          ('reference_tolerance_ppm', 0, 1),
                          ('reference_tolerance_seconds', 0, 1),
                          ('minimum_monthly_day_coverage', 0, 1)]:
        value = rule.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError('Explicit finite rule parameter required: ' + name)
    start, end = pd.Timestamp(rule['applicable_from']), pd.Timestamp(rule['applicable_to'])
    if pd.isna(start) or pd.isna(end) or start > end or start != start.normalize() or end != end.normalize() or start.tzinfo or end.tzinfo:
        raise ValueError('Applicable date bounds must be ordered local dates')
    mapping = pd.DataFrame(rule.get('stack_map', []))
    if set(['plant', 'unit', 'stack_id']) - set(mapping) or mapping.empty:
        raise ValueError('Confirmed stack-to-unit mapping required')
    if mapping.duplicated(['plant', 'stack_id']).any() or mapping[['plant', 'stack_id']].isna().any().any():
        raise ValueError('Duplicate or missing stack mapping')
    units = pd.to_numeric(mapping.unit, errors='raise')
    if not np.isfinite(units).all() or (units <= 0).any() or (units != np.floor(units)).any():
        raise ValueError('Mapped units must be positive integers')
    codebook = rule.get('status_codebook', {})
    if not codebook or any(v not in ('valid', 'invalid', 'unknown') for v in codebook.values()):
        raise ValueError('Confirmed provider codebook required')
    required = rule.get('required_case_types', [])
    if not {'normal', 'spike'} <= set(required) or not set(required) <= {'normal', 'spike', 'stopped', 'startup', 'calibration'}:
        raise ValueError('Normal and spike reference cases are required')
    return documents


def replay_intervals(frame, rule):
    """Use a complete local-day grid; exclude invalid slots, retain raw records."""
    required = ['plant', 'unit', 'stack_id', 'interval_start_kst',
                'interval_end_kst', rule['nox_column'], 'valid_seconds', 'provider_status_code']
    if set(required) - set(frame) or frame.empty:
        raise ValueError('Missing interval columns or empty provider intervals')
    out = frame.copy()
    if out[required[:5] + ['provider_status_code']].isna().any().any():
        raise ValueError('Missing interval identifiers/status')
    # Reject naive timestamps rather than silently treating them as UTC.
    for col in ['interval_start_kst', 'interval_end_kst']:
        parsed = [pd.Timestamp(v) for v in out[col]]
        if any(v.tzinfo is None for v in parsed):
            raise ValueError('Interval timestamps require an explicit timezone')
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert('Asia/Seoul')
    out['date'] = out.interval_start_kst.dt.tz_localize(None).dt.normalize()
    end_local = out.interval_end_kst.dt.tz_localize(None)
    out['duration_seconds'] = (out.interval_end_kst - out.interval_start_kst).dt.total_seconds()
    if (out.duration_seconds <= 0).any() or (end_local > out.date + pd.Timedelta(days=1)).any():
        raise ValueError('Nonpositive interval or cross-midnight interval; request split provider records')
    if not out.date.between(rule['applicable_from'], rule['applicable_to']).all():
        raise ValueError('Intervals outside the documented applicable dates')
    out['unit'] = pd.to_numeric(out.unit, errors='raise')
    mapping = pd.DataFrame(rule['stack_map']).rename(columns={'unit': 'mapped_unit'})
    mapping['mapped_unit'] = pd.to_numeric(mapping.mapped_unit, errors='raise').astype(int)
    out = out.merge(mapping, on=['plant', 'stack_id'], how='left', validate='many_to_one')
    if out.mapped_unit.isna().any() or not out.unit.eq(out.mapped_unit).all():
        raise ValueError('Interval unit/stack differs from confirmed mapping')
    out['unit'] = out.unit.astype(int)
    out['measurement_status'] = out.provider_status_code.astype(str).map(rule['status_codebook'])
    if out.measurement_status.isna().any() or out.measurement_status.eq('unknown').any():
        raise ValueError('Unmapped/unknown provider status; verification remains blocked')
    out['valid_seconds'] = pd.to_numeric(out.valid_seconds, errors='raise')
    if not np.isfinite(out.valid_seconds).all() or (out.valid_seconds < 0).any() or (out.valid_seconds > out.duration_seconds).any():
        raise ValueError('Invalid valid_seconds or valid time exceeds interval duration')
    if (out.measurement_status.eq('invalid') & out.valid_seconds.ne(0)).any():
        raise ValueError('Invalid interval must have zero eligible measured seconds')
    out['selected_nox_ppm'] = pd.to_numeric(out[rule['nox_column']], errors='coerce')
    use = out.measurement_status.eq('valid') & out.valid_seconds.gt(0)
    if not np.isfinite(out.loc[use, 'selected_nox_ppm']).all() or (out.loc[use, 'selected_nox_ppm'] < 0).any():
        raise ValueError('Nonfinite/negative eligible concentration')
    ordered = out.sort_values(['plant', 'stack_id', 'interval_start_kst'])
    prev_end = ordered.groupby(['plant', 'stack_id']).interval_end_kst.shift()
    if (ordered.interval_start_kst < prev_end).any():
        raise ValueError('Duplicate/overlapping intervals')
    grid = ordered.groupby(STACK_KEYS).duration_seconds.sum()
    if not np.allclose(grid, 86400, rtol=0, atol=1e-6):
        raise ValueError('Incomplete daily interval grid; request explicit invalid/missing slots')
    out['weighted_nox'] = np.where(use, out.selected_nox_ppm * out.valid_seconds, 0.)
    daily = out.groupby([*STACK_KEYS, 'unit'], as_index=False).agg(
        weighted_nox=('weighted_nox', 'sum'), valid_seconds=('valid_seconds', 'sum'),
        interval_rows=('duration_seconds', 'size'))
    eligible = daily.valid_seconds.gt(0) & daily.valid_seconds.ge(rule['minimum_daily_valid_seconds'])
    daily['nox_ppm'] = daily.weighted_nox.div(daily.valid_seconds.where(eligible))
    daily['daily_eligible'] = eligible
    return out.drop(columns=['mapped_unit']), daily.drop(columns=['weighted_nox'])


def compare_reference(daily, reference, rule):
    if set(REFERENCE_COLUMNS) - set(reference) or reference.empty:
        raise ValueError('Provider daily reference and annotated case types required')
    ref = reference.copy()
    ref['date'] = pd.to_datetime(ref.date, errors='raise')
    if ref.date.isna().any() or not ref.date.eq(ref.date.dt.normalize()).all() or ref.date.dt.tz is not None:
        raise ValueError('Reference dates must be local dates without time')
    if not set(rule['required_case_types']) <= set(ref.case_type):
        raise ValueError('Reference cases do not cover the documented required states')
    ref['nox_ppm'] = pd.to_numeric(ref.nox_ppm, errors='raise')
    ref['valid_seconds'] = pd.to_numeric(ref.valid_seconds, errors='raise')
    if not np.isfinite(ref.valid_seconds).all() or (ref.valid_seconds < 0).any() or (ref.valid_seconds > 86400).any():
        raise ValueError('Invalid provider reference duration')
    if np.isinf(ref.nox_ppm).any() or (ref.nox_ppm.dropna() < 0).any():
        raise ValueError('Invalid provider reference concentration')
    joined = daily.merge(ref, on=STACK_KEYS, how='outer', suffixes=('_replayed', '_provider'),
                         validate='one_to_one', indicator=True)
    joined['absolute_error_ppm'] = abs(joined.nox_ppm_replayed - joined.nox_ppm_provider)
    joined['duration_error_seconds'] = abs(joined.valid_seconds_replayed - joined.valid_seconds_provider)
    concentration_match = joined.absolute_error_ppm.le(rule['reference_tolerance_ppm']) | (
        joined.nox_ppm_replayed.isna() & joined.nox_ppm_provider.isna())
    joined['matches'] = joined._merge.eq('both') & concentration_match & joined.duration_error_seconds.le(rule['reference_tolerance_seconds'])
    return joined


def monthly_targets(daily, rule):
    """Research daily arithmetic mean; incomplete A/B days never substitute one stack."""
    records = []
    mapping = pd.DataFrame(rule['stack_map'])
    mapping['unit'] = pd.to_numeric(mapping.unit, errors='raise').astype(int)
    for (plant, unit), group in daily.groupby(['plant', 'unit']):
        expected = set(mapping.loc[(mapping.plant == plant) & (mapping.unit == unit), 'stack_id'])
        unit_days = []
        for date, day in group.groupby('date'):
            complete = set(day.stack_id) == expected and day.daily_eligible.all()
            unit_days.append({'date': date, 'target': day.nox_ppm.mean() if complete else np.nan})
        days = pd.DataFrame(unit_days)
        days['month'] = days.date.dt.to_period('M').dt.to_timestamp()
        for month, values in days.groupby('month'):
            count = int(values.target.notna().sum())
            coverage = count / month.days_in_month
            eligible = count > 0 and coverage >= rule['minimum_monthly_day_coverage']
            records.append({'plant': plant, 'unit': int(unit), 'month': month,
                'target': float(values.target.mean()) if eligible else np.nan,
                'eligible_days': count, 'calendar_days': month.days_in_month,
                'day_coverage': coverage, 'quality_status': 'verified_research' if eligible else 'unavailable',
                'target_rule_id': rule['rule_id']})
    return pd.DataFrame(records)


def verify(rule_path, intervals_path, reference_path, output):
    rule_path, intervals_path, reference_path = map(Path, [rule_path, intervals_path, reference_path])
    rule = json.loads(rule_path.read_text())
    documents = validate_rule(rule, rule_path.parent)
    intervals = pd.read_csv(intervals_path, dtype={'provider_status_code': str, 'stack_id': str})
    reference = pd.read_csv(reference_path, dtype={'stack_id': str})
    normalized, daily = replay_intervals(intervals, rule)
    comparison = compare_reference(daily, reference, rule)
    output = fresh_output(output)
    normalized.to_csv(output / 'normalized_intervals.csv', index=False)
    daily.to_csv(output / 'replayed_daily.csv', index=False)
    comparison.to_csv(output / 'daily_reference_comparison.csv', index=False)
    matched = bool(comparison.matches.all())
    metadata = {'schema': VERSION_SCHEMA, 'status': 'verified_research' if matched else 'replay_mismatch',
        'target_unit': 'ppm', 'official_mass_or_legal_certification': False,
        'rule_id': rule['rule_id'], 'verified_daily_rows': len(daily),
        'reference_mismatch_rows': int((~comparison.matches).sum()),
        'rule': rule, 'evidence_documents': documents,
        'source_files': [{'file': str(p.resolve()), 'sha256': sha256(p)}
                         for p in [rule_path, intervals_path, reference_path]], 'files': {}}
    if matched:
        monthly_targets(daily, rule).to_csv(output / 'monthly_targets.csv', index=False, date_format='%Y-%m-%d')
    metadata['files'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'target_version.json', metadata)
    return metadata


def load_verified_version(path):
    path = Path(path)
    meta = json.loads((path / 'target_version.json').read_text())
    if meta.get('schema') != VERSION_SCHEMA or meta.get('status') != 'verified_research' or meta.get('target_unit') != 'ppm':
        raise ValueError('Target version not verified; model training blocked')
    for item in [*meta.get('source_files', []), *meta.get('evidence_documents', [])]:
        if sha256(item['file']) != item['sha256']:
            raise ValueError('Target source/evidence changed')
    for name, digest in meta['files'].items():
        if sha256(path / name) != digest:
            raise ValueError('Frozen target version changed: ' + name)
    return pd.read_csv(path / 'monthly_targets.csv', parse_dates=['month']), meta


def compare_saved(output):
    """Recompute identical-row existing comparisons; no fitting or target edits."""
    output = fresh_output(output)
    early_path = ROOT / 'reports/temporal_validation_20261001/unit_actual_outer_predictions.csv'
    seed_path = ROOT / 'reports/readiness_20261002/candidates/seed_predictions.csv'
    common_path = ROOT / 'reports/retraining/common_period_predictions.csv'
    early = pd.read_csv(early_path)
    early = early.loc[early.supported].copy()
    seeds = pd.read_csv(seed_path)
    seeds = seeds.loc[seeds.candidate.eq('no_weather_log_lgb') & seeds.input_policy.eq('values_only') & seeds.split.eq('test') & seeds.supported]
    keys = ['fold', *KEYS]
    grouped = seeds.groupby(keys, as_index=False).agg(target=('target', 'first'),
        target_count=('target', 'nunique'), seed_count=('seed', 'nunique'), prediction=('prediction', 'mean'))
    aligned = early.merge(grouped, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    if len(aligned) != len(early) or len(aligned) != len(grouped) or not np.array_equal(aligned.target_old, aligned.target_new) or not grouped.seed_count.eq(5).all() or not grouped.target_count.eq(1).all():
        raise ValueError('Saved target/cohort/seed mismatch; refuse a misleading comparison')
    aligned.to_csv(output / 'aligned_development_predictions.csv', index=False)
    scores = []
    for plant, rows in [('all', aligned), *list(aligned.groupby('plant'))]:
        for method in ['pred_xgboost', 'pred_lightgbm', 'pred_ensemble', 'pred_last_month', 'prediction']:
            scores.append({'period': '2024-07 through 2025-12', 'plant': plant,
                           'method': method, **safe_metrics(rows.target_old, rows[method])})
    frame = pd.DataFrame(scores)
    frame.to_csv(output / 'scores.csv', index=False)
    common = pd.read_csv(common_path)
    common = common.loc[common.baseline_supported]
    details = {name: safe_metrics(common.target, common[col]) for name, col in
               [('initial', 'pred_baseline'), ('retrained', 'pred_candidate')]}
    report = {'research_target_validity': 'unverified', 'development_same_rows': len(aligned),
        'retraining_same_rows': len(common), 'retraining_2026': details,
        'independent_final_evaluation': False, 'new_training_performed': False,
        'service_replaced': False, 'source_files': [{'file': str(p), 'sha256': sha256(p)}
            for p in [early_path, seed_path, common_path]]}
    write_json(output / 'comparison.json', report)
    return report


def prepare(raw_dir, output):
    from .data import read, generation
    from .quality import inspect_targets
    output = fresh_output(output)
    raw_dir = Path(raw_dir)
    daily, monthly, quality = inspect_targets(read('emissions_daily.csv', raw_dir), generation(raw_dir))
    monthly.to_csv(output / 'research_monthly_v0.csv', index=False, date_format='%Y-%m-%d')
    details, summaries = [], []
    for plant, unit, month in CASES:
        subset = daily.loc[daily.plant.eq(plant) & daily.unit.eq(unit) & daily.month.eq(pd.Timestamp(month + '-01'))].copy()
        days = subset.loc[subset.included_in_target].groupby('date').nox_numeric.mean()
        subset['request_role'] = np.where(subset.nox_numeric.gt(500), 'spike_candidate', 'comparison_context_not_confirmed_normal')
        details.append(subset)
        summaries.append({'plant': plant, 'unit': unit, 'month': month,
            'observed_days': len(days), 'source_rows': len(subset),
            'monthly_mean_ppm': float(days.mean()), 'daily_median_ppm': float(days.median()),
            'top5_share_pct': float(100 * days.nlargest(5).sum() / days.sum()),
            'provider_unknown_rows': int(subset.measurement_status.eq('unknown').sum())})
    pd.concat(details).to_csv(output / 'case_daily_attachment.csv', index=False, date_format='%Y-%m-%d')
    pd.DataFrame(summaries).to_csv(output / 'case_summary.csv', index=False)
    pd.DataFrame(columns=INTERVAL_COLUMNS).to_csv(output / 'provider_intervals_TEMPLATE.csv', index=False)
    pd.DataFrame(columns=REFERENCE_COLUMNS).to_csv(output / 'provider_daily_reference_TEMPLATE.csv', index=False)
    write_json(output / 'aggregation_rule_TEMPLATE.json', template_rule())
    write_json(output / 'target_version.json', {'schema': VERSION_SCHEMA, 'status': 'pending_provider_confirmation',
        'target_unit': 'ppm', 'training_allowed': False, 'research_v0_sha256': sha256(output / 'research_monthly_v0.csv'),
        'source_files': [{'file': str((raw_dir / 'emissions_daily.csv').resolve()),
                          'sha256': sha256(raw_dir / 'emissions_daily.csv')}],
        'unverified_rows': quality['daily_status_counts'].get('unverified', 0),
        'review_rows': quality['daily_status_counts'].get('review_required', 0)})
    (output / 'NOx_provider_request.txt').write_text(
        '제목: 발전소 NOx 공개 일자료의 측정 유효성 및 집계 정의 확인 요청\n\n'
        '한국남동발전 공개자료 담당부서 및 삼천포·영동 환경/TMS 담당자께\n'
        '공개 NOx 농도(ppm)로 연구용 예측 모델을 개발하고 있습니다. 첨부 수치는 '
        '오류로 판정한 값이 아니라 측정·보정·집계 정의를 확인할 사례입니다.\n\n'
        '우선 영동 1호기 2025-10-03~11, 2024-10-14~18 및 삼천포 4A/4B '
        '2025-03-04~09, 3A/3B 2025-03-06의 원 측정간격 자료와 인접 정상 사례를 요청합니다. '
        '비교를 위해 해당 월 전체의 측정/무효/미수신 슬롯을 함께 받을 수 있으면 좋겠습니다.\n'
        '1. 공개 NOx가 산소 보정 전/후 중 무엇인지, 기준 산소농도·건식/습식 기준·보정 순서\n'
        '2. 산소·유량·온도 단위, 유량 기준 상태와 시간 평균/적산 여부\n'
        '3. 정상·기동·정지·교정·고장·미수신·무효 상태 코드표와 각 구간 판정, 유효 측정시간\n'
        '4. 일평균 산출 간격·분모·최소 유효시간·반올림·결측 및 대체자료 처리와 계산 예시\n'
        '5. 사업소·호기·보일러·배출구 A/B·TMS 식별자 대응표 및 변경 적용일\n'
        '6. 보일러 연료 투입/점화/정지, SCR 가동·우회·약품 주입·입구온도 기록\n'
        '7. 당시 적용한 규칙 버전과 공개자료의 수정·재게시 이력\n\n'
        '정상, 급등, 정지, 기동, 교정 사례별 원 구간자료와 기관 일평균을 함께 받아 '
        '집계 재현을 확인하고자 합니다. 상태 코드는 추정하지 않고 제공된 코드표로 변환하겠습니다.\n'
        '템플릿은 희망 항목 예시이며 원래 형식과 코드표로 제공하셔도 됩니다. 월 총질량 kg는 '
        '이 농도 연구와 별개이며 가능하면 기관 검증 월 질량과 정의를 별도로 요청합니다.\n'
        '첨부: case_daily_attachment.csv, case_summary.csv, provider_intervals_TEMPLATE.csv, '
        'provider_daily_reference_TEMPLATE.csv\n', encoding='utf-8')
    return {'status': 'pending_provider_confirmation', 'cases': summaries, 'training_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--raw-dir', type=Path, default=ROOT / 'data/readiness_bundle_20261002')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify')
    for name in ['rule', 'intervals', 'reference', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('compare-saved')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.raw_dir, args.output)
        elif args.command == 'verify':
            result = verify(args.rule, args.intervals, args.reference, args.output)
        else:
            result = compare_saved(args.output)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, 'Blocked: ' + str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get('status') == 'replay_mismatch':
        parser.exit(2, 'Daily replay does not match provider reference; training blocked\n')


if __name__ == '__main__':
    main()
