"""Auditable target screening. Heuristics request review; they do not delete data."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import KEYS, ROOT


@dataclass(frozen=True)
class QualityRules:
    # Review thresholds, NOT emissions standards or proof of sensor failure.
    high_nox: float = 500.0
    high_oxygen: float = 20.0
    low_stack_temperature: float = 60.0
    minimum_monthly_coverage: float = 0.8

    def __post_init__(self):
        if not (np.isfinite(self.high_nox) and self.high_nox > 0
                and 0 < self.high_oxygen <= 100
                and np.isfinite(self.low_stack_temperature)
                and 0 < self.minimum_monthly_coverage <= 1):
            raise ValueError('Invalid quality review thresholds')


RULE_NOTES = {
    'invalid_plant': '발전소 식별자 누락. 집계 불가.',
    'invalid_date': '날짜 형식 오류. 집계 불가.',
    'duplicate_stack_day': '같은 사업소·배출구·날짜의 중복. 원자료 확인 필요.',
    'unmapped_unit': '배출구와 호기 대응 근거 없음. 개별 호기 집계 불가.',
    'invalid_nox': 'NOX가 결측·비수치·무한대. 집계 불가.',
    'negative_nox': '음수 NOX. 집계 불가.',
    'provider_invalid': '명시적 measurement_status=invalid. 해당 측정은 집계 제외.',
    'zero_nox': '0의 정지·결측·실측 의미 확인. 자동 제외하지 않음.',
    'high_nox': '연구용 검토 임계값 초과. 법정 기준이 아님.',
    'high_oxygen': '높은 산소 기록. 산소 보정·운전 상태 확인.',
    'invalid_auxiliary': '산소·유량·온도 기록의 결측/비수치 또는 범위 의심.',
    'positive_nox_zero_flow': '양수 NOX와 0 유량이 동시 기록됨. 자동 제외하지 않음.',
    'high_nox_low_temperature': '큰 NOX와 낮은 배기가스 온도가 동시 기록됨.',
    'reported_stopped': '명시적 stopped/maintenance 상태. 농도를 임의의 0으로 바꾸지 않음.',
    'low_emission_coverage': '관측일 충족률 부족. 유효 가동시간 충족률과 다름.',
    'no_emission_rows': '해당 호기·월의 대응 가능한 NOx 자료가 없음.',
    'generation_missing': '해당 호기·월의 발전실적 자료가 없음.',
    'positive_nox_zero_generation': '월 발전량 0과 양수 월 NOX가 함께 존재함.',
    'zero_efficiency_positive_generation': '양수 발전량과 0 열효율. 미보고 표기 확인.',
    'invalid_efficiency': '열효율이 0~100% 범위를 벗어남. 원값 보존, 모델 특징은 결측 처리.',
    'missing_efficiency': '열효율 관측 누락. 모델 특징은 학습 중앙값으로 보완.',
    'weather_missing': '월 기상자료 없음.',
    'low_weather_coverage': '월 기온 관측일 충족률 부족.',
}


def _flags_text(flags):
    return json.dumps(sorted(flags), ensure_ascii=False)


def inspect_daily(frame, rules=None):
    """Return all original rows, normalized values and independent review reasons.

    Optional statuses are normalized import fields, not a guessed vendor QC-code
    mapping. Invalid vendor codes must be translated using a confirmed codebook.
    """
    rules = rules or QualityRules()
    required = {'사업소', '호기', '일자', 'NOX'}
    if required - set(frame):
        raise ValueError(f'Missing emissions columns: {sorted(required - set(frame))}')
    out = frame.copy().reset_index(drop=True)
    out['source_row'] = np.arange(len(out)) + 2
    out['plant'] = out['사업소'].fillna('').astype(str).str.strip().replace({'분당화력': '분당'})
    stack = out['호기'].astype(str).str.strip()
    out['unit'] = pd.to_numeric(stack.str.extract(r'^(\d+)(?:[AB])?호기$')[0], errors='coerce')
    dates = out['일자'].astype(str).str.strip()
    out['date'] = pd.to_datetime(dates.where(dates.str.fullmatch(r'\d{8}')), format='%Y%m%d', errors='coerce')
    out['month'] = out['date'].dt.to_period('M').dt.to_timestamp()
    out['nox_numeric'] = pd.to_numeric(out['NOX'], errors='coerce')
    for source, col in [('산소', 'oxygen_numeric'), ('유량', 'flow_numeric'), ('온도', 'stack_temperature_numeric')]:
        out[col] = pd.to_numeric(out[source], errors='coerce') if source in out else np.nan
    for column, choices in [('measurement_status', {'unknown', 'valid', 'invalid'}),
                            ('operating_status', {'unknown', 'running', 'stopped', 'startup', 'maintenance'})]:
        if column not in out:
            out[column] = 'unknown'
        out[column] = out[column].fillna('unknown').astype(str).str.strip()
        if not out[column].isin(choices).all():
            raise ValueError(f'{column}: use normalized statuses {sorted(choices)}')
    flags = [set() for _ in range(len(out))]

    def flag(mask, code):
        for i in np.flatnonzero(np.asarray(mask.fillna(False) if isinstance(mask, pd.Series) else mask)):
            flags[i].add(code)

    finite_nox = np.isfinite(out['nox_numeric']) & ~out['NOX'].map(lambda v: isinstance(v, (bool, np.bool_)))
    duplicate = out.assign(normalized_stack=stack).duplicated(['plant', 'normalized_stack', 'date'], keep=False)
    flag(out['date'].isna(), 'invalid_date')
    flag(out['plant'].eq(''), 'invalid_plant')
    flag(out['date'].notna() & duplicate, 'duplicate_stack_day')
    flag(out['unit'].isna() | out['unit'].le(0), 'unmapped_unit')
    flag(~finite_nox, 'invalid_nox')
    flag(out['nox_numeric'].lt(0), 'negative_nox')
    flag(out['measurement_status'].eq('invalid'), 'provider_invalid')
    flag(out['nox_numeric'].eq(0), 'zero_nox')
    flag(out['nox_numeric'].gt(rules.high_nox), 'high_nox')
    flag(out['oxygen_numeric'].gt(rules.high_oxygen), 'high_oxygen')
    auxiliary = (~np.isfinite(out['oxygen_numeric']) | ~np.isfinite(out['flow_numeric'])
                 | ~np.isfinite(out['stack_temperature_numeric'])
                 | ~out['oxygen_numeric'].between(0, 100) | out['flow_numeric'].lt(0))
    flag(auxiliary, 'invalid_auxiliary')
    flag(out['nox_numeric'].gt(0) & out['flow_numeric'].eq(0), 'positive_nox_zero_flow')
    flag(out['nox_numeric'].gt(rules.high_nox)
         & out['stack_temperature_numeric'].lt(rules.low_stack_temperature), 'high_nox_low_temperature')
    flag(out['operating_status'].isin(['stopped', 'maintenance']), 'reported_stopped')
    invalid_codes = {'invalid_plant', 'invalid_date', 'duplicate_stack_day', 'unmapped_unit',
                     'invalid_nox', 'negative_nox', 'provider_invalid'}
    out['included_in_target'] = [not bool(f & invalid_codes) for f in flags]
    out['quality_status'] = ['excluded' if f & invalid_codes else 'review_required' if f
                             else 'unverified' if status == 'unknown' else 'valid'
                             for f, status in zip(flags, out['measurement_status'])]
    out['quality_reasons'] = [_flags_text(f) for f in flags]
    return out


def inspect_targets(frame, generation=None, rules=None):
    """Calculate the unchanged research target plus its measurement coverage.

    Suspicion flags remain included. Only malformed measurements and explicit
    provider-invalid records are excluded. Missing/invalid rows remain in reports.
    """
    rules = rules or QualityRules()
    daily = inspect_daily(frame, rules)
    mapped = daily.loc[daily['unit'].notna() & daily['unit'].gt(0) & daily['month'].notna()].copy()
    mapped['unit'] = mapped['unit'].astype(int)
    accepted = mapped.loc[mapped['included_in_target']]
    day_values = accepted.groupby(['plant', 'unit', 'date'], as_index=False)['nox_numeric'].mean()
    day_values['month'] = day_values['date'].dt.to_period('M').dt.to_timestamp()
    averages = day_values.groupby(KEYS).agg(target=('nox_numeric', 'mean'), emission_days=('date', 'nunique'))
    rows = []
    for key, group in mapped.groupby(KEYS, sort=True):
        reasons = set().union(*(set(json.loads(v)) for v in group['quality_reasons']))
        target, days = (averages.loc[key, ['target', 'emission_days']].tolist()
                        if key in averages.index else (np.nan, 0))
        coverage = float(days / key[2].days_in_month)
        if coverage < rules.minimum_monthly_coverage:
            reasons.add('low_emission_coverage')
        rows.append(dict(zip(KEYS, key), target=target, emission_days=int(days), emission_coverage=coverage,
                         source_rows=len(group), excluded_rows=int((~group['included_in_target']).sum()),
                         review_rows=int(group['quality_status'].eq('review_required').sum()),
                         validity_unknown_rows=int(group['measurement_status'].eq('unknown').sum()),
                         quality_reasons=_flags_text(reasons)))
    columns = [*KEYS, 'target', 'emission_days', 'emission_coverage', 'source_rows',
               'excluded_rows', 'review_rows', 'validity_unknown_rows', 'quality_reasons']
    monthly = pd.DataFrame(rows, columns=columns)
    monthly['month'] = pd.to_datetime(monthly['month'])
    monthly['unit'] = monthly['unit'].astype(int)
    if generation is not None:
        monthly = monthly.merge(generation[KEYS + ['generation_mwh', 'thermal_efficiency_pct']],
                                on=KEYS, how='outer', validate='one_to_one')
        for col in ['emission_days','source_rows','excluded_rows','review_rows','validity_unknown_rows']:
            monthly[col] = monthly[col].fillna(0).astype(int)
        monthly['emission_coverage'] = monthly['emission_coverage'].fillna(0)
        for i, row in monthly.iterrows():
            reasons = set(json.loads(row['quality_reasons'])) if pd.notna(row['quality_reasons']) else set()
            if row['source_rows']==0:
                reasons.update(['no_emission_rows','low_emission_coverage'])
            if pd.isna(row['generation_mwh']):
                reasons.add('generation_missing')
            if row['generation_mwh'] == 0 and row['target'] > 0:
                reasons.add('positive_nox_zero_generation')
            if row['generation_mwh'] > 0 and row['thermal_efficiency_pct'] == 0:
                reasons.add('zero_efficiency_positive_generation')
            if pd.isna(row['thermal_efficiency_pct']):
                reasons.add('missing_efficiency')
            elif not 0 <= row['thermal_efficiency_pct'] <= 100:
                reasons.add('invalid_efficiency')
            monthly.at[i, 'quality_reasons'] = _flags_text(reasons)
        monthly = monthly.drop(columns=['generation_mwh', 'thermal_efficiency_pct'])
    monthly['quality_status'] = ['unavailable' if pd.isna(r['target']) else
                                 'review_required' if json.loads(r['quality_reasons']) else
                                 'unverified' if r['validity_unknown_rows'] else 'valid'
                                 for _, r in monthly.iterrows()]
    counts = {code: int(daily['quality_reasons'].map(lambda v: code in json.loads(v)).sum())
              for code in RULE_NOTES}
    summary = {'schema_version': 1, 'rules': asdict(rules), 'rule_notes': RULE_NOTES,
               'thresholds_are_review_heuristics': True, 'source_rows': len(daily),
               'included_rows': int(daily['included_in_target'].sum()),
               'excluded_rows': int((~daily['included_in_target']).sum()),
               'daily_status_counts': daily['quality_status'].value_counts().to_dict(),
               'daily_reason_counts': counts,
               'monthly_status_counts': monthly['quality_status'].value_counts().to_dict(),
               'structural_error_rows': int(daily['quality_reasons'].map(
                   lambda v: bool(set(json.loads(v)) & {'invalid_plant', 'invalid_date', 'duplicate_stack_day', 'invalid_nox', 'negative_nox'})).sum()),
               'limitations': ['Unknown provider validity is not certified valid.',
                               'Daily records cannot establish subdaily valid operating duration.',
                               'A/B concentrations remain arithmetic means, not flow-weighted means.']}
    return daily, monthly, summary


def update_weather_quality(frame):
    """Apply the same weather review policy to dataset and standalone reports."""
    frame = frame.copy()
    for i,row in frame.iterrows():
        reasons = set(json.loads(row['quality_reasons'])) if pd.notna(row['quality_reasons']) else set()
        if pd.isna(row['temperature_c']):
            reasons.add('weather_missing')
        elif row['weather_coverage'] < .8:
            reasons.add('low_weather_coverage')
        frame.at[i,'quality_reasons'] = _flags_text(reasons)
        if pd.isna(row['target']):
            frame.at[i,'quality_status'] = 'unavailable'
        elif reasons:
            frame.at[i,'quality_status'] = 'review_required'
    return frame


def write_quality_reports(daily, monthly, summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / 'daily_quality.csv', index=False, date_format='%Y-%m-%d')
    monthly.to_csv(output / 'monthly_quality.csv', index=False, date_format='%Y-%m-%d')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/quality')
    parser.add_argument('--high-nox', type=float, default=500)
    args = parser.parse_args()
    from .data import generation, monthly_weather, read
    daily, monthly, summary = inspect_targets(read('emissions_daily.csv', args.raw_dir),
                                              generation(args.raw_dir), QualityRules(high_nox=args.high_nox))
    weather,_ = monthly_weather(args.raw_dir)
    monthly = monthly.merge(weather[['plant','month','temperature_c','weather_days','weather_coverage']],
                            on=['plant','month'],how='left',validate='many_to_one')
    monthly = update_weather_quality(monthly)
    summary['monthly_status_counts'] = monthly['quality_status'].value_counts().to_dict()
    summary['monthly_reason_counts'] = {code:int(monthly['quality_reasons'].map(lambda v:code in json.loads(v)).sum()) for code in RULE_NOTES}
    write_quality_reports(daily, monthly, summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
