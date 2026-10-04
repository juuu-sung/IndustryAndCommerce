"""Preserve KOEN hourly exports and derive unit-independent generation patterns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests

from .schema import KEYS, ROOT
def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

FEATURES = ['gen_hour_coverage', 'gen_zero_hour_fraction', 'gen_zero_day_fraction',
            'gen_longest_zero_run_fraction', 'gen_zero_positive_transitions',
            'gen_positive_hour_cv', 'gen_peak_to_positive_mean', 'gen_hour_ramp_relative_mean']
HOURS = [f'{h}시 발전량(MWh)' for h in range(1, 25)]
POLICY = {'version': 1, 'plants': ['영동'], 'features': FEATURES,
          'minimum_hour_coverage': .95, 'scope': 'realized current-month electrical generation patterns',
          'unit_status': 'scale-independent features; hourly physical unit remains unconfirmed',
          'zero_status': 'zero reported electrical generation; not verified boiler/SCR shutdown',
          'reconciliation_limits_mwh': {'daily_total': .02, 'hourly_sum': .5},
          'unavailable_input': 'explicit null; train-median fill plus availability indicator; target retained',
          'provider_publication_time': 'unverified'}
BASE = 'https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_hourly(path, month):
    """Preserve blank hours as NaN and reject shifted columns or unexpected keys."""
    records = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    if not records:
        raise ValueError('Empty hourly response')
    header = [v.strip() for v in records[0]]
    if len(set(header)) != len(header) or set(['발전구분', '호기', '일자', '총량(KW)', *HOURS]) - set(header):
        raise ValueError('Hourly provider schema changed')
    rows = []
    for line, raw in enumerate(records[1:], 2):
        if not raw:
            continue
        if len(raw) == len(header) + 1 and not raw[-1].strip():
            raw = raw[:-1]
        if len(raw) != len(header):
            raise ValueError(f'Hourly column count mismatch at line {line}')
        rows.append([v.strip() for v in raw])
    frame = pd.DataFrame(rows, columns=header)
    if frame.empty or not frame['발전구분'].eq('영동').all() or not frame['호기'].isin(['1', '2']).all():
        raise ValueError('Unexpected hourly plant/unit/empty data')
    frame['unit'] = frame['호기'].astype(int)
    frame['date'] = pd.to_datetime(frame['일자'], errors='raise').dt.normalize()
    if frame.date.isna().any() or not frame.date.dt.strftime('%Y-%m').eq(month).all():
        raise ValueError('Unexpected hourly dates')
    if frame.duplicated(['unit', 'date']).any():
        raise ValueError('Duplicate hourly unit/date')
    for col in [*HOURS, '총량(KW)']:
        numeric = pd.to_numeric(frame[col].replace({'': np.nan, '-': np.nan}), errors='raise')
        if (numeric.dropna() < 0).any() or not np.isfinite(numeric.dropna()).all():
            raise ValueError('Negative or nonfinite hourly observation')
        frame[col] = numeric
    frame['source_file'] = Path(path).name
    return frame.sort_values(['unit', 'date']).reset_index(drop=True)


def longest_zero_run(values):
    longest = running = 0
    for value in values:
        running = running + 1 if np.isfinite(value) and value == 0 else 0
        longest = max(longest, running)
    return longest


def aggregate_unit(frame, unit, month):
    period = pd.Period(month, 'M')
    dates = pd.date_range(period.start_time, period.end_time.normalize(), freq='D')
    group = frame.loc[frame.unit.eq(unit)].set_index('date').reindex(dates)
    matrix = group[HOURS].to_numpy(dtype=float)
    values = matrix.ravel()  # Ascending dates, provider hours 1..24, never across units.
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    complete = np.isfinite(matrix).all(axis=1)
    valid_pairs = finite[1:] & finite[:-1]
    transitions = int(((values[1:] == 0) != (values[:-1] == 0))[valid_pairs].sum())
    all_mean = values[finite].mean() if finite.any() else np.nan
    positive_mean = positive.mean() if len(positive) else 0.
    record = {'plant': '영동', 'unit': unit, 'month': period.start_time,
        'gen_hour_coverage': float(finite.sum() / len(values)),
        'gen_zero_hour_fraction': float((values[finite] == 0).mean()) if finite.any() else np.nan,
        'gen_zero_day_fraction': float((matrix[complete] == 0).all(axis=1).mean()) if complete.any() else np.nan,
        'gen_longest_zero_run_fraction': float(longest_zero_run(values) / len(values)),
        'gen_zero_positive_transitions': transitions,
        'gen_positive_hour_cv': float(positive.std(ddof=0) / positive_mean) if positive_mean > 0 else 0.,
        'gen_peak_to_positive_mean': float(positive.max() / positive_mean) if positive_mean > 0 else 0.,
        'gen_hour_ramp_relative_mean': float(np.abs(np.diff(values))[valid_pairs].mean() / all_mean)
            if valid_pairs.any() and all_mean > 0 else 0.,
        'expected_hours': len(values), 'observed_hours': int(finite.sum()),
        'expected_days': len(dates), 'observed_days': int(group['unit'].notna().sum()),
        'complete_days': int(complete.sum()), 'missing_hours': int((~finite).sum()),
        'daily_totals_observed': int(group['총량(KW)'].notna().sum()),
        'hourly_sum_raw': float(values[finite].sum()),
        'daily_total_sum_raw': float(group['총량(KW)'].sum(min_count=1)),
        'max_daily_sum_difference_raw': float((group[HOURS].sum(axis=1, min_count=24) - group['총량(KW)']).abs().max()),
        'operation_pattern_basis': 'koen_hourly_reported_electrical_generation',
        'operation_source_file': frame.source_file.iloc[0],
    }
    return record


def validate_pattern(pattern, month):
    if not isinstance(pattern, dict) or set(pattern) != set(FEATURES):
        raise ValueError('operation_pattern must contain exactly the configured generation features')
    out = {}
    expected_hours = pd.Period(month, 'M').days_in_month * 24
    for name, value in pattern.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name}: finite nonnegative value required')
        if (name.endswith('_fraction') or name == 'gen_hour_coverage') and value > 1:
            raise ValueError(f'{name}: fraction must be between 0 and 1')
        if name == 'gen_zero_positive_transitions' and (int(value) != value or value > expected_hours - 1):
            raise ValueError('gen_zero_positive_transitions must be an integer within the month')
        out[name] = float(value)
    if out['gen_hour_coverage'] < POLICY['minimum_hour_coverage']:
        raise ValueError('Hourly observation coverage must be at least 95%; missing observations are not zeros')
    if out['gen_positive_hour_cv'] > np.sqrt(expected_hours):
        raise ValueError('Positive-hour variation is inconsistent with the month length')
    if out['gen_peak_to_positive_mean'] > expected_hours or 0 < out['gen_peak_to_positive_mean'] < 1:
        raise ValueError('Peak/positive mean is outside its mathematical range')
    if out['gen_hour_ramp_relative_mean'] > 2 * expected_hours:
        raise ValueError('Relative hourly change is outside its mathematical range')
    if out['gen_longest_zero_run_fraction'] > out['gen_zero_hour_fraction'] * out['gen_hour_coverage'] + 1e-12:
        raise ValueError('Longest zero run exceeds observed zero hours')
    if out['gen_zero_hour_fraction'] in (0., 1.) and out['gen_zero_positive_transitions'] != 0:
        raise ValueError('Zero/positive transitions conflict with the zero-hour fraction')
    if out['gen_zero_hour_fraction'] == 1. and any(out[k] != 0 for k in ['gen_positive_hour_cv', 'gen_peak_to_positive_mean', 'gen_hour_ramp_relative_mean']):
        raise ValueError('All-zero reported generation requires zero positive-hour statistics')
    return out


def attach_patterns(df, raw_dir, manifest):
    """Optional bundle extension; old bundles and target values are unchanged."""
    path = Path(raw_dir) / 'operation_monthly.csv'
    policy = manifest.get('operation_input_policy')
    if policy is None:
        if path.exists():
            raise ValueError('Operation file requires an explicit source policy')
        return df
    if policy != POLICY or not path.exists():
        raise ValueError('Unrecognized or incomplete operation input policy')
    frame = pd.read_csv(path, parse_dates=['month'])
    if set([*KEYS, *FEATURES, 'operation_pattern_basis', 'operation_source_file', 'operation_quality_reason']) - set(frame):
        raise ValueError('Missing operation columns')
    if frame.empty or frame.duplicated(KEYS).any() or not frame.plant.isin(policy['plants']).all():
        raise ValueError('Invalid/duplicate operation keys')
    if not frame.unit.isin([1, 2]).all():
        raise ValueError('Operation units must match the verified 1/2 provider mapping')
    if frame.month.isna().any() or not frame.month.dt.is_month_start.all():
        raise ValueError('Operation months must be month starts')
    if not frame.operation_pattern_basis.isin(['koen_hourly_reported_electrical_generation', 'review_required']).all() or frame.operation_source_file.isna().any():
        raise ValueError('Missing/mismatched operation provenance')
    for _, row in frame.iterrows():
        if row.operation_pattern_basis == 'review_required':
            if not row[FEATURES].isna().all() or not isinstance(row.get('operation_quality_reason'), str) or not row.operation_quality_reason:
                raise ValueError('Quarantined patterns require all-null features and an explicit reason')
        else:
            validate_pattern(row[FEATURES].to_dict(), row.month)
    source_keys = pd.MultiIndex.from_frame(frame[KEYS])
    if not source_keys.isin(pd.MultiIndex.from_frame(df[KEYS])).all():
        raise ValueError('Unrepresented operation keys')
    return df.merge(frame[KEYS + FEATURES + ['operation_pattern_basis', 'operation_source_file', 'operation_quality_reason']],
                    on=KEYS, how='left', validate='one_to_one')


def require_patterns(frame, policy):
    if policy != POLICY:
        raise ValueError('Unknown operation input policy')
    relevant = frame.loc[frame.plant.isin(policy['plants'])]
    if 'operation_pattern_basis' not in relevant or relevant.operation_pattern_basis.isna().any():
        raise ValueError('Every configured plant/month requires an observed or quarantined operation record')


def validate_operation_history(frame, policy):
    require_patterns(frame, policy)
    for _, row in frame.loc[frame.plant.isin(policy['plants'])].iterrows():
        if 'operation_source_file' not in row or pd.isna(row.operation_source_file):
            raise ValueError('Operation history requires source provenance')
        if row.operation_pattern_basis == 'review_required':
            if not row.reindex(FEATURES).isna().all() or pd.isna(row.get('operation_quality_reason')) or not row.get('operation_quality_reason'):
                raise ValueError('Quarantined operation history requires all-null features and a reason')
        elif row.operation_pattern_basis == 'koen_hourly_reported_electrical_generation':
            validate_pattern(row.reindex(FEATURES).to_dict(), row.month)
        else:
            raise ValueError('Invalid operation history basis')


def fetch(name, url, params, output, method='POST'):
    path = output / name
    sidecar = path.with_name(path.name + '.provenance.json')
    if path.exists():
        receipt = json.loads(sidecar.read_text())
        if receipt.get('sha256') != digest(path):
            raise ValueError('Cached source checksum mismatch')
        if receipt.get('url') != url or receipt.get('method') != method or receipt.get('parameters') != params:
            raise ValueError('Cached source request differs from the requested URL/method/parameters')
        return path
    stamp = datetime.now(timezone.utc).isoformat()
    response = requests.request(method, url, data=params if method == 'POST' else None,
        params=params if method == 'GET' else None, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    write_json(sidecar, {'url': url, 'resolved_url': response.url, 'method': method, 'parameters': params,
        'retrieved_at_utc': stamp, 'provider_publication_time': 'unknown; retrieval time is not publication time',
        'http_status': response.status_code, 'content_type': response.headers.get('Content-Type'),
        'bytes': len(response.content), 'sha256': digest(path)})
    return path


def collect(source_dir, *, start_month='2023-01', end_month='2025-12'):
    periods = pd.period_range(start_month, end_month, freq='M')
    current = pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).to_period('M')
    if not len(periods) or periods[-1] >= current:
        raise ValueError('Hourly collection requires ordered, completed months')
    source_dir.mkdir(parents=True, exist_ok=True)
    # Confirm the current public code list rather than treating two code systems as interchangeable.
    plants = json.loads(fetch('plant_codes.json', BASE + 'getFirePowerList.do', {'code_cd1': '01'}, source_dir).read_text())['resultList']
    code = next(r['group_cd'] for r in plants if r['group_nm'] == '영동')
    units = json.loads(fetch('unit_codes.json', BASE + 'getHokiList.do', {'code_cd1': '01', 'group_cd': code}, source_dir).read_text())['resultList']
    if code != '8451' or not {'1', '2'}.issubset({r['code_cd'] for r in units}):
        raise ValueError('Provider plant/unit mapping changed')
    jobs = []
    for period in periods:
        ym = period.strftime('%Y%m')
        params = {'pageIndex': '1', 'strOrgNo': code, 'strHokiS': '1', 'strHokiE': '2',
                  'strDateS': ym + '01', 'strDateE': ym + str(period.days_in_month)}
        jobs.append((f'hourly_yd_{ym}.csv', BASE + 'csvDown.do', params, source_dir))
    failures = []
    def download(job):
        try:
            path = fetch(*job)
            read_hourly(path, job[2]['strDateS'][:4] + '-' + job[2]['strDateS'][4:6])
            print(path.name, path.stat().st_size, flush=True)
        except Exception as error:
            failures.append({'file': job[0], 'error': str(error)})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, jobs))
    write_json(source_dir.parent / 'collection_status.json', {
        'months_requested': len(periods), 'requested_months': [start_month, end_month], 'failures': failures})
    if failures:
        raise ValueError(f'{len(failures)} hourly requests failed; inspect collection_status.json')


def prepare(source_dir, output, base_raw):
    if output.exists():
        raise ValueError('Refusing to overwrite operation bundle')
    from .data import build_dataset
    baseline, _ = build_dataset('concentration', base_raw)
    months = []
    for period in pd.period_range('2023-01', '2025-12', freq='M'):
        path = source_dir / f"hourly_yd_{period.strftime('%Y%m')}.csv"
        receipt = json.loads(path.with_name(path.name + '.provenance.json').read_text())
        if digest(path) != receipt['sha256']:
            raise ValueError('Hourly source checksum mismatch')
        frame = read_hourly(path, str(period))
        for unit in (1, 2):
            record = aggregate_unit(frame, unit, str(period))
            validate_pattern({k: record[k] for k in FEATURES}, str(period))
            months.append(record)
    monthly = pd.DataFrame(months)
    reconciled = monthly.merge(baseline[KEYS + ['generation_mwh']], on=KEYS, validate='one_to_one')
    reconciled['daily_sum_div_1000_minus_monthly_mwh'] = reconciled.daily_total_sum_raw / 1000 - reconciled.generation_mwh
    reconciled['hour_sum_div_1000_minus_monthly_mwh'] = reconciled.hourly_sum_raw / 1000 - reconciled.generation_mwh
    reconciled['scale_status'] = 'raw/1000 inferred from monthly reconciliation; physical unit unconfirmed'
    full = reconciled.gen_hour_coverage.eq(1) & reconciled.daily_totals_observed.eq(reconciled.expected_days)
    rejected = ~full | reconciled.daily_sum_div_1000_minus_monthly_mwh.abs().gt(.02) | reconciled.hour_sum_div_1000_minus_monthly_mwh.abs().gt(.5)
    # Rules are fixed before model training. Quarantine features, never edit NOx targets or raw cells.
    monthly['operation_quality_reason'] = ''
    for i in monthly.index[rejected]:
        monthly.loc[i, 'operation_quality_reason'] = 'Incomplete observations or hourly/monthly generation reconciliation failed'
    reconciled['operation_quality_status'] = np.where(rejected, 'review_required', 'accepted')
    reconciled['observed_hour_coverage'] = reconciled.gen_hour_coverage
    monthly.loc[rejected, FEATURES] = np.nan
    monthly.loc[rejected, 'operation_pattern_basis'] = 'review_required'
    output.mkdir(parents=True)
    manifest = json.loads((base_raw / 'manifest.json').read_text())
    for item in manifest['files']:
        if digest(base_raw / item['file']) != item['sha256']:
            raise ValueError('Base bundle checksum mismatch')
        shutil.copy2(base_raw / item['file'], output / item['file'])
    monthly.to_csv(output / 'operation_monthly.csv', index=False, date_format='%Y-%m-%d')
    reconciled.to_csv(source_dir.parent / 'monthly_reconciliation.csv', index=False, date_format='%Y-%m-%d')
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), operation_input_policy=POLICY,
        operation_sources={'months': ['2023-01', '2025-12'], 'unit_months': len(monthly),
            'source_directory': str(source_dir.resolve()),
            'files': [{'file': p.name, 'sha256': digest(p)} for p in sorted(source_dir.iterdir()) if p.is_file()],
            'features_exclude_nox': True, 'boiler_scr_state_not_inferred': True})
    manifest['files'].append({'file': 'operation_monthly.csv', 'sha256': digest(output / 'operation_monthly.csv'), 'rows': len(monthly)})
    write_json(output / 'manifest.json', manifest)
    write_json(source_dir.parent / 'preparation_summary.json', {'unit_months': len(monthly),
        'observed_days': int(monthly.observed_days.sum()), 'observed_hours': int(monthly.observed_hours.sum()),
        'missing_hours': int(monthly.missing_hours.sum()),
        'accepted_unit_months': int((~rejected).sum()), 'quarantined_unit_months': int(rejected.sum()),
        'max_daily_month_difference_mwh': float(reconciled.daily_sum_div_1000_minus_monthly_mwh.abs().max()),
        'max_hour_month_difference_mwh': float(reconciled.hour_sum_div_1000_minus_monthly_mwh.abs().max()),
        'minimum_coverage': float(monthly.gen_hour_coverage.min()), 'target_changed': False, 'policy': POLICY})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/hourly_operations_20261002/sources')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/operations_20261002')
    parser.add_argument('--base-raw', type=Path, default=ROOT / 'data/unit_fuel_20261001')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only:
        collect(args.source_dir)
    if not args.collect_only:
        prepare(args.source_dir, args.output, args.base_raw)


if __name__ == '__main__':
    main()
