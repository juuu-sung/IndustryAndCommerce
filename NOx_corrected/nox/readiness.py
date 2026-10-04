"""Collect subsequent operating inputs and build a separate, auditable bundle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .collect import normalize_generation, parse_export
from .data import build_dataset, generation
from .operating_patterns import (FEATURES, POLICY, aggregate_unit, collect, digest,
                                 fetch, read_hourly, validate_pattern)
from .schema import KEYS, ROOT
from .train import write_json
from .weather_quality import POLICY as WEATHER_POLICY, recover_weather, monthly_metadata


def collect_sources(source_dir):
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in range(2023, 2027):
        last = f'{year}1231' if year < 2026 else '20260831'
        params = {'pageIndex': '1', 'strOrgNo': '', 'strHokiS': '', 'strHokiE': '',
                  'strDateS': f'{year}0101', 'strDateE': last}
        jobs.append((f'weather_{year}.csv',
                     'https://www.koenergy.kr/kosep/gv/nf/dt/nfdt18/csvDown.do', params, 'weather'))
    jobs.append(('generation_202609.csv',
                 'https://www.koenergy.kr/kosep/gv/nf/dt/nfdt01/csvDown.do',
                 {'pageIndex': '1', 'strOrgNo': '', 'strHokiS': '', 'strHokiE': '',
                  'strDateS': '202609', 'strDateE': '202609'}, 'generation'))
    statuses = []

    def download(job):
        name, url, params, kind = job
        try:
            path = fetch(name, url, params, source_dir)
            frame = parse_export(path.read_bytes(), kind)
            dates = pd.to_datetime(frame['일자'].astype(str),
                                   format='%Y%m%d' if kind == 'weather' else '%Y%m')
            lo = pd.to_datetime(params['strDateS'], format='%Y%m%d' if kind == 'weather' else '%Y%m')
            hi = pd.to_datetime(params['strDateE'], format='%Y%m%d' if kind == 'weather' else '%Y%m')
            if not dates.between(lo, hi).all():
                raise ValueError('Provider returned dates outside the requested range')
            result = {'file': name, 'status': 'ok', 'rows': len(frame), 'sha256': digest(path)}
        except Exception as error:
            result = {'file': name, 'status': 'failed', 'error_type': type(error).__name__,
                      'reason': str(error)[:250]}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(download, jobs))
    try:
        collect(source_dir, start_month='2026-01', end_month='2026-09')
        hourly = {'status': 'complete'}
    except Exception as error:
        hourly = {'status': 'partial_or_failed', 'error_type': type(error).__name__, 'reason': str(error)[:250]}
    result = {'exports': statuses, 'hourly': hourly,
              'retrieval_is_not_publication': True, 'new_nox_targets_requested': False}
    write_json(source_dir.parent / 'source_status.json', result)
    return result


def checked_source(path):
    receipt = json.loads(path.with_name(path.name + '.provenance.json').read_text())
    if receipt.get('sha256') != digest(path):
        raise ValueError('Source checksum mismatch: ' + path.name)
    return receipt


def extend_patterns(base, gen, sources):
    old = pd.read_csv(base / 'operation_monthly.csv', parse_dates=['month'])
    results, reconciliations, failures = [], [], []
    for path in sorted(sources.glob('hourly_yd_2026*.csv')):
        month = path.stem[-6:][:4] + '-' + path.stem[-2:]
        try:
            checked_source(path)
            frame = read_hourly(path, month)
        except Exception as error:
            failures.append({'file': path.name, 'reason': str(error)[:250]})
            continue
        for unit in (1, 2):
            record = aggregate_unit(frame, unit, month)
            row = gen.loc[gen.plant.eq('영동') & gen.unit.eq(unit) & gen.month.eq(pd.Timestamp(month + '-01'))]
            observed_coverage = record['gen_hour_coverage']
            reason = ''
            if row.empty:
                status, reason = 'unmatched_generation', 'Monthly generator observation unavailable; retain source without model attachment'
                daily_diff = hourly_diff = None
            else:
                monthly_mwh = float(row.generation_mwh.iloc[0])
                daily_diff = record['daily_total_sum_raw'] / 1000 - monthly_mwh
                hourly_diff = record['hourly_sum_raw'] / 1000 - monthly_mwh
                full = observed_coverage == 1 and record['daily_totals_observed'] == record['expected_days']
                try:
                    validate_pattern({k: record[k] for k in FEATURES}, pd.Timestamp(month + '-01'))
                except ValueError as error:
                    reason = str(error)
                if not full or not np.isfinite([daily_diff, hourly_diff]).all() or abs(daily_diff) > .02 or abs(hourly_diff) > .5:
                    reason = reason or 'Incomplete observations or hourly/monthly generation reconciliation failed'
                status = 'review_required' if reason else 'accepted'
                record['operation_quality_reason'] = reason
                if reason:
                    record.update({k: np.nan for k in FEATURES})
                    record['operation_pattern_basis'] = 'review_required'
                results.append(record)
            reconciliations.append({'plant': '영동', 'unit': unit, 'month': month,
                'source_file': path.name, 'operation_quality_status': status,
                'reason': reason, 'observed_hour_coverage': observed_coverage,
                'daily_sum_div_1000_minus_monthly_mwh': daily_diff,
                'hour_sum_div_1000_minus_monthly_mwh': hourly_diff,
                'scale_status': 'raw/1000 numerical reconciliation; physical unit unconfirmed'})
    added = pd.DataFrame(results)
    merged = pd.concat([old, added], ignore_index=True)
    if merged.duplicated(KEYS).any():
        raise ValueError('Operation extension overlaps existing keys')
    return merged, pd.DataFrame(reconciliations), failures


def prospective_completeness(base_gen, source):
    if not source.exists():
        return {'status': 'unavailable', 'target_scored': False}
    checked_source(source)
    observed, excluded = normalize_generation(parse_export(source.read_bytes(), 'generation'))
    expected = base_gen.loc[base_gen.month.eq(base_gen.month.max()), ['plant', 'unit']]
    fresh = observed.rename(columns={'사업소': 'plant', '호기': 'unit'})
    expected_keys = set(expected.itertuples(index=False, name=None))
    observed_keys = set(fresh[['plant', 'unit']].itertuples(index=False, name=None))
    return {'month': '2026-09', 'status': 'complete_generator_cohort' if expected_keys == observed_keys else 'incomplete_or_changed_cohort',
            'expected_units': len(expected_keys), 'observed_units': len(observed_keys),
            'missing_units': [[p, int(u)] for p, u in sorted(expected_keys - observed_keys)],
            'unexpected_units': [[p, int(u)] for p, u in sorted(observed_keys - expected_keys)],
            'excluded_steam_rows': len(excluded), 'target_scored': False,
            'note': 'Generator cohort completeness alone does not establish fuel/NOx/weather completeness or a final holdout'}


def prepare_bundle(base, sources, output, report):
    base, sources, output, report = map(Path, (base, sources, output, report))
    if output.exists():
        raise ValueError('Refusing to overwrite a readiness bundle')
    manifest = json.loads((base / 'manifest.json').read_text())
    for item in manifest['files']:
        if digest(base / item['file']) != item['sha256']:
            raise ValueError('Base bundle checksum mismatch')
    gen = generation(base)
    status = json.loads((sources.parent / 'source_status.json').read_text())
    allowed = {r['file'] for r in status['exports'] if r['status'] == 'ok'}
    exports, catalog = [], {'baseline_weather': {'file': str((base / 'weather_daily.csv').resolve()),
        'sha256': digest(base / 'weather_daily.csv'), 'provider': 'KOEN',
        'retrieved_at_utc': 'see baseline manifest; not a provider publication timestamp'}}
    for path in sorted(sources.glob('weather_*.csv')):
        if path.name not in allowed:
            continue
        receipt = checked_source(path)
        exports.append((path.stem, parse_export(path.read_bytes(), 'weather')))
        catalog[path.stem] = {'file': str(path.resolve()), **receipt}
    original = pd.read_csv(base / 'weather_daily.csv')
    weather, cells, recovery = recover_weather(original, exports)
    metadata = monthly_metadata(weather, gen, cells)
    patterns, reconciled, failures = extend_patterns(base, gen, sources)
    report.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='readiness_pending_') as directory:
        pending = Path(directory)
        for item in manifest['files']:
            shutil.copy2(base / item['file'], pending / item['file'])
        weather.to_csv(pending / 'weather_daily.csv', index=False)
        patterns.to_csv(pending / 'operation_monthly.csv', index=False, date_format='%Y-%m-%d')
        metadata.to_csv(pending / 'weather_monthly_metadata.csv', index=False, date_format='%Y-%m-%d')
        cells.to_csv(pending / 'weather_cell_sources.csv', index=False)
        write_json(pending / 'weather_source_catalog.json', catalog)
        write_json(pending / 'weather_recovery.json', recovery)
        files = [item['file'] for item in manifest['files']]
        files += ['weather_monthly_metadata.csv', 'weather_cell_sources.csv', 'weather_source_catalog.json', 'weather_recovery.json']
        manifest['files'] = [{'file': name, 'sha256': digest(pending / name)} for name in files]
        manifest.update(created_at=datetime.now(timezone.utc).isoformat(), weather_quality_policy=WEATHER_POLICY,
                        base_manifest_sha256=digest(base / 'manifest.json'),
                        readiness_source_directory=str(sources.resolve()))
        manifest['operation_sources']['extension_months'] = ['2026-01', '2026-09']
        manifest['operation_sources']['unit_months'] = len(patterns)
        manifest['operation_sources']['extension_files'] = [{'file': p.name, 'sha256': digest(p)}
            for p in sorted(sources.glob('hourly_yd_2026*.csv'))]
        write_json(pending / 'manifest.json', manifest)
        old, _ = build_dataset('concentration', base)
        new, audit = build_dataset('concentration', pending)
        pd.testing.assert_frame_equal(old[KEYS + ['target']], new[KEYS + ['target']], check_exact=True)
        for name in ['generation.csv', 'emissions_daily.csv', 'fuel_site.csv', 'fuel_unit.csv']:
            if digest(base / name) != digest(pending / name):
                raise ValueError('Protected source changed: ' + name)
        coverage = []
        for label, frame in [('before', old), ('after', new)]:
            observed = frame.loc[frame.target.notna()].copy()
            observed['year'] = observed.month.dt.year
            for (plant, year), group in observed.groupby(['plant', 'year']):
                coverage.append({'stage': label, 'plant': plant, 'year': int(year), 'n': len(group),
                                 'missing_temperature': int(group.temperature_c.isna().sum())})
        pd.DataFrame(coverage).to_csv(report / 'weather_coverage_by_plant_year.csv', index=False)
        prospective = prospective_completeness(gen, sources / 'generation_202609.csv') if 'generation_202609.csv' in allowed else {'status': 'unavailable', 'target_scored': False}
        summary = {'recovered_cells': len(recovery['recovered_cells']),
                   'preserved_weather_conflicts': len(recovery['preserved_conflicts']),
                   'unmapped_new_station_rows': len(recovery['unmapped_stations']),
                   'operation_total_unit_months': len(patterns), 'operation_added_unit_months': len(patterns) - len(old.loc[old.plant.eq('영동') & old.month.lt('2026-01-01')]),
                   'new_operation_status_counts': reconciled.operation_quality_status.value_counts().to_dict() if len(reconciled) else {},
                   'hourly_parse_failures': failures, 'prospective': prospective,
                   'target_values_unchanged': True, 'generation_and_fuel_unchanged': True,
                   'source_publication_times_unverified': True, 'audit': audit}
        reconciled.to_csv(report / 'new_operation_reconciliation.csv', index=False)
        write_json(report / 'data_readiness.json', summary)
        shutil.copytree(pending, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'data/operations_20261002')
    parser.add_argument('--sources', type=Path, default=ROOT / 'data/readiness_20261002/sources')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/readiness_bundle_20261002')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/readiness_20261002')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only:
        collect_sources(args.sources)
    if not args.collect_only:
        summary = prepare_bundle(args.base, args.sources, args.output, args.report)
        print(json.dumps({k: v for k, v in summary.items() if k != 'audit'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
