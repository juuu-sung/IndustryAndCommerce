"""Versioned observation imports. SQLite transactions never overwrite model files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .schema import CONDITIONS, KEYS, ROOT


QUALITY_COLUMNS = ['quality_status', 'quality_reasons', 'source_rows', 'excluded_rows',
                   'review_rows', 'validity_unknown_rows', 'emission_days',
                   'emission_coverage', 'weather_days', 'weather_coverage', 'reported_thermal_efficiency_pct']
RAW_FILES = ['generation.csv', 'fuel_site.csv', 'weather_daily.csv', 'emissions_daily.csv']
DEFAULT_DB = ROOT / 'data/runtime/history.sqlite'


class HistoryError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def timestamp(value):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HistoryError('Use an ISO timestamp with timezone') from error
    if pd.isna(result) or result.tzinfo is None:
        raise HistoryError('Use an ISO timestamp with timezone, e.g. 2026-10-01T05:00:00+09:00')
    return result.tz_convert('UTC').isoformat(timespec='microseconds')


def month_key(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value):
        raise HistoryError('Month must use YYYY-MM')
    try:
        return pd.Timestamp(value + '-01')
    except (ValueError, OverflowError) as error:
        raise HistoryError('Month is outside the supported date range') from error


def period_end(month):
    """A Korean reporting month is complete at the next month's local midnight."""
    return (pd.Timestamp(month) + pd.offsets.MonthBegin(1)).tz_localize('Asia/Seoul').tz_convert('UTC')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def model_key(artifact):
    artifact = Path(artifact)
    meta = json.loads((artifact / 'metadata.json').read_text())
    if meta['target_mode'] != 'concentration':
        raise HistoryError('Observation imports support concentration only; legacy units are unverified')
    identity = {'target': meta['target'], 'models': meta['bundle_sha256'], 'known_units': meta['known_units'],
                'baseline_sha256': hashlib.sha256((artifact / 'history.csv').read_bytes()).hexdigest()}
    if 'history_bootstrap_through' in meta:
        identity['history_bootstrap_through'] = meta['history_bootstrap_through']
    if 'fuel_input_basis' in meta:
        identity['fuel_input_basis'] = meta['fuel_input_basis']
    if 'operation_input_policy' in meta:
        identity['operation_input_policy'] = meta['operation_input_policy']
    return hashlib.sha256(_json(identity).encode()).hexdigest(), meta


def _payload(row):
    result = {}
    for col in [*KEYS, *CONDITIONS, 'target', *QUALITY_COLUMNS]:
        value = row.get(col)
        if col == 'month':
            result[col] = pd.Timestamp(value).strftime('%Y-%m')
        elif col == 'unit':
            result[col] = int(value)
        elif value is None or pd.isna(value):
            result[col] = None
        elif isinstance(value, np.generic):
            result[col] = value.item()
        else:
            result[col] = value
    if result['quality_status'] is None:
        result['quality_status'] = 'unverified'
    if result['target'] is None:
        result['quality_status'] = 'unavailable'
    if result['quality_reasons'] is None:
        result['quality_reasons'] = '[]'
    from .operating_patterns import FEATURES
    for col in ('fuel_basis', 'fuel_source_file', *FEATURES,
                'operation_pattern_basis', 'operation_source_file', 'operation_quality_reason'):
        if col in row:
            result[col] = None if pd.isna(row[col]) else row[col].item() if isinstance(row[col], np.generic) else row[col]
    return result


def _same(a, b):
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not np.isclose(x, y, rtol=1e-9, atol=1e-12):
                return False
        elif x != y:
            return False
    return True


def validate_observations(frame, known_units):
    required = {*KEYS, *CONDITIONS, 'target'}
    if required - set(frame):
        raise HistoryError(f'Missing history columns: {sorted(required - set(frame))}')
    if frame.empty or frame.duplicated(KEYS).any():
        raise HistoryError('History must be nonempty with unique plant/unit/month keys')
    if not pd.api.types.is_datetime64_dtype(frame['month']):
        raise HistoryError('History months must be parsed calendar dates')
    if frame['month'].isna().any() or not frame['month'].dt.is_month_start.all():
        raise HistoryError('History keys must use the first day of a calendar month')
    for _, row in frame.iterrows():
        if isinstance(row['unit'], (bool, np.bool_)) or not isinstance(row['unit'], (int, np.integer)):
            raise HistoryError('History unit must be an integer')
        if (row['plant'], row['unit']) not in known_units:
            raise HistoryError('Unrepresented plant/unit cannot be added to this model')
        for col in [*CONDITIONS, 'target']:
            value = row[col]
            if pd.isna(value):
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                raise HistoryError(f'{col}: invalid observation')
            if col not in ['temperature_c', 'wind_sin', 'wind_cos'] and value < 0:
                raise HistoryError(f'{col}: negative observation')
        if pd.isna(row['capacity_mw']) or row['capacity_mw'] <= 0 or pd.isna(row['generation_mwh']):
            raise HistoryError('Observed capacity and generation are required')
        if pd.notna(row['thermal_efficiency_pct']) and not 0 <= row['thermal_efficiency_pct'] <= 100:
            raise HistoryError('Observed efficiency must be between 0 and 100 or explicitly missing')
        if pd.isna(row['utilization_pct']):
            raise HistoryError('Observed utilization is required')
        for col, low, high in [('temperature_c', -80, 60), ('humidity_pct', 0, 100),
                               ('wind_speed_ms', 0, 100), ('wind_sin', -1, 1), ('wind_cos', -1, 1)]:
            if pd.notna(row[col]) and not low <= row[col] <= high:
                raise HistoryError(f'{col}: invalid observation range')
        if row.get('quality_status', 'unverified') not in {'valid', 'unverified', 'review_required', 'unavailable'}:
            raise HistoryError('Invalid target quality status')
        if 'quality_reasons' in row:
            try:
                reasons = json.loads(row['quality_reasons'])
            except (TypeError, ValueError) as error:
                raise HistoryError('quality_reasons must be a JSON list of codes') from error
            if not isinstance(reasons, list) or any(not isinstance(v, str) for v in reasons):
                raise HistoryError('quality_reasons must be a JSON list of codes')


class HistoryStore:
    def __init__(self, path=DEFAULT_DB, expected_model_key=None):
        self.path = Path(path)
        self.expected_model_key = expected_model_key

    def _connect(self, readonly=False):
        if readonly:
            connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True,
                                         timeout=5, isolation_level=None)
        else:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def _meta(self, connection):
        meta = dict(connection.execute('SELECT key, value FROM metadata').fetchall())
        if meta.get('schema_version') != '1':
            raise HistoryError('Unsupported history schema')
        if self.expected_model_key and meta.get('model_key') != self.expected_model_key:
            raise HistoryError('History does not match the frozen model/baseline; use a matching database')
        return meta

    def initialize(self, artifact):
        artifact = Path(artifact)
        identity, meta = model_key(artifact)
        self.expected_model_key = identity
        baseline = pd.read_csv(artifact / 'history.csv', parse_dates=['month'])
        known = {(r['plant'], r['unit']) for r in meta['known_units']}
        baseline = baseline.loc[[(p, u) in known for p, u in baseline[['plant', 'unit']].itertuples(index=False, name=None)]]
        if 'history_bootstrap_through' in meta:
            # Newly fetched observations must be imported with their actual local
            # recording times. Do not backdate them as a research bootstrap.
            baseline = baseline.loc[baseline['month'] <= month_key(meta['history_bootstrap_through'])]
        # Attach inspection results without altering frozen target/condition values.
        if not {'quality_status', 'quality_reasons'} <= set(baseline):
            from .data import build_dataset
            inspected, _ = build_dataset('concentration')
            quality = inspected[KEYS + [c for c in QUALITY_COLUMNS if c in inspected]]
            baseline = baseline.drop(columns=[c for c in QUALITY_COLUMNS if c in baseline]).merge(
                quality, on=KEYS, how='left', validate='one_to_one')
        validate_observations(baseline, known)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS batches (revision INTEGER PRIMARY KEY, batch_key TEXT UNIQUE NOT NULL, source TEXT NOT NULL, imported_at TEXT NOT NULL, available_at TEXT NOT NULL, source_files TEXT NOT NULL, stats TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS records (plant TEXT NOT NULL, unit INTEGER NOT NULL, month TEXT NOT NULL, revision INTEGER NOT NULL REFERENCES batches(revision), payload TEXT NOT NULL, PRIMARY KEY(plant,unit,month,revision))')
            connection.execute('CREATE INDEX IF NOT EXISTS record_lookup ON records(plant,unit,month,revision)')
            if connection.execute('SELECT COUNT(*) FROM metadata').fetchone()[0]:
                self._meta(connection)
                connection.commit()
                return {'status': 'already_initialized', **self.status()}
            imported = utc_now()
            values = {'schema_version': '1', 'model_key': identity, 'target': _json(meta['target']),
                      'known_units': _json(meta['known_units']), 'revision': '1'}
            if 'fuel_input_basis' in meta:
                from .predict import validate_fuel_history
                validate_fuel_history(baseline, meta['fuel_input_basis'])
                values['fuel_input_basis'] = _json(meta['fuel_input_basis'])
            if 'operation_input_policy' in meta:
                from .operating_patterns import validate_operation_history
                validate_operation_history(baseline, meta['operation_input_policy'])
                values['operation_input_policy'] = _json(meta['operation_input_policy'])
            connection.executemany('INSERT INTO metadata VALUES (?,?)', values.items())
            stats = {'rows': len(baseline), 'bootstrap_availability_assumed': True,
                     'note': 'Research snapshot; month-end availability is assumed, not confirmed publication history.'}
            connection.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)',
                               (1, identity, 'saved_research_snapshot', imported, imported,
                                _json({'history.csv': hashlib.sha256((artifact / 'history.csv').read_bytes()).hexdigest()}), _json(stats)))
            connection.executemany('INSERT INTO records VALUES (?,?,?,?,?)',
                                   [(r['plant'], int(r['unit']), r['month'].strftime('%Y-%m'), 1, _json(_payload(r)))
                                    for _, r in baseline.iterrows()])
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return {'status': 'initialized', **self.status()}

    def _records(self, connection, as_of=None):
        sql = 'SELECT r.*,b.source,b.imported_at,b.available_at FROM records r JOIN batches b USING(revision)'
        params = []
        if as_of is not None:
            # Bootstrap is an explicitly labelled research assumption. Imports and
            # corrections must have been both published and recorded by as_of.
            sql += ' WHERE (r.revision=1 OR (b.available_at<=? AND b.imported_at<=?))'
            params = [as_of, as_of]
        sql += ' ORDER BY r.revision DESC'
        latest = {}
        for row in connection.execute(sql, params):
            key = (row['plant'], row['unit'], row['month'])
            if key in latest:
                continue
            payload = json.loads(row['payload'])
            payload.update(history_revision=row['revision'], history_source=row['source'],
                           history_recorded_at=row['imported_at'], history_available_at=row['available_at'],
                           history_availability_assumed=row['revision'] == 1)
            latest[key] = payload
        return latest

    def snapshot(self, as_of=None):
        as_of = timestamp(as_of or utc_now())
        connection = self._connect(readonly=True)
        try:
            connection.execute('BEGIN')
            meta = self._meta(connection)
            rows = list(self._records(connection, as_of).values())
            frame = pd.DataFrame(rows)
            frame['month'] = pd.to_datetime(frame['month'], format='%Y-%m')
            from .operating_patterns import FEATURES
            for col in [*CONDITIONS, 'target', *[c for c in FEATURES if c in frame]]:
                frame[col] = pd.to_numeric(frame[col], errors='raise')
            # A bootstrap observation also cannot precede the end of its month.
            eligible = [not r['history_availability_assumed'] or period_end(r['month']) <= pd.Timestamp(as_of)
                        for _, r in frame.iterrows()]
            frame = frame.loc[eligible].sort_values(KEYS).reset_index(drop=True)
            return frame, {'revision': int(frame['history_revision'].max()) if len(frame) else 0,
                           'current_revision': int(meta['revision']), 'as_of': as_of,
                           'bootstrap_availability_assumed': True}
        finally:
            connection.close()

    def status(self):
        connection = self._connect(readonly=True)
        try:
            connection.execute('BEGIN')
            meta = self._meta(connection)
            rows = list(self._records(connection).values())
            latest = connection.execute('SELECT * FROM batches ORDER BY revision DESC LIMIT 1').fetchone()
            return {'storage': 'sqlite', 'revision': int(meta['revision']), 'row_count': len(rows),
                    'latest_observed_month': max((r['month'] for r in rows if r['target'] is not None), default=None),
                    'latest_imported_at': latest['imported_at'], 'last_source': latest['source'],
                    'quality_status_counts': pd.Series([r['quality_status'] for r in rows]).value_counts().to_dict(),
                    'bootstrap_availability_assumed': True}
        finally:
            connection.close()

    def import_observations(self, frame, *, source, source_files, batch_key, available_at=None,
                            expected_revision=None, allow_corrections=False, dry_run=False):
        if not isinstance(source, str) or not source.strip() or len(source) > 500:
            raise HistoryError('A nonempty source description is required (max 500 characters)')
        source = source.strip()
        imported = utc_now()
        available = timestamp(available_at or imported)
        if pd.Timestamp(available) > pd.Timestamp(imported):
            raise HistoryError('Future availability cannot be imported as observed data')
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            meta = self._meta(connection)
            current_revision = int(meta['revision'])
            if expected_revision is not None and expected_revision != current_revision:
                raise HistoryError(f'History revision changed: expected {expected_revision}, current {current_revision}')
            known = {(r['plant'], r['unit']) for r in json.loads(meta['known_units'])}
            validate_observations(frame, known)
            if 'fuel_input_basis' in meta:
                from .predict import validate_fuel_history
                validate_fuel_history(frame, json.loads(meta['fuel_input_basis']))
            if 'operation_input_policy' in meta:
                from .operating_patterns import validate_operation_history
                validate_operation_history(frame, json.loads(meta['operation_input_policy']))
            if any(period_end(v) > pd.Timestamp(available) for v in frame['month']):
                raise HistoryError('Only complete monthly observations available by available_at may be imported')
            if connection.execute('SELECT 1 FROM batches WHERE batch_key=?', (batch_key,)).fetchone():
                return {'status': 'already_imported', 'revision': current_revision, 'inserted': 0, 'corrected': 0}
            existing = self._records(connection)
            updates, inserted, corrected, unchanged = [], 0, 0, 0
            for _, row in frame.iterrows():
                value = _payload(row)
                key = tuple(value[c] for c in KEYS)
                old = existing.get(key)
                if old is not None and _same(value, {k: old.get(k) for k in value}):
                    unchanged += 1
                    continue
                if old is not None:
                    if not allow_corrections:
                        raise HistoryError(f'Existing month differs: {key}; inspect and use --allow-corrections')
                    corrected += 1
                else:
                    inserted += 1
                updates.append((key, value))
            stats = {'inserted': inserted, 'corrected': corrected, 'unchanged': unchanged,
                     'candidate_rows': len(frame),
                     'quality_status_counts': frame['quality_status'].value_counts().to_dict() if 'quality_status' in frame else {}}
            if not updates or dry_run:
                return {'status': 'dry_run' if dry_run else 'no_changes', 'revision': current_revision, **stats}
            revision = current_revision + 1
            connection.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)',
                               (revision, batch_key, source, imported, available, _json(source_files), _json(stats)))
            connection.executemany('INSERT INTO records VALUES (?,?,?,?,?)',
                                   [(key[0], key[1], key[2], revision, _json(value)) for key, value in updates])
            connection.execute('UPDATE metadata SET value=? WHERE key=?', (str(revision), 'revision'))
            connection.commit()
            return {'status': 'imported', 'revision': revision, **stats}
        except Exception:
            connection.rollback()
            raise
        finally:
            # dry-run, conflicts and idempotent reads never leave a transaction open.
            if connection.in_transaction:
                connection.rollback()
            connection.close()


def import_bundle(store, raw_dir, *, start_month, end_month, source, report_dir,
                  available_at=None, expected_revision=None, allow_corrections=False, dry_run=False):
    from .data import build_dataset, generation, read
    from .quality import inspect_targets, write_quality_reports, RULE_NOTES
    raw_dir = Path(raw_dir)
    start, end = month_key(start_month), month_key(end_month)
    if start > end:
        raise HistoryError('start-month must not follow end-month')
    raw_files = RAW_FILES + (['fuel_unit.csv'] if (raw_dir/'fuel_unit.csv').exists() else [])
    hashes = {name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() for name in raw_files}
    daily, monthly, summary = inspect_targets(read('emissions_daily.csv', raw_dir), generation(raw_dir))
    write_quality_reports(daily, monthly, summary, report_dir)
    if summary['structural_error_rows']:
        raise HistoryError('Malformed/duplicate measurements: import rejected; inspect quality reports')
    frame, audit = build_dataset('concentration', raw_dir)
    summary['monthly_status_counts'] = frame['quality_status'].value_counts().to_dict()
    summary['monthly_reason_counts'] = {code:int(frame['quality_reasons'].map(lambda v:code in json.loads(v)).sum()) for code in RULE_NOTES}
    write_quality_reports(daily, frame[KEYS+['target']+[c for c in QUALITY_COLUMNS if c in frame]],summary,report_dir)
    frame = frame.loc[frame['month'].between(start, end)].copy()
    connection = store._connect(readonly=True)
    try:
        meta = store._meta(connection)
    finally:
        connection.close()
    known = {(r['plant'], r['unit']) for r in json.loads(meta['known_units'])}
    represented = pd.Series([(p, u) in known for p, u in frame[['plant', 'unit']].itertuples(index=False, name=None)], index=frame.index)
    unrepresented = frame.loc[~represented, KEYS].astype(str).to_dict('records')
    frame = frame.loc[represented]
    summary['unrepresented_monthly_rows'] = unrepresented
    summary['dataset_audit'] = audit
    (Path(report_dir) / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Detect source replacement during import rather than committing mixed snapshots.
    if hashes != {name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() for name in raw_files}:
        raise HistoryError('Source files changed while being inspected; retry a stable export')
    identity = hashlib.sha256(_json({'source_files': hashes, 'source': source.strip(),
                                    'start': start_month, 'end': end_month, 'rules': summary['rules']}).encode()).hexdigest()
    result = store.import_observations(frame, source=source, source_files=hashes, batch_key=identity,
                                      available_at=available_at, expected_revision=expected_revision,
                                      allow_corrections=allow_corrections, dry_run=dry_run)
    result['unrepresented_rows'] = len(unrepresented)
    (Path(report_dir) / 'import_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / 'artifacts/concentration')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('status')
    command = sub.add_parser('import')
    command.add_argument('--raw-dir', type=Path, required=True)
    command.add_argument('--start-month', required=True)
    command.add_argument('--end-month', required=True)
    command.add_argument('--source', required=True)
    command.add_argument('--available-at')
    command.add_argument('--expected-revision', type=int)
    command.add_argument('--allow-corrections', action='store_true')
    command.add_argument('--dry-run', action='store_true')
    command.add_argument('--report-dir', type=Path, default=ROOT / 'reports/history_import')
    args = parser.parse_args()
    try:
        identity, _ = model_key(args.artifact_dir)
        store = HistoryStore(args.db, identity)
        if args.command == 'init':
            result = store.initialize(args.artifact_dir)
        elif args.command == 'status':
            result = store.status()
        else:
            result = import_bundle(store, args.raw_dir, start_month=args.start_month, end_month=args.end_month,
                                   source=args.source, report_dir=args.report_dir, available_at=args.available_at,
                                   expected_revision=args.expected_revision, allow_corrections=args.allow_corrections,
                                   dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f'History operation failed: {error}\n')


if __name__ == '__main__':
    main()
