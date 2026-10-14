"""Register immutable, source-bound cases for replaying saved NOx evaluations.

This module reads frozen artifacts; it never trains models or imports observations.
A successful registration establishes file/row consistency, not measurement validity.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from ..schema import ROOT, TARGETS

FILES = (
    'metadata.json', 'history.csv', 'test_predictions.csv',
    'validation_predictions.csv', 'validation_weight_search.json',
    'selection_lock.json', 'preprocessor.json', 'xgb_model.json', 'lgb_model.txt',
)
MODEL_FILES = ('xgb_model.json', 'lgb_model.txt', 'preprocessor.json')
LOCK_FILES = ('metadata.json', *MODEL_FILES)
METHODS = (
    'xgboost', 'lightgbm', 'ensemble', 'train_global_median',
    'train_unit_mean', 'last_month', 'previous_3months',
)
DEFAULT_REGISTRY = ROOT / 'data/runtime/copilot/cases'
QUALITY_STATES = {'valid', 'unverified', 'review_required', 'unavailable'}
MANIFEST_FIELDS = {
    'schema_version', 'case_key', 'case', 'analysis_mode', 'registered_at',
    'artifact_path', 'files', 'bundle_version_id', 'model', 'protocol',
    'target', 'target_definition_sha256', 'target_verification_status',
    'source_manifest', 'snapshot', 'limitations', 'case_version_id',
}


class RegistrationError(ValueError):
    """A case cannot be trusted as a consistent replay of the saved evaluation."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _month(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value):
        raise RegistrationError('Month must use YYYY-MM')
    try:
        date(int(value[:4]), int(value[5:]), 1)
    except ValueError as error:
        raise RegistrationError('Invalid calendar month') from error
    return value


def _plant(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegistrationError('Plant must be a nonempty, exact identifier')
    return value


def _unit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RegistrationError('Unit must be a positive integer, not a boolean')
    return value


@dataclass(frozen=True)
class CaseSpec:
    case_key: str
    plant: str
    unit: int
    month: str

    def __post_init__(self):
        _case_key(self.case_key)
        _plant(self.plant)
        _unit(self.unit)
        _month(self.month)

    @property
    def key(self):
        return self.plant, self.unit, self.month

    def as_dict(self):
        return {'plant': self.plant, 'unit': self.unit, 'month': self.month}


def _case_key(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', value):
        raise RegistrationError('case_key must use 1-80 lowercase letters, digits, _ or -')
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RegistrationError(f'{label} must be a finite number')
    return float(value)


def _csv_number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RegistrationError(f'{label}: invalid CSV number') from error
    if not math.isfinite(result):
        raise RegistrationError(f'{label}: nonfinite CSV number')
    return result


def _csv_integer(value, label, minimum=1):
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RegistrationError(f'{label}: invalid CSV integer') from error
    if not number.is_finite() or number != number.to_integral_value() or number < minimum:
        raise RegistrationError(f'{label}: invalid CSV integer')
    return int(number)


def _csv_boolean(value, label):
    if value not in ('True', 'False', 'true', 'false'):
        raise RegistrationError(f'{label}: invalid CSV boolean')
    return value.lower() == 'true'


def _timestamp(value, label):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise RegistrationError(f'{label} must be an ISO timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistrationError(f'{label} must include a timezone')
    return value


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise RegistrationError(f'{label} must be a SHA-256 digest')
    return value


def _json(data, label):
    def reject_constant(value):
        raise RegistrationError(f'{label}: nonfinite JSON constant')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RegistrationError(f'{label}: duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RegistrationError(f'{label}: invalid JSON') from error


def _object(value, label):
    if not isinstance(value, dict):
        raise RegistrationError(f'{label} must be a JSON object')
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise RegistrationError(f'{label} must be nonempty text')
    return value


def _text_list(value, label):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise RegistrationError(f'{label} must be a list of nonempty strings')
    return value


def _protocol(value):
    value = _object(value, 'split')
    if set(value) != {'train', 'validation', 'test'}:
        raise RegistrationError('Exactly train, validation and test periods are required')
    previous = None
    for name in ('train', 'validation', 'test'):
        bounds = value[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise RegistrationError('Each period needs two YYYY-MM bounds')
        start, stop = map(_month, bounds)
        if start > stop or (previous is not None and start <= previous):
            raise RegistrationError('Periods must be ordered and nonoverlapping')
        previous = stop
    return value


def _csv_rows(data, label, required):
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required).issubset(columns):
            raise RegistrationError(f'{label}: missing or duplicate columns')
        rows = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise RegistrationError(f'{label}: malformed CSV row')
            raw_month = row['month']
            if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])-01', raw_month):
                raise RegistrationError(f'{label}: months must use the first calendar day')
            key = (_plant(row['plant']), _csv_integer(row['unit'], label + '.unit'),
                   _month(raw_month[:7]))
            if key in rows:
                raise RegistrationError(f'{label}: duplicate plant/unit/month key')
            rows[key] = row
    except (UnicodeError, csv.Error) as error:
        raise RegistrationError(f'{label}: invalid CSV') from error
    if not rows:
        raise RegistrationError(f'{label}: empty CSV')
    return rows


def _read_bundle(artifact):
    artifact = Path(artifact).resolve()
    data, records = {}, {}
    try:
        for name in FILES:
            source = artifact / name
            if source.resolve() != artifact / name or not source.is_file():
                raise RegistrationError(f'Bundle file missing or linked outside bundle: {name}')
            data[name] = source.read_bytes()
            records[name] = {'sha256': _sha(data[name]), 'size_bytes': len(data[name])}
    except OSError as error:
        raise RegistrationError('Frozen evaluation bundle is unavailable') from error
    return artifact, data, records


def _unchanged(artifact, records):
    try:
        for name, record in records.items():
            source = artifact / name
            if source.resolve() != artifact / name or _sha(source.read_bytes()) != record['sha256']:
                raise RegistrationError(f'Bundle changed while being checked: {name}')
    except OSError as error:
        raise RegistrationError('Bundle became unavailable while being checked') from error


def _model_info(data, records):
    meta = _object(_json(data['metadata.json'], 'metadata.json'), 'metadata')
    if type(meta.get('schema_version')) is not int or meta['schema_version'] != 1:
        raise RegistrationError('Unsupported model metadata schema')
    target = _object(meta.get('target'), 'target')
    if (meta.get('target_mode') != 'concentration'
            or target.get('name') != TARGETS['concentration']['name']
            or target.get('unit') != 'ppm'):
        raise RegistrationError('Registration supports the research concentration target in ppm only')
    _text(target.get('description'), 'target.description')
    _text_list(target.get('limitations'), 'target.limitations')
    split = _protocol(meta.get('split'))
    if meta.get('trained_through') != split['train'][1]:
        raise RegistrationError('trained_through differs from the training period')
    _timestamp(meta.get('trained_at'), 'trained_at')
    _text(meta.get('evaluation_mode'), 'evaluation_mode')
    _text(meta.get('selection_scope'), 'selection_scope')
    if meta.get('weights_locked_before_test') is not True:
        raise RegistrationError('Model must declare weights locked before test evaluation')
    hashes = _object(meta.get('bundle_sha256'), 'bundle_sha256')
    if set(hashes) != set(MODEL_FILES):
        raise RegistrationError('Unexpected model/preprocessor bundle files')
    for name, expected in hashes.items():
        if _hash(expected, name) != records[name]['sha256']:
            raise RegistrationError(f'Model bundle checksum mismatch: {name}')
    lock = _object(_json(data['selection_lock.json'], 'selection_lock.json'), 'selection lock')
    locked = _object(lock.get('sha256'), 'selection_lock.sha256')
    if set(locked) != set(LOCK_FILES) or lock.get('test_predictions_created') is not False:
        raise RegistrationError('Invalid pre-evaluation selection lock')
    for name, expected in locked.items():
        if _hash(expected, name) != records[name]['sha256']:
            raise RegistrationError(f'Selection lock checksum mismatch: {name}')
    pre = _object(_json(data['preprocessor.json'], 'preprocessor.json'), 'preprocessor')
    columns = _text_list(pre.get('columns'), 'preprocessor.columns')
    if not columns or len(columns) != len(set(columns)):
        raise RegistrationError('Preprocessor columns must be nonempty and unique')
    expected_scope = f"{split['train'][0]} through {split['train'][1]} only"
    if pre.get('fit_scope') != expected_scope:
        raise RegistrationError('Preprocessor fit scope differs from the training period')
    medians = _object(pre.get('medians'), 'preprocessor.medians')
    if set(medians) != set(columns):
        raise RegistrationError('Preprocessor medians do not match its feature schema')
    for name, value in medians.items():
        _number(value, 'median.' + name)
    xm = _object(_json(data['xgb_model.json'], 'xgb_model.json'), 'XGBoost model')
    learner = _object(xm.get('learner'), 'XGBoost learner')
    if learner.get('feature_names') != columns:
        raise RegistrationError('XGBoost/preprocessor feature schema mismatch')
    try:
        lgb_names = [line.split('=', 1)[1].split() for line in
                     data['lgb_model.txt'].decode('utf-8').splitlines()
                     if line.startswith('feature_names=')]
    except UnicodeError as error:
        raise RegistrationError('Invalid LightGBM feature header') from error
    if lgb_names != [columns]:
        raise RegistrationError('LightGBM/preprocessor feature schema mismatch')
    known = meta.get('known_units')
    if not isinstance(known, list) or not known:
        raise RegistrationError('known_units must be nonempty')
    units = set()
    for item in known:
        item = _object(item, 'known unit')
        pair = (_plant(item.get('plant')), _unit(item.get('unit')))
        if pair in units:
            raise RegistrationError('Duplicate known unit')
        units.add(pair)
    xw = _number(meta.get('xgboost_weight'), 'xgboost_weight')
    lw = _number(meta.get('lightgbm_weight'), 'lightgbm_weight')
    if not (0 <= xw <= 1 and 0 <= lw <= 1 and math.isclose(xw + lw, 1, abs_tol=1e-12)):
        raise RegistrationError('Invalid ensemble weights')
    trials = _json(data['validation_weight_search.json'], 'validation_weight_search.json')
    if not isinstance(trials, list) or not trials:
        raise RegistrationError('Validation weight search must be nonempty')
    checked = []
    for trial in trials:
        trial = _object(trial, 'weight trial')
        weight = _number(trial.get('xgboost_weight'), 'trial weight')
        rmse = _number(trial.get('rmse'), 'trial RMSE')
        if not 0 <= weight <= 1 or rmse < 0:
            raise RegistrationError('Invalid validation weight trial')
        checked.append((weight, rmse))
    if not math.isclose(xw, min(checked, key=lambda t: t[1])[0], abs_tol=1e-12):
        raise RegistrationError('Saved weight differs from the validation-selected weight')
    _hash(meta.get('source_manifest_sha256'), 'source_manifest_sha256')
    model = {
        'model_version_id': _digest({name: records[name]['sha256'] for name in LOCK_FILES}),
        'trained_through': meta['trained_through'], 'created_at': meta['trained_at'],
        'weights': {'xgboost': xw, 'lightgbm': lw},
        'preprocessor_fit_scope': pre['fit_scope'], 'feature_count': len(columns),
        'selection_scope': meta['selection_scope'], 'evaluation_mode': meta['evaluation_mode'],
        'historical_deployment_verified': False,
        'independent_final_evaluation_verified': False,
    }
    return meta, split, target, units, model


def _case_payload(spec, data, records):
    meta, split, target, known, model = _model_info(data, records)
    if spec.key[:2] not in known:
        raise RegistrationError('Case plant/unit was not represented in model training')
    if not split['test'][0] <= spec.month <= split['test'][1]:
        raise RegistrationError('Case month must be in the saved test period')
    history = _csv_rows(data['history.csv'], 'history.csv',
                        {'plant', 'unit', 'month', 'target', 'quality_status', 'quality_reasons'})
    if any(k[2] > split['test'][1] for k in history):
        raise RegistrationError('Frozen history contains months beyond the evaluation period')
    for row in history.values():
        if row['target'] != '' and _csv_number(row['target'], 'history.target') < 0:
            raise RegistrationError('Concentration observations cannot be negative')
    observed_train = {k[:2] for k, r in history.items()
                      if split['train'][0] <= k[2] <= split['train'][1] and r['target'] != ''}
    if known != observed_train:
        raise RegistrationError('Known units differ from observed training history')
    required = {'plant', 'unit', 'month', 'target', 'supported', 'api_input_accepted',
                'nox_history_count', *('pred_' + n for n in METHODS[:3])}
    prediction_sets = {}
    for name, period in (('validation_predictions.csv', 'validation'), ('test_predictions.csv', 'test')):
        predictions = _csv_rows(data[name], name, required)
        for key, row in predictions.items():
            if not split[period][0] <= key[2] <= split[period][1]:
                raise RegistrationError(f'{name}: row outside its declared period')
            if key not in history:
                raise RegistrationError(f'{name}: prediction has no corresponding observation')
            observed = _csv_number(history[key]['target'], 'history.target')
            actual = _csv_number(row['target'], name + '.target')
            if actual < 0 or not math.isclose(observed, actual, rel_tol=1e-10, abs_tol=1e-10):
                raise RegistrationError(f'{name}: prediction target differs from frozen observation')
            if _csv_boolean(row['supported'], 'supported') != (key[:2] in known):
                raise RegistrationError(f'{name}: supported status differs from model cohort')
            _csv_boolean(row['api_input_accepted'], 'api_input_accepted')
            count = _csv_integer(row['nox_history_count'], 'nox_history_count', minimum=0)
            if count > 3:
                raise RegistrationError('nox_history_count cannot exceed three calendar months')
            serial = int(key[2][:4]) * 12 + int(key[2][5:]) - 1
            previous_keys = []
            for lag in (1, 2, 3):
                year, offset = divmod(serial - lag, 12)
                previous_keys.append((*key[:2], f'{year:04d}-{offset + 1:02d}'))
            observed_count = sum(k in history and history[k]['target'] != '' for k in previous_keys)
            if count != observed_count:
                raise RegistrationError('Stored history count differs from the three previous calendar months')
            for method in METHODS:
                col = 'pred_' + method
                if col in row and _csv_number(row[col], col) < 0:
                    raise RegistrationError('Concentration predictions cannot be negative')
        prediction_sets[name] = predictions
    if spec.key not in prediction_sets['test_predictions.csv']:
        raise RegistrationError('Case has no unique saved test prediction')
    row = history[spec.key]
    prediction = prediction_sets['test_predictions.csv'][spec.key]
    if not _csv_boolean(prediction['api_input_accepted'], 'api_input_accepted'):
        raise RegistrationError('Case input was rejected in the stored API validation')
    status = row['quality_status']
    if status not in QUALITY_STATES - {'unavailable'}:
        raise RegistrationError('Observed case needs an explicit available quality status')
    reasons = _text_list(_json(row['quality_reasons'], 'quality_reasons'), 'quality_reasons')
    snapshot = {
        'observed_target': _csv_number(row['target'], 'observed target'),
        'quality_status': status, 'quality_reasons': reasons,
        'fuel_basis': row.get('fuel_basis') or None,
        'saved_predictions': {n: _csv_number(prediction['pred_' + n], 'pred_' + n)
                              for n in METHODS if 'pred_' + n in prediction},
        'supported': True, 'stored_api_input_accepted': True,
        'observed_previous_months': _csv_integer(prediction['nox_history_count'], 'nox_history_count', 0),
        'observation_row_sha256': _digest(row), 'prediction_row_sha256': _digest(prediction),
    }
    payload = {
        'case': spec.as_dict(), 'analysis_mode': 'evaluation_replay',
        'bundle_version_id': _digest({n: r['sha256'] for n, r in records.items()}),
        'model': model, 'protocol': split, 'target': target,
        'target_definition_sha256': _digest(target),
        'target_verification_status': 'pending_provider_confirmation',
        'source_manifest': {'declared_sha256': meta['source_manifest_sha256'],
                            'original_manifest_file_verified': False},
        'snapshot': snapshot,
        'limitations': list(dict.fromkeys(target['limitations'] + [
            '과거 자료로 사후 재현한 평가이며 해당 월 당시 서비스의 예측 기록이 아닙니다.',
            '입력 자료의 당시 공개·입수 시점은 검증되지 않았습니다.',
            '등록은 파일과 관측·예측의 대응 검사이며 측정 유효성 인증이 아닙니다.',
        ])),
    }
    payload['case_version_id'] = _digest(payload)
    return payload


def _verified_bundle(path):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RegistrationError('Registered case manifest is unavailable') from error
    manifest = _object(_json(data, 'case manifest'), 'case manifest')
    if set(manifest) != MANIFEST_FIELDS or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise RegistrationError('Unsupported or incomplete case manifest schema')
    fields = _object(manifest['case'], 'case')
    if set(fields) != {'plant', 'unit', 'month'}:
        raise RegistrationError('Invalid case key fields')
    spec = CaseSpec(manifest['case_key'], **fields)
    if path.stem != spec.case_key:
        raise RegistrationError('Manifest filename differs from case_key')
    _timestamp(manifest['registered_at'], 'registered_at')
    source = _text(manifest['artifact_path'], 'artifact_path')
    if Path(source).is_absolute():
        raise RegistrationError('artifact_path must be relative to the case manifest')
    artifact, blobs, records = _read_bundle(path.parent / source)
    expected = _object(manifest['files'], 'files')
    if set(expected) != set(FILES):
        raise RegistrationError('Registered file set differs from the replay bundle')
    for name in FILES:
        record = _object(expected[name], name)
        if set(record) != {'sha256', 'size_bytes'} or type(record['size_bytes']) is not int:
            raise RegistrationError('Invalid registered file descriptor')
        _hash(record['sha256'], name)
        if record != records[name]:
            raise RegistrationError(f'Registered file checksum mismatch: {name}')
    payload = _case_payload(spec, blobs, records)
    if any(manifest.get(key) != value for key, value in payload.items()):
        raise RegistrationError('Registered case snapshot, version or protocol differs from its sources')
    _unchanged(artifact, records)
    return manifest, blobs


def _verify_path(path):
    return _verified_bundle(path)[0]


def _write_new(path, manifest):
    """Atomically create a private manifest without replacing any existing entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='case_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            output.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)  # Exclusive creation, also safe against a concurrent writer.
    finally:
        Path(temporary).unlink(missing_ok=True)


class CaseRegistry:
    def __init__(self, directory=DEFAULT_REGISTRY):
        self.directory = Path(directory).resolve()

    def path_for(self, case_key):
        return self.directory / (_case_key(case_key) + '.json')

    def verify(self, case_key):
        return _verify_path(self.path_for(case_key))

    def read_verified(self, case_key):
        """Return the manifest and exactly the bytes used to verify its sources.

        Consumers must parse these captured bytes rather than reopen source paths.
        The existing v1 manifest and its content version remain unchanged.
        """
        return _verified_bundle(self.path_for(case_key))

    def register(self, artifact_dir, *, case_key, plant, unit, month):
        spec = CaseSpec(case_key, plant, unit, month)
        artifact = Path(artifact_dir).resolve()
        path = self.path_for(case_key)
        if self.directory.is_relative_to(artifact):
            raise RegistrationError('Registry must be outside the frozen artifact directory')
        if path.exists():
            existing = self.verify(case_key)
            if existing['case'] != spec.as_dict() or (path.parent / existing['artifact_path']).resolve() != artifact:
                raise RegistrationError('case_key is already registered for a different case or bundle')
            return existing
        artifact, blobs, records = _read_bundle(artifact)
        payload = _case_payload(spec, blobs, records)
        manifest = {
            'schema_version': 1, 'case_key': case_key,
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'artifact_path': os.path.relpath(artifact, path.parent),
            'files': records, **payload,
        }
        _unchanged(artifact, records)
        try:
            _write_new(path, manifest)
        except FileExistsError:
            existing = self.verify(case_key)
            if (existing['case_version_id'] != manifest['case_version_id']
                    or (path.parent / existing['artifact_path']).resolve() != artifact):
                raise RegistrationError('Concurrent registration used a different case or bundle')
            return existing
        return self.verify(case_key)


def summary(manifest):
    """Safe CLI summary; source rows and numeric plant observations stay in the local manifest."""
    return {
        'case_key': manifest['case_key'], 'case': manifest['case'],
        'analysis_mode': manifest['analysis_mode'],
        'case_version_id': manifest['case_version_id'],
        'registered_file_count': len(manifest['files']),
        'protocol': manifest['protocol'], 'model_created_at': manifest['model']['created_at'],
        'quality_status': manifest['snapshot']['quality_status'],
        'target_verification_status': manifest['target_verification_status'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Register and verify frozen NOx evaluation cases')
    sub = parser.add_subparsers(dest='action', required=True)
    register = sub.add_parser('register')
    register.add_argument('--artifact-dir', type=Path, required=True)
    register.add_argument('--case-key', required=True)
    register.add_argument('--plant', required=True)
    register.add_argument('--unit', type=int, required=True)
    register.add_argument('--month', required=True)
    register.add_argument('--registry-dir', type=Path, default=DEFAULT_REGISTRY)
    verify = sub.add_parser('verify')
    verify.add_argument('--case-key', required=True)
    verify.add_argument('--registry-dir', type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(argv)
    try:
        registry = CaseRegistry(args.registry_dir)
        if args.action == 'register':
            manifest = registry.register(args.artifact_dir, case_key=args.case_key,
                                         plant=args.plant, unit=args.unit, month=args.month)
        else:
            manifest = registry.verify(args.case_key)
        print(json.dumps({'status': 'ok', 'manifest': str(registry.path_for(args.case_key)),
                          **summary(manifest)}, ensure_ascii=False, allow_nan=False))
    except (RegistrationError, OSError) as error:
        print(json.dumps({'status': 'blocked', 'error': str(error)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

