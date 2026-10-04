"""Freeze source/test evidence, preserve prior bundles, and verify every ZIP entry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

from .operating_patterns import digest
from .schema import ROOT
from .train import write_json

CHANGES = {'README.md', 'nox/data.py', 'nox/features.py', 'nox/history.py',
           'nox/llm.py', 'nox/predict.py', 'nox/temporal_validation.py', 'nox/train.py'}


def main():
    report = ROOT / 'reports/operations_validation_20261002'
    archive = ROOT.parent / 'NOx_operations_validation_20261002.zip'
    if archive.exists():
        raise ValueError('Refusing to overwrite a delivery archive')
    prior_path = report / 'previous_delivery_manifest.json'
    if not prior_path.exists():
        shutil.copy2(ROOT / 'delivery_manifest.json', prior_path)
    prior = json.loads(prior_path.read_text())
    changed = [r['file'] for r in prior['files'] if digest(ROOT / r['file']) != r['sha256']]
    if set(changed) != CHANGES:
        raise ValueError('Unexpected changes to prior delivery: ' + str(changed))
    source = ROOT / 'data/hourly_operations_20261002'
    prep = json.loads((source / 'preparation_summary.json').read_text())
    refetch = json.loads((source / 'refetch_verification.json').read_text())
    if prep['unit_months'] != 72 or prep['observed_hours'] != 52608 or prep['missing_hours'] != 0 or prep['quarantined_unit_months'] != 8:
        raise ValueError('Unexpected source quality counts')
    if refetch['monthly_files'] != 8 or not all(r['numeric_values_identical'] for r in refetch['results']):
        raise ValueError('Refetch verification incomplete')
    for path in (source / 'sources').glob('*.provenance.json'):
        receipt = json.loads(path.read_text())
        raw = path.with_name(path.name.removesuffix('.provenance.json'))
        if digest(raw) != receipt['sha256']:
            raise ValueError('Source receipt mismatch')
    suite = ET.parse(report / 'tests.xml').getroot().find('testsuite')
    if suite is None or any(int(suite.get(k, '0')) for k in ['failures', 'errors', 'skipped']) or int(suite.get('tests')) != 131:
        raise ValueError('Regression tests are incomplete')
    http = json.loads((report / 'http_verification.json').read_text())
    if http['status'] != 'passed' or http['predict_status'] != 200:
        raise ValueError('HTTP verification incomplete')
    summary = json.loads((report / 'summary.json').read_text())
    verification = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_delivery_files': len(prior['files']), 'preserved_previous_files': len(prior['files']) - len(changed),
        'intentional_code_changes': changed, 'old_raw_targets_models_and_sqlite_unchanged': True,
        'source_month_files': 36, 'source_unit_months': 72, 'source_hour_values': 52608,
        'quarantined_feature_months': 8, 'refetch_months': 8,
        'test_cases_passed': int(suite.get('tests')), 'test_failures': 0,
        'supported_outer_rows': summary['pooled_supported_rows'], 'unseen_outer_rows': summary['pooled_unseen_rows'],
        'loaded_model_rows': sum(r['inference_verification']['loaded_model_rows'] for r in summary['folds'].values()),
        'api_prediction_rows': sum(r['inference_verification']['api_prediction_rows'] for r in summary['folds'].values()),
        'api_rejected_rows': sum(r['inference_verification']['api_rejected_rows'] for r in summary['folds'].values()),
        'actual_http_verified': True, 'gemini_live_call': False,
        'actual_scr_time_series_obtained': False, 'mass_target_created': False, 'not_deployed': True,
        'source_quality_refetch_does_not_prove_sensor_validity': True}
    write_json(report / 'final_verification.json', verification)
    old_names = {r['file'] for r in prior['files']}
    excluded = prior.get('excluded_intermediate_artifacts', [])
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == 'delivery_manifest.json' or '__pycache__' in path.parts:
            continue
        if any(part.startswith('.') for part in path.relative_to(ROOT).parts) and rel not in old_names:
            continue
        if any(rel == p or rel.startswith(p + '/') for p in excluded):
            continue
        files.append({'file': rel, 'sha256': digest(path), 'bytes': path.stat().st_size})
    write_json(ROOT / 'delivery_manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'previous_delivery_manifest_sha256': digest(prior_path),
        'intentional_previous_file_changes': sorted(CHANGES),
        'excluded_intermediate_artifacts': excluded, 'files': files})
    entries = [r['file'] for r in files] + ['delivery_manifest.json']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for rel in entries:
            bundle.write(ROOT / rel, ROOT.name + '/' + rel)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != len(entries):
            raise ValueError('ZIP structure/checksum verification failed')
        import hashlib
        for rel in entries:
            if hashlib.sha256(bundle.read(ROOT.name + '/' + rel)).hexdigest() != digest(ROOT / rel):
                raise ValueError('Archive entry differs from local source: ' + rel)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
                      'manifest_files': len(files), 'zip_entries': len(entries), 'verification': verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
