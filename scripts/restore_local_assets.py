"""Attach existing local NOx assets; never downloads or uploads data."""
import argparse
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'NOx_corrected'
DIRECTORIES = [
    'examples',
    'data/raw', 'data/retrieved/20261001', 'data/extended_20261001',
    'data/unit_fuel_20261001', 'data/operations_20261002', 'data/readiness_bundle_20261002',
    'artifacts/concentration', 'artifacts/legacy', 'artifacts/concentration_20261001',
    'artifacts/concentration_unit_fuel_20261001',
    'reports/temporal_validation_20261001', 'reports/operations_validation_20261002',
    'reports/pre_request_experiments_20261002', 'reports/readiness_20261002',
    'reports/retraining', 'reports/unit_fuel_retraining', 'reports/target_validation_20261003',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


def restore(source, dry_run=False):
    source = source.expanduser().resolve()
    if source == DESTINATION.resolve():
        raise ValueError('Source must be the existing project, not this destination')
    if not (source / 'data/raw/manifest.json').is_file():
        raise ValueError('Source must contain the existing NOx_corrected data and model assets')
    pending, identical, missing = [], 0, []
    for relative in DIRECTORIES:
        directory = source / relative
        if not directory.is_dir():
            missing.append(relative)
            continue
        for original in sorted(directory.rglob('*')):
            # Code/docs, credentials, runtime DB and bulk experiment models stay untouched.
            if (not original.is_file() or '__pycache__' in original.parts
                    or original.name.startswith('.') or original.suffix in {'.md', '.py', '.pyc', '.zip'}
                    or '.sqlite' in original.name or 'provider_private' in original.parts):
                continue
            rel = original.relative_to(source)
            if rel == Path('examples/request.json'):
                continue  # Keep the intentionally synthetic public request.
            if 'models' in rel.parts and any(x in rel.parts for x in
                    ['readiness_20261002', 'pre_request_experiments_20261002']):
                continue
            target = DESTINATION / rel
            if target.exists():
                if not target.is_file() or digest(target) != digest(original):
                    raise ValueError(f'Refusing to overwrite a different asset: {rel}')
                identical += 1
            else:
                pending.append((original, target))
    if not dry_run:
        for original, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    print(f'Local assets: {len(pending)} to copy, {identical} already identical; dry_run={dry_run}')
    if missing:
        print('Optional snapshots unavailable: ' + ', '.join(missing))
    print('Data/models/results are ignored by Git; no network request was made.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        restore(args.source, args.dry_run)
    except ValueError as error:
        parser.exit(2, str(error) + '\n')
