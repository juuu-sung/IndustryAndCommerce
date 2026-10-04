"""One fixed feature experiment compared with frozen A/B/C unit-fuel results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .operating_patterns import FEATURES, POLICY
from .schema import KEYS, ROOT
from .temporal_validation import (FOLDS, METHODS, describe_fold, prepare_data,
    safe_metrics, sha256, train_fold, verify_saved_predictions)
from .train import write_json


def compare(before, after):
    joined = before.merge(after, on=['fold', *KEYS], suffixes=('_before', '_after'), validate='one_to_one')
    if len(joined) != len(before) or len(joined) != len(after):
        raise ValueError('Evaluation keys changed')
    for col in ['target', 'supported', 'api_input_accepted', *['pred_' + m for m in METHODS[3:]]]:
        np.testing.assert_allclose(joined[col + '_before'], joined[col + '_after'], rtol=0, atol=1e-12)
    summaries = []
    for scope, selected in [('supported', joined.loc[joined.supported_after]),
                            ('unseen_diagnostic', joined.loc[~joined.supported_after])]:
        for fold, period in [('pooled', selected), *list(selected.groupby('fold'))]:
            for plant, group in [('all', period), *list(period.groupby('plant'))]:
                for method in METHODS:
                    b = safe_metrics(group.target_after, group['pred_' + method + '_before'])
                    a = safe_metrics(group.target_after, group['pred_' + method + '_after'])
                    summaries.append({'scope': scope, 'fold': fold, 'plant': plant, 'method': method,
                        'n': len(group), 'before_rmse': b['rmse'], 'after_rmse': a['rmse'],
                        'before_mae': b['mae'], 'after_mae': a['mae'],
                        'rmse_improvement_pct': 100 * (1 - a['rmse'] / b['rmse']) if b['rmse'] else None,
                        'mae_improvement_pct': 100 * (1 - a['mae'] / b['mae']) if b['mae'] else None})
    return joined, summaries


def run(output, raw_dir, baseline):
    if output.exists():
        raise ValueError('Refusing to overwrite frozen experiment')
    if not (baseline / 'COMPLETE').exists():
        raise ValueError('Complete frozen temporal baseline required')
    frozen = {str(p.relative_to(baseline)): sha256(p) for p in baseline.rglob('*') if p.is_file()}
    df, manifest, source = prepare_data(raw_dir)
    if manifest.get('operation_input_policy') != POLICY:
        raise ValueError('Configured operating-feature bundle required')
    base_df, _, _ = prepare_data(ROOT / 'data/unit_fuel_20261001')
    pd.testing.assert_frame_equal(df[base_df.columns], base_df)
    output.mkdir(parents=True)
    write_json(output / 'experiment_lock.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source': source,
        'baseline_directory': str(baseline), 'baseline_file_sha256': frozen,
        'folds': FOLDS, 'features': FEATURES, 'operation_policy': POLICY,
        'model_parameters': 'unchanged shared make_models(); no search',
        'target_changed': False, 'test_periods_excluded_from_feature_selection': True,
        'availability': {name: describe_fold(df, p) for name, p in FOLDS.items()},
        'not_deployed': True, '2026_evaluation': 'excluded; not used to select features or weights'})
    original, extended, fold_results = [], [], {}
    for name, protocol in FOLDS.items():
        folder = output / 'models' / name
        score = train_fold(df, manifest, source, protocol, folder)
        score['inference_verification'] = verify_saved_predictions(folder)
        fold_results[name] = score
        for collection, path in [(original, baseline / 'models/unit_actual' / name), (extended, folder)]:
            rows = pd.read_csv(path / 'test_predictions.csv', parse_dates=['month'])
            rows['fold'] = name
            collection.append(rows)
        print('completed fold', name, 'weight', score['weight'], flush=True)
    original, extended = pd.concat(original, ignore_index=True), pd.concat(extended, ignore_index=True)
    joined, comparisons = compare(original, extended)
    joined.to_csv(output / 'paired_predictions.csv', index=False, date_format='%Y-%m-%d')
    pd.DataFrame(comparisons).to_csv(output / 'comparison.csv', index=False)
    write_json(output / 'summary.json', {'source': source, 'folds': fold_results, 'comparisons': comparisons,
        'pooled_supported_rows': int(extended.supported.sum()),
        'pooled_unseen_rows': int((~extended.supported).sum()), 'not_deployed': True})
    for rel, expected in frozen.items():
        if sha256(baseline / rel) != expected:
            raise ValueError('Frozen baseline changed')
    (output / 'COMPLETE').write_text('Fixed operation-feature experiment complete; not deployed.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/operations_validation_20261002')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/operations_20261002')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/temporal_validation_20261001')
    parser.add_argument('--max-training-seconds', type=float, default=300.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.max_training_seconds) or args.max_training_seconds <= 0:
        parser.error('Training time limit must be finite and positive')
    if args.output.exists():
        parser.error('Choose a new output; existing experiments cannot be overwritten')
    if args.dry_run:
        df, _, _ = prepare_data(args.raw_dir)
        print(json.dumps({n: describe_fold(df, p) for n, p in FOLDS.items()}, ensure_ascii=False, indent=2))
        return
    if args.worker:
        run(args.output, args.raw_dir, args.baseline)
        return
    command = [sys.executable, '-m', 'nox.operations_validation', '--worker', '--output', str(args.output),
               '--raw-dir', str(args.raw_dir), '--baseline', str(args.baseline)]
    started = time.perf_counter()
    try:
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=args.max_training_seconds)
        status, code, console = ('success' if child.returncode == 0 else 'failed'), child.returncode, child.stdout + child.stderr
    except subprocess.TimeoutExpired as error:
        status, code = 'timeout', 2
        console = ''.join(v.decode() if isinstance(v, bytes) else v or '' for v in [error.stdout, error.stderr])
    seconds = time.perf_counter() - started
    if args.output.exists():
        (args.output / 'training_console.txt').write_text(console)
        write_json(args.output / 'timing.json', {'seconds': seconds, 'status': status, 'returncode': code,
            'limit_seconds': args.max_training_seconds, 'command': command,
            'scope': 'entire worker: three folds, scoring, model reload and per-row API verification'})
        if code:
            (args.output / 'COMPLETE').unlink(missing_ok=True)
            (args.output / 'INCOMPLETE').write_text(status)
    print(console, end='')
    print(json.dumps({'status': status, 'seconds': seconds, 'output': str(args.output)}, ensure_ascii=False))
    if code:
        parser.exit(code, 'Experiment incomplete; inspect logs.\n')


if __name__ == '__main__':
    main()
