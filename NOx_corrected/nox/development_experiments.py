"""Fixed development ablations and target-transform comparisons, never deployed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from .data import read
from .features import build_features, Preprocessor
from .operating_patterns import FEATURES as HOURLY
from .quality import inspect_daily
from .schema import FUELS, KEYS, ROOT
from .temporal_validation import (FOLDS, EVALUATION_MODE, baseline_predictions,
    describe_fold, fold_masks, prepare_data, safe_metrics, sha256)
from .train import make_models, write_json

HISTORY = ['nox_lag1', 'nox_roll3', 'nox_history_count', 'nox_lag1_missing']
FUEL = [c + suffix for c in FUELS.values() for suffix in ['', '_change', '_per_mwh']]
WEATHER = ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'wind_sin', 'wind_cos']
PATTERNS = [*HOURLY, 'operation_pattern_available']
GROUPS = {
    'full': [],
    'without_nox_history': HISTORY,
    'without_fuel': FUEL,
    'without_weather': WEATHER,
    'without_hourly_patterns': PATTERNS,
    # Fuel allocation and per-MWh ratios contain generation information. Remove
    # that whole family together; capacity remains a static unit descriptor.
    'without_current_operations': [*FUEL, *PATTERNS, 'generation_mwh',
                                   'thermal_efficiency_pct', 'utilization_pct'],
}
TRANSFORMS = ['log1p', 'ppm']
METHODS = ['xgboost', 'lightgbm', 'ensemble', 'last_month',
           'previous_3months', 'train_unit_mean', 'train_global_median']
LABELS = {'full': '전체 입력', 'without_nox_history': 'NOx 이력 제외',
          'without_fuel': '연료 및 연료 파생변수 제외', 'without_weather': '기상 제외',
          'without_hourly_patterns': '시간별 발전 패턴 제외',
          'without_current_operations': '현재 월 운전·연료·시간 패턴 제외'}


def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso', force_ascii=False))


def inverse(values, transform):
    if transform not in TRANSFORMS:
        raise ValueError('Unknown target transform')
    values = np.asarray(values, dtype=float)
    values = np.expm1(values) if transform == 'log1p' else values
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')
    return np.maximum(values, 0)


def blend_native(a, b, weight, transform):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Invalid blend weight')
    return inverse(weight * a + (1 - weight) * b, transform)


def select_weight(y, a, b, transform):
    trials = [{'xgboost_weight': float(w),
               **safe_metrics(y, blend_native(a, b, w, transform))}
              for w in np.linspace(0, 1, 101)]
    if not len(y):
        raise ValueError('No selection observations')
    return min(trials, key=lambda r: r['rmse'])['xgboost_weight'], trials


def candidate_features(X, variant):
    if variant not in GROUPS:
        raise ValueError('Unknown ablation variant')
    return X.drop(columns=[c for c in GROUPS[variant] if c in X])


def fit_candidate(df, protocol, variant, transform, output, source=None):
    """Fit/select/freeze only. This function never predicts outer score rows."""
    if transform not in TRANSFORMS:
        raise ValueError('Unknown target transform')
    if output.exists():
        raise ValueError('Refusing to overwrite a candidate')
    masks, supported, known = fold_masks(df, protocol)
    full_X = build_features(df)
    X = candidate_features(full_X, variant)
    pre = Preprocessor().fit(X.loc[masks['train']],
        fit_scope=f"{protocol['train'][0]} through {protocol['train'][1]} only")
    y = df.loc[masks['train'], 'target'].to_numpy()
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training target')
    train_y = np.log1p(y) if transform == 'log1p' else y
    xm, lm = make_models()
    xm.fit(pre.transform(X.loc[masks['train']]), train_y)
    lm.fit(pre.transform(X.loc[masks['train']]), train_y)
    selected = masks['validation'] & supported
    val_X = pre.transform(X.loc[selected])
    weight, trials = select_weight(df.loc[selected, 'target'],
        xm.predict(val_X), lm.predict(val_X), transform)
    output.mkdir(parents=True)
    xm.save_model(output / 'xgb_model.json')
    lm.booster_.save_model(str(output / 'lgb_model.txt'))
    write_json(output / 'preprocessor.json', pre.to_dict())
    write_json(output / 'selection_trials.json', trials)
    artifact_hashes = {n: sha256(output / n) for n in
                      ['xgb_model.json', 'lgb_model.txt', 'preprocessor.json', 'selection_trials.json']}
    # Intentionally no metadata.json: the service Predictor must not interpret
    # ppm-space experimental artifacts as its existing log-space service bundle.
    write_json(output / 'experiment.json', {
        'schema': 'development_experiment_v1', 'variant': variant, 'transform': transform,
        'target_unit': 'ppm', 'blend_space': transform, 'weight': weight,
        'protocol': protocol, 'dropped_columns': [c for c in GROUPS[variant] if c in full_X],
        'model_columns': pre.columns, 'known_units': [[p, int(u)] for p, u in sorted(known)],
        'selection_scope': 'supported validation observations only',
        'selection_rows': int(selected.sum()), 'artifact_sha256': artifact_hashes,
        'source': source, 'not_deployed': True, 'evaluation_mode': EVALUATION_MODE,
        'outer_predictions_created': False,
    })
    return output


def predict_saved(output, X):
    meta = json.loads((output / 'experiment.json').read_text())
    if meta.get('schema') != 'development_experiment_v1':
        raise ValueError('Unsupported experiment schema')
    for name, digest in meta['artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Experiment artifact changed: ' + name)
    pre = Preprocessor.from_dict(json.loads((output / 'preprocessor.json').read_text()))
    xm = xgb.XGBRegressor(n_jobs=1)
    xm.load_model(output / 'xgb_model.json')
    lm = lgb.Booster(model_file=str(output / 'lgb_model.txt'))
    transformed = pre.transform(X)
    a, b = xm.predict(transformed), lm.predict(transformed, num_threads=1)
    return {'xgboost': inverse(a, meta['transform']), 'lightgbm': inverse(b, meta['transform']),
            'ensemble': blend_native(a, b, meta['weight'], meta['transform'])}


def score_candidate(df, protocol, output):
    meta = json.loads((output / 'experiment.json').read_text())
    masks, supported, _ = fold_masks(df, protocol)
    full_X = build_features(df)
    X = candidate_features(full_X, meta['variant'])
    combined = []
    training = df.loc[masks['train']]
    q90 = training.groupby(['plant', 'unit']).target.quantile(.9)
    for split in ['validation', 'test']:
        mask = masks[split]
        rows = df.loc[mask, KEYS + ['target']].copy()
        rows['supported'] = supported.loc[mask]
        rows['high_target_threshold_ppm'] = [float(q90.get((p, u), training.target.quantile(.9)))
            for p, u in rows[['plant', 'unit']].itertuples(index=False, name=None)]
        rows['high_observed_target'] = rows.target > rows.high_target_threshold_ppm
        # Comparators intentionally keep their original NOx inputs, even when
        # those columns are removed from the candidate model.
        predictions, flags = baseline_predictions(df, full_X, masks['train'], mask)
        predictions.update(predict_saved(output, X.loc[mask]))
        for method, values in predictions.items():
            rows['pred_' + method] = values
        for name, values in flags.items():
            rows[name] = values
        rows['split'], rows['variant'], rows['transform'] = split, meta['variant'], meta['transform']
        combined.append(rows)
    return pd.concat(combined, ignore_index=True)


def score_rows(rows):
    result = []
    for (variant, transform, fold, split), period in rows.groupby(['variant', 'transform', 'fold', 'split']):
        for scope, known in [('supported', period.loc[period.supported]),
                             ('unseen_diagnostic', period.loc[~period.supported])]:
            for plant, group in [('all', known), *list(known.groupby('plant'))]:
                for band, selected in [('all', group), ('high_observed_target', group.loc[group.high_observed_target])]:
                    for method in METHODS:
                        pred = selected['pred_' + method]
                        result.append({'variant': variant, 'transform': transform, 'fold': fold,
                            'split': split, 'scope': scope, 'plant': plant, 'band': band, 'method': method,
                            **safe_metrics(selected.target, pred),
                            'bias_ppm': float((pred - selected.target).mean()) if len(selected) else None})
    return pd.DataFrame(result)


def daily_stats(days, training_days):
    values = days['nox_numeric']
    threshold = float(training_days.nox_numeric.quantile(.95)) if len(training_days) else None
    total = float(values.sum())
    return {'observed_days': len(days), 'zero_days': int(values.eq(0).sum()),
            'minimum_ppm': float(values.min()), 'median_ppm': float(values.median()),
            'maximum_ppm': float(values.max()), 'mean_ppm': float(values.mean()),
            'top5_days_share_of_daily_value_sum': float(values.nlargest(5).sum() / total) if total else None,
            'training_daily_q95_ppm': threshold,
            'days_above_training_daily_q95': int(values.gt(threshold).sum()) if threshold is not None else None}


def analyze_errors(df, raw_dir, baseline, output):
    """Describe fixed existing errors; no target correction or sample exclusion."""
    paired = pd.read_csv(baseline / 'paired_predictions.csv', parse_dates=['month'])
    selected = paired.loc[paired.supported_after].copy()
    selected['error_ppm'] = selected.pred_ensemble_after - selected.target_after
    selected['squared_error'] = selected.error_ppm ** 2
    selected = selected.sort_values('squared_error', ascending=False)
    sse = selected.squared_error.sum()
    contributions = selected.groupby('plant').agg(n=('error_ppm', 'size'),
        squared_error=('squared_error', 'sum'), bias_ppm=('error_ppm', 'mean'))
    contributions['squared_error_share'] = contributions.squared_error / sse
    contributions.to_csv(output / 'error_contribution_by_plant.csv')
    daily = inspect_daily(read('emissions_daily.csv', raw_dir))
    daily = daily.loc[daily.date.lt(pd.Timestamp('2026-01-01')) & daily.included_in_target]
    # Match the target's A/B arithmetic daily aggregation before describing days.
    days = daily.groupby(['plant', 'unit', 'date'], as_index=False).nox_numeric.mean()
    case_records, top_daily = [], []
    X = build_features(df)
    for case_id, (_, row) in enumerate(selected.head(10).iterrows(), 1):
        month, plant, unit = row.month, row.plant, int(row.unit)
        unit_days = days.loc[days.plant.eq(plant) & days.unit.eq(unit)]
        current = unit_days.loc[unit_days.date.dt.to_period('M').eq(month.to_period('M'))]
        protocol = FOLDS[row.fold]
        historical = unit_days.loc[unit_days.date.between(pd.Timestamp(protocol['train'][0]+'-01'),
                                  pd.Period(protocol['train'][1], 'M').end_time)]
        stats = daily_stats(current, historical)
        if not np.isclose(stats['mean_ppm'], row.target_after, rtol=0, atol=1e-9):
            raise ValueError('Daily target reconstruction mismatch')
        idx = df.index[df.plant.eq(plant) & df.unit.eq(unit) & df.month.eq(month)].item()
        observed = df.loc[idx]
        meta = json.loads((baseline / 'models' / row.fold / 'metadata.json').read_text())
        pre = Preprocessor.from_dict(json.loads((baseline / 'models' / row.fold / 'preprocessor.json').read_text()))
        filled = pre.transform(X.loc[[idx]])
        outside = [c for c, limits in meta['feature_ranges'].items()
                   if not limits['min'] <= filled[c].iloc[0] <= limits['max']]
        record = {'case_id': case_id, 'plant': plant, 'unit': unit, 'month': month.strftime('%Y-%m'),
            'fold': row.fold, 'observed_ppm': float(row.target_after),
            'predicted_ppm': float(row.pred_ensemble_after), 'error_ppm': float(row.error_ppm),
            'squared_error_share': float(row.squared_error / sse), **stats,
            'calendar_day_coverage': len(current) / month.days_in_month,
            'quality_status': observed.quality_status, 'quality_reasons': json.loads(observed.quality_reasons),
            'imputed_features': [c for c in pre.columns if pd.isna(X.at[idx, c])],
            'outside_training_range': outside,
            'generation_mwh': float(observed.generation_mwh), 'fuel_basis': observed.fuel_basis,
            'weather_coverage': None if pd.isna(observed.weather_coverage) else float(observed.weather_coverage),
            'nox_lag1': None if pd.isna(X.at[idx, 'nox_lag1']) else float(X.at[idx, 'nox_lag1']),
            'nox_roll3': None if pd.isna(X.at[idx, 'nox_roll3']) else float(X.at[idx, 'nox_roll3']),
            'recent_observed_months': int(X.at[idx, 'nox_history_count']),
            'current_conditions': records(df.loc[[idx], KEYS + [c for c in FUELS.values()]])[0]}
        case_records.append(record)
        subset = daily.loc[daily.plant.eq(plant) & daily.unit.eq(unit)
                          & daily.month.eq(month)].copy()
        subset.insert(0, 'case_id', case_id)
        top_daily.append(subset)
    pd.concat(top_daily, ignore_index=True).to_csv(output / 'large_error_source_days.csv', index=False)
    write_json(output / 'large_error_cases.json', case_records)
    write_json(output / 'error_analysis.json', {
        'source': str(baseline), 'observations': len(selected),
        'top_squared_error_shares': {str(k): float(selected.head(k).squared_error.sum()/sse) for k in [3,5,10]},
        'by_plant': records(contributions.reset_index()), 'cases': case_records,
        'notes': ['Diagnostic only: extreme values remain in all model scores.',
                  'Shares refer to squared prediction errors or sums of daily concentration values, not mass.',
                  'Unknown source validity is not confirmed sensor correctness.']})
    return case_records


def verify_reference(rows, baseline):
    ref = pd.read_csv(baseline / 'paired_predictions.csv', parse_dates=['month'])
    current = rows.loc[rows.variant.eq('full') & rows['transform'].eq('log1p') & rows.split.eq('test')]
    joined = current.merge(ref, on=['fold', *KEYS], validate='one_to_one')
    if len(joined) != len(ref) or len(current) != len(ref):
        raise ValueError('Reference evaluation keys changed')
    for method in METHODS:
        np.testing.assert_allclose(joined['pred_'+method], joined['pred_'+method+'_after'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(joined.target, joined.target_after, rtol=0, atol=1e-12)
    return {'rows': len(joined), 'supported_rows': int(current.supported.sum()),
            'unseen_rows': int((~current.supported).sum()), 'reference_reproduced': True}


def run(output, raw_dir, baseline):
    if output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    started = time.perf_counter()
    frozen = {r['file']: r['sha256'] for r in json.loads((ROOT/'delivery_manifest.json').read_text())['files']}
    frozen['delivery_manifest.json'] = sha256(ROOT/'delivery_manifest.json')
    for rel, digest in frozen.items():
        if sha256(ROOT / rel) != digest:
            raise ValueError('Prior delivery changed before experiment: ' + rel)
    df, manifest, source = prepare_data(raw_dir)
    output.mkdir(parents=True)
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': source,
        'folds': FOLDS, 'variants': GROUPS, 'transforms': TRANSFORMS, 'candidate_folds': 36,
        'model_settings': 'shared make_models(), unchanged; fixed seed and one thread',
        'blend': 'weighted native training space; inverse transform then nonnegative clamp',
        'selection': '101 weights, validation RMSE on known units only',
        'target_unchanged': True, '2026_excluded': True, 'not_deployed': True,
        'interpretation': 'Previously inspected development periods; not a new independent final test. '
                          'Single-model comparisons isolate transform changes; ensemble comparisons also change blend space.'}
    write_json(output / 'experiment_plan.json', plan)
    cases = analyze_errors(df, raw_dir, baseline, output)
    model_dirs, locks = [], {}
    # Freeze every candidate before reading any newly computed outer scores.
    training_started = time.perf_counter()
    for variant in GROUPS:
        for transform in TRANSFORMS:
            for fold, protocol in FOLDS.items():
                folder = output / 'models' / variant / transform / fold
                fit_candidate(df, protocol, variant, transform, folder, source)
                model_dirs.append((fold, protocol, folder))
                locks[str(folder.relative_to(output))] = {
                    p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
                print('frozen', variant, transform, fold, flush=True)
    fit_seconds = time.perf_counter() - training_started
    write_json(output / 'all_candidates_lock.json', locks)
    predictions = []
    for fold, protocol, folder in model_dirs:
        period = score_candidate(df, protocol, folder)
        period['fold'] = fold
        period.to_csv(folder/'predictions.csv', index=False, date_format='%Y-%m-%d')
        predictions.append(period)
    all_rows = pd.concat(predictions, ignore_index=True)
    ref = verify_reference(all_rows, baseline)
    all_rows.to_csv(output/'all_predictions.csv', index=False, date_format='%Y-%m-%d')
    # Pool individual observations, never average fold-level RMSEs.
    pooled = all_rows.copy(); pooled['fold'] = 'pooled'
    scores = score_rows(pd.concat([all_rows, pooled], ignore_index=True))
    scores.to_csv(output/'scores.csv', index=False)
    for rel, files in locks.items():
        for filename, digest in files.items():
            if sha256(output/rel/filename) != digest:
                raise ValueError('Candidate changed during evaluation')
    for rel, digest in frozen.items():
        if sha256(ROOT/rel) != digest:
            raise ValueError('Prior delivery changed during experiment: '+rel)
    write_json(output/'verification.json', {**ref, 'fit_seconds': fit_seconds,
        'total_seconds': time.perf_counter()-started, 'frozen_prior_files_verified': len(frozen),
        'all_candidates_locked_before_outer_scoring': True, 'models_reloaded_for_all_scores': True,
        'target_corrected_or_outliers_removed': False, 'service_models_changed': False,
        'candidate_folds': len(model_dirs), 'not_deployed': True, 'cases': len(cases)})
    (output/'COMPLETE').write_text('Development experiments complete; not deployed.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/pre_request_experiments_20261002')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/operations_20261002')
    parser.add_argument('--baseline', type=Path, default=ROOT/'reports/operations_validation_20261002')
    parser.add_argument('--max-seconds', type=float, default=300.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Time limit must be finite and positive')
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.dry_run:
        df, _, _ = prepare_data(args.raw_dir)
        print(json.dumps({'candidate_folds': len(GROUPS)*len(TRANSFORMS)*len(FOLDS),
            'folds': {n: describe_fold(df,p) for n,p in FOLDS.items()}}, ensure_ascii=False, indent=2))
        return
    if args.worker:
        run(args.output, args.raw_dir, args.baseline)
        return
    command = [sys.executable,'-m','nox.development_experiments','--worker',
               '--output',str(args.output),'--raw-dir',str(args.raw_dir),'--baseline',str(args.baseline)]
    started = time.perf_counter()
    try:
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=args.max_seconds)
        code, console = child.returncode, child.stdout+child.stderr
    except subprocess.TimeoutExpired as error:
        code = 2
        console = ''.join(v.decode() if isinstance(v,bytes) else v or '' for v in [error.stdout,error.stderr])
    if args.output.exists():
        (args.output/'console.txt').write_text(console)
        write_json(args.output/'timing.json', {'seconds':time.perf_counter()-started,
            'limit_seconds':args.max_seconds,'returncode':code,'command':command})
        if code:
            (args.output/'COMPLETE').unlink(missing_ok=True)
            (args.output/'INCOMPLETE').write_text('Check console; no results are final.\n')
    print(console,end='')
    if code:
        parser.exit(code,'Development experiment incomplete.\n')


if __name__ == '__main__':
    main()
