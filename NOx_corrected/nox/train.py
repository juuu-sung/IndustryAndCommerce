"""Time-ordered training. Preprocessing and weights never fit on test data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .data import build_dataset
from .features import build_features, Preprocessor
from .schema import KEYS, ROOT, TARGETS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    r2 = float(r2_score(y, pred)) if len(y) >= 2 and np.ptp(y) > 0 else None
    return {'n': len(y), 'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae': float(mean_absolute_error(y, pred)), 'r2': r2}


def blend(xlog, llog, weight):
    return np.maximum(0, np.expm1(weight * xlog + (1 - weight) * llog))


def choose_weight(y, xlog, llog):
    trials = []
    for w in np.linspace(0, 1, 101):
        trials.append({'xgboost_weight': float(w), **metrics(y, blend(xlog, llog, w))})
    best = min(trials, key=lambda row: row['rmse'])
    return best['xgboost_weight'], trials


def make_models():
    """Shared fixed settings for single-period and temporal-validation runs."""
    xm = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, reg_alpha=1., reg_lambda=2.,
        random_state=0, n_jobs=1, objective='reg:squarederror')
    lm = lgb.LGBMRegressor(n_estimators=300, max_depth=3, num_leaves=8,
        learning_rate=.05, subsample=.8, subsample_freq=1, colsample_bytree=.8,
        random_state=0, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
    return xm, lm


DEFAULT_SPLIT = {'train': ['2023-01', '2023-12'], 'validation': ['2024-01', '2024-06'],
                 'test': ['2024-07', '2024-12']}


def validate_split(protocol):
    from .history import month_key
    if set(protocol) != {'train','validation','test'}:
        raise ValueError('Exactly train/validation/test periods are required')
    previous_end = None
    for name in ('train','validation','test'):
        bounds = protocol[name]
        if len(bounds) != 2:
            raise ValueError('Each split requires two YYYY-MM bounds')
        start, end = map(month_key, bounds)
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError('Time splits must be ordered and nonoverlapping')
        previous_end = end


def split_masks(df, protocol=None):
    protocol = protocol or DEFAULT_SPLIT
    validate_split(protocol)
    observed = df['target'].notna()
    return {name: observed & df['month'].between(bounds[0]+'-01', bounds[1]+'-01')
            for name, bounds in protocol.items()}


def train(mode, output=None, *, raw_dir=None, protocol=None):
    protocol = protocol or DEFAULT_SPLIT
    validate_split(protocol)
    output = Path(output or ROOT / 'artifacts' / mode)
    if (output / 'metadata.json').exists():
        raise ValueError('Refusing to overwrite a frozen artifact; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    df, audit = build_dataset(mode, raw_dir)
    source_manifest = json.loads(((raw_dir or ROOT / 'data/raw') / 'manifest.json').read_text())
    if 'fuel_input_basis' in source_manifest:
        for item in source_manifest['files']:
            if hashlib.sha256(((raw_dir or ROOT/'data/raw') / item['file']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Source bundle checksum mismatch')
        for plant in source_manifest['fuel_input_basis']['reported_unit_plants']:
            if not df.loc[df.plant.eq(plant), 'fuel_basis'].eq('provider_reported_unit_month').all():
                raise ValueError('Reported unit fuel must cover every modeled month for the configured plant')
    from .history import validate_observations
    validate_observations(df, set(df[['plant','unit']].itertuples(index=False,name=None)))
    if (df['target'].dropna() < 0).any():
        raise ValueError('log1p target must be nonnegative')
    masks = split_masks(df, protocol)
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('All three time splits require observed targets')
    if 'operation_input_policy' in source_manifest:
        from .operating_patterns import require_patterns
        require_patterns(df.loc[masks['train'] | masks['validation'] | masks['test']], source_manifest['operation_input_policy'])
    X = build_features(df)
    pre = Preprocessor().fit(X.loc[masks['train']], fit_scope=f"{protocol['train'][0]} through {protocol['train'][1]} only")
    inputs = {s: pre.transform(X.loc[mask]) for s, mask in masks.items()}
    y = {s: df.loc[mask, 'target'] for s, mask in masks.items()}
    # Explicit thread and random-state controls for reproducible runs.
    xm, lm = make_models()
    xm.fit(inputs['train'], np.log1p(y['train']))
    lm.fit(inputs['train'], np.log1p(y['train']))
    xv, lv = xm.predict(inputs['validation']), lm.predict(inputs['validation'])
    weight, trials = choose_weight(y['validation'], xv, lv)

    # Save frozen inference artifacts BEFORE asking either model for test predictions.
    xm.save_model(output / 'xgb_model.json')
    lm.booster_.save_model(str(output / 'lgb_model.txt'))
    write_json(output / 'preprocessor.json', pre.to_dict())
    metadata = {
        'schema_version': 1, 'target_mode': mode, 'target': TARGETS[mode],
        'xgboost_weight': weight, 'lightgbm_weight': 1 - weight,
        'blend_space': 'log1p target; expm1 after blending; nonnegative clamp',
        'split': protocol,
        'trained_through': protocol['train'][1],
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'selection_scope': 'validation only; no refit with validation/test',
        'evaluation_mode': 'monthly rolling origin using already observed previous-month targets and realized current-month conditions; not a 12-month forecast',
        'weights_locked_before_test': True,
        'known_units': df.loc[masks['train'], ['plant', 'unit']].drop_duplicates().to_dict('records'),
        'feature_ranges': {c: {'min': float(inputs['train'][c].min()), 'max': float(inputs['train'][c].max())}
                           for c in pre.columns},
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'flask']},
        'source_manifest_sha256': hashlib.sha256(((raw_dir or ROOT / 'data/raw') / 'manifest.json').read_bytes()).hexdigest(),
        'bundle_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ['xgb_model.json', 'lgb_model.txt', 'preprocessor.json']},
    }
    if 'bootstrap_observation_month' in source_manifest:
        metadata['history_bootstrap_through'] = source_manifest['bootstrap_observation_month']
    if 'fuel_input_basis' in source_manifest:
        metadata['fuel_input_basis'] = source_manifest['fuel_input_basis']
    if 'operation_input_policy' in source_manifest:
        metadata['operation_input_policy'] = source_manifest['operation_input_policy']
    write_json(output / 'metadata.json', metadata)
    df.to_csv(output / 'history.csv', index=False, date_format='%Y-%m-%d')
    write_json(output / 'data_audit.json', audit)

    group_mean = df.loc[masks['train']].groupby(['plant', 'unit'])['target'].mean()
    global_median = float(y['train'].median())
    report = {'target': TARGETS[mode], 'protocol': metadata['split'], 'weight': weight,
              'feature_count': len(pre.columns), 'data_audit': audit,
              'evaluation_mode': metadata['evaluation_mode'], 'scores': {}, 'test_by_plant': {}}
    for split in ('validation', 'test'):
        subset = df.loc[masks[split]].copy()
        a, b = xm.predict(inputs[split]), lm.predict(inputs[split])
        fallback = pd.Series([group_mean.get((p, u), global_median)
                              for p, u in subset[['plant', 'unit']].itertuples(index=False, name=None)],
                              index=subset.index)
        prediction = {
            'train_global_median': np.full(len(subset), global_median),
            'train_unit_mean': fallback.to_numpy(),
            'last_month': X.loc[masks[split], 'nox_lag1'].fillna(fallback).to_numpy(),
            'previous_3months': X.loc[masks[split], 'nox_roll3'].fillna(fallback).to_numpy(),
            'xgboost': blend(a, b, 1.), 'lightgbm': blend(a, b, 0.),
            'ensemble': blend(a, b, weight),
        }
        report['scores'][split] = {name: metrics(y[split], pred) for name, pred in prediction.items()}
        for name, pred in prediction.items():
            subset[f'pred_{name}'] = pred
        subset[KEYS + ['target'] + [f'pred_{n}' for n in prediction]].to_csv(
            output / f'{split}_predictions.csv', index=False, date_format='%Y-%m-%d')
        if split == 'test':
            for plant, group in subset.groupby('plant'):
                report['test_by_plant'][plant] = metrics(group['target'], group['pred_ensemble'])
    write_json(output / 'validation_weight_search.json', trials)
    write_json(output / 'metrics.json', report)
    print(json.dumps({'mode': mode, 'weight': weight, 'rows': {k: int(v.sum()) for k, v in masks.items()},
                      'test': report['scores']['test']}, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['legacy', 'concentration', 'both'], default='both')
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-start', default='2023-01')
    parser.add_argument('--train-end', default='2023-12')
    parser.add_argument('--validation-start', default='2024-01')
    parser.add_argument('--validation-end', default='2024-06')
    parser.add_argument('--test-start', default='2024-07')
    parser.add_argument('--test-end', default='2024-12')
    args = parser.parse_args()
    protocol = {name:[getattr(args, name+'_start'),getattr(args, name+'_end')]
                for name in ('train','validation','test')}
    if args.target=='both' and args.output:
        parser.error('--output requires a single target mode')
    for mode in ('legacy', 'concentration') if args.target == 'both' else (args.target,):
        train(mode, args.output, raw_dir=args.raw_dir, protocol=protocol)
