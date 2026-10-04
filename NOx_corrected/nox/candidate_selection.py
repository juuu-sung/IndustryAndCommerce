"""Four preregistered candidates; validation-only nomination and locked scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .development_experiments import WEATHER, inverse, blend_native, select_weight
from .features import build_features, Preprocessor
from .schema import KEYS, ROOT
from .temporal_validation import prepare_data, fold_masks, baseline_predictions, safe_metrics, sha256
from .train import make_models, write_json
from .weather_quality import QUALITY_FEATURES


def validate_protocol(protocol):
    expected = [
        {'id': 'full_log_lgb', 'drop_weather': False, 'transform': 'log1p', 'method': 'lightgbm'},
        {'id': 'no_weather_log_lgb', 'drop_weather': True, 'transform': 'log1p', 'method': 'lightgbm'},
        {'id': 'full_ppm_xgb', 'drop_weather': False, 'transform': 'ppm', 'method': 'xgboost'},
        {'id': 'full_ppm_ensemble', 'drop_weather': False, 'transform': 'ppm', 'method': 'ensemble'}]
    if protocol.get('schema') != 'nox_readiness_protocol_v1' or protocol.get('candidate_configs') != expected:
        raise ValueError('Unknown selection protocol/candidate definitions')
    selection = protocol.get('selection', {})
    guards = {'overall_rmse': 'not greater than baseline', 'overall_mae': 'not greater than baseline',
              'plant_mae': 'not greater than baseline + max(0.2 ppm, 10% of baseline MAE)'}
    if selection.get('primary') != 'MAE' or selection.get('secondary') != 'RMSE' or selection.get('guards') != guards:
        raise ValueError('Selection criteria differ from the supported frozen protocol')
    if protocol.get('input_policies') != ['values_only', 'weather_quality']:
        raise ValueError('Both registered input policies are required')
    seeds = protocol.get('seeds', [])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Unique nonnegative integer seeds are required')


def select_features(X, candidate, input_policy):
    if input_policy not in ('values_only', 'weather_quality'):
        raise ValueError('Unknown input policy')
    drops = list(QUALITY_FEATURES) if input_policy == 'values_only' else []
    if candidate['drop_weather']:
        drops += [*WEATHER, *QUALITY_FEATURES]
    return X.drop(columns=[c for c in set(drops) if c in X])


def fit_trial(df, period, candidate, policy, seed, output):
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen trial')
    masks, supported, known = fold_masks(df, period)
    X = select_features(build_features(df), candidate, policy)
    pre = Preprocessor().fit(X.loc[masks['train']], fit_scope=f"{period['train'][0]} through {period['train'][1]} only")
    target = df.loc[masks['train'], 'target'].to_numpy()
    train_y = np.log1p(target) if candidate['transform'] == 'log1p' else target
    xm, lm = make_models()
    models = {'xgboost': xm, 'lightgbm': lm}
    names = ['xgboost', 'lightgbm'] if candidate['method'] == 'ensemble' else [candidate['method']]
    selected = masks['validation'] & supported
    val_X = pre.transform(X.loc[selected])
    native = {}
    fitted_seconds = 0.
    for name in names:
        models[name].set_params(random_state=seed)
        started = time.perf_counter()
        models[name].fit(pre.transform(X.loc[masks['train']]), train_y)
        fitted_seconds += time.perf_counter() - started
        native[name] = models[name].predict(val_X)
    if candidate['method'] == 'ensemble':
        weight, trials = select_weight(df.loc[selected, 'target'], native['xgboost'], native['lightgbm'], candidate['transform'])
        prediction = blend_native(native['xgboost'], native['lightgbm'], weight, candidate['transform'])
    else:
        weight, trials = (1. if names == ['xgboost'] else 0.), []
        prediction = inverse(native[names[0]], candidate['transform'])
    output.mkdir(parents=True)
    if 'xgboost' in names:
        xm.save_model(output / 'xgb_model.json')
    if 'lightgbm' in names:
        lm.booster_.save_model(str(output / 'lgb_model.txt'))
    write_json(output / 'preprocessor.json', pre.to_dict())
    write_json(output / 'weight_trials.json', trials)
    hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    metadata = {'schema': 'nox_candidate_selection_v1', 'candidate': candidate, 'input_policy': policy,
                'seed': seed, 'period': period, 'weight': weight, 'files': hashes,
                'known_units': [[p, int(u)] for p, u in sorted(known)], 'fitted_seconds': fitted_seconds,
                'transform': candidate['transform'], 'not_deployed': True,
                'selection_rows': int(selected.sum()), 'selection_scope': 'supported validation only'}
    write_json(output / 'candidate.json', metadata)
    rows = prediction_rows(df, period, selected, supported, prediction)
    rows['split'] = 'validation'
    return rows, fitted_seconds


def saved_predict(output, X):
    metadata = json.loads((output / 'candidate.json').read_text())
    if metadata['schema'] != 'nox_candidate_selection_v1':
        raise ValueError('Unknown candidate schema')
    for name, expected in metadata['files'].items():
        if sha256(output / name) != expected:
            raise ValueError('Frozen candidate changed: ' + name)
    pre = Preprocessor.from_dict(json.loads((output / 'preprocessor.json').read_text()))
    X = pre.transform(select_features(X, metadata['candidate'], metadata['input_policy']))
    native = {}
    if 'xgb_model.json' in metadata['files']:
        model = xgb.XGBRegressor(n_jobs=1)
        model.load_model(output / 'xgb_model.json')
        native['xgboost'] = model.predict(X)
    if 'lgb_model.txt' in metadata['files']:
        model = lgb.Booster(model_file=str(output / 'lgb_model.txt'))
        native['lightgbm'] = model.predict(X, num_threads=1)
    if metadata['candidate']['method'] == 'ensemble':
        return blend_native(native['xgboost'], native['lightgbm'], metadata['weight'], metadata['transform'])
    return inverse(native[metadata['candidate']['method']], metadata['transform'])


def prediction_rows(df, period, mask, supported, prediction):
    masks, _, _ = fold_masks(df, period)
    full_X = build_features(df)
    rows = df.loc[mask, KEYS + ['target']].copy()
    rows['supported'] = supported.loc[mask]
    rows['prediction'] = prediction
    rows['weather_missing'] = df.loc[mask, 'temperature_c'].isna()
    train = df.loc[masks['train']]
    q90 = train.groupby(['plant', 'unit']).target.quantile(.9)
    rows['high_target'] = [y > q90.get((p, u), train.target.quantile(.9)) for p, u, y in
        rows[['plant', 'unit', 'target']].itertuples(index=False, name=None)]
    baselines, _ = baseline_predictions(df, full_X, masks['train'], mask)
    rows['previous_month'] = baselines['last_month']
    return rows


def seed_mean(rows):
    keys = ['candidate', 'input_policy', 'fold', 'split', *KEYS]
    for _, group in rows.groupby(keys, dropna=False):
        if group.target.nunique(dropna=False) != 1 or group.supported.nunique() != 1:
            raise ValueError('Seed target/support keys differ')
    return rows.groupby(keys, as_index=False).agg(target=('target', 'first'), prediction=('prediction', 'mean'),
        previous_month=('previous_month', 'first'), supported=('supported', 'first'),
        weather_missing=('weather_missing', 'first'), high_target=('high_target', 'first'))


def nominate(rows):
    """Never access outer targets, including when a caller passes both splits."""
    validation = seed_mean(rows.loc[rows['split'].eq('validation') & rows.supported].copy())
    groups = {key: group for key, group in validation.groupby(['candidate', 'input_policy'])}
    baseline_key = ('full_log_lgb', 'values_only')
    if baseline_key not in groups:
        raise ValueError('Missing validation comparator')
    baseline = groups[baseline_key]
    baseline_metrics = safe_metrics(baseline.target, baseline.prediction)
    baseline_plants = {p: safe_metrics(g.target, g.prediction) for p, g in baseline.groupby('plant')}
    results = []
    for (candidate, policy), group in groups.items():
        left = group.sort_values(['fold', *KEYS])
        right = baseline.sort_values(['fold', *KEYS])
        if left[['fold', *KEYS]].to_records(index=False).tolist() != right[['fold', *KEYS]].to_records(index=False).tolist() or not np.array_equal(left.target.to_numpy(), right.target.to_numpy()):
            raise ValueError('Candidate validation population differs')
        metrics = safe_metrics(group.target, group.prediction)
        reasons = []
        for metric in ['rmse', 'mae']:
            if metrics[metric] > baseline_metrics[metric] + 1e-12:
                reasons.append('overall_' + metric)
        for plant, subset in group.groupby('plant'):
            reference = baseline_plants[plant]['mae']
            if safe_metrics(subset.target, subset.prediction)['mae'] > reference + max(.2, .1 * reference) + 1e-12:
                reasons.append('plant_mae:' + plant)
        results.append({'candidate': candidate, 'input_policy': policy, **metrics,
                        'eligible': not reasons, 'guard_failures': reasons})
    eligible = [r for r in results if r['eligible']]
    choice = min(eligible, key=lambda r: (r['mae'], r['rmse'], r['candidate'], r['input_policy']))
    return {'nominee': choice, 'validation_comparison': results,
            'criteria': 'MAE then RMSE; overall and plant guards; validation only; seed-mean predictions',
            'not_deployed': True, 'independent_final_evaluation': False}


def score_rows(rows):
    scores = []
    for (candidate, policy, split), period in rows.groupby(['candidate', 'input_policy', 'split']):
        for scope, known in [('supported', period.loc[period.supported]), ('unseen', period.loc[~period.supported])]:
            for plant, group in [('all', known), *list(known.groupby('plant'))]:
                for band, subset in [('all', group), ('high_target', group.loc[group.high_target]),
                                     ('missing_weather', group.loc[group.weather_missing])]:
                    score = safe_metrics(subset.target, subset.prediction)
                    errors = subset.prediction - subset.target
                    scores.append({'candidate': candidate, 'input_policy': policy, 'split': split,
                                   'scope': scope, 'plant': plant, 'band': band, **score,
                                   'bias_ppm': float(errors.mean()) if len(errors) else None,
                                   'underprediction_mae_ppm': float((-errors).clip(lower=0).mean()) if len(errors) else None})
    return pd.DataFrame(scores)


def run(raw, protocol_path, output):
    if output.exists():
        raise ValueError('Refusing to overwrite candidate results')
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    protocol_hash = sha256(protocol_path)
    df, manifest, source = prepare_data(raw)
    if set(QUALITY_FEATURES) - set(df):
        raise ValueError('Registered weather quality comparison requires source metadata')
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'source.json', source)
    started, fitted, trials, validation, locks = time.perf_counter(), 0., [], [], {}
    fit_count = 0
    for policy in protocol['input_policies']:
        for candidate in protocol['candidate_configs']:
            for fold, period in protocol['folds'].items():
                for seed in protocol['seeds']:
                    path = output / 'models' / policy / candidate['id'] / fold / str(seed)
                    rows, seconds = fit_trial(df, period, candidate, policy, seed, path)
                    rows['candidate'], rows['input_policy'], rows['fold'], rows['seed'] = candidate['id'], policy, fold, seed
                    validation.append(rows)
                    trials.append((path, fold, period, candidate['id'], policy, seed))
                    fitted += seconds
                    fit_count += 2 if candidate['method'] == 'ensemble' else 1
                    for file in path.iterdir():
                        locks[str(file.relative_to(output))] = sha256(file)
                print(f'frozen {policy}/{candidate["id"]}/{fold}', flush=True)
    val = pd.concat(validation, ignore_index=True)
    # The candidate choice is serialized and locked before ANY outer prediction.
    nomination = nominate(val)
    write_json(output / 'nomination.json', nomination)
    locks['nomination.json'] = sha256(output / 'nomination.json')
    locks['protocol.json'] = sha256(output / 'protocol.json')
    locks['source.json'] = sha256(output / 'source.json')
    write_json(output / 'selection_lock.json', {'files': locks, 'outer_predictions_created': False})
    evaluated, max_difference = [], 0.
    full_X = build_features(df)
    for path, fold, period, candidate, policy, seed in trials:
        masks, supported, _ = fold_masks(df, period)
        mask = masks['validation'] & supported
        predictions = saved_predict(path, full_X.loc[mask])
        reference = val.loc[val.candidate.eq(candidate) & val.input_policy.eq(policy) & val.fold.eq(fold) & val.seed.eq(seed)]
        difference = float(np.max(np.abs(predictions - reference.prediction.to_numpy())))
        max_difference = max(max_difference, difference)
        if difference > 1e-10:
            raise ValueError('Saved/reloaded validation prediction differs')
        mask = masks['test']
        rows = prediction_rows(df, period, mask, supported, saved_predict(path, full_X.loc[mask]))
        rows['split'], rows['candidate'], rows['input_policy'], rows['fold'], rows['seed'] = 'test', candidate, policy, fold, seed
        evaluated.append(rows)
    all_rows = pd.concat([val, *evaluated], ignore_index=True)
    all_rows.to_csv(output / 'seed_predictions.csv', index=False, date_format='%Y-%m-%d')
    means = seed_mean(all_rows)
    means.to_csv(output / 'seed_mean_predictions.csv', index=False, date_format='%Y-%m-%d')
    score_rows(means).to_csv(output / 'scores.csv', index=False)
    per_seed = []
    for (candidate, policy, seed, split), group in all_rows.loc[all_rows.supported].groupby(['candidate', 'input_policy', 'seed', 'split']):
        per_seed.append({'candidate': candidate, 'input_policy': policy, 'seed': int(seed), 'split': split,
                         **safe_metrics(group.target, group.prediction)})
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(output / 'metrics_per_seed.csv', index=False)
    per_seed.groupby(['candidate', 'input_policy', 'split'])[['mae', 'rmse']].agg(['mean', 'std', 'min', 'max']).to_csv(output / 'seed_variability.csv')
    if protocol_hash != sha256(protocol_path) or any(sha256(output / path) != digest for path, digest in locks.items()):
        raise ValueError('Protocol or frozen selection changed during scoring')
    verification = {'protocol_sha256': protocol_hash, 'candidate_trials': len(trials), 'model_fit_count': fit_count,
                    'fitted_seconds': fitted, 'worker_seconds': time.perf_counter() - started,
                    'saved_validation_max_abs_difference': max_difference,
                    'prediction_rows': len(all_rows), 'seed_mean_rows': len(means),
                    'frozen_files_unchanged': True, 'nomination_created_before_outer_predictions': True,
                    'service_changed': False, 'independent_final_evaluation': False}
    write_json(output / 'verification.json', verification)
    (output / 'COMPLETE').write_text('completed; development evaluation only\n')
    print(json.dumps({'verification': verification, 'nominee': nomination['nominee']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/readiness_bundle_20261002')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'reports/readiness_20261002/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/readiness_20261002/candidates')
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite candidate output')
    if not 1 <= args.max_seconds <= 300:
        raise ValueError('Execution deadline must be between 1 and 300 seconds')
    if args.dry_run:
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol)
        count = len(protocol['candidate_configs']) * len(protocol['input_policies']) * len(protocol['folds']) * len(protocol['seeds'])
        print(json.dumps({'candidate_trials': count, 'max_seconds': args.max_seconds,
                          'new_final_holdout_used': False}, ensure_ascii=False))
        return
    if args.worker:
        run(args.raw, args.protocol, args.output)
        return
    command = [sys.executable, '-m', 'nox.candidate_selection', '--raw', str(args.raw),
               '--protocol', str(args.protocol), '--output', str(args.output), '--worker']
    try:
        subprocess.run(command, check=True, timeout=args.max_seconds)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'INCOMPLETE').write_text('Execution failed or deadline exceeded; do not use incomplete results\n')
        raise


if __name__ == '__main__':
    main()
