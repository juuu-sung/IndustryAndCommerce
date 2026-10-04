"""Compare fixed model families only after an evidence-backed target replay.

This is development evaluation on already inspected periods. It never replaces
service models. The parent enforces a runtime budget for the training worker.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .candidate_selection import (fit_trial, saved_predict, prediction_rows,
    seed_mean, nominate, score_rows)
from .features import build_features
from .schema import KEYS, ROOT
from .target_validation import fresh_output, load_verified_version
from .temporal_validation import FOLDS, prepare_data, fold_masks, sha256
from .train import write_json

CONFIGS = [
    {'id': 'full_log_xgb', 'drop_weather': False, 'transform': 'log1p', 'method': 'xgboost'},
    {'id': 'full_log_lgb', 'drop_weather': False, 'transform': 'log1p', 'method': 'lightgbm'},
    {'id': 'full_log_ensemble', 'drop_weather': False, 'transform': 'log1p', 'method': 'ensemble'},
    {'id': 'no_weather_log_lgb', 'drop_weather': True, 'transform': 'log1p', 'method': 'lightgbm'}]


def validated_data(raw, version):
    target, metadata = load_verified_version(version)
    df, _, source = prepare_data(raw)
    target = target.loc[target.month.le(df.month.max())].copy()
    if set([*KEYS, 'target', 'quality_status', 'target_rule_id']) - set(target):
        raise ValueError('Incomplete versioned monthly target schema')
    if target.duplicated(KEYS).any() or target[KEYS].isna().any().any():
        raise ValueError('Duplicate or missing monthly target keys')
    if not target.month.eq(target.month.dt.to_period('M').dt.to_timestamp()).all():
        raise ValueError('Monthly target keys must be first-of-month dates')
    observed = target.target.notna()
    if not np.isfinite(target.loc[observed, 'target']).all() or (target.loc[observed, 'target'] < 0).any():
        raise ValueError('Invalid verified monthly concentrations')
    if not target.loc[observed, 'quality_status'].eq('verified_research').all():
        raise ValueError('Unverified monthly target remains')
    if not target.target_rule_id.eq(metadata['rule_id']).all():
        raise ValueError('Mixed target rule versions')
    expected = df.loc[df.target.notna(), KEYS]
    check = expected.merge(target[KEYS], on=KEYS, how='left', indicator=True, validate='one_to_one')
    missing = check.loc[check._merge.eq('left_only'), KEYS]
    if len(missing):
        raise ValueError(f'Confirmed targets cover only a subset; {len(missing)} observed development months still missing')
    extras = target[KEYS].merge(df[KEYS], on=KEYS, how='left', indicator=True, validate='one_to_one')
    if not extras._merge.eq('both').all():
        raise ValueError('Confirmed target has keys outside the operating-data cohort')
    original = df[KEYS + ['target']].rename(columns={'target': 'original_research_target'})
    df = df.drop(columns=['target', 'quality_status'], errors='ignore').merge(
        target[KEYS + ['target', 'quality_status', 'target_rule_id']], on=KEYS,
        how='left', validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    differences = original.merge(target[KEYS + ['target']], on=KEYS, how='outer', validate='one_to_one')
    differences['target_change_ppm'] = differences.target - differences.original_research_target
    source.update(target_version_directory=str(Path(version).resolve()),
        target_version_sha256=sha256(Path(version) / 'target_version.json'),
        target_rule_id=metadata['rule_id'], research_target_replaced_for_this_experiment_only=True,
        initial_artifacts_and_runtime_db_unchanged=True)
    return df, source, differences


def run(raw, version, output, seeds=(0, 1, 2, 3, 4), folds=None):
    if not seeds or len(seeds) != len(set(seeds)) or any(type(v) is not int or not 0 <= v < 2**32 for v in seeds):
        raise ValueError('Unique nonnegative seeds required')
    df, source, differences = validated_data(raw, version)
    folds = folds or FOLDS
    for period in folds.values():
        fold_masks(df, period)
    output = fresh_output(output)
    started = time.perf_counter()
    write_json(output / 'source.json', source)
    write_json(output / 'protocol.json', {'schema': 'nox_verified_target_comparison_v1',
        'folds': folds, 'seeds': list(seeds), 'configs': CONFIGS,
        'evaluation_mode': 'retrospective_development_only_realized_monthly_conditions',
        'selection': 'Supported validation MAE then RMSE with existing plant guards',
        'service_replaced': False, 'independent_final_evaluation': False})
    differences.to_csv(output / 'target_changes.csv', index=False)
    df.to_csv(output / 'dataset_snapshot.csv', index=False)
    validation, trials, files, fitted = [], [], {}, 0.
    for candidate in CONFIGS:
        for fold, period in folds.items():
            for seed in seeds:
                path = output / 'models' / candidate['id'] / fold / str(seed)
                rows, seconds = fit_trial(df, period, candidate, 'values_only', seed, path)
                rows['candidate'], rows['input_policy'], rows['fold'], rows['seed'] = candidate['id'], 'values_only', fold, seed
                validation.append(rows)
                trials.append((path, candidate, fold, period, seed, rows))
                fitted += seconds
                for file in path.iterdir():
                    files[str(file.relative_to(output))] = sha256(file)
            print(f'frozen {candidate["id"]}/{fold}', flush=True)
    val = pd.concat(validation, ignore_index=True)
    write_json(output / 'nomination.json', nominate(val))
    for name in ['nomination.json', 'protocol.json', 'source.json', 'dataset_snapshot.csv']:
        files[name] = sha256(output / name)
    write_json(output / 'selection_lock.json', {'files': files, 'outer_predictions_created': False})
    load_verified_version(version)
    if sha256(Path(version) / 'target_version.json') != source['target_version_sha256']:
        raise ValueError('Target version changed during training')
    if sha256(Path(raw) / 'manifest.json') != source['manifest_sha256']:
        raise ValueError('Operating source manifest changed during training')
    X = build_features(df)
    outer, max_reload_error = [], 0.
    for path, candidate, fold, period, seed, prior in trials:
        masks, supported, _ = fold_masks(df, period)
        selected = masks['validation'] & supported
        reloaded = saved_predict(path, X.loc[selected])
        difference = float(np.max(abs(reloaded - prior.prediction.to_numpy())))
        max_reload_error = max(max_reload_error, difference)
        if not np.allclose(reloaded, prior.prediction, rtol=0, atol=1e-8):
            raise ValueError('Frozen model reload changed validation predictions')
        rows = prediction_rows(df, period, masks['test'], supported,
                               saved_predict(path, X.loc[masks['test']]))
        rows['candidate'], rows['input_policy'], rows['fold'], rows['seed'], rows['split'] = candidate['id'], 'values_only', fold, seed, 'test'
        outer.append(rows)
    if any(sha256(output / file) != digest for file, digest in files.items()):
        raise ValueError('Frozen selection/model files changed during evaluation')
    rows = pd.concat([val, *outer], ignore_index=True)
    means = seed_mean(rows)
    rows.to_csv(output / 'seed_predictions.csv', index=False)
    means.to_csv(output / 'mean_predictions.csv', index=False)
    score_rows(means).to_csv(output / 'scores.csv', index=False)
    # Explicitly verify every candidate has exactly the same outcomes/population.
    reference = None
    for _, group in means.groupby(['candidate', 'input_policy']):
        current = group.sort_values(['fold', 'split', *KEYS])[['fold', 'split', *KEYS, 'target', 'supported']].reset_index(drop=True)
        if reference is None:
            reference = current
        else:
            pd.testing.assert_frame_equal(reference, current)
    result = {'schema': 'nox_verified_target_training_result_v1',
        'target_rule_id': source['target_rule_id'], 'trial_count': len(trials),
        'model_fit_count': len(folds) * len(seeds) * 5, 'fitted_seconds': fitted,
        'worker_seconds': time.perf_counter() - started, 'saved_reload_max_abs_error': max_reload_error,
        'same_target_and_cohort_all_candidates': True, 'selection_frozen_before_outer': True,
        'service_replaced': False, 'independent_final_evaluation': False}
    write_json(output / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/readiness_bundle_20261002')
    parser.add_argument('--target-version', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--deadline-seconds', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.deadline_seconds <= 0:
            raise ValueError('Positive training deadline required')
        if not args.worker:
            # Reject an unverified/partial version before allocating any worker.
            validated_data(args.raw_dir, args.target_version)
            if args.output.exists():
                raise ValueError('Choose a new training output directory')
            command = [sys.executable, '-m', 'nox.validated_training', *sys.argv[1:], '--worker']
            try:
                completed = subprocess.run(command, timeout=args.deadline_seconds, cwd=ROOT)
                raise SystemExit(completed.returncode)
            except subprocess.TimeoutExpired:
                parser.exit(124, 'Training budget exceeded; partial output preserved. '
                            'Rerun in VS Code with a NEW --output and --deadline-seconds 3600\n')
        print(json.dumps(run(args.raw_dir, args.target_version, args.output, tuple(args.seeds)), ensure_ascii=False))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, 'Blocked: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
