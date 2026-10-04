"""Fixed calendar-month development validation; no tuning on outer score periods."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .data import build_dataset
from .features import build_features, Preprocessor
from .history import validate_observations
from .schema import CONDITIONS, KEYS, ROOT, TARGETS
from .train import blend, choose_weight, make_models, metrics, split_masks, write_json


FOLDS = {
    'A': {'train': ['2023-01', '2023-12'], 'validation': ['2024-01', '2024-06'],
          'test': ['2024-07', '2024-12']},
    'B': {'train': ['2023-01', '2024-06'], 'validation': ['2024-07', '2024-12'],
          'test': ['2025-01', '2025-06']},
    'C': {'train': ['2023-01', '2024-12'], 'validation': ['2025-01', '2025-06'],
          'test': ['2025-07', '2025-12']},
}
CUTOFF = '2025-12'
METHODS = ['xgboost', 'lightgbm', 'ensemble', 'last_month',
           'previous_3months', 'train_unit_mean', 'train_global_median']
EVALUATION_MODE = (
    'monthly rolling evaluation with observed earlier targets and realized current-month '
    'conditions; publication latency unverified; not a six-month or advance forecast'
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_settings(model):
    # XGBoost's default missing=np.nan is a sentinel, not a JSON number.
    return {k: 'NaN (missing-value sentinel)' if isinstance(v, float) and np.isnan(v) else v
            for k, v in model.get_params().items()}


def prepare_data(raw_dir):
    raw_dir = Path(raw_dir)
    manifest = json.loads((raw_dir / 'manifest.json').read_text())
    for item in manifest['files']:
        if sha256(raw_dir / item['file']) != item['sha256']:
            raise ValueError(f"Source bundle checksum mismatch: {item['file']}")
    full, audit = build_dataset('concentration', raw_dir)
    # Do this BEFORE constructing features, selecting columns, or splitting targets.
    df = full.loc[full.month <= pd.Timestamp(CUTOFF + '-01')].sort_values(KEYS).reset_index(drop=True)
    validate_observations(df, set(df[['plant', 'unit']].itertuples(index=False, name=None)))
    if (df.target.dropna() < 0).any():
        raise ValueError('log1p target must be nonnegative')
    if 'fuel_input_basis' in manifest:
        for plant in manifest['fuel_input_basis']['reported_unit_plants']:
            if not df.loc[df.plant.eq(plant), 'fuel_basis'].eq('provider_reported_unit_month').all():
                raise ValueError('Reported unit fuel must cover every development month')
    if 'operation_input_policy' in manifest:
        from .operating_patterns import require_patterns
        require_patterns(df, manifest['operation_input_policy'])
    source = {'raw_dir': str(raw_dir.resolve()), 'manifest_sha256': sha256(raw_dir / 'manifest.json'),
              'file_sha256': {r['file']: r['sha256'] for r in manifest['files']},
              'source_audit': audit, 'excluded_after': CUTOFF,
              'excluded_future_rows': int(len(full) - len(df)),
              'development_rows': len(df), 'observed_development_rows': int(df.target.notna().sum())}
    return df, manifest, source


def fold_masks(df, protocol):
    masks = split_masks(df, protocol)
    if any(not m.any() for m in masks.values()):
        raise ValueError('Each split requires observed targets')
    # Calendar-month membership is independent of row order, unit, and missing targets.
    month_sets = {s: set(df.loc[m, 'month']) for s, m in masks.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if month_sets[a] & month_sets[b]:
            raise ValueError('A calendar month cannot cross splits')
    known = set(df.loc[masks['train'], ['plant', 'unit']].itertuples(index=False, name=None))
    supported = pd.Series([(p, int(u)) in known for p, u in
                           df[['plant', 'unit']].itertuples(index=False, name=None)], index=df.index)
    if not (masks['validation'] & supported).any():
        raise ValueError('Weight selection requires supported validation units')
    return masks, supported, known


def safe_metrics(y, pred):
    if len(y) == 0:
        return {'n': 0, 'rmse': None, 'mae': None, 'r2': None}
    return metrics(y, pred)


def score_table(rows):
    return {name: safe_metrics(rows.target, rows['pred_' + name]) for name in METHODS}


def scoped_scores(rows):
    return {'supported': score_table(rows.loc[rows.supported]),
            'unseen_diagnostic': score_table(rows.loc[~rows.supported]),
            'all_diagnostic': score_table(rows)}


def baseline_predictions(df, X, training_mask, score_mask):
    training = df.loc[training_mask]
    unit_mean = training.groupby(['plant', 'unit']).target.mean()
    median = float(training.target.median())
    subset = df.loc[score_mask]
    fallback = pd.Series([unit_mean.get((p, u), median) for p, u in
                          subset[['plant', 'unit']].itertuples(index=False, name=None)], index=subset.index)
    last, recent = X.loc[score_mask, 'nox_lag1'], X.loc[score_mask, 'nox_roll3']
    return {'train_global_median': np.full(len(subset), median), 'train_unit_mean': fallback.to_numpy(),
            'last_month': last.fillna(fallback).to_numpy(),
            'previous_3months': recent.fillna(fallback).to_numpy()}, {
                'last_month_fallback': last.isna().to_numpy(),
                'previous_3months_fallback': recent.isna().to_numpy(),
            }


def request_for(row):
    out = {c: row[c] for c in ['plant', 'unit', *CONDITIONS] if c not in ('wind_sin', 'wind_cos')}
    out['unit'] = int(row.unit)
    out['month'] = row.month.strftime('%Y-%m')
    out['wind_direction_deg'] = float(np.degrees(np.arctan2(row.wind_sin, row.wind_cos)) % 360)
    if 'fuel_basis' in row:
        out['fuel_basis'] = row.fuel_basis
    out = {k: None if pd.isna(v) else float(v) if isinstance(v, np.floating) else v for k, v in out.items()}
    from .operating_patterns import FEATURES
    if 'gen_hour_coverage' in row and row.plant == '영동':
        out['operation_pattern'] = {c: float(row[c]) for c in FEATURES} if pd.notna(row.gen_hour_coverage) else None
    return out


def api_rejection(row, known, operation_policy=None):
    from .predict import validate_model_input
    try:
        validate_model_input(request_for(row), known, operation_policy)
        return ''
    except ValueError as error:
        return str(error)


def describe_fold(df, protocol):
    masks, supported, known = fold_masks(df, protocol)
    return {'protocol': protocol, 'known_unit_count': len(known),
            'rows': {s: int(m.sum()) for s, m in masks.items()},
            'supported_rows': {s: int((m & supported).sum()) for s, m in masks.items()},
            'unseen_rows': {s: int((m & ~supported).sum()) for s, m in masks.items()},
            'unseen_units': df.loc[(masks['validation'] | masks['test']) & ~supported,
                                   ['plant', 'unit']].drop_duplicates().to_dict('records'),
            'months': {s: sorted(df.loc[m, 'month'].dt.strftime('%Y-%m').unique()) for s, m in masks.items()}}


def train_fold(df, manifest, source, protocol, output):
    started = time.perf_counter()
    output.mkdir(parents=True)
    # Each fold only retains history through its own outer-period end.
    df = df.loc[df.month <= pd.Timestamp(protocol['test'][1] + '-01')].reset_index(drop=True)
    masks, supported, known = fold_masks(df, protocol)
    X = build_features(df)
    pre = Preprocessor().fit(X.loc[masks['train']], fit_scope=f"{protocol['train'][0]} through {protocol['train'][1]} only")
    train_X = pre.transform(X.loc[masks['train']])
    xm, lm = make_models()
    xm.fit(train_X, np.log1p(df.loc[masks['train'], 'target']))
    lm.fit(train_X, np.log1p(df.loc[masks['train'], 'target']))
    selection_mask = masks['validation'] & supported
    selection_X = pre.transform(X.loc[selection_mask])
    weight, trials = choose_weight(df.loc[selection_mask, 'target'],
                                  xm.predict(selection_X), lm.predict(selection_X))
    # Freeze models, preprocessing, and weight BEFORE any outer-period predictions.
    xm.save_model(output / 'xgb_model.json')
    lm.booster_.save_model(str(output / 'lgb_model.txt'))
    write_json(output / 'preprocessor.json', pre.to_dict())
    metadata = {
        'schema_version': 1, 'target_mode': 'concentration', 'target': TARGETS['concentration'],
        'xgboost_weight': weight, 'lightgbm_weight': 1 - weight,
        'blend_space': 'log1p target; expm1 after blending; nonnegative clamp',
        'split': protocol, 'trained_through': protocol['train'][1],
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'selection_scope': 'supported units in validation only; no refit with validation/test',
        'selection_rows': int(selection_mask.sum()),
        'evaluation_mode': EVALUATION_MODE, 'weights_locked_before_test': True,
        'known_units': [{'plant': p, 'unit': int(u)} for p, u in sorted(known)],
        'feature_ranges': {c: {'min': float(train_X[c].min()), 'max': float(train_X[c].max())} for c in pre.columns},
        'packages': {p: importlib.metadata.version(p) for p in
                     ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'flask']},
        'source_manifest_sha256': source['manifest_sha256'],
        'history_bootstrap_through': protocol['train'][1],
        'bundle_sha256': {n: sha256(output / n) for n in ['xgb_model.json', 'lgb_model.txt', 'preprocessor.json']},
        'model_settings': {'xgboost': model_settings(xm), 'lightgbm': model_settings(lm)},
        'usage': 'development experiment; not deployed; unseen-unit predictions are diagnostics only',
    }
    if 'fuel_input_basis' in manifest:
        metadata['fuel_input_basis'] = manifest['fuel_input_basis']
    if 'operation_input_policy' in manifest:
        metadata['operation_input_policy'] = manifest['operation_input_policy']
    write_json(output / 'metadata.json', metadata)
    df.to_csv(output / 'history.csv', index=False, date_format='%Y-%m-%d')
    write_json(output / 'validation_weight_search.json', trials)
    locks = {n: sha256(output / n) for n in
             ['metadata.json', 'xgb_model.json', 'lgb_model.txt', 'preprocessor.json']}
    write_json(output / 'selection_lock.json', {'sha256': locks, 'test_predictions_created': False})

    report = {**describe_fold(df, protocol), 'weight': weight, 'feature_count': len(pre.columns),
              'evaluation_mode': EVALUATION_MODE, 'scores': {}, 'by_plant': {}, 'api_input_rejections': {}}
    for split in ('validation', 'test'):
        mask = masks[split]
        model_X = pre.transform(X.loc[mask])
        a, b = xm.predict(model_X), lm.predict(model_X)
        predictions, flags = baseline_predictions(df, X, masks['train'], mask)
        predictions.update(xgboost=blend(a, b, 1), lightgbm=blend(a, b, 0), ensemble=blend(a, b, weight))
        rows = df.loc[mask, KEYS + ['target']].copy()
        rows['supported'] = supported.loc[mask]
        rows['nox_history_count'] = X.loc[mask, 'nox_history_count']
        rows['api_rejection_reason'] = [api_rejection(r, known, manifest.get('operation_input_policy')) for _, r in df.loc[mask].iterrows()]
        rows['api_input_accepted'] = rows.api_rejection_reason.eq('')
        for name, values in predictions.items():
            rows['pred_' + name] = values
        for name, values in flags.items():
            rows[name] = values
        rows.to_csv(output / f'{split}_predictions.csv', index=False, date_format='%Y-%m-%d')
        report['scores'][split] = scoped_scores(rows)
        report['by_plant'][split] = {plant: scoped_scores(group) for plant, group in rows.groupby('plant')}
        # These rows remain in batch scores; API limitations are reported independently.
        report['api_input_rejections'][split] = rows.loc[~rows.api_input_accepted,
            KEYS + ['supported', 'api_rejection_reason']].assign(month=lambda d: d.month.dt.strftime('%Y-%m')).to_dict('records')
    if locks != {n: sha256(output / n) for n in locks}:
        raise ValueError('Frozen selection changed during outer evaluation')
    report['seconds'] = time.perf_counter() - started
    write_json(output / 'metrics.json', report)
    return report


def verify_saved_predictions(output):
    from .predict import Predictor
    predictor = Predictor(output, history_db=False)
    history = predictor.history
    results = {'loaded_model_rows': 0, 'api_prediction_rows': 0, 'api_rejected_rows': 0,
               'maximum_api_absolute_difference': 0.}
    for split in ('validation', 'test'):
        saved = pd.read_csv(output / f'{split}_predictions.csv', parse_dates=['month'])
        rows = history.merge(saved[KEYS], on=KEYS, how='inner', validate='one_to_one')
        rows = saved[KEYS].merge(rows, on=KEYS, how='left', validate='one_to_one')
        X = predictor.pre.transform(build_features(rows, history))
        loaded = blend(predictor.xgb.predict(X), predictor.lgb.predict(X, num_threads=1), predictor.weight)
        np.testing.assert_allclose(loaded, saved.pred_ensemble, rtol=1e-7, atol=1e-7)
        results['loaded_model_rows'] += len(saved)
        for (_, row), (_, prediction) in zip(rows.iterrows(), saved.iterrows()):
            if prediction.api_input_accepted:
                actual = predictor.predict(request_for(row))['prediction']['value']
                difference = abs(actual - prediction.pred_ensemble)
                if not np.isclose(actual, prediction.pred_ensemble, rtol=1e-7, atol=1e-7):
                    raise ValueError('API and saved temporal-validation prediction differ')
                results['api_prediction_rows'] += 1
                results['maximum_api_absolute_difference'] = max(results['maximum_api_absolute_difference'], difference)
            else:
                try:
                    predictor.predict(request_for(row))
                except ValueError:
                    results['api_rejected_rows'] += 1
                else:
                    raise ValueError('Unexpected API acceptance of an unsupported/invalid input')
    return results


def paired_comparison(estimated, actual):
    joined = estimated.merge(actual, on=['fold', *KEYS], suffixes=('_estimated', '_actual'), validate='one_to_one')
    if len(joined) != len(estimated) or len(joined) != len(actual):
        raise ValueError('Fuel variants have different evaluation keys')
    if not joined.target_estimated.equals(joined.target_actual) or not joined.supported_estimated.equals(joined.supported_actual):
        raise ValueError('Fuel variants must have identical targets and unit support')
    rows = []
    for scope, subset in [('supported', joined.loc[joined.supported_actual]),
                          ('unseen_diagnostic', joined.loc[~joined.supported_actual])]:
        for fold, group in [('pooled', subset), *list(subset.groupby('fold'))]:
            for method in METHODS:
                e = safe_metrics(group.target_actual, group[f'pred_{method}_estimated'])
                a = safe_metrics(group.target_actual, group[f'pred_{method}_actual'])
                rows.append({'scope': scope, 'fold': fold, 'method': method, 'n': len(group),
                    'estimated_rmse': e['rmse'], 'actual_rmse': a['rmse'],
                    'rmse_change_actual_minus_estimated': None if not len(group) else a['rmse'] - e['rmse'],
                    'estimated_mae': e['mae'], 'actual_mae': a['mae']})
    return rows


def run_validation(output, raw_dirs):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite validation output; choose a new --output')
    data = {name: prepare_data(raw) for name, raw in raw_dirs.items()}
    reference = next(iter(data.values()))[0][KEYS + ['target']]
    for df, _, _ in data.values():
        pd.testing.assert_frame_equal(reference, df[KEYS + ['target']])
    output.mkdir(parents=True)
    write_json(output / 'run_plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'folds': FOLDS, 'cutoff': CUTOFF, 'primary_metric': 'supported-unit outer-period RMSE in ppm',
        'weight_selection': '0..1 step .01, validation supported-unit RMSE only; tie chooses smaller XGB weight',
        'baselines': 'unit training mean, with global training median fallback; prior values use exact calendar months',
        'evaluation_mode': EVALUATION_MODE, 'interpretation': 'historical development validation, not a fresh holdout',
        'source': {name: source for name, (_, _, source) in data.items()}})
    summaries, pooled_frames, metric_rows, top_errors, verification = {}, {}, [], [], {}
    for variant, (df, manifest, source) in data.items():
        reports, frames = {}, []
        for fold, protocol in FOLDS.items():
            folder = output / 'models' / variant / fold
            reports[fold] = train_fold(df, manifest, source, protocol, folder)
            verification[f'{variant}/{fold}'] = verify_saved_predictions(folder)
            rows = pd.read_csv(folder / 'test_predictions.csv', parse_dates=['month'])
            rows.insert(0, 'fold', fold)
            frames.append(rows)
            print(json.dumps({'variant': variant, 'fold': fold, 'weight': reports[fold]['weight'],
                'seconds': reports[fold]['seconds'], 'outer_supported': reports[fold]['scores']['test']['supported']['ensemble']},
                ensure_ascii=False), flush=True)
            for scope, scores in reports[fold]['scores']['test'].items():
                for method, values in scores.items():
                    metric_rows.append({'variant': variant, 'fold': fold, 'plant': '전체', 'scope': scope,
                                        'method': method, **values})
            for plant, scopes in reports[fold]['by_plant']['test'].items():
                for scope, scores in scopes.items():
                    for method, values in scores.items():
                        metric_rows.append({'variant': variant, 'fold': fold, 'plant': plant, 'scope': scope,
                                            'method': method, **values})
        pooled = pd.concat(frames, ignore_index=True)
        if pooled.duplicated(KEYS).any():
            raise ValueError('Outer score periods must not overlap')
        pooled.to_csv(output / f'{variant}_outer_predictions.csv', index=False, date_format='%Y-%m-%d')
        pooled_frames[variant] = pooled
        summaries[variant] = {'folds': reports, 'pooled_scores': scoped_scores(pooled),
            'pooled_by_plant': {plant: scoped_scores(g) for plant, g in pooled.groupby('plant')},
            'pooled_api_rejected_rows': int((~pooled.api_input_accepted).sum())}
        for scope, scores in summaries[variant]['pooled_scores'].items():
            for method, values in scores.items():
                metric_rows.append({'variant': variant, 'fold': 'pooled', 'plant': '전체', 'scope': scope,
                                    'method': method, **values})
        for plant, scopes in summaries[variant]['pooled_by_plant'].items():
            for scope, scores in scopes.items():
                for method, values in scores.items():
                    metric_rows.append({'variant': variant, 'fold': 'pooled', 'plant': plant, 'scope': scope,
                                        'method': method, **values})
        worst = pooled.assign(absolute_error=(pooled.target - pooled.pred_ensemble).abs()).nlargest(15, 'absolute_error')
        for record in worst.assign(month=worst.month.dt.strftime('%Y-%m')).to_dict('records'):
            top_errors.append({'variant': variant, **record})
    comparison = paired_comparison(pooled_frames['site_estimated'], pooled_frames['unit_actual'])
    pd.DataFrame(comparison).to_csv(output / 'fuel_basis_comparison.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(output / 'outer_metrics.csv', index=False)
    pd.DataFrame(top_errors).to_csv(output / 'largest_errors.csv', index=False)
    write_json(output / 'verification.json', verification)
    write_json(output / 'summary.json', {'target': TARGETS['concentration'], 'variants': summaries,
        'fuel_basis_comparison': comparison, 'evaluation_mode': EVALUATION_MODE,
        'limitations': ['development periods have previously been examined; not a new untouched test',
                        'folds share training data and are not independent replications',
                        'unseen-unit diagnostics are not supported API performance',
                        'current-month realized inputs and assumed prior-target availability are used',
                        'target spikes retained; provider measurement validity remains unverified']})
    (output / 'COMPLETE').write_text('All six runs and saved-model/API consistency checks succeeded.\n')


def write_report(output):
    """Regenerate the readable report without refitting or changing frozen bundles."""
    output = Path(output)
    if not (output / 'COMPLETE').exists():
        raise ValueError('Only a completed validation run can be reported')
    summary = json.loads((output / 'summary.json').read_text())
    verification = json.loads((output / 'verification.json').read_text())
    timing = json.loads((output / 'timing.json').read_text()) if (output / 'timing.json').exists() else {}
    plan = json.loads((output / 'run_plan.json').read_text())
    labels = {'site_estimated': '기존 사업소 연료 배분 추정', 'unit_actual': '영동 실제 호기별 연료'}
    methods = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'ensemble': '검증 선정 앙상블',
               'last_month': '전월 NOx', 'previous_3months': '직전 3개월 평균',
               'train_unit_mean': '학습 호기 평균', 'train_global_median': '학습 전체 중앙값'}
    actual = summary['variants']['unit_actual']
    primary = actual['pooled_scores']['supported']['ensemble']
    unseen = actual['pooled_scores']['unseen_diagnostic']['ensemble']
    n_all = actual['pooled_scores']['all_diagnostic']['ensemble']['n']
    weight_text = '; '.join(f"{labels[v]}: " + ', '.join(f"{f} XGB {r['weight']:.2f}/LGB {1-r['weight']:.2f}"
        for f, r in data['folds'].items()) for v, data in summary['variants'].items())
    lines = ['NOx A·B·C 시간 순서 검증 결과', '검증일: ' + plan['created_at'][:10], '',
        '먼저 읽을 결과',
        '달력 월을 기준으로 학습·가중치 선정·점검 기간을 분리한 여섯 번의 실험을 완료했습니다.',
        '선정 가중치: ' + weight_text,
        '실제 연료 기준의 점검 통합 RMSE/MAE는 아래 표와 같습니다. 모든 관측 타깃과 급등 기록을 유지했습니다.',
        f"전체 실행 {timing.get('seconds', 0):.3f}초 (자료 준비·6회 학습·점검·추론 대조 포함, 제한 {timing.get('limit_seconds', 300):g}초).", '',
        '1. 분할과 해석',
        '반복 | 학습 | 가중치 선정 | 점검 | 학습/선정/점검 전체 관측 행 | 선정/점검 지원 호기 행',
    ]
    for fold, r in summary['variants']['unit_actual']['folds'].items():
        p = r['protocol']
        periods = ['~'.join(p[s]) for s in ('train', 'validation', 'test')]
        counts = '/'.join(str(r['rows'][s]) for s in ('train', 'validation', 'test'))
        lines.append(f"{fold} | {' | '.join(periods)} | {counts} | {r['supported_rows']['validation']}/{r['supported_rows']['test']}")
    lines += ['',
        '반복별 지원 호기 수와 신규 호기 행 수는 summary.json의 known_unit_count/unseen_rows에 기록했습니다.',
        '선정에는 지원 호기만 사용합니다. 신규 호기는 진단 점수를 따로 저장하며 해당 반복의 API는 400으로 거절합니다.',
        f"점검 기간 {n_all}행 중 지원 호기 {primary['n']}행이 주 비교 대상입니다. 신규 호기 {unseen['n']}행을 섞어 서비스 성능으로 소개하지 않습니다.",
        '같은 달의 모든 호기를 같은 기간에 배치했습니다. 행 번호로 시계열 분할하지 않았습니다.',
        '전처리 중앙값과 특징 선택은 학습 행만 사용합니다. 가중치는 선정 기간에서 0~1을 0.01 간격으로 탐색해 RMSE를 최소화했습니다.',
        '학습한 두 모델은 선정 기간으로 다시 학습하지 않으며, 선정 파일·모델·전처리를 점검 예측 전에 저장하고 해시를 잠갔습니다.',
        'log1p 공간에서 혼합한 뒤 expm1으로 ppm를 복원합니다. ppm 예측의 직접 산술평균과 다릅니다.',
        '같은 호기의 직전 1~3개월 실제 관측을 각 월에 순차적으로 사용합니다. 앞 달이 없으면 더 오래된 값을 전월로 당기지 않습니다.',
        '기준 모델의 이력 부족은 학습 호기 평균, 신규 호기는 학습 전체 중앙값으로 보완합니다.',
        '이는 해당 월의 실제 발전·연료·기상 입력에 대한 연구 농도 추정입니다. 6개월 한 번에 예보하거나 월 시작 전 예보한 성능이 아닙니다.',
        '실제 자료 공표 지연은 확인되지 않았습니다. 과거 월 관측이 사용 가능한 조건을 가정한 개발 검증입니다.',
        '2026년 행 176개는 특징 구성 전에 제외했습니다. 앞선 실험에서 이미 살펴본 개발 기간이므로 새로운 미사용 최종 시험은 아닙니다.',
        '반복들은 학습 기간을 공유합니다. 세 개의 독립된 통계 표본으로 해석하지 않습니다.', '',
        '2. 기간별 성능 (지원 호기, ppm; 오차는 작을수록 좋음)',
        '연료 기준 | 반복 | 선정 XGB/LGB 가중치 | 점검 n | XGB RMSE | LGB RMSE | 선정 앙상블 RMSE | 앙상블 MAE | 최선 단순 기준/RMSE',
    ]
    for variant, r in summary['variants'].items():
        for fold, fr in r['folds'].items():
            s = fr['scores']['test']['supported']
            baseline = min(['last_month', 'previous_3months', 'train_unit_mean', 'train_global_median'], key=lambda k: s[k]['rmse'])
            lines.append(f"{labels[variant]} | {fold} | {fr['weight']:.2f}/{1-fr['weight']:.2f} | {s['ensemble']['n']} | "
                f"{s['xgboost']['rmse']:.4f} | {s['lightgbm']['rmse']:.4f} | {s['ensemble']['rmse']:.4f} | "
                f"{s['ensemble']['mae']:.4f} | {methods[baseline]}/{s[baseline]['rmse']:.4f}")
    lines += ['', '선정 가중치가 뒤 점검에서도 최선인지는 별도로 확인해야 합니다. 뒤 점검 결과로 가중치를 다시 바꾸지 않았습니다.',
        '작은 순위 차이에 큰 의미를 두지 않고 기간별·발전소별 차이를 함께 봅니다.', '',
        f"3. 점검 통합 성능 ({primary['n']}행; 기간 RMSE 평균이 아니라 저장한 모든 예측으로 재계산)",
        '방법 | 추정 연료 RMSE/MAE | 실제 연료 RMSE/MAE',
    ]
    for method in METHODS:
        e = summary['variants']['site_estimated']['pooled_scores']['supported'][method]
        a = summary['variants']['unit_actual']['pooled_scores']['supported'][method]
        lines.append(f"{methods[method]} | {e['rmse']:.4f}/{e['mae']:.4f} | {a['rmse']:.4f}/{a['mae']:.4f}")
    e = summary['variants']['site_estimated']['pooled_scores']['supported']['ensemble']
    a = summary['variants']['unit_actual']['pooled_scores']['supported']['ensemble']
    lines += ['', f"실제 연료 앙상블: 추정 연료 대비 통합 RMSE {(a['rmse']/e['rmse']-1)*100:+.2f}%, MAE {(a['mae']/e['mae']-1)*100:+.2f}% 변화 (음수는 오차 감소).",
        f"실제 연료 앙상블 R²={a['r2']:.4f}. 정확도 백분율이 아닙니다.",
        '실제 연료의 같은 모델 종류끼리도 비교표를 저장했습니다. 타깃/점검 키/지원 호기는 두 연료 기준에서 동일합니다.',
        '통합표를 보고 다시 고른 모델의 미사용 성능을 주장할 수는 없습니다.', '',
        '4. 발전소별 실제 연료 앙상블 (지원 호기)',
        '발전소 | n | 앙상블 RMSE/MAE | 전월 기준 RMSE/MAE | 학습 호기 평균 RMSE/MAE',
    ]
    for plant, scopes in summary['variants']['unit_actual']['pooled_by_plant'].items():
        s = scopes['supported']
        fmt = lambda k: f"{s[k]['rmse']:.4f}/{s[k]['mae']:.4f}"
        lines.append(f"{plant} | {s['ensemble']['n']} | {fmt('ensemble')} | {fmt('last_month')} | {fmt('train_unit_mean')}")
    lines += ['', '전역 모델이므로 영동 입력 변경이 다른 발전소의 예측에도 영향을 줄 수 있습니다. 발전소별 변화를 함께 확인합니다.',
        '지원/신규 호기 구분과 발전소별 단순 기준 대비 점수를 함께 검토합니다.',
        '큰 오차 순위 15개는 largest_errors.csv에 저장했습니다.', '',
        '5. 신규 여수 2호기와 API 입력 제한',
    ]
    for variant, r in summary['variants'].items():
        s = r['pooled_scores']['unseen_diagnostic']['ensemble']
        lines.append(f"{labels[variant]} 신규 호기 점검 {s['n']}행: RMSE {s['rmse']:.4f}, MAE {s['mae']:.4f}ppm (지원 성능과 분리).")
    lines += ['분당 2025-01 열효율 132.88%, 458.58%, 107.05%는 앞선 품질 검사에서 특징 결측으로 처리됐습니다.',
        '이 3행은 지원 호기 일괄 점수에 포함하고 학습 중앙값으로 보완했지만, 현재 API는 필수 열효율 null을 거절합니다.',
        f"실제 연료 점검 {n_all}행 중 API 입력 거절 {actual['pooled_api_rejected_rows']}행, 입력 가능 {n_all-actual['pooled_api_rejected_rows']}행입니다. 상세 사유는 예측 CSV에 있습니다.",
        '서버 입력 정책은 이번 평가 구현에서 변경하지 않았습니다. 결측 열효율 입력의 별도 정책 정리가 필요합니다.', '',
        '6. 검증과 파일',
        f"저장 모델 재로드 대조 {sum(r['loaded_model_rows'] for r in verification.values())}건; API 예측 대조 {sum(r['api_prediction_rows'] for r in verification.values())}건; "
        f"예상 API 거절 확인 {sum(r['api_rejected_rows'] for r in verification.values())}건.",
        f"API/저장 예측 최대 절대 차이 {max(r['maximum_api_absolute_difference'] for r in verification.values()):.3g}ppm.",
        '이 건수는 두 연료 기준 및 반복의 선정/점검을 포함해 중복 월이 있습니다. 독립 표본 수가 아닙니다.',
        '추론은 기존 Predictor와 Flask 테스트 클라이언트로 대조했으며, 새 HTTP 서버나 Gemini 외부 호출은 이 작업 범위에 포함하지 않았습니다.',
        '새 테스트는 점검 타깃·입력 변조에 대한 학습/가중치 불변성, 신규 호기 선정 제외, 달력 월 분할, 이력 부족 보완, 보존 해시와 API 정책을 확인합니다.',
        '전체 회귀 테스트 실행 결과: tests_console.txt, tests.xml.',
        '기존 원자료·타깃·서버 선택·이력 DB·기존 모델 파일을 보존했습니다. 새 실험 파일은 이 폴더에 저장했습니다.',
        'summary.json: 기간별/통합/발전소별 RMSE·MAE·R²와 지원/신규 구분.',
        'outer_metrics.csv: 모든 방법·기간·발전소·지원 범위 점수.',
        '*_outer_predictions.csv: 모든 점검 예측과 지원 여부, API 입력 상태.',
        'fuel_basis_comparison.csv: 모델 종류를 고정한 실제/추정 연료 비교.',
        'models/<연료 기준>/<A,B,C>: 동결 모델, 전처리, 선정 탐색 101개, 해시 잠금, 이력, 기간별 예측.',
        'verification.json, timing.json: 저장/추론 대조와 300초 실행 제한 결과.', '',
        '7. 다음 판단',
        '시간 분할 평가 체계는 구현됐습니다. 단일 선정 기간의 점수만으로 모델 교체를 판단하지 않습니다.',
        '다음은 계획대로 영동 2023~2025 시간별 발전 실적을 확보해 운전 특징을 추가하고, 동일한 여섯 분할로 비교하는 작업입니다.',
        '여수 1호기 악화와 삼천포 전역 모델 영향을 함께 점검합니다. 단순 호기 평균/전월 기준은 계속 비교 대상으로 유지합니다.',
        '급등의 유효성은 공식 자료로 확인해야 하며, 이번 결과를 이유로 삭제하지 않습니다. 이후 미사용 기간의 검증은 별도로 필요합니다.', '',
        '재현 명령 (프로젝트 루트, 기존 출력은 덮어쓰지 않음)',
        'MPLCONFIGDIR=/tmp/nox-mpl /opt/anaconda3/bin/python -m nox.temporal_validation --dry-run',
        'MPLCONFIGDIR=/tmp/nox-mpl /opt/anaconda3/bin/python -m nox.temporal_validation --output reports/temporal_validation_run2 --max-training-seconds 300',
        'MPLCONFIGDIR=/tmp/nox-mpl /opt/anaconda3/bin/python -m nox.temporal_validation --output reports/temporal_validation_20261001 --report-only',
    ]
    tests = output / 'tests_console.txt'
    if tests.exists():
        ending = tests.read_text().strip().splitlines()[-1]
        lines.append('\n이번 회귀 테스트: ' + ending)
    (output / 'NOx_temporal_validation_result.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/temporal_validation_20261001')
    parser.add_argument('--estimated-raw-dir', type=Path, default=ROOT / 'data/extended_20261001')
    parser.add_argument('--actual-raw-dir', type=Path, default=ROOT / 'data/unit_fuel_20261001')
    parser.add_argument('--max-training-seconds', type=float, default=300.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.max_training_seconds) or args.max_training_seconds <= 0:
        parser.error('--max-training-seconds must be finite and positive')
    raw_dirs = {'site_estimated': args.estimated_raw_dir, 'unit_actual': args.actual_raw_dir}
    if args.report_only:
        write_report(args.output)
        print(args.output / 'NOx_temporal_validation_result.txt')
        return
    if args.dry_run:
        plan = {}
        for variant, raw in raw_dirs.items():
            df, _, source = prepare_data(raw)
            plan[variant] = {'source': source, 'folds': {n: describe_fold(df, p) for n, p in FOLDS.items()}}
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.output.exists():
        parser.error('Output exists; choose a new --output')
    if args.worker:
        run_validation(args.output, raw_dirs)
        return
    command = [sys.executable, '-m', 'nox.temporal_validation', '--worker', '--output', str(args.output),
               '--estimated-raw-dir', str(args.estimated_raw_dir), '--actual-raw-dir', str(args.actual_raw_dir)]
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 timeout=args.max_training_seconds, check=False)
        status, code, console = ('success' if process.returncode == 0 else 'failed'), process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as error:
        status, code = 'timeout', 2
        console = (error.stdout or b'').decode() + (error.stderr or b'').decode()
    seconds = time.perf_counter() - started
    if args.output.exists():
        (args.output / 'training_console.txt').write_text(console, encoding='utf-8')
        write_json(args.output / 'timing.json', {'seconds': seconds, 'status': status,
            'returncode': code, 'limit_seconds': args.max_training_seconds, 'command': command,
            'scope': 'entire worker: preparation, six fits, scoring and inference consistency checks'})
        if status != 'success':
            (args.output / 'COMPLETE').unlink(missing_ok=True)
            (args.output / 'INCOMPLETE').write_text(status + '\n')
    print(console, end='')
    print(json.dumps({'status': status, 'seconds': seconds, 'output': str(args.output)}, ensure_ascii=False))
    if code:
        parser.exit(code, 'Validation incomplete. Inspect logs and use a new output directory for a rerun.\n')
    write_report(args.output)


if __name__ == '__main__':
    main()
