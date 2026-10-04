import json

import numpy as np
import pandas as pd
import pytest

from nox.app import create_app
from nox.features import build_features
from nox.predict import Predictor
from nox.schema import CONDITIONS, KEYS, ROOT
from nox.temporal_validation import (FOLDS, METHODS, api_rejection, baseline_predictions,
    describe_fold, fold_masks, paired_comparison, prepare_data, request_for,
    run_validation, safe_metrics, scoped_scores, sha256, train_fold)
from nox.train import choose_weight

OUTPUT = ROOT / 'reports/temporal_validation_20261001'


@pytest.fixture(scope='module')
def development():
    return prepare_data(ROOT / 'data/unit_fuel_20261001')


@pytest.mark.parametrize('fold,counts,unseen', [
    ('A', [252, 126, 126], [0, 0, 0]),
    ('B', [378, 126, 132], [0, 0, 6]),
    ('C', [504, 132, 124], [0, 6, 6]),
])
def test_calendar_splits_and_unseen_units_are_order_independent(development, fold, counts, unseen):
    df = development[0].sample(frac=1, random_state=7).reset_index(drop=True)
    report = describe_fold(df, FOLDS[fold])
    assert list(report['rows'].values()) == counts
    assert list(report['unseen_rows'].values()) == unseen
    assert report['known_unit_count'] == 21
    masks, supported, known = fold_masks(df, FOLDS[fold])
    assert ('여수', 2) not in known
    assert int((masks['validation'] & supported).sum()) == 126
    # Every observed unit from a calendar month has exactly the same split.
    labels = pd.Series('', index=df.index)
    for name, mask in masks.items():
        labels.loc[mask] = name
    assert labels.loc[df.target.notna()].groupby(df.loc[df.target.notna(), 'month']).nunique().max() == 1


def test_2026_targets_and_inputs_excluded_before_feature_construction(development):
    df, _, source = development
    assert df.month.max() == pd.Timestamp('2025-12-01')
    assert source['excluded_future_rows'] == 176
    assert len(df) == 792 and int(df.target.notna().sum()) == 760


def test_baseline_fallback_uses_training_only_and_exact_previous_month():
    def row(month, target, unit=1):
        return {'plant': '분당', 'unit': unit, 'month': pd.Timestamp(month), 'target': target,
                **{c: 1. for c in CONDITIONS}}
    df = pd.DataFrame([row('2023-01-01', 2), row('2023-02-01', 6),
                       row('2023-04-01', 99999), row('2023-04-01', 99999, 2)])
    X = build_features(df)
    predictions, flags = baseline_predictions(df, X, df.index < 2, df.index >= 2)
    # March is missing; February cannot become April's previous month.
    np.testing.assert_array_equal(predictions['last_month'], [4., 4.])
    np.testing.assert_array_equal(predictions['previous_3months'], [4., 4.])
    np.testing.assert_array_equal(predictions['train_unit_mean'], [4., 4.])
    np.testing.assert_array_equal(flags['last_month_fallback'], [True, True])
    np.testing.assert_array_equal(flags['previous_3months_fallback'], [False, True])


def test_outer_target_and_inputs_do_not_change_fitted_preprocessor_models_or_weight(development, tmp_path):
    df, manifest, source = development
    original, changed = tmp_path / 'original', tmp_path / 'changed'
    first = train_fold(df, manifest, source, FOLDS['C'], original)
    altered = df.copy()
    outer = altered.month >= pd.Timestamp('2025-07-01')
    altered.loc[outer, 'target'] = 1e8
    altered.loc[outer, 'temperature_c'] = 1e6
    altered.loc[outer, 'generation_mwh'] = 1e12
    second = train_fold(altered, manifest, source, FOLDS['C'], changed)
    assert first['weight'] == second['weight']
    for name in ['preprocessor.json', 'xgb_model.json', 'lgb_model.txt', 'validation_weight_search.json']:
        assert sha256(original / name) == sha256(changed / name)
    assert first['scores']['test']['supported']['ensemble'] != second['scores']['test']['supported']['ensemble']


def test_unseen_validation_target_cannot_select_weight(development, tmp_path):
    df, manifest, source = development
    _, supported, _ = fold_masks(df, FOLDS['C'])
    original, changed = tmp_path / 'original', tmp_path / 'changed'
    a = train_fold(df, manifest, source, FOLDS['C'], original)
    altered = df.copy()
    altered.loc[~supported & altered.target.notna(), 'target'] = 1e9
    b = train_fold(altered, manifest, source, FOLDS['C'], changed)
    assert a['weight'] == b['weight']
    assert sha256(original / 'validation_weight_search.json') == sha256(changed / 'validation_weight_search.json')
    assert a['scores']['validation']['unseen_diagnostic']['ensemble']['n'] == 6
    assert a['scores']['validation']['unseen_diagnostic']['ensemble']['rmse'] != b['scores']['validation']['unseen_diagnostic']['ensemble']['rmse']


def test_empty_scope_has_null_metrics_and_pooled_metrics_use_all_rows():
    assert safe_metrics([], []) == {'n': 0, 'rmse': None, 'mae': None, 'r2': None}
    rows = pd.DataFrame({'target': [0., 0., 0.], 'supported': [True, True, False],
                         **{'pred_' + m: [3., 4., 100.] for m in METHODS}})
    score = scoped_scores(rows)
    assert score['supported']['ensemble']['rmse'] == pytest.approx(np.sqrt(12.5))
    assert score['supported']['ensemble']['r2'] is None
    assert score['unseen_diagnostic']['ensemble']['mae'] == 100


def test_overwrite_is_rejected_before_loading_any_sources(tmp_path):
    with pytest.raises(ValueError, match='overwrite'):
        run_validation(tmp_path, {'site_estimated': tmp_path / 'missing'})


def test_source_checksum_failure_is_not_silently_accepted(tmp_path):
    (tmp_path / 'x.csv').write_text('changed')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': [{'file': 'x.csv', 'sha256': 'bad'}]}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare_data(tmp_path)


def test_paired_comparison_rejects_different_target_or_support():
    frame = pd.DataFrame({'fold': ['A'], 'plant': ['분당'], 'unit': [1],
        'month': [pd.Timestamp('2024-07-01')], 'target': [10.], 'supported': [True],
        **{'pred_' + m: [12.] for m in METHODS}})
    assert paired_comparison(frame, frame)[0]['rmse_change_actual_minus_estimated'] == 0
    for column, value in [('target', 11.), ('supported', False)]:
        changed = frame.copy()
        changed[column] = value
        with pytest.raises(ValueError, match='identical targets'):
            paired_comparison(frame, changed)


@pytest.mark.parametrize('variant', ['site_estimated', 'unit_actual'])
def test_saved_actual_experiment_protocol_scores_and_api_policy(variant):
    if not (OUTPUT / 'COMPLETE').exists():
        pytest.skip('Run the temporal-validation experiment first')
    summary = json.loads((OUTPUT / 'summary.json').read_text())['variants'][variant]
    pooled = pd.read_csv(OUTPUT / f'{variant}_outer_predictions.csv')
    assert len(pooled) == 382 and int(pooled.supported.sum()) == 370
    assert summary['pooled_scores']['unseen_diagnostic']['ensemble']['n'] == 12
    assert summary['pooled_api_rejected_rows'] == 15  # 12 unseen + 3 invalid thermal efficiencies.
    for fold in FOLDS:
        folder = OUTPUT / 'models' / variant / fold
        metadata = json.loads((folder / 'metadata.json').read_text())
        assert metadata['selection_rows'] == 126
        assert metadata['split'] == FOLDS[fold]
        assert metadata['weights_locked_before_test'] is True
        saved = pd.read_csv(folder / 'validation_predictions.csv')
        supported = saved.loc[saved.supported]
        # Reproduce selection from stored standalone predictions in the correct log space.
        weight, _ = choose_weight(supported.target, np.log1p(supported.pred_xgboost), np.log1p(supported.pred_lightgbm))
        assert weight == metadata['xgboost_weight']
        predictor = Predictor(folder, history_db=False)
        assert predictor.history.month.max() <= pd.Timestamp(FOLDS[fold]['test'][1] + '-01')
        for name, expected in json.loads((folder / 'selection_lock.json').read_text())['sha256'].items():
            assert sha256(folder / name) == expected
        assert api_rejection(predictor.history.loc[predictor.history.plant.eq('여수') & predictor.history.unit.eq(2)].iloc[-1],
                             predictor.known_units) == 'Plant/unit was not represented in model training'
    predictor = Predictor(OUTPUT / 'models' / variant / 'B', history_db=False)
    unseen = predictor.history.loc[predictor.history.plant.eq('여수') & predictor.history.unit.eq(2)].iloc[-1]
    assert create_app(predictor).test_client().post('/api/predict', json=request_for(unseen)).status_code == 400
