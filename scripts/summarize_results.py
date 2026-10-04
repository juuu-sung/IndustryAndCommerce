"""Recompute documented metrics from paired saved predictions; no model training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'NOx_corrected'
OUTPUT = ROOT / 'docs/generated'
KEYS = ['fold', 'plant', 'unit', 'month']


def score(actual, predicted):
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError('Metrics require nonempty, finite paired observations')
    return {'n': len(error), 'rmse_ppm': float(np.sqrt(np.mean(error ** 2))),
            'mae_ppm': float(np.mean(np.abs(error)))}


def main(chart=False):
    old = pd.read_csv(PROJECT / 'reports/temporal_validation_20261001/unit_actual_outer_predictions.csv')
    new = pd.read_csv(PROJECT / 'reports/readiness_20261002/candidates/seed_mean_predictions.csv')
    new = new.loc[(new.candidate == 'no_weather_log_lgb') &
                  (new.input_policy == 'values_only') & (new.split == 'test')]
    paired = old.merge(new[KEYS + ['target', 'supported', 'prediction']], on=KEYS,
                       suffixes=('_old', '_new'), validate='one_to_one', how='outer', indicator=True)
    assert paired['_merge'].eq('both').all(), 'Candidates must cover the same rows'
    assert np.allclose(paired.target_old, paired.target_new, rtol=0, atol=1e-10)
    assert paired.supported_old.equals(paired.supported_new)
    paired = paired.loc[paired.supported_old].copy()
    assert len(paired) == 370
    methods = {
        'XGBoost': 'pred_xgboost', 'LightGBM': 'pred_lightgbm',
        'Period-selected ensemble': 'pred_ensemble', 'Training unit mean': 'pred_train_unit_mean',
        'Previous 3-month mean': 'pred_previous_3months', 'Last month': 'pred_last_month',
        'Training global median': 'pred_train_global_median',
        'Candidate: no-weather log LightGBM (5-seed mean)': 'prediction',
    }
    overall = pd.DataFrame([{'method': name, **score(paired.target_old, paired[column])}
                            for name, column in methods.items()])
    regions = []
    for plant, group in paired.groupby('plant'):
        for name, column in methods.items():
            regions.append({'plant': plant, 'method': name, **score(group.target_old, group[column])})
    common = pd.read_csv(PROJECT / 'reports/retraining/common_period_predictions.csv')
    common = common.loc[common.baseline_supported]
    assert len(common) == 168
    retrained = [{'model': name, **score(common.target, common[column])}
                 for name, column in [('initial', 'pred_baseline'), ('retrained', 'pred_candidate')]]
    recorded = json.loads((PROJECT / 'reports/target_validation_20261003/saved_comparison_v2/comparison.json').read_text())
    for result in retrained:
        expected = recorded['retraining_2026'][result['model']]
        assert np.isclose(result['rmse_ppm'], expected['rmse'], rtol=0, atol=1e-9)
        assert np.isclose(result['mae_ppm'], expected['mae'], rtol=0, atol=1e-9)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT / 'development_metrics.csv', index=False)
    pd.DataFrame(regions).to_csv(OUTPUT / 'plant_metrics.csv', index=False)
    pd.DataFrame(retrained).to_csv(OUTPUT / 'same_cohort_2026_metrics.csv', index=False)
    meta = {'target': 'monthly_daily_stack_mean_nox', 'unit': 'ppm',
            'development_rows': 370, 'development_period': '2024-07 through 2025-12',
            'retraining_comparison_rows': 168, 'retraining_comparison_period': '2026-01 through 2026-08',
            'target_validity': 'agency confirmation pending', 'independent_final_evaluation': False,
            'candidate_deployed': False}
    (OUTPUT / 'comparison_scope.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    if chart:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = overall.set_index('method').loc[
            ['Last month', 'XGBoost', 'LightGBM', 'Candidate: no-weather log LightGBM (5-seed mean)']]
        labels = ['Last month', 'XGBoost', 'LightGBM', 'Candidate\n(no weather, 5 seeds)']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        for ax, metric, title in zip(axes, ['rmse_ppm', 'mae_ppm'], ['RMSE (ppm)', 'MAE (ppm)']):
            bars = ax.bar(range(4), selected[metric], color=['#94a3b8', '#60a5fa', '#0d9488', '#f59e0b'])
            ax.set_xticks(range(4), labels, fontsize=9)
            ax.set_title(title, fontsize=12)
            ax.set_ylim(0, selected[metric].max() * 1.22)
            ax.spines[['top', 'right']].set_visible(False)
            ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, alpha=.18)
        fig.suptitle('Development comparison: same 370 unit-months', fontsize=14)
        fig.supxlabel('2024H2–2025; research concentration; reused evaluation periods', fontsize=9)
        fig.savefig(OUTPUT / 'development_comparison.png', dpi=160)
        plt.close(fig)
    print(overall.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', action='store_true', help='Requires requirements-docs.txt')
    main(parser.parse_args().chart)
