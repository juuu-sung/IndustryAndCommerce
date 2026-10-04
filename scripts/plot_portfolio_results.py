"""Render the portfolio chart from public aggregate metrics, without raw data."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/generated/development_metrics.csv'
OUTPUT = ROOT / 'docs/images'
METHODS = [
    'Last month', 'XGBoost', 'LightGBM', 'Period-selected ensemble',
    'Candidate: no-weather log LightGBM (5-seed mean)',
]


def main():
    data = pd.read_csv(SOURCE).set_index('method').loc[METHODS]
    if not data['n'].eq(370).all():
        raise ValueError('This figure requires the documented 370-row comparison')
    if not data[['rmse_ppm', 'mae_ppm']].notna().all().all():
        raise ValueError('Metrics must be complete')
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ['AppleGothic', 'Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic']
                 if name in installed), None)
    korean = font is not None
    if korean:
        plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'path'
    baseline, candidate = data.iloc[0], data.iloc[-1]
    improvement = (baseline.rmse_ppm - candidate.rmse_ppm) / baseline.rmse_ppm * 100
    labels = (['전월 값', 'XGBoost', 'LightGBM', '기간별 선정 앙상블', '개선 후보 · LightGBM'] if korean else
              ['Last month', 'XGBoost', 'LightGBM', 'Period-selected ensemble', 'Candidate: LightGBM'])
    fig = plt.figure(figsize=(11, 5.8), facecolor='white')
    ink, muted, blue = '#172b4d', '#64748b', '#2563eb'
    fig.text(.05, .92, '전월 값 기준 대비 예측 오차 감소' if korean else 'Lower prediction error than last-month baseline',
             fontsize=17, color=ink, weight='bold')
    fig.text(.95, .91, f'{improvement:.1f}%', fontsize=28, color=blue, weight='bold', ha='right')
    fig.text(.05, .855, ('동일한 370개 호기·월 관측 비교  |  2024년 하반기~2025년  |  낮을수록 우수' if korean else
                        'Same 370 unit-months  |  2024H2–2025  |  Lower is better'), fontsize=10.5, color=muted)
    ax = fig.add_axes([.28, .21, .55, .52])
    colors = ['#cbd5e1', '#94a3b8', '#94a3b8', '#94a3b8', blue]
    bars = ax.barh(range(5), data.rmse_ppm, color=colors, height=.57)
    ax.set_yticks(range(5), labels, fontsize=11, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.tick_params(axis='both', length=0, labelcolor=muted)
    ax.tick_params(axis='y', pad=12, labelcolor=ink)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.xaxis.grid(True, color='#e2e8f0', linewidth=.7)
    ax.set_axisbelow(True)
    for i, (bar, rmse, mae) in enumerate(zip(bars, data.rmse_ppm, data.mae_ppm)):
        color = blue if i == 4 else ink
        ax.text(rmse + .65, bar.get_y() + bar.get_height() / 2, f'{rmse:.2f}',
                va='center', color=color, fontsize=12, weight='bold' if i == 4 else 'normal')
        ax.text(46, i, f'{mae:.2f}', va='center', color=color, fontsize=12,
                weight='bold' if i == 4 else 'normal', clip_on=False)
    ax.text(0, -.7, 'RMSE (ppm)', color=muted, fontsize=10)
    ax.text(46, -.7, 'MAE (ppm)', color=muted, fontsize=10, clip_on=False)
    fig.text(.05, .09, ('개선 후보: 기상 변수 제외 · 로그 변환 · LightGBM 5개 시드 평균' if korean else
                       'Candidate: no weather features · log target · mean of 5 LightGBM seeds'),
             color=ink, fontsize=10.5)
    fig.text(.05, .04, ('연구용 NOx 농도에 대한 개발 평가 결과입니다. 후보 모델은 서비스에 적용하지 않았습니다.' if korean else
                       'Development evaluation of research NOx concentration. Candidate is not deployed.'),
             color=muted, fontsize=9)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / 'model_performance.png', dpi=180, facecolor=fig.get_facecolor())
    vector = OUTPUT / 'model_performance.svg'
    fig.savefig(vector, facecolor=fig.get_facecolor())
    # Matplotlib adds spaces before newlines in SVG paths; keep the artifact tidy.
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines()) + '\n')
    plt.close(fig)
    print(f'RMSE {baseline.rmse_ppm:.4f} -> {candidate.rmse_ppm:.4f}; improvement {improvement:.4f}%')


if __name__ == '__main__':
    main()
