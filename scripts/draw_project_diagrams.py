"""Draw data and service diagrams from public schema metadata only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/images'
INK, MUTED, BLUE, GREEN = '#172b4d', '#64748b', '#2563eb', '#0f766e'


def setup():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ['AppleGothic', 'Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic']
                   if name in fonts), None)
    if family is None:
        raise RuntimeError('Diagram generation requires a Korean font; existing PNG/SVG files are portable')
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    OUTPUT.mkdir(parents=True, exist_ok=True)


def canvas(title, subtitle, height):
    fig, ax = plt.subplots(figsize=(10, height), facecolor='white')
    fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.02)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.025, .965, title, fontsize=18, color=INK, va='top', weight='bold')
    ax.text(.025, .895, subtitle, fontsize=10.5, color=MUTED, va='top')
    return fig, ax


def box(ax, x, y, width, height, title, body='', *, color=BLUE, fill='#eff6ff',
        title_size=11, body_size=9.5, dashed=False):
    patch = FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.009,rounding_size=0.012',
                          linewidth=1, edgecolor=color, facecolor=fill,
                          linestyle='--' if dashed else '-', zorder=2)
    ax.add_patch(patch)
    if body:
        ax.text(x + width / 2, y + height * .75, title, ha='center', va='center',
                fontsize=title_size, color=color, weight='bold', zorder=3)
        ax.text(x + width / 2, y + height * .34, body, ha='center', va='center',
                fontsize=body_size, color=INK, linespacing=1.35, zorder=3)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha='center', va='center',
                fontsize=title_size, color=color, weight='bold', linespacing=1.35, zorder=3)


def arrow(ax, start, end, *, color='#94a3b8', connection='arc3,rad=0', dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                                connectionstyle=connection, linewidth=1.2, color=color,
                                linestyle='--' if dashed else '-', zorder=3))


def save(fig, stem):
    fig.savefig(OUTPUT / f'{stem}.png', dpi=180, facecolor='white')
    vector = OUTPUT / f'{stem}.svg'
    fig.savefig(vector, facecolor='white')
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines()) + '\n')
    plt.close(fig)


def data_pipeline(meta):
    summary = meta['summary']
    fig, ax = canvas('서로 다른 자료를 호기·월 단위로 통합',
                     '일별·월별·시간별 원천 자료 → 키와 집계 기준 정리 → 모델 입력과 학습 정답 분리', 5.3)
    sources = [
        ('NOx 관측', '일별 굴뚝 농도\n날짜·호기·ppm', '호기·일로 정리\n관측일의 월 산술평균'),
        ('발전 실적', '호기별 월 자료\n용량·발전량·효율', '발전소·호기·월\n기준 키로 유지'),
        ('연료 소비', '사업소별 월 자료\n영동 호기별 실제값', '발전량 비중으로 배분\n영동은 실제값 사용'),
        ('기상 관측', '사업소별 일 자료\n기온·습도·바람', '사업소·월별 집계\n같은 사업소 호기 연결'),
        ('영동 시간별 발전', '시간별 전기 발전량\n운전 패턴 실험용', '0값 비중·변동성 등\n월별 특징으로 변환'),
    ]
    width = .172
    for i, (title, body, aggregation) in enumerate(sources):
        x = .025 + i * .195
        center = x + width / 2
        box(ax, x, .635, width, .175, title, body, title_size=10.5, body_size=9)
        box(ax, x, .44, width, .105, aggregation, color=MUTED, fill='#f8fafc', title_size=9.5)
        arrow(ax, (center, .627), (center, .555))
        arrow(ax, (center, .431), (center, .375))
    box(ax, .025, .095, .952, .265, '', color=BLUE, fill='#f8fafc')
    ax.text(.5, .315, '통합 테이블  |  1행 = 발전소 × 호기 × 월',
            ha='center', va='center', fontsize=13, color=INK, weight='bold', zorder=3)
    box(ax, .05, .135, .59, .105, '입력 X',
        '월 운전·연료·기상 + 과거 NOx + 계절성\n시간별 발전 패턴은 후보별로 추가', body_size=9, title_size=10.5)
    box(ax, .67, .135, .278, .105, '정답 y', '해당 월 NOx 농도 (ppm)',
        color=GREEN, fill='#ecfdf5', body_size=10, title_size=10.5)
    ax.text(.025, .047,
            f"{summary['plant_count']}개 발전소 · {summary['plant_unit_count']}개 발전소/호기 조합 · "
            f"{summary['month_count']}개월 · {summary['rows']:,}행  |  "
            f"NOx 타깃 {summary['observed_targets']:,}행 / 결측 {summary['missing_targets']}행",
            fontsize=10, color=MUTED, va='center')
    save(fig, 'data_pipeline')


def system_architecture():
    fig, ax = canvas('머신러닝 모델 개발에서 관리정보 API까지',
                     '학습·모델 비교와 서비스 추론을 구분하고, LLM 실패 시에도 예측 수치를 유지', 6.0)
    box(ax, .025, .14, .435, .68, '', color='#e2e8f0', fill='#f8fafc')
    box(ax, .53, .14, .445, .68, '', color='#e2e8f0', fill='#f8fafc')
    ax.text(.242, .78, '모델 개발 · 평가', ha='center', color=INK, fontsize=13, weight='bold', zorder=3)
    ax.text(.753, .78, 'Flask 예측 · 설명 API', ha='center', color=INK, fontsize=13, weight='bold', zorder=3)

    box(ax, .055, .635, .375, .092, '입력 X + 정답 y', '과거 학습 → 검증 선정 → 이후 평가',
        title_size=10.5, body_size=9)
    box(ax, .055, .48, .165, .095, 'XGBoost', '회귀 모델', title_size=11)
    box(ax, .265, .48, .165, .095, 'LightGBM', '회귀 모델', title_size=11)
    arrow(ax, (.19, .627), (.14, .585))
    arrow(ax, (.295, .627), (.35, .585))
    box(ax, .055, .34, .375, .083, '검증 오차로 앙상블 가중치 선정', 'XGBoost 비중 0~1 탐색',
        title_size=10.5, body_size=9)
    arrow(ax, (.14, .472), (.19, .433))
    arrow(ax, (.35, .472), (.295, .433))
    box(ax, .075, .197, .335, .085, '개선 후보 비교', '기상 제외 · 로그 LightGBM (서비스 미적용)',
        color=MUTED, fill='white', title_size=10.5, body_size=8.8, dashed=True)
    arrow(ax, (.24, .331), (.24, .292), dashed=True)

    box(ax, .56, .653, .385, .072, '운전 입력 JSON + 관측 이력', title_size=10.5)
    box(ax, .56, .548, .385, .065, '입력 검증 · 달력 기준 특징 구성', title_size=10.5)
    box(ax, .56, .439, .385, .068, '기존 저장 모델 → NOx 농도 예측',
        color=GREEN, fill='#ecfdf5', title_size=10.5)
    arrow(ax, (.752, .644), (.752, .622))
    arrow(ax, (.752, .539), (.752, .516))
    arrow(ax, (.44, .38), (.549, .473), color=GREEN)
    ax.text(.486, .465, '기존\n모델', ha='center', va='center', fontsize=8.5, color=GREEN)
    box(ax, .56, .294, .215, .091, 'Gemini 설명', '연동 코드 · 모의 검증', title_size=10.5, body_size=9)
    box(ax, .812, .294, .134, .091, '기본 설명', 'OFF / 실패', color=MUTED,
        fill='white', title_size=10.5, body_size=9)
    arrow(ax, (.69, .43), (.667, .394))
    arrow(ax, (.864, .43), (.879, .394))
    arrow(ax, (.782, .34), (.801, .34), dashed=True)
    box(ax, .56, .177, .385, .065, '예측 · 설명 · 입력 품질 JSON 반환', title_size=10.2)
    arrow(ax, (.667, .285), (.69, .25))
    arrow(ax, (.879, .285), (.864, .25))
    ax.text(.025, .071, '개선 후보는 서비스 모델과 별도입니다. Gemini 실제 외부 호출 성공은 추가 검증이 필요합니다.',
            fontsize=9.5, color=MUTED)
    save(fig, 'system_architecture')


if __name__ == '__main__':
    meta = json.loads((ROOT / 'docs/generated/data_structure.json').read_text())
    assert meta['summary']['rows'] == meta['summary']['observed_targets'] + meta['summary']['missing_targets']
    assert meta['summary']['duplicate_keys'] == 0
    setup()
    data_pipeline(meta)
    system_architecture()
    print('Generated data_pipeline and system_architecture PNG/SVG files from aggregate metadata')
