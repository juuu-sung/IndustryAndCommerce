"""Render validated report JSON as escaped, self-contained Korean HTML."""
from html import escape

from .validators import validate_report

METHOD_LABELS = {
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'ensemble': '선정 앙상블',
    'train_global_median': '학습 전체 중앙값', 'train_unit_mean': '학습 호기 평균',
    'last_month': '전월 값', 'previous_3months': '직전 3개월 평균',
}
QUALITY_LABELS = {'valid': '유효 상태 기록', 'unverified': '유효성 미확인',
                  'review_required': '검토 필요', 'unavailable': '관측 없음'}


def text(value):
    return escape(str(value), quote=True)


def number(value, suffix='', signed=False):
    if value is None:
        return '확인 불가'
    return text(format(value, '+,.2f' if signed else ',.2f') + suffix)


def coverage_text(value):
    if value['status'] != 'available':
        return '확인 불가'
    return text(f"{value['observed_days']}/{value['calendar_days']}일 ({value['coverage_ratio'] * 100:.1f}%)")


def render_html(report):
    validate_report(report)
    case = report['case']
    current = report['observed']
    comparisons = report['comparisons']
    previous = comparisons['previous_month']
    rolling = comparisons['previous_3months']
    coverages = {item['month']: item['emission'] for item in report['source_quality']['by_month']}
    title = f"{case['plant']} {case['unit']}호기 · {case['month']} NOx 분석"
    cards = ''.join(f'<article class="card"><span>{text(label)}</span><strong>{value}</strong></article>' for label, value in (
        ('현재 월 관측', number(current['value'], ' ppm')),
        ('전월 대비 차이', number(previous['delta_ppm'], ' ppm', signed=True)),
        ('전월 대비 변화율', number(previous['change_percent'], '%', signed=True)),
        ('직전 3개월 평균', number(rolling['mean_ppm'], ' ppm')),
    ))
    history = ''.join(
        '<tr><td>'+text(o['month'])+'</td><td>'+number(o['value'], ' ppm')+'</td><td>'
        +text(QUALITY_LABELS[o['quality_status']])+'</td><td>'+coverage_text(coverages[o['month']])+'</td><td>'
        +text(' · '.join(o['quality_reasons']) or o['missing_reason'] or '기록된 사유 없음')+'</td></tr>'
        for o in [*report['history'], current])
    predictions = ''.join(
        '<tr><th scope="row">'+text(METHOD_LABELS[method])+'</th><td>'+number(value['prediction_ppm'])
        +'</td><td>'+number(value['signed_error_ppm'], signed=True)+'</td><td>'
        +number(value['absolute_error_ppm'])+'</td></tr>' for method in report['selected_methods']
        for value in [report['predictions'][method]])
    checks = ''.join('<li><strong>'+text(check['code'])+'</strong> · '+text(', '.join(check['months']))
                     +'<p>'+text(check['message'])+'</p></li>' for check in report['checks'])
    limitations = ''.join('<li>'+text(value)+'</li>' for value in report['limitations'])
    periods = ''.join('<tr><th scope="row">'+text(label)+'</th><td>'+text(' ~ '.join(report['protocol'][key]))
                      +'</td></tr>' for key, label in [('train', '학습'), ('validation', '가중치 선정'), ('test', '평가')])
    provenance = ''.join('<tr><th scope="row">'+text(name)+'</th><td class="hash">'+text(value)+'</td></tr>'
                         for name, value in (
                             ('사례 버전', report['provenance']['case_version_id']),
                             ('계산 버전', report['calculation']['calculation_id']),
                             ('모델 버전', report['provenance']['model_version_id']),
                             ('타깃 정의', report['provenance']['target_definition_sha256']),
                             ('관측 파일', report['provenance']['history_file_sha256']),
                             ('예측 파일', report['provenance']['prediction_file_sha256']),
                             ('보고서 ID', report['report_id']), ('내용 기준', report['context_hash']),
                         ))
    reason = ', '.join(previous['change_percent_reasons'])
    undefined_rate = ('<p class="muted">변화율 확인 불가 사유: '+text(reason)+'</p>') if reason else ''
    fuel = {'provider_reported_unit_month': '제공기관의 실제 호기별 월 실적',
            'allocated_from_site_total': '사업소 합계의 호기별 배분 추정'}.get(report['input_quality']['fuel_basis'],
                                                                            report['input_quality']['fuel_basis'] or '근거 미기록')
    model = report['model']
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>{text(title)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#16243b;font-family:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;line-height:1.65}}main{{max-width:1100px;margin:auto;padding:36px 24px}}header{{margin-bottom:22px}}h1{{font-size:28px;line-height:1.4;margin:12px 0}}h2{{font-size:20px;margin:0 0 14px}}p{{margin:6px 0}}.badge{{display:inline-block;background:#e8e0ff;color:#513b80;border-radius:20px;padding:4px 12px;font-size:13px;font-weight:700}}.muted{{color:#596b81;font-size:14px}}.warning{{border-left:4px solid #b77910;background:#fff5d9;padding:16px 20px;border-radius:8px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,section{{background:white;border:1px solid #dce4ee;border-radius:14px;padding:20px}}.card span{{display:block;font-size:13px;color:#60718a}}.card strong{{display:block;font-size:23px;margin-top:7px;line-height:1.4}}section{{margin-top:20px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 12px;border-bottom:1px solid #e4eaf2;text-align:left;vertical-align:top}}thead th{{background:#edf3fa;font-weight:700;white-space:nowrap}}tbody th{{font-weight:600}}.hash{{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:12px}}li{{margin:10px 0}}li p{{color:#485b73}}footer{{font-size:12px;color:#60718a;margin:22px 0;overflow-wrap:anywhere}}details{{margin-top:16px}}summary{{cursor:pointer;color:#365d89;font-weight:600}}@media(max-width:780px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:24px 14px}}h1{{font-size:24px}}.card strong{{font-size:20px}}}}@media(max-width:420px){{.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{padding:0}}section,.card{{break-inside:avoid}}.table-wrap{{overflow:visible}}}}
</style></head><body><main>
<header><span class="badge">연구용 · 검토 전 초안</span><h1>{text(title)}</h1>
<p>{text(report['target_definition']['description'])}</p><p class="muted">저장 평가 재현 · 생성 시각 {text(report['report_generated_at'])}</p></header>
<div class="warning"><strong>측정 유효성은 제공기관 확인 대기입니다.</strong><p>현재 품질 상태: {text(QUALITY_LABELS[current['quality_status']])}. 이 보고서는 사후 평가 재현이며 해당 월 당시 서비스가 생성한 미래 예측 기록이 아닙니다.</p></div>
<div class="cards">{cards}</div>{undefined_rate}
<p class="muted">직전 3개월 관측: {text(rolling['observed_count'])}/3개 달 · 0값과 검토 필요 관측도 원본대로 포함합니다. 수치는 소수 둘째 자리 표시이며 JSON에는 원 정밀도를 유지합니다.</p>
<section><h2>1. 관측 추세와 품질</h2><div class="table-wrap"><table><thead><tr><th>월</th><th>NOx 농도</th><th>품질 상태</th><th>관측일 충족</th><th>검토·누락 사유</th></tr></thead><tbody>{history}</tbody></table></div><p class="muted">관측일 충족률은 일자료 존재 비율이며 유효 측정시간·가동시간의 충족률이 아닙니다.</p></section>
<section><h2>2. 저장 예측과 관측 비교</h2><p class="muted">모든 수치의 단위는 ppm입니다. 부호 오차 = 예측 − 관측. 한 사례의 절대 오차는 RMSE가 아닙니다.</p>
<div class="table-wrap"><table><thead><tr><th>방법</th><th>저장 예측</th><th>부호 오차</th><th>절대 오차</th></tr></thead><tbody>{predictions}</tbody></table></div>
<p class="muted">앙상블 가중치: XGBoost {number(model['weights']['xgboost'])} / LightGBM {number(model['weights']['lightgbm'])}. 개선 후보 모델의 새 추론과 구분합니다.</p></section>
<section><h2>3. 입력 자료와 추가 확인 사항</h2><p>연료 자료 근거: {text(fuel)}. ton과 kL를 혼합하거나 NOx를 kg으로 환산하지 않습니다.</p>
<p>현재 월 NOx 관측일 충족: {coverage_text(report['input_quality']['daily_observation_coverage'])} · 기온 관측일 충족: {coverage_text(report['input_quality']['weather_temperature_coverage'])}</p>
<p class="muted">특징별 결측 보완·학습 범위 경고는 이 고정 보고서 계약에서 확인하지 않았습니다. 확인하지 않은 항목을 0이나 정상으로 표시하지 않습니다.</p><ul>{checks}</ul></section>
<section><h2>4. 모델·기간·출처</h2><p>모델 생성 시각: {text(model['created_at'])}</p><div class="table-wrap"><table><tbody>{periods}</tbody></table></div>
<details><summary>파일 및 내용 버전 확인</summary><div class="table-wrap"><table><tbody>{provenance}</tbody></table></div></details>
<p class="muted">원천 manifest 해시는 모델 메타데이터의 선언 값을 보존합니다. 원천 manifest 파일 자체를 재검증한 것으로 표시하지 않습니다.</p></section>
<section><h2>5. 분석의 한계</h2><ul>{limitations}</ul><p class="muted">문서 근거: 아직 연결하지 않음 · LLM: 사용하지 않음 · 사용자 승인: 아직 진행하지 않음</p></section>
<footer>보고서 {text(report['report_id'])} · {text(report['report_version'])} · 개정 1</footer>
</main></body></html>'''
