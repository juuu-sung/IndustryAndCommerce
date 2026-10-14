"""Escaped HTML separating backend facts, general sources and review decisions."""
from html import escape
from urllib.parse import urlparse

from .reports import validate_envelope


def e(value):
    return escape(str(value),quote=True)


def render_html(report,review=None):
    validate_envelope(report)
    base=report['base_report']
    case=base['case']
    labels={'review_required':'검토 필요','unverified':'유효성 미확인',
            'pending_provider_confirmation':'기관 확인 대기','unavailable':'관측 없음'}
    rows=''
    for fact in report['facts']:
        value=fact['value']
        shown='확인 불가' if value is None else format(value,',.2f') if type(value) is float else labels.get(value,str(value))
        rows+=f'<tr><th>{e(fact["label"])}</th><td>{e(shown)}</td><td>{e(fact["unit"] or "—")}</td><td><code>{e(fact["fact_id"])}</code></td></tr>'
    claims=''.join('<li><p>'+e(s['text'])+'</p><small>'+e(' · '.join(s['fact_refs']+s['citation_refs']))+'</small></li>'
                   for s in report['analysis']['content']['statements'])
    sources=''
    for citation in report['citations']:
        loc=citation['locator']
        position=('PDF 실제 '+str(loc['pdf_page_number'])+'페이지; 인쇄 표기 '+str(loc.get('printed_page_label') or '미확인')
                  if loc['kind']=='pdf_page' else '절/추출 위치: '+loc.get('section',str(loc.get('block_number',''))))
        url=citation['source_url']
        parsed=urlparse(url)
        title=(f'<a href="{e(url)}" rel="noopener noreferrer" target="_blank">{e(citation["title"])}</a>'
               if parsed.scheme=='https' and parsed.netloc and not parsed.username else e(citation['title']))
        scope={'project_method':'프로젝트 연구 방법','general_technical':{'US':'미국','KR':'한국'}[citation['country']]+' 일반 기술자료',
               'korean_tms_general':'국내 TMS 개요'}[citation['scope']]
        sources+=f'<tr><th>{e(citation["citation_id"])}</th><td>{title}<p class="muted">{e(position)}</p></td><td>{e(scope)}<p>{e(citation["version_label"])}</p></td></tr>'
    limitations=''.join('<li>'+e(s)+'</li>' for s in report['limitations'])
    status='검토 전 초안'
    if review is not None:
        if review.get('report_id')!=report['report_id'] or review.get('report_revision')!=report['revision']:
            raise ValueError('Review metadata does not match the report revision')
        status={'waiting_review':'사용자 검토 대기','succeeded':'사용자 승인 후 완료',
                'cancelled':'사용자 거절로 종료'}.get(review['status'],review['status'])
    note=f'<section><h2>사용자 개정 메모</h2><p>{e(report["review_note"])}</p></section>' if report['review_note'] else ''
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(case['plant'])} NOx 근거 보고서</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#17263a;font-family:system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif;line-height:1.7}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:27px}}h2{{font-size:20px}}section{{background:white;border:1px solid #dce4ee;border-radius:14px;padding:22px;margin:18px 0}}.badge{{background:#e8e0ff;padding:6px 12px;border-radius:18px;font-weight:700}}.warning{{background:#fff3d6;border-left:4px solid #b87c19;padding:18px}}.muted,small{{color:#53647b}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #e3e9f0}}thead{{background:#edf3fa}}code{{overflow-wrap:anywhere}}li{{margin:12px 0}}a{{color:#245e98}}footer{{overflow-wrap:anywhere;font-size:12px;color:#53647b}}@media(max-width:650px){{main{{padding:14px}}section{{padding:15px}}h1{{font-size:23px}}}}
</style></head><body><main><span class="badge">{e(status)}</span>
<h1>{e(case['plant'])} {e(case['unit'])}호기 · {e(case['month'])} NOx 근거 보고서</h1>
<p>{e(base['target_definition']['description'])}</p>
<div class="warning"><strong>측정 유효성은 기관 확인 대기입니다.</strong><p>저장된 과거 평가를 재현했습니다. 일반 문서가 실제 호기의 고장이나 측정 유효성을 증명하지 않습니다.</p></div>
<section><h2>서버가 계산한 관측·예측·비교</h2><div class="table"><table><thead><tr><th>항목</th><th>값</th><th>단위</th><th>수치 근거 ID</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>사실과 근거를 연결한 설명</h2><p class="muted">설명 상태: {e(report['analysis']['status'])} · 의미적 근거 충실성은 사용자 검토 필요</p><ul>{claims}</ul></section>
<section><h2>문서 근거와 적용 범위</h2><div class="table"><table><thead><tr><th>근거 ID</th><th>문서·원문 위치</th><th>범위·버전</th></tr></thead><tbody>{sources}</tbody></table></div><p class="muted">원문 링크와 페이지/절을 대조하세요. 자동 추출된 표·수치는 원문 확인이 필요합니다.</p></section>
{note}<section><h2>확인할 사항과 한계</h2><ul>{limitations}</ul></section>
<footer>보고서 {e(report['report_id'])} · 개정 {e(report['revision'])}<br>수치·문서 내용 기준 {e(report['context_hash'])}<br>사용자 승인은 보고서 검토이며 기관 측정 인증이나 설비 제어 승인이 아닙니다.</footer>
</main></body></html>'''
