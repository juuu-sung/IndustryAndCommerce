"""A small, source-checked initial benchmark; expert review remains pending."""
import re

from .common import RUNTIME, RagError, write_new
from .ingest import Corpus
from ..registry import _digest

GOLD_PATH = RUNTIME / 'gold_v1.json'


def ref(unit, needle):
    return {'unit_id': unit, 'needle': needle}


def questions():
    # Development and test questions are assigned before running search comparisons.
    dev = [
        ('NOx 연구 타깃은 무엇을 월별 평균한 값인가요?', 'target', [ref('project_data:section_48','일평균값의 월별 산술평균')]),
        ('ppm에 발전량을 곱해서 월 배출량 kg을 만들 수 있나요?', 'units', [ref('project_data:section_48','현재 자료만으로 ppm에 발전량을 곱해 kg 타깃을 만들지 않습니다.')]),
        ('연료의 ton과 유류 kL를 왜 합산하지 않나요?', 'units', [ref('project_data:section_19','질량 연료와 합산하지 않음')]),
        ('전월 자료가 누락되면 이전 관측 행을 사용하나요?', 'calendar', [ref('project_data:section_60','누락된 달을 이전 관측 행으로 대체하지 않음')]),
        ('NOx 정답이 없는 달을 0이나 중앙값으로 채워 학습하나요?', 'quality', [ref('project_data:section_19','타깃 결측 32행을 0이나 중앙값으로 채우지 않습니다.')]),
        ('결측 보완값과 특징 선택 기준은 어느 학습 구간에서 정하나요?', 'validation', [ref('project_data:section_60','전처리·열 선택·중앙값·가중치는 학습/선정 구간에서만 산출')]),
        ('사업소 연료 배분 추정과 실제 호기 연료는 어떻게 구분하나요?', 'quality', [ref('project_data:section_19','880행은 배분 추정')]),
        ('시간별 발전 패턴을 SCR 운전 상태로 해석할 수 있나요?', 'scope', [ref('project_data:section_3','전기 발전 패턴이며 SCR 운전 상태가 아님')]),
        ('월 NOx 질량 타깃에는 농도 외에 어떤 자료가 필요하나요?', 'units', [ref('project_data:section_48','같은 시간대의 농도·배출가스 유량·유효 시간·표준 상태')]),
        ('반복 확인한 개발 기간을 독립 최종 평가로 부를 수 있나요?', 'validation', [ref('project_evaluation:section_18','이 기간은 후속 개선에도 반복해서 확인했으므로 독립 최종 평가로 부르지 않습니다.')]),
        ('새 호기와 기존 호기의 평가 자료는 어떻게 분리했나요?', 'validation', [ref('project_evaluation:section_18','학습에 없는 여수 2호기 12개를 분리한 기존 호기')]),
        ('RMSE와 MAE는 큰 급등 오차에 어떻게 반응하나요?', 'metric', [ref('project_evaluation:section_18','RMSE는 제곱 오차 평균의 제곱근이라 급등을 크게 놓칠수록 커집니다.')]),
        ('Gemini 실제 외부 호출은 완료됐나요 아니면 모의 응답 검사인가요?', 'scope', [ref('project_api:section_3','모의 응답')]),
        ('LLM이 NOx 수치와 단위를 임의로 바꿀 수 있나요?', 'service', [ref('project_api:section_84','백엔드는 NOx 수치와 단위를 직접 관리')]),
        ('예측 API의 as_of는 어떤 이력 기준을 의미하나요?', 'service', [ref('project_api:section_19','그때 확보한 이력만 사용')]),
        ('관측일 충족률은 유효 측정시간 충족률과 같은가요?', 'quality', [ref('project_reports:section_17','관측일 충족률은 일자료 존재 비율이며 유효 측정·운전시간 충족률이 아닙니다.')]),
        ('기관 공식 일평균을 재현할 때 원시 구간은 어떻게 가중하나요?', 'validation', [ref('project_target_validation:section_45','유효 초 가중 일평균')]),
        ('저장 예측의 부호 오차는 예측에서 관측을 빼나요?', 'metric', [ref('project_calculations:section_5','저장 예측 − 현재 관측')]),
        ('전월 NOx가 0이면 변화율을 어떻게 처리하나요?', 'calendar', [ref('project_calculations:section_5','previous_month_zero')]),
        ('급등 유효성 확인과 실제 운전 기록은 무엇을 요청해야 하나요?', 'multi', [ref('project_data:section_85','NOx 보정 전·후 값'),ref('project_target_validation:section_23','기동')]),
        ('SCR에서 사용하는 NOx 환원제는 암모니아와 요소 중 무엇인가요?', 'technical', [ref('epa_scr_manual:page_5','Either ammonia or urea may be used as the NOx reduction reagent in SCR systems.')]),
        ('SCR 반응은 질소산화물을 어떤 생성물로 바꾸나요?', 'technical', [ref('epa_scr_manual:page_14','reduce it to N2 and H2O')]),
        ('SCR 성능에 영향을 주는 반응 온도와 체류시간은 어디에서 확인하나요?', 'technical', [ref('epa_scr_manual:page_19','Reaction temperature range;'),ref('epa_scr_manual:page_19','Residence time available in the optimum temperature range;')]),
        ('SNCR 반응 온도가 너무 낮거나 높으면 어떤 문제가 발생하나요?', 'technical', [ref('epa_sncr_manual:page_17','at lower temperatures, the reaction kinetics are slow: at higher temperatures, the reagent oxidizes and additional NOx is generated.')]),
        ('SCR와 SNCR의 성능에 영향을 주는 운전 요소를 각각 찾아주세요.', 'multi', [ref('epa_scr_manual:page_19','major design and operational factors'),ref('epa_sncr_manual:page_4','Temperature, residence time, type of NOx reducing reagent')]),
        ('굴뚝 TMS CleanSYS는 어떤 시스템인가요?', 'technical', [ref('keco_cleansys:cleansys_overview','관제센터와 온라인으로 연결')]),
        ('영동 2025년 10월 NOx 예측값을 문서에서 찾아주세요.', 'unsupported', 'requires_numeric_tool'),
        ('영동 SCR 고장 원인을 확정해 주세요.', 'unsupported', 'requires_provider_evidence'),
        ('국내 NOx 법정 허용기준 수치를 알려주세요.', 'unsupported', 'requires_verified_regulation'),
        ('김치볶음밥 조리 순서를 알려주세요.', 'unsupported', 'insufficient_evidence'),
    ]
    test = [
        ('NOx 기록의 0을 정지 상태로 보고 삭제해도 되나요?', 'quality', [ref('project_data:section_48','값 0은 정지·무효·실측 0을 구분할 플래그가 없어 임의로 삭제하지 않았습니다.')]),
        ('직전 3개월 평균에 이번 달의 NOx도 포함되나요?', 'calendar', [ref('project_calculations:section_5','현재 달을 제외한 정확한 3개 달')]),
        ('월별 풍향을 단순 산술평균하지 않고 어떻게 처리했나요?', 'quality', [ref('project_data:section_60','풍향은 원형 평균')]),
        ('관측일 100% 충족이면 센서 측정값도 유효하다고 인증된 건가요?', 'scope', [ref('project_reports:section_17','유효 측정·운전시간 충족률이 아닙니다.')]),
        ('사례 파일의 해시가 맞으면 측정 유효성도 확인된 것인가요?', 'scope', [ref('project_calculations:section_5','측정 유효성 인증으로 해석하지 않습니다.')]),
        ('이력의 입수 기준 시각을 제한하는 API 필드는 무엇인가요?', 'service', [ref('project_api:section_19','선택적 시간대 포함 입수 기준 시각')]),
        ('NOx 농도는 발전 실적이 늘면 반드시 비례해서 커지나요?', 'target', [ref('project_data:section_48','농도와 발전량이 반드시 비례하지 않습니다.')]),
        ('NOx 500 초과 같은 검토 조건을 법적 무효 기준으로 사용해도 되나요?', 'scope', [ref('project_data:section_77','법적 무효 기준이 아닙니다.')]),
        ('기상 제외 모델이 유리했다면 기상은 NOx와 무관하다고 결론내릴 수 있나요?', 'validation', [ref('project_data:section_60','기상 자체가 무관하다는 결론이 아닙니다.')]),
        ('시간 검증에서 같은 달의 여러 호기를 다른 기간에 나눠도 되나요?', 'validation', [ref('project_evaluation:section_18','같은 달의 모든 호기를 함께 나눕니다.')]),
        ('LLM 설명에서 ppm을 kg으로 환산하도록 사용했나요?', 'service', [ref('project_api:section_84','LLM은 ppm→kg 환산')]),
        ('SCR의 NOx 저감 반응도 유효 온도 범위가 있나요?', 'technical', [ref('epa_scr_manual:page_19','The NOx reduction reaction is effective only within a given temperature range.')]),
        ('SCR 성능 점검에는 촉매 비활성과 촉매 관리도 포함되나요?', 'technical', [ref('epa_scr_manual:page_19','Catalyst deactivation;'),ref('epa_scr_manual:page_19','Catalyst management.')]),
        ('SNCR에서 저부하 운전이나 연료 변경은 온도 분포에 어떤 영향을 주나요?', 'technical', [ref('epa_sncr_manual:page_18','Combustion units operated at low load or with different fuels may result in changes in the temperature profile')]),
        ('암모니아 슬립 ammonia slip의 뜻은 무엇인가요?', 'technical', [ref('epa_sncr_fact:block_1','Ammonia slip refers to emissions of unreacted ammonia')]),
        ('타깃 자료와 LLM 응답 처리에서 임의 질량 환산을 어떻게 제한하나요?', 'multi', [ref('project_data:section_48','현재 자료만으로 ppm에 발전량을 곱해 kg 타깃을 만들지 않습니다.'),ref('project_api:section_84','LLM은 ppm→kg 환산')]),
        ('삼천포 2025년 10월 NOx 농도를 찾아 알려주세요.', 'unsupported', 'requires_numeric_tool'),
        ('여수의 실제 촉매 교체 일시를 알려주세요.', 'unsupported', 'requires_provider_evidence'),
        ('국내 발전소의 NOx 허용기준 수치와 위반 확정을 해주세요.', 'unsupported', 'requires_verified_regulation'),
        ('초콜릿 케이크 만드는 레시피를 알려주세요.', 'unsupported', 'insufficient_evidence'),
    ]
    return [('development', f'D{i:02d}', *item) for i, item in enumerate(dev,1)] + [('test',f'T{i:02d}',*item) for i,item in enumerate(test,1)]


def build_gold(corpus=None, path=GOLD_PATH):
    corpus = corpus if corpus is not None else Corpus()
    manifest, units = corpus.verify()
    by_unit = {unit['unit_id']: unit for unit in units}
    records = []
    for split, qid, query, category, evidence in questions():
        groups = []
        status = evidence if isinstance(evidence, str) else 'answerable'
        if not isinstance(evidence, str):
            for i, reference in enumerate(evidence,1):
                unit = by_unit[reference['unit_id']]
                pattern = r'\s+'.join(re.escape(word) for word in reference['needle'].split())
                match = re.search(pattern, unit['text'], flags=re.IGNORECASE)
                if not match:
                    raise RagError('gold_span_missing', f'Gold evidence was not found: {qid}/{reference["unit_id"]}')
                groups.append({'group_id': qid + f'_g{i}', 'alternatives': [{
                    'unit_id': unit['unit_id'], 'unit_sha256': unit['unit_sha256'],
                    'document_version_id': unit['document_version_id'],
                    'char_start': match.start(), 'char_end': match.end(),
                }]})
        records.append({'question_id': qid, 'split': split, 'query': query, 'category': category,
                        'expected_status': status, 'required_evidence_groups': groups})
    content = {'schema_version': 1, 'corpus_version_id': manifest['corpus_version_id'],
               'annotation_status': 'assistant_source_span_checked; independent_expert_review_pending',
               'split_policy': '30 development / 20 test, assigned before search comparison',
               'questions': records}
    content['gold_version_id'] = _digest(content)
    write_new(path, content)
    return content


if __name__ == '__main__':
    import json
    result = build_gold()
    print(json.dumps({'questions': len(result['questions']), 'gold_version_id': result['gold_version_id']}))
