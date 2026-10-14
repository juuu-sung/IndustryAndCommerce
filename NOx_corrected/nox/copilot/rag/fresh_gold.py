"""Additional source-bound questions fixed before the new search comparison."""
import re
from .common import RUNTIME, RagError, write_new
from .ingest import Corpus
from ..registry import _digest

PATH = RUNTIME / 'fresh_gold_20261014.json'


def build(corpus=None, path=PATH):
    corpus = corpus or Corpus()
    manifest, units = corpus.verify()
    units = {u['unit_id']: u for u in units}
    answerable = [
        ('프로젝트의 NOx 정답 단위와 월평균 산정 방식은?', 'project_data:section_48', '일평균값의 월별 산술평균'),
        ('월 발전 실적과 ppm만으로 질량 kg 정답을 계산할 수 있습니까?', 'project_data:section_48', '현재 자료만으로 ppm에 발전량을 곱해 kg 타깃을 만들지 않습니다.'),
        ('유류 kL를 석탄 ton과 더해서 총연료로 쓰면 안 되는 이유는?', 'project_data:section_19', '질량 연료와 합산하지 않음'),
        ('전월 관측이 비었을 때 두 달 전 행을 전월 특징으로 쓰나요?', 'project_data:section_60', '누락된 달을 이전 관측 행으로 대체하지 않음'),
        ('NOx 정답 결측 월도 중앙값으로 채워 모델 학습에 포함합니까?', 'project_data:section_19', '타깃 결측 32행을 0이나 중앙값으로 채우지 않습니다.'),
        ('풍향을 월 단위 특징으로 만들 때 방향의 원형 특성을 고려했나요?', 'project_data:section_60', '풍향은 원형 평균'),
        ('이력 조회 API에서 입수 당시의 정보를 제한하는 시각 필드는?', 'project_api:section_19', '선택적 시간대 포함 입수 기준 시각'),
        ('NOx 500이라는 검토 숫자로 측정을 법적 무효 처리해도 되나요?', 'project_data:section_77', '법적 무효 기준이 아닙니다.'),
        ('NOx 관측일 충족률이 높으면 유효 측정시간도 충족한 것인가요?', 'project_reports:section_17', '유효 측정·운전시간 충족률이 아닙니다.'),
        ('기상 제외 후보의 성능이 좋아졌다고 날씨 영향이 없다고 할 수 있나요?', 'project_data:section_60', '기상 자체가 무관하다는 결론이 아닙니다.'),
        ('LLM 출력에서 연구 농도를 질량으로 변환하는 기능이 있나요?', 'project_api:section_84', 'LLM은 ppm→kg 환산'),
        ('같은 월에 관측된 여러 호기의 모델 평가 기간을 따로 나눌 수 있나요?', 'project_evaluation:section_18', '같은 달의 모든 호기를 함께 나눕니다.'),
        ('SCR의 reducing reagent로 요소도 암모니아처럼 사용할 수 있나요?', 'epa_scr_manual:page_5', 'Either ammonia or urea may be used as the NOx reduction reagent in SCR systems.'),
        ('SCR가 NOx를 환원했을 때 질소와 물이 생성되나요?', 'epa_scr_manual:page_14', 'reduce it to N2 and H2O'),
        ('SCR reaction temperature와 residence time이 운영 성능 요소인가요?', 'epa_scr_manual:page_19', 'Residence time available in the optimum temperature range;'),
        ('SNCR 온도가 낮거나 높을 때 kinetics와 oxidation에 무슨 변화가 있나요?', 'epa_sncr_manual:page_17', 'at lower temperatures, the reaction kinetics are slow: at higher temperatures, the reagent oxidizes and additional NOx is generated.'),
        ('SNCR low load나 different fuels가 temperature profile에 영향을 줍니까?', 'epa_sncr_manual:page_18', 'Combustion units operated at low load or with different fuels may result in changes in the temperature profile'),
        ('SCR catalyst deactivation과 catalyst management를 점검해야 하나요?', 'epa_scr_manual:page_19', 'Catalyst management.'),
        ('SNCR의 ammonia slip은 반응하지 않은 암모니아 배출을 뜻하나요?', 'epa_sncr_fact:block_1', 'Ammonia slip refers to emissions of unreacted ammonia'),
        ('CleanSYS의 굴뚝 측정은 관제센터와 어떻게 연결됩니까?', 'keco_cleansys:cleansys_overview', '관제센터와 온라인으로 연결'),
    ]
    unsupported = [
        ('영동 2025년 10월 월 NOx 농도와 예측값을 알려주세요.', 'requires_numeric_tool'),
        ('삼천포 2026년 실제 발전량은?', 'requires_numeric_tool'),
        ('영동의 센서 무효 기록을 확정해 주세요.', 'requires_provider_evidence'),
        ('영동 촉매의 실제 수명은 몇 년인가요?', 'requires_provider_evidence'),
        ('여수 SCR 고장 발생 여부를 단정해 주세요.', 'requires_provider_evidence'),
        ('국내 법적 NOx 기준과 위반 여부는?', 'requires_verified_regulation'),
        ('NOx 문서와 무관한 케이크 레시피를 알려주세요.', 'insufficient_evidence'),
        ('주말 남해 여행지를 추천해 주세요.', 'insufficient_evidence'),
        ('오늘 축구 결과를 알려주세요.', 'insufficient_evidence'),
        ('복권 번호를 골라 주세요.', 'insufficient_evidence'),
    ]
    records = []
    for i, (query, uid, needle) in enumerate(answerable, 1):
        unit = units[uid]
        match = re.search(r'\s+'.join(re.escape(w) for w in needle.split()), unit['text'], re.I)
        if match is None:
            raise RagError('gold_span_missing', f'Additional source span missing: F{i:02d}')
        records.append({'question_id': f'F{i:02d}', 'split': 'test', 'query': query,
                        'category': 'technical' if uid.startswith('epa_') else 'project', 'expected_status': 'answerable',
                        'required_evidence_groups': [{'group_id': f'F{i:02d}_g1', 'alternatives': [{
                            'unit_id': uid, 'unit_sha256': unit['unit_sha256'],
                            'document_version_id': unit['document_version_id'],
                            'char_start': match.start(), 'char_end': match.end()}]}]})
    for i, (query, status) in enumerate(unsupported, 21):
        records.append({'question_id': f'F{i:02d}', 'split': 'test', 'query': query,
                        'category': 'unsupported', 'expected_status': status, 'required_evidence_groups': []})
    core = {'schema_version': 1, 'corpus_version_id': manifest['corpus_version_id'],
            'annotation_status': 'assistant_source_span_checked; independent_expert_review_pending',
            'purpose': 'Additional wording on the same source concepts; not an independent expert benchmark.',
            'questions': records}
    core['gold_version_id'] = _digest(core)
    write_new(path, core)
    return core


if __name__ == '__main__':
    import json
    result = build()
    print(json.dumps({'questions': len(result['questions']), 'gold_version_id': result['gold_version_id']}))
