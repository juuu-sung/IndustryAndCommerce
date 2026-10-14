"""All displayed numbers are copied from the verified deterministic report."""
from ..validators import validate_report


def make_facts(base):
    validate_report(base)
    previous = base['comparisons']['previous_month']
    rolling = base['comparisons']['previous_3months']
    values = [
        ('F_CURRENT', '현재 월 NOx 관측', base['observed']['value'], 'ppm', 'observed'),
        ('F_PREVIOUS', '정확한 전월 NOx 관측', previous['observation']['value'], 'ppm', 'comparisons.previous_month'),
        ('F_DELTA', '전월 대비 차이', previous['delta_ppm'], 'ppm', 'comparisons.previous_month'),
        ('F_CHANGE', '전월 대비 변화율', previous['change_percent'], '%', 'comparisons.previous_month'),
        ('F_MEAN', '직전 3개월 평균', rolling['mean_ppm'], 'ppm', 'comparisons.previous_3months'),
        ('F_HISTORY_COUNT', '직전 3개월 관측 수', rolling['observed_count'], 'months', 'comparisons.previous_3months'),
        ('F_QUALITY', '현재 관측 품질', base['observed']['quality_status'], None, 'observed'),
        ('F_PROVIDER', '기관 측정 유효성 확인', base['target_verification_status'], None, 'target_verification_status'),
    ]
    for method in base['selected_methods']:
        prediction = base['predictions'][method]
        values += [
            ('F_PRED_' + method.upper(), method + ' 저장 예측', prediction['prediction_ppm'], 'ppm', 'predictions.' + method),
            ('F_ERROR_' + method.upper(), method + ' 절대 오차', prediction['absolute_error_ppm'], 'ppm', 'predictions.' + method),
        ]
    return [{'fact_id': fid, 'label': label, 'value': value, 'unit': unit,
             'source_field': source, 'base_report_id': base['report_id']} for fid, label, value, unit, source in values]


def citations_for(contexts):
    unique = {}
    for section in contexts:
        for hit in section['response']['results']:
            key = (hit['document_version_id'], hit['chunk_id'])
            if key not in unique:
                unique[key] = {k: hit[k] for k in ('chunk_id', 'unit_id', 'document_id', 'document_version_id',
                                                  'source_sha256', 'title', 'source_url', 'version_label',
                                                  'language', 'country', 'scope', 'plant_specific_evidence',
                                                  'locator', 'text', 'warnings')}
                unique[key]['section_ids'] = []
            if section['section_id'] not in unique[key]['section_ids']:
                unique[key]['section_ids'].append(section['section_id'])
    return [{'citation_id': f'C{i:02d}', **value} for i, (_, value) in enumerate(sorted(unique.items()), 1)]
