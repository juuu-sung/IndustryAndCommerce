from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['plant', 'unit', 'month']
FUELS = {
    '유연탄': 'bituminous_ton', '무연탄': 'anthracite_ton',
    '유류': 'oil_kl', 'LNG': 'lng_ton',
    '고형연료': 'solid_ton', '우드펠릿': 'pellet_ton',
}
OPERATIONS = {
    '용량(MW)': 'capacity_mw', '발전량(MWh)': 'generation_mwh',
    '열효율(%)': 'thermal_efficiency_pct', '이용률(%)': 'utilization_pct',
}
CONDITIONS = [*OPERATIONS.values(), *FUELS.values(),
              'temperature_c', 'humidity_pct', 'wind_speed_ms',
              'wind_sin', 'wind_cos']
TARGETS = {
    'legacy': {
        'name': 'legacy_nox_proxy', 'unit': 'unverified',
        'description': '기존 NOX_kg 열의 연구용 값. 월 배출 질량으로 검증되지 않음.',
        'limitations': ['유량 단위 및 시간 적산 근거 미확인',
                        '기존 병합 파일의 운전/타깃 선택 편향이 남아 있음',
                        'kg 단위 배출량 또는 법규 판단에 사용할 수 없음'],
    },
    'concentration': {
        'name': 'monthly_daily_stack_mean_nox', 'unit': 'ppm',
        'description': '관측된 일평균 NOx 농도를 호기/날짜별로 평균한 뒤 월별 산술평균한 연구용 타깃',
        'limitations': ['일평균값의 월별 산술평균이며 공식 월평균/월 질량과 다름',
                        '삼천포 A/B 농도는 일별 산술평균이며 유량 가중 농도가 아님',
                        '여수의 호기 미표기(-) 자료는 대응 근거가 없어 제외',
                        '0의 정지/결측 구분 및 측정 유효 플래그가 없음',
                        '농도 ppm는 원 제공기관 농도 표기를 근거로 함; 일자료 상세 정의 추가 확인 필요'],
    },
}
