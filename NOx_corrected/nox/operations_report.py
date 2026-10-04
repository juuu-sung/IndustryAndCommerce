"""Write the fixed experiment report and concrete, unsent agency data request."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .operating_patterns import FEATURES, HOURS, digest, read_hourly
from .schema import ROOT
from .temporal_validation import prepare_data, request_for
from .train import write_json


def main():
    report = ROOT / 'reports/operations_validation_20261002'
    hourly = ROOT / 'data/hourly_operations_20261002'
    if not (report / 'COMPLETE').exists():
        raise ValueError('Completed feature experiment required')
    summary = json.loads((report / 'summary.json').read_text())
    prep = json.loads((hourly / 'preparation_summary.json').read_text())
    timing = json.loads((report / 'timing.json').read_text())
    http = json.loads((report / 'http_verification.json').read_text())
    comparison = pd.read_csv(report / 'comparison.csv')
    audit = pd.read_csv(hourly / 'monthly_reconciliation.csv', parse_dates=['month'])
    rejected = audit.loc[audit.operation_quality_status.eq('review_required')]
    rejected.to_csv(report / 'hourly_quality_request_attachment.csv', index=False, date_format='%Y-%m-%d')
    frame = read_hourly(hourly / 'sources/hourly_yd_202412.csv', '2024-12')
    i, j = np.unravel_index(np.argmax(frame[HOURS].to_numpy()), frame[HOURS].shape)
    extreme = {'date': frame.iloc[i].date.strftime('%Y-%m-%d'), 'unit': int(frame.iloc[i].unit),
        'column': HOURS[j], 'provider_raw_value': float(frame.iloc[i][HOURS[j]]),
        'daily_reported_total_raw': float(frame.iloc[i]['총량(KW)']), 'physical_unit_verified': False}
    write_json(report / 'hourly_extreme_observation.json', extreme)
    prior = ROOT.parent / 'NOx_yeongdong_audit_20261001'
    evidence = hourly / 'public_evidence/prior_specs'
    evidence.mkdir(parents=True, exist_ok=True)
    inventory = []
    for source in ['20530', '20628', '20732', '20985', '21117']:
        for path in sorted((prior / 'sources').glob('spec*' + source + '*')):
            shutil.copy2(path, evidence / path.name)
            inventory.append({'file': str((evidence / path.name).relative_to(ROOT)), 'sha256': digest(path),
                'kind': 'prior public procurement/specification; original retrieval receipt retained',
                'actual_operating_record': False})
        for path in sorted((prior / 'results').glob('spec_' + source + '*_extracted.txt')):
            shutil.copy2(path, evidence / path.name)
    pd.DataFrame(inventory).to_csv(report / 'prior_spec_inventory.csv', index=False)
    gaps = [
        {'field': 'nox_concentration_unit', 'status': 'confirmed_as_ppm_in_provider_metadata',
         'evidence': 'data/hourly_operations_20261002/public_evidence/daily_nox_metadata.html'},
        {'field': 'flow_unit_and_rate_or_integrated_volume', 'status': 'unconfirmed'},
        {'field': 'dry_wet_standard_temperature_pressure', 'status': 'unconfirmed'},
        {'field': 'raw_vs_oxygen_corrected_nox_and_reference_oxygen', 'status': 'unconfirmed'},
        {'field': 'no2_equivalent_and_conversion_coefficient', 'status': 'unconfirmed'},
        {'field': 'matched_interval_c_and_flow', 'status': 'not_obtained; daily marginal means are insufficient'},
        {'field': 'valid_invalid_calibration_replacement_and_valid_duration', 'status': 'not_obtained'},
        {'field': 'official_unit_stack_tms_sensor_mapping', 'status': 'not_obtained'},
        {'field': 'actual_scr_on_off_bypass_urea_rate_temperature_catalyst_events', 'status': 'not_obtained_for_2023_2025'},
        {'field': 'official_monthly_nox_mass_kg', 'status': 'not_obtained'},
    ]
    write_json(report / 'measurement_readiness.json', {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'concentration_target_retained': True,
        'monthly_mass_target_ready': False, 'mass_target_created': False, 'actual_scr_features_added': False,
        'fields': gaps, 'cleansys_status': 'metadata retrieved; actual site unavailable due TLS certificate verification failure',
        'conversion_rule': 'Do not multiply daily mean concentration by daily mean flow; matched interval products required',
        'next_action': 'Request verified monthly masses or matched TMS concentration/volume intervals and definitions; request actual SCR time series'})
    df, _, _ = prepare_data(ROOT / 'data/operations_20261002')
    example = df.loc[df.plant.eq('영동') & df.unit.eq(1) & df.month.eq(pd.Timestamp('2025-10-01'))].iloc[0]
    unavailable = df.loc[df.plant.eq('영동') & df.unit.eq(2) & df.month.eq(pd.Timestamp('2024-12-01'))].iloc[0]
    write_json(ROOT / 'examples/request_operations_202510.json', request_for(example))
    write_json(ROOT / 'examples/request_operations_unavailable_202412.json', request_for(unavailable))
    template_cols = ['plant', 'unit', 'stack_id', 'tms_id', 'interval_start_kst', 'interval_end_kst',
        'nox_ppm_raw', 'nox_ppm_oxygen_corrected', 'oxygen_pct', 'gas_flow_rate', 'flow_unit',
        'integrated_gas_volume_sm3', 'valid_seconds', 'measurement_status', 'status_code_definition',
        'boiler_operating_state', 'scr_operating_state', 'scr_bypass_state', 'urea_solution_injection_kg_h',
        'urea_concentration_pct', 'ammonia_injection_kg_h', 'scr_inlet_temperature_c',
        'scr_inlet_nox_ppm', 'scr_outlet_nox_ppm', 'catalyst_event_id', 'source_available_at']
    pd.DataFrame(columns=template_cols).to_csv(report / 'requested_interval_schema_TEMPLATE.csv', index=False)
    lines = [
        'NOx 시간별 발전 패턴 추가 및 동일 기간 검증 결과',
        '실행 완료: ' + datetime.now(timezone.utc).isoformat(),
        '코드·자료·검증을 추가했으며 기존 실행 모델과 이력 DB는 교체하지 않았습니다.',
        '', '1. 결과 해석',
        '전체 앙상블 MAE는 약 1.69% 개선됐지만 RMSE는 약 0.58% 악화됐습니다.',
        '영동 앙상블 RMSE 개선은 약 0.10%, MAE 개선은 약 2.66%입니다. 급등 예측이 해결된 결과가 아닙니다.',
        '검증 기간별 결과가 다르고 큰 오차가 남아 현재 모델을 교체할 근거로 삼지 않았습니다.',
        '외부 점검 기간을 보고 특징 조합이나 가중치를 다시 조정하지 않았습니다.',
        '', '2. 확보 자료와 품질 처리',
        f"영동 1·2호기 2023-01~2025-12, 36개 월 파일 / 72개 호기·월 / {prep['observed_days']:,}개 호기·일 / {prep['observed_hours']:,}개 시간 관측값.",
        '시간 관측 누락 0개. 단, 관측값 존재와 관측값의 정확성은 다릅니다.',
        '일 합계를 1,000으로 나눈 값은 월 발전 실적과 최대 0.00264MWh 차이입니다.',
        'CSV 시간 열은 MWh, 일 합계 열은 KW로 표기되어 있어 물리 단위는 제공기관 확인이 필요합니다.',
        '따라서 이번 특징에는 비율·변동계수·전환 횟수처럼 일정한 단위 배율에 영향을 받지 않는 값만 사용했습니다.',
        'raw/1000은 합계를 수치로 대조하기 위한 가설이며 확정된 환산 단위로 쓰지 않았습니다.',
        '학습 전에 정한 시간 합계 대조 한계 0.5MWh, 일 합계 한계 0.02MWh에 따라 64개 호기·월을 수용하고 8개를 격리했습니다.',
        '격리된 월의 시간 특징 8개만 결측 처리합니다. 원본 시간값, 기존 NOx 타깃과 평가 행은 유지합니다.',
        '결측 특징은 학습 기간 중앙값으로 채우고 별도 관측 가능 여부 표시를 제공합니다.',
        f"가장 큰 원값: 영동 {extreme['unit']}호기 {extreme['date']} {extreme['column']} = {extreme['provider_raw_value']:,.0f}. 같은 날 일 합계 원값은 {extreme['daily_reported_total_raw']:,.0f}.",
        '8개 문제가 있는 월을 다시 내려받아 처음 파일과 원본 바이트·숫자가 모두 같음을 확인했습니다.',
        '이는 제공기관이 같은 값을 반복 제공한다는 확인이며 센서의 정확성 확인은 아닙니다.',
        '문제 목록: hourly_quality_request_attachment.csv; 재다운로드 근거: data/hourly_operations_20261002/refetch_verification.json',
    ]
    for _, r in rejected.iterrows():
        lines.append(f"  영동 {int(r.unit)}호기 {r.month:%Y-%m}: 시간 합계 대조 차이 {r.hour_sum_div_1000_minus_monthly_mwh:.6f}MWh; 시간 특징 격리")
    lines += ['', '3. 추가 특징과 입력 흐름',
        '시간 관측 충족률, 발전 실적 0인 시간 비율, 하루 전체가 0인 날짜 비율, 가장 긴 연속 0 시간 비율,',
        '0↔양수 시간 전환 횟수, 양수 시간 출력의 변동계수, 최대/양수 평균 비율, 인접 시간 출력 변화의 상대평균을 추가했습니다.',
        '전환 횟수는 실제 보일러 기동·정지 횟수가 아닙니다. 발전 실적 0은 SCR·보일러 정지 판정이 아닙니다.',
        '8개 원 특징과 관측 가능 여부 표시 1개를 추가해 각 반복의 모델 입력 특징은 44→53개가 됐습니다.',
        '모든 발전소를 함께 학습하는 모델이라 영동 특징 추가가 다른 발전소의 예측에도 영향을 줄 수 있습니다.',
        '새 실험 모델의 영동 API는 operation_pattern에 8개 특징을 받습니다. 격리·미확보 시에는 명시적인 null을 받습니다.',
        '영동 입력에서 이 필드를 빼면 400 오류를 반환합니다. 다른 발전소나 기존 모델에 임의 패턴을 넣어도 거부합니다.',
        '입력 범위 검사 → 동일 특징 생성 → 저장 전처리 → XGB/LGB → 고정 가중 예측 → LLM 사실 전달까지 연결했습니다.',
        '실험 SQLite 초기화·갱신에서도 특징·출처·격리 사유를 보존하고 입력 정책이 다른 이력을 거부합니다.',
        '이 특징은 예측 대상 월이 끝난 뒤 알 수 있는 실제 발전 실적입니다. 월 시작 전 미래 예측으로 해석할 수 없습니다.',
        '월 시작 전 예측을 하려면 예상 발전 스케줄이나 별도 운전 패턴 예측값을 사용하고 별도로 검증해야 합니다.',
        '', '4. 시간 분할과 누출 방지',
        'A: 학습 2023 / 가중치 선정 2024 상반기 / 점검 2024 하반기',
        'B: 학습 2023~2024 상반기 / 선정 2024 하반기 / 점검 2025 상반기',
        'C: 학습 2023~2024 / 선정 2025 상반기 / 점검 2025 하반기',
        '기존 호기 평가 370개, 학습에 없던 여수 2호기 12개는 별도 진단으로 분리합니다.',
        '전체 2026년 176행은 특징 생성 전에 제외했습니다. 이번 실험에서 2026년 성능을 다시 보고 선택하지 않았습니다.',
        '원 NOx 타깃·운전 조건·연료·기상·평가 키·단순 기준 예측이 모두 기존 실험과 일치합니다.',
        'XGB/LGB 각 300개 트리, 깊이 3, 학습률 0.05, 난수 고정, 스레드 1 및 log1p 결합을 유지합니다.',
        '전처리는 각 반복의 학습 구간에서만 계산하고, 학습에 존재한 호기의 선정 구간 RMSE로 가중치를 정합니다.',
        '가중치·전처리·모델을 외부 점검 예측 전에 잠그고 파일 해시로 변하지 않았음을 확인했습니다.',
        '선택된 XGBoost 비중: A 0.00 / B 1.00 / C 0.45. C의 이전 비중은 0.00이었습니다.',
        '', '5. 전체 모델별 결과: 기존 → 시간 패턴 추가 (ppm, 동일 370행)',
        '방법 | RMSE 이전 | RMSE 추가 | MAE 이전 | MAE 추가',
    ]
    pooled = comparison.loc[(comparison.scope == 'supported') & (comparison.fold == 'pooled')]
    for _, r in pooled.loc[pooled.plant.eq('all')].iterrows():
        lines.append(f'{r.method} | {r.before_rmse:.6f} | {r.after_rmse:.6f} | {r.before_mae:.6f} | {r.after_mae:.6f}')
    lines += ['', '6. 발전소별 선정 앙상블: 기존 → 시간 패턴 추가', '발전소 | 행 수 | RMSE 이전 | RMSE 추가 | MAE 이전 | MAE 추가']
    for _, r in pooled.loc[~pooled.plant.eq('all') & pooled.method.eq('ensemble')].iterrows():
        lines.append(f'{r.plant} | {int(r.n)} | {r.before_rmse:.6f} | {r.after_rmse:.6f} | {r.before_mae:.6f} | {r.after_mae:.6f}')
    lines += ['', '7. 기간별 선정 앙상블과 급등 사례', '기간 | RMSE 이전 | RMSE 추가 | MAE 이전 | MAE 추가']
    for _, r in comparison.loc[(comparison.scope == 'supported') & (comparison.fold != 'pooled') & (comparison.plant == 'all') & (comparison.method == 'ensemble')].iterrows():
        lines.append(f'{r.fold} | {r.before_rmse:.6f} | {r.after_rmse:.6f} | {r.before_mae:.6f} | {r.after_mae:.6f}')
    paired = pd.read_csv(report / 'paired_predictions.csv')
    paired['absolute_error_after'] = (paired.target_after - paired.pred_ensemble_after).abs()
    largest = paired.loc[paired.supported_after].nlargest(10, 'absolute_error_after')
    largest.to_csv(report / 'largest_errors.csv', index=False)
    for _, r in largest.head(3).iterrows():
        lines.append(f'{r.plant} {int(r.unit)}호기 {r.month[:7]}: 실제 {r.target_after:.3f}, 이전 {r.pred_ensemble_before:.3f}, 추가 {r.pred_ensemble_after:.3f}ppm')
    lines += ['', '8. 실제 저감설비 자료 확보 상황',
        '탈질설비 운영현황 공식 게시판을 확인했으나 올라온 암모니아 사용실적은 2009~2015년입니다.',
        '공공데이터포털 검색과 공식 공개자료에서 2023~2025년 영동 호기별 실제 SCR 가동·주입량 시계열을 확보하지 못했습니다.',
        '앞서 확보한 2026년 환원제·밸브·제어설비·유량계·촉매 구매 규격서와 원본 영수증을 이번 묶음에도 보존했습니다.',
        '이 자료는 설비·규격의 근거입니다. 실제 설치 완료일·환원제 소비량·우회·SCR 가동 로그의 근거가 아닙니다.',
        '규격서의 산소 6% 또는 요소 농도 40%를 일별 배출 CSV나 실제 주입량의 조건으로 대입하지 않았습니다.',
        'CleanSYS 공개시스템 안내 메타데이터는 읽었지만 실제 사이트 접속은 TLS 인증서 검증 오류로 실패했습니다.',
        '접속 실패 원인과 조회 URL은 public_evidence/retrieval_status.json에 보존했습니다. 30분 영동 계측 데이터는 확보하지 못했습니다.',
        '', '9. 월 총배출량 kg 타깃 검토',
        'NOx는 제공기관 메타데이터에 ppm로 명시되어 있습니다. 유량은 단위·유량률/적산량·표준상태·건습 기준이 확인되지 않았습니다.',
        '보정 전/후 NOx 농도, 기준 산소, NO2 환산 정의, 유효 측정 시간, 무효자료 대체 규칙과 굴뚝 대응도 필요합니다.',
        '일 평균 농도와 일 평균 유량의 곱은 같은 시간 구간별 농도×가스 부피 합계를 대신할 수 없습니다.',
        '같은 호기·굴뚝·구간의 확인된 농도와 적산 표준가스량, 환산계수가 있을 때만 기존 mass_from_intervals 함수를 사용합니다.',
        '자료가 확보되지 않아 kg 타깃이나 kg 모델은 만들지 않았습니다. 현재 예측 타깃은 기존 월별 산술평균 ppm 그대로입니다.',
        '가장 빠른 대안은 기관이 검증한 호기/굴뚝별 월 NOx kg를 받는 것입니다. 그 정의와 기간을 확인한 뒤 별도 모델을 구성합니다.',
        '질량 준비 상태: measurement_readiness.json. 요청용 헤더만 있는 TEMPLATE CSV는 관측 자료가 아닙니다.',
        '', '10. 검증·실행',
        f"학습·원본 비교·저장 모델 재로딩·API 행별 추론 전체 {timing['seconds']:.2f}초, 제한 {timing['limit_seconds']:.0f}초.",
        '저장 모델 예측 766건, API 허용 742건, 의도된 거부 24건을 검증했습니다. 최대 예측 차이는 약 1.42e-14입니다.',
        f"임시 실제 HTTP 서버: {http['status']}; health/predict HTTP 상태 {http.get('health_status')}/{http.get('predict_status')}. 확인 후 서버를 종료했습니다.",
        '기존 5001 서버·서비스 모델·관측 DB를 변경하지 않았습니다. 실제 Gemini 외부 호출은 실행하지 않았습니다.',
    ]
    tests = (report / 'tests_console.txt').read_text().strip().splitlines()
    lines.append('전체 회귀 테스트: ' + tests[-1])
    lines += ['재실행은 새 출력 폴더를 지정하세요:',
        'MPLCONFIGDIR=/tmp/nox-mpl /opt/anaconda3/bin/python -m nox.operations_validation --output reports/operations_validation_rerun',
        '원본 시간 자료 수집: python -m nox.operating_patterns --collect-only',
        '추가 자료 구성: python -m nox.operating_patterns --prepare-only --output data/operations_rerun',
        '기존 파일의 덮어쓰기는 거부합니다. 범위를 확장하려면 수집 월 범위와 품질 규칙을 먼저 명시해야 합니다.',
        '', '11. 다음에 할 일',
        '자료 요청 초안을 검토해 제공기관에 8개 월 원값 확인, 2023~2025 SCR/TMS 기록과 최신 급등 구간 기록을 요청하세요.',
        '받은 시계열의 단위·상태 코드·호기/굴뚝·시간 구간을 확인한 후 동일 A/B/C 기간에서 운전+SCR 특징을 비교합니다.',
        '기관 검증 월 질량이나 대응 가능한 구간 자료가 확보될 때 kg 모델을 ppm 모델과 분리해 구성합니다.',
        '현재 큰 오차를 해결했다고 표현하거나 새 실험을 서비스 모델로 교체하지 않습니다.',
        '', '공식 근거 URL',
        '시간별 발전 실적: https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221',
        '일자별 NOx 단위 안내: https://www.data.go.kr/data/15131510/fileData.do',
        '탈질설비 운영현황: https://www.koenergy.kr/kosep/fr/bo/board/main.do?menuCd=FN02011006',
        '굴뚝자동측정 공개 안내: https://www.data.go.kr/data/15136837/fileData.do',
        '총량 자료 확인·검증: https://www.keco.or.kr/web/lay1/S1T166C1020/contents.do',
    ]
    (report / 'NOx_operations_validation_result.txt').write_text('\n'.join(lines) + '\n')
    request_text = '''영동 NOx 예측 연구용 데이터 정의 확인 및 운전·저감설비 자료 요청 초안
상태: 작성만 완료. 기관에 제출하거나 전송하지 않았습니다.

제목: 영동 1·2호기 시간별 발전자료 품질 확인 및 SCR/TMS 운전 기록 제공 요청
대상: 한국남동발전 공공데이터 담당부서 및 영동에코발전본부 운전·환경 담당부서

안녕하세요. 공개 발전·연료·일별 NOx 자료를 활용해 NOx 예측 연구를 진행하고 있습니다.
발전 실적 0을 보일러·SCR 정지로 추정하거나, 정의가 확인되지 않은 유량으로 kg를 계산하지 않기 위해 아래 확인과 자료 제공을 요청드립니다.

1. 필요한 기간 및 범위
- 영동 1·2호기 2023-01-01~2025-12-31: 기존 학습·검증 기간을 설명할 운전 및 실제 저감설비 기록.
- 가능하면 2026-01-01~최신 완료 월: 2026년 3·6·7월 등 급등 구간의 후속 품질 확인. 연구와 별도로 평가할 예정입니다.
- 우선순위: 시간 단위 또는 30분 단위. 공개가 어려우면 호기·일 단위 집계와 집계 정의를 부탁드립니다.
- NOx 질량은 기관 검증 호기/굴뚝·월 단위 kg 실적을 제공받을 수 있다면 가장 우선적으로 사용하려 합니다.

2. 시간별 발전 실적 CSV 단위·원값 확인
공식 시간별 CSV는 시간 열을 '1시 발전량(MWh)' 등으로, 일 합계 열을 '총량(KW)'로 표기합니다.
일 합계 원값/1,000과 공식 월 발전량 MWh는 최대 약 0.00264MWh 차이로 일치하지만, 이 수치 일치만으로 시간 열의 실제 단위를 확정하지 않았습니다.
- 각 시간 열이 kWh/MWh 에너지인지, 평균 kW/MW 출력인지, 시간 길이 및 계량기준을 확인 부탁드립니다.
- 1시~24시는 구간 시작/종료 중 어느 시각인가요? 일자와 24시의 대응, 시간대, 정산 전후·수정 이력을 알려주세요.
- 공란·0·음수·비정상적으로 큰 숫자의 의미와 입력·전송 오류 여부를 알려주세요.
- 8개 호기·월의 시간 합계가 일·월 합계와 맞지 않아 연구에서는 시간 특징만 미사용 처리했습니다. NOx 관측값은 삭제하지 않았습니다.
- 첨부 hourly_quality_request_attachment.csv에 월별 차이를 기록했습니다.
- 영동 2호기 2024-12-17에 시간 열의 원값 3,864,824,192가 있는데, 같은 날짜 일 합계 원값은 180,000입니다. 올바른 원값과 발생 이유를 확인 부탁드립니다.
- 문제가 있는 월을 재다운로드했을 때도 원본 바이트·수치가 같았습니다. 웹 자료의 단순 재조회가 정확성 확인을 대신하지 않는다는 점을 이해하고 있습니다.

3. 실제 보일러 및 SCR 운전 기록
- 보일러 운전/정지/기동·정지 과정, 최소부하·저부하, 계획·불시 정비의 실제 시작·종료 시각과 상태 코드 정의.
- SCR 가동/정지/우회, 환원제 공급 상태, 주입 제어 모드와 알람·고장·계측 점검 시각.
- 시간별 요소수 또는 암모니아 실제 주입량 및 단위, 요소수 농도, 환원제 종류·전환 시점.
- SCR 입구/출구 NOx 농도, 가스 온도, 차압, 실제 측정 위치와 센서 ID.
- 촉매 교체·재생·정비의 실제 완료일과 층/호기/설비 대응, 가능한 경우 운전 시간·활성도 기록.
- 계량기·유량계 교정, 데이터 전송 장애와 TMS 연결 관계.
구매·규격서에는 실제 소비량·가동 이력이 없으므로, 구매 예정일이나 요구 규격을 실측 운전 특징으로 쓰지 않았습니다.
월 총 환원제 사용량만 가능하면 재고 입출고·구매량과 실제 소비량의 차이 및 호기 배분 방식을 설명 부탁드립니다.

4. 일별 NOx·유량·산소·온도 정의와 질량 구성 근거
- NOx ppm은 보정 전 농도인가요, 기준 산소 보정 농도인가요? 기준 산소와 NO/NO2 또는 NO2 환산 정의를 알려주세요.
- 유량의 단위, 순간/시간 평균 유량률인지 일 적산량인지, 표준 온도·압력과 건식/습식 조건을 알려주세요.
- 일 평균 계산은 전체 24시간 산술평균인가요, 유효 측정 시간만의 평균인가요? 정지·교정 구간 처리와 관측 간격을 알려주세요.
- 유효/무효/교정/점검/결측 상태와 무효자료 대체 규칙, 구간별 유효 초 수 및 원 코드표가 필요합니다.
- 농도와 유량이 같은 배출구·같은 시간 구간에 대응하는 30분/시간 자료를 제공할 수 있나요?
- 발전 호기 1·2와 굴뚝·TMS·분석기 ID 대응, 공통 굴뚝 여부 및 대응 변경 이력을 부탁드립니다.
- 공식 월 NOx kg가 있다면 산정 방법, 합산 배출구 범위, 유효시간·대체자료 처리 및 확정/잠정 여부를 알려주세요.
일 평균 농도×일 평균 유량으로는 실제 시간별 곱의 합계를 복원할 수 없으므로, 현재 공개 일 평균만으로 kg를 만들지 않았습니다.

5. 제공 형식 및 이력
- CSV/XLSX, UTF-8 권장. plant/unit/stack_id/TMS_ID/구간 시작·종료 시각/원값/단위/상태를 분리한 자료가 유용합니다.
- requested_interval_schema_TEMPLATE.csv는 요청 항목 예시이며, 값이 없는 헤더 템플릿입니다. 기관 원래 형식도 가능합니다.
- 원 관측일, 최초 공개 시각, 수정 공개 시각 및 수정 사유를 가능하면 함께 부탁드립니다.
- 비공개 사유가 있는 상세 항목은 제공 가능한 단위와 대체 집계 항목을 안내 부탁드립니다.

요청자가 입력할 항목: 성명·소속·연구 목적 상세·회신 받을 연락처.
첨부 후보: hourly_quality_request_attachment.csv, hourly_extreme_observation.json, requested_interval_schema_TEMPLATE.csv.
기관 공개·제공 신청 창구: https://www.koenergy.kr/kosep/gv/dt/nfdt51/main.do?menuCd=FN091210
실제 신청/제출은 별도의 사용자 지시를 받아 진행해야 합니다.
'''
    request_path = report / 'NOx_operations_data_request.txt'
    request_path.write_text(request_text)
    print(report / 'NOx_operations_validation_result.txt')
    print(request_path)


if __name__ == '__main__':
    main()
