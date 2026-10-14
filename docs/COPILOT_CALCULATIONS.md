# 등록 사례의 달력 이력·예측 오차 계산

등록한 영동 사례에서 전월 변화, 직전 3개월 평균, 저장된 방법별 예측 오차를 계산하는 기능을 구현했습니다. LLM·신규 학습·새 모델 추론 없이 고정 관측과 저장 예측을 사용합니다. 완성된 보고서의 계약·검증·HTML·API는 다음 단계입니다.

## 계산 기준

| 항목 | 정의 | 자료가 부족한 경우 |
| --- | --- | --- |
| 전월 | 정확한 직전 달의 같은 발전소·호기 관측 | 더 오래된 행으로 대체하지 않음 |
| 전월 차이 | 현재 관측 − 전월 관측, ppm | 둘 중 하나가 없으면 null과 이유 |
| 전월 변화율 | 전월 차이 ÷ 전월 관측 × 100, % | 전월이 0이면 null과 previous_month_zero |
| 직전 3개월 평균 | 현재 달을 제외한 정확한 3개 달의 이용 가능한 관측 평균 | observed_count·coverage·달별 누락 이유를 함께 반환 |
| 평균 대비 차이 | 현재 관측 − 직전 3개월 평균, ppm | 현재 관측 또는 평균이 없으면 null |
| 부호 오차 | 저장 예측 − 현재 관측, ppm | 관측·예측이 없으면 unavailable |
| 절대 오차 | 부호 오차의 절댓값, ppm | 한 사례의 절대 오차를 RMSE로 표시하지 않음 |

예를 들어 합성 관측이 7월 8, 8월 9, 9월 10, 10월 12ppm이면 전월 차이 +2ppm, 변화율 +20%, 직전 평균 9ppm입니다. 예측이 11ppm이면 부호 오차 −1ppm, 절대 오차 1ppm입니다. 이 예시는 실제 발전소 수치가 아닙니다.

0값과 review_required 값도 관측 그대로 계산합니다. 품질 상태·검토 이유·출처를 유지하며 측정 유효성 인증으로 해석하지 않습니다. 내부 값을 반올림하지 않으며 JSON의 NaN·Infinity와 잘못된 숫자·비달력 이력을 차단합니다.

## 읽기와 버전 연결

CaseRegistry.read_verified는 등록부를 원본과 대조한 바로 그 바이트를 반환합니다. ArchiveRepository는 이 바이트에서 같은 발전소·호기의 현재 달과 정확한 직전 세 달만 읽습니다. 검증 후 원본 경로를 다시 열어 다른 관측을 섞지 않습니다. 기존 v1 등록부의 스키마·시각·사례 버전은 변경하지 않았습니다.

계산 JSON은 다음 근거를 포함합니다.

- 사례·파일 묶음·모델·타깃 정의 버전
- 고정 관측·예측 CSV 해시 및 원본 행 해시
- 각 달의 관측값·품질 상태·검토 이유·행 해시·누락 이유
- 학습·가중치 선정·평가 기간과 모델 생성 시각
- 연료 자료의 실제/추정 구분, 원천 manifest 검증 범위
- 계산 규칙 버전 calendar_metrics_v1 및 등록·읽기·계산 코드의 SHA-256

calculation_id는 생성 시각과 머신의 경로를 제외한 내용에서 계산합니다. 같은 버전의 재실행은 기존 출력과 시각을 유지합니다. 내용이 달라지거나 코드가 변경되면 별도 버전 파일을 생성합니다. 원본 해시 변경, 저장 도중 사례 버전 변경, 기존 결과 변조, 원본/등록부 안으로 출력하려는 요청은 차단합니다.

실제 사례는 evaluation_replay이며 당시 서비스가 생성한 예측 이력이 아닙니다. 원천 manifest의 선언 해시를 보존하되 그 원천 파일 자체를 재검증한 것으로 표시하지 않습니다. target_verification_status는 pending_provider_confirmation을 유지합니다.

## 실행

프로젝트의 NOx_corrected 폴더에서 실행합니다. 기존 등록부가 필요하며 [사례 등록 안내](COPILOT_CASE_REGISTRY.md)를 먼저 참고하세요.

```bash
python -m nox.copilot.calculations \
  --case-key yeongdong_u1_2025_10_fold_c_v1
```

기본 출력은 data/runtime/copilot/calculations/ 아래의 사례 키·계산 버전 이름 JSON입니다. 다른 등록부/출력 폴더는 --registry-dir와 --output-dir로 지정합니다. CLI 로그는 성공 여부·버전·품질 상태·출력 경로만 표시하고 실제 관측·예측값은 출력 파일에서 확인합니다. 실제 계산 JSON과 등록부는 Git 제외 대상으로 유지합니다.

Python에서는 다음과 같이 사용합니다.

```python
from nox.copilot.calculations import calculate_case, save_calculation

result = calculate_case('yeongdong_u1_2025_10_fold_c_v1')
path, persisted = save_calculation('yeongdong_u1_2025_10_fold_c_v1')
```

calculate_metrics는 합성 Observation 입력으로 독립 경계 검사를 실행할 수 있습니다. 현재 등록 API는 관측이 존재하고 저장 평가가 승인된 사례만 받습니다. 따라서 관측 없음·예측 없음 처리는 순수 계산 계약으로 검증했으며, 실제 미관측 사례 등록을 새로 지원한 것은 아닙니다.

## 검증 결과

2026년 10월 13일 계산 관련 신규 테스트 32개를 추가했고, 실제 자료를 포함한 전체 245개 테스트가 57.96초에 통과했습니다. 건너뛴 검사는 없으며 기존 불리언 부적합 입력 검사에서 Pandas FutureWarning 1건이 발생했습니다.

```bash
MPLCONFIGDIR=/tmp/nox-mpl python -m pytest -q --require-local-assets
```

검사는 전월 누락·0값·연도 경계·부분 관측·현재 관측 없음·품질 경고·타 호기/미래 자료 배제·숫자 오류·원본/결과 변조·검증 바이트 재사용·경로 및 버전 재현성을 포함합니다. 실제 영동 사례는 원본 CSV에서 독립 계산한 전월 차이·직전 평균·모든 저장 방법의 오차와 대조했고 원본 파일 해시 유지도 확인했습니다. 기본 서비스 모델을 바꾸거나 Gemini API를 실제 호출하지 않았습니다.

## 다음 단계

2026년 10월 14일 ReportData 계약·ReportBuilder·Validator·JSON/HTML·Flask 보고서 API를 구현했습니다. [보고서 안내](COPILOT_REPORTS.md)를 참고하세요. 이후 문서 RAG, Agent 도구 선택과 검토 후 재개를 연결합니다.

- [전체 확장 계획](COPILOT_IMPLEMENTATION_PLAN.md)
- [노션 구조·진행 현황](https://app.notion.com/p/3f8a7839359181ee8286cf13ccbfba31)
- [읽기 어댑터](../NOx_corrected/nox/copilot/archive_repository.py)
- [계산 구현](../NOx_corrected/nox/copilot/calculations.py)
- [계산 검증](../NOx_corrected/tests/test_copilot_calculations.py)
