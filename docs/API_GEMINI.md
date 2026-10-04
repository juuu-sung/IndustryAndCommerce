# Flask API와 Gemini 연동

## 실제 확인한 범위

입력 검증→관측 이력 조회→특징 구성→저장 모델 예측→Gemini 프롬프트→외부 REST 호출→구조화 응답 검증→JSON 반환 코드가 구성돼 있습니다. 테스트는 성공·실패·형식 오류의 **모의 응답**으로 계약과 예측값 유지 여부를 검증합니다.

저장된 실제 호출 점검 기록 (로컬 `NOx_corrected/reports/readiness_20261002/gemini_check.json`)은 `actual_external_call_verified: false`, `status: not_run`입니다. 당시 `GEMINI_API_KEY`와 `GEMINI_MODEL`이 없어 실제 외부 호출을 수행하지 않았습니다. 기본 서버는 `gemini_enabled: false`로 동작합니다. Gemini 연동 구현과 실제 API 성공 확인을 구분합니다.

## 엔드포인트

| 요청 | 내용 |
| --- | --- |
| `GET /api/health` | 정상 상태, 타깃 설명·단위·제약, Gemini 설정, 이력 상태 |
| `GET /api/history/status` | 모델 식별에 맞는 DB의 월 범위·개정·입수 상태 |
| `POST /api/predict` | 검증된 운전 조건에서 예측·품질·설명 반환 |

`/api/predict`는 `application/json`을 요구하며 최대 요청 크기는 32 KiB입니다. 잘못된 입력 400, 잘못된 Content-Type 415, 너무 큰 본문 413, 검증할 수 없는 이력 503, 내부 오류 500을 반환합니다.

## 입력

[공개 예시](../NOx_corrected/examples/request.json)는 형식 확인용 **가상 운전 조건**이며 실제 발전 실적이 아닙니다. 필수 수치 필드는 유한한 값을 요구하고 허용되지 않은 필드·미학습 호기·잘못된 달은 거절합니다.

| 필드 | 의미 |
| --- | --- |
| `plant`, `unit`, `month` | 발전소, 정수 호기, `YYYY-MM` |
| `capacity_mw`, `generation_mwh` | 설비용량 MW, 해당 월 발전량 MWh |
| `thermal_efficiency_pct`, `utilization_pct` | 월 실적 효율·이용률 % |
| `bituminous_ton`, `anthracite_ton`, `lng_ton`, `solid_ton`, `pellet_ton` | 연료별 소비량 ton |
| `oil_kl` | 유류 소비량 kL |
| `temperature_c`, `humidity_pct`, `wind_speed_ms`, `wind_direction_deg` | 선택적 기상 입력; 없거나 `null`이면 결측 처리 |
| `as_of` | 선택적 시간대 포함 입수 기준 시각; 그때 확보한 이력만 사용 |
| `fuel_basis` | 실제 호기 연료 / 사업소 배분 추정의 명시적 구분 |
| `operation_pattern` | 운전 특징 실험 모델에서 요구; 기본 모델에서는 받지 않음 |

영동 실제 연료 모델은 `provider_reported_unit_month` 근거를 요구합니다. 운전 특징 모델은 유효한 특징 사전 또는 명시적 `null`을 요구하고, 생략은 거절합니다. 기본 모델의 예시에 운전 패턴을 임의로 붙이면 400 응답입니다. 최신·실제 연료·운전 패턴 요청은 기존 로컬 프로젝트의 `examples/`에서 연결할 수 있으며 공개 대상에 포함하지 않습니다.

## 응답

```json
{
  "prediction": {
    "target": "monthly_daily_stack_mean_nox",
    "unit": "ppm",
    "value": 1.2345
  },
  "analysis": {
    "status": "disabled",
    "content": {
      "summary": "예측값은 입력 조건에 대한 연구용 추정입니다.",
      "observations": ["예측 원인을 이 입력만으로 확정할 수 없습니다."],
      "suggested_checks": ["실제 NOx 계측값과 월별 입력 조건을 대조하세요."],
      "limitations": ["실제 응답에는 타깃 제약과 입력 품질 경고가 포함됩니다."]
    }
  }
}
```

이는 수치를 임의로 둔 응답 구조 예시이며 실행 결과가 아닙니다. 실제 응답에는 `model`, `input_quality`, `warnings`도 있습니다. 동일 입력이라도 DB 개정·관측 이력·모델이 달라지면 예측이 달라질 수 있습니다. `analysis.status`는 기본 설명 `disabled`, Gemini 성공 `gemini`, 외부/형식 오류 시 `fallback`입니다.

## Gemini 설정과 점검

환경 변수로만 키를 전달합니다. 코드가 `.env`를 자동으로 읽지는 않습니다. `.env`는 Git에서 제외합니다. 아래 입력은 zsh에서 키를 화면에 표시하지 않고 읽으며, 셸 기록에 키를 직접 입력하지 않습니다.

```bash
# NOx_corrected 안에서, zsh
read -s 'GEMINI_API_KEY?Gemini API key: '
export GEMINI_API_KEY
read 'GEMINI_MODEL?사용 가능한 Gemini model id: '
export GEMINI_MODEL
python -m nox.gemini_check --live \
  --artifact artifacts/concentration \
  --request examples/request.json \
  --output "reports/local_gemini_check_$(date +%Y%m%d_%H%M%S).json"
```

`--live`가 있을 때 1회 외부 호출을 시도합니다. 성공하면 `status: passed`, `actual_external_call_verified: true`와 스키마·예측값 유지 검사 결과가 남습니다. `failed_or_fallback`이면 실제 연동 성공으로 기록하지 않습니다. 모델 ID는 해당 계정에서 사용 가능한 값을 직접 지정합니다.

```bash
ENABLE_GEMINI=1 python -m nox.app
```

Flask를 켜기 전에 키와 모델 환경 변수를 같은 터미널에서 설정해야 합니다. `/api/health`의 `gemini_enabled: true`는 사용 설정만 뜻하며 호출 성공 증거가 아닙니다. `/api/predict`의 `analysis.status: gemini`와 호출 점검 기록을 함께 확인합니다.

## 응답 처리

`llm.py`는 Gemini REST `generateContent`에 JSON 응답 스키마를 전달합니다. 종료 상태 `STOP`, 응답 크기 1 MiB, 제한 시간 12초, 텍스트 JSON·필드·타입·문자열 및 배열 길이를 검사합니다. LLM이 반환한 문자열을 무검증으로 사용하지 않습니다.

백엔드는 NOx 수치와 단위를 직접 관리하고 데이터 제약을 설명에 유지합니다. LLM은 ppm→kg 환산, 임의 법적 한계값, 급등의 확정 원인, 발전량 0→SCR 정지 판정을 생성하도록 사용하지 않습니다. 현재 설명은 연구 결과 이해를 위한 참고 정보이며 실제 설비 제어와 규제 판정은 검증 범위에 포함하지 않습니다.
