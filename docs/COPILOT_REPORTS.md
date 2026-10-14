# NOx 보고서 JSON·HTML·Flask API

등록 사례의 고정 관측·저장 예측·계산 결과를 보고서 초안으로 연결했습니다. LLM 없이 수치·품질·출처가 재현되는 구조이며, 문서 RAG·Agent·사용자 승인·새 조건 추론은 다음 단계입니다.

## 진행 방식과 구현 결과

| 순서 | 구현 | 목적 |
| --- | --- | --- |
| 1. 계약 | 엄격한 요청·보고서 필드, 단위·분석 모드·상태 | 잘못된 입력과 설명 범위를 고정 |
| 2. 생성 | ReportBuilder가 관측·예측·비교·품질·출처·확인 사항 구성 | 검토할 수 있는 초안 생성 |
| 3. 검증 | 지표 재계산·타깃/기간/가중치·원본 대응·내용 해시 검사 | 수치·출처·품질 변경 감지 |
| 4. 출력 | 같은 검증 JSON으로 독립 실행 가능한 HTML 생성 | API와 화면의 내용 일치 |
| 5. 저장·API | 내용 ID로 배타적 저장, 생성·조회·미리보기 API | 반복 요청·파일 변경·오류 처리 |

Pydantic·새 DB·새 학습 없이 표준 라이브러리와 기존 Flask를 재사용했습니다. JSON은 원 정밀도를 유지하고 HTML의 수치만 소수 둘째 자리로 표시합니다.

## 보고서에 포함한 내용

- 대상 발전소·호기·월, 연구용 ppm 타깃 정의, 평가 재현 모드
- 현재 관측과 정확한 직전 세 달, 전월 변화·변화율·관측 충족 수
- 선택한 저장 예측 방법, 부호/절대 오차, 모델·앙상블 가중치
- 원본에 있는 NOx·기온 관측일 수/충족률, 품질 상태·검토 이유
- 연료의 실제/추정 근거, 모델 생성 시각과 학습·선정·평가 기간
- 관측·예측·모델·타깃·계산·보고서 버전과 원천 manifest 검증 범위
- 자료와 대조할 확인 사항, 통과 검사·미확인 항목·한계

관측일 충족률은 일자료 존재 비율이며 유효 측정·운전시간 충족률이 아닙니다. 일 수와 비율이 일치하는지 달력 기준으로 검사합니다. 자료에 필드가 없거나 누락되면 unavailable을 표시합니다. 특징별 결측 보완·학습 범위 경고는 이 보고서 계약에서 읽지 않았으므로 확인하지 않은 상태를 명시합니다.

0값·급등·검토 필요 자료를 자동 삭제하거나 정상/무효로 확정하지 않습니다. RAG 근거는 빈 목록, LLM은 not_used, 보고서는 draft/개정 1입니다. 한 사례의 절대 오차를 RMSE로 부르지 않습니다.

## 핵심 코드

| 모듈 | 책임 |
| --- | --- |
| contracts.py | ReportRequest, 지원 방법·모드·엄격한 입력, 공개 오류 |
| report_sources.py | 같은 등록 원본의 관측일/기온 충족률 및 행 해시 |
| report_builder.py | ReportData 생성·확인 사항·context_hash·보고서 ID |
| validators.py | 구조·지표 재계산·타깃/기간/원본 대응·변조 검사 |
| renderers.py | 이스케이프한 한국어 HTML, 표시 정밀도·안내 |
| repositories.py | 원본 재검증·배타적 JSON/HTML 저장·기존 초안 반환 |
| routes.py | 기존 Flask의 독립 Blueprint와 HTTP 계약 |
| reports.py | 서버·LLM 없이 JSON/HTML을 생성하는 CLI |

기존 create_app에 Blueprint를 연결했습니다. 비어 있는 사례 등록부나 비활성 RAG 때문에 /api/health 및 /api/predict가 실패하지 않습니다. 생성자에서 enable_copilot=False를 주면 보고서 API를 끌 수 있고 테스트용 ReportService도 주입할 수 있습니다.

## API 계약

| 요청 | 결과 |
| --- | --- |
| GET /api/copilot/cases | 검증된 사례·방법·대상·기간; 잘못된 자산은 blocked 항목 |
| POST /api/reports/drafts | 생성된 report, created, JSON/HTML 상대 링크 |
| GET /api/reports/{report_id} | 원본과 다시 대조한 보고서 JSON |
| GET /api/reports/{report_id}/html | 같은 보고서의 HTML, text/html |

새 초안은 201, 같은 내용의 기존 초안은 200입니다. 입력은 case_key가 필수이며 analysis_mode는 evaluation_replay만 지원합니다. methods를 생략하면 이용 가능한 저장 방법을 모두 선택합니다. 목록 순서는 정규화하고 중복·미지원/자료에 없는 방법을 거부합니다. expected_case_version_id를 주면 조회 시 확인한 사례 버전과의 일치도 검사합니다.

```json
{
  "case_key": "yeongdong_u1_2025_10_fold_c_v1",
  "analysis_mode": "evaluation_replay",
  "methods": ["xgboost", "lightgbm", "ensemble"]
}
```

클라이언트의 파일 경로·모델 경로·예측값·단위·상태 변경 필드는 받지 않습니다. 중복 JSON 키와 NaN/Infinity, 32KiB 초과 요청도 차단합니다.

| 상태 | 의미 |
| --- | --- |
| 400 | 잘못된 JSON·식별자·필드·방법·모드 |
| 404 | 미등록 사례 또는 없는 보고서 |
| 409 | 원본/보고서 변조, 사례·구현 버전 충돌 |
| 413 / 415 | 요청 크기 초과 / 부적합 Content-Type |
| 503 | 고정 자산·등록 원본·보고서 저장소 사용 불가 |

외부 응답에는 서버 파일 경로와 내부 오류를 노출하지 않습니다. HTML의 모든 동적 문자열을 이스케이프하고 외부 스크립트/리소스를 사용하지 않습니다. HTTP에는 no-store·nosniff, HTML에는 CSP·no-referrer를 적용했습니다. 현재는 기존 로컬 Flask 실행을 대상으로 하며 사용자별 접근 권한·승인과 Agent 작업 상태는 이후에 구현합니다.

## 저장과 재현

기본 출력: NOx_corrected/data/runtime/copilot/reports/{report_id}.json 및 .html. 실제 행별 보고서는 Git에서 제외합니다. 임시 파일·fsync·배타적 생성으로 완성된 파일만 저장하고 기존 내용은 덮어쓰지 않습니다.

report_id는 보고서 본문의 내용 해시입니다. context_hash는 사례·계산·선택 방법·보고서 코드·관측 충족 자료 및 문서/검색 설정을 연결합니다. 생성 시각과 머신 경로는 내용 동일성 비교에서 제외합니다. 같은 요청은 최초 JSON과 생성 시각을 반환하며 HTML도 바이트까지 동일합니다.

HTML은 JSON 객체의 키 순서에 의존하지 않고 selected_methods 순서로 그립니다. 이로써 파일 저장 시 키 정렬과 API의 JSON 직렬화 방식이 달라도 표 순서가 유지됩니다. 원본 변경, 결과 변조, 저장 중 사례 변경, 심볼릭 링크 보고서와 원본 폴더로의 출력은 차단합니다.

현재 v1 검증기는 보고서 코드 버전이 달라진 옛 초안을 409로 차단합니다. CLI 또는 생성 API로 새 초안을 만들고 기존 파일은 보존합니다. 여러 개정의 승인·과거 버전 전용 읽기는 Agent 단계에서 확장합니다.

## 직접 실행

작업한 코드는 IndustryAndCommerce 저장소의 NOx_corrected 안에 있습니다.

```bash
cd NOx_corrected
python -m nox.copilot.reports \
  --case-key yeongdong_u1_2025_10_fold_c_v1
```

CLI가 JSON·HTML 경로와 보고서 ID를 출력합니다. --methods, --registry-dir, --report-dir로 선택을 바꿀 수 있습니다. 이미 등록한 사례와 고정 자산이 필요합니다.

HTTP로 사용하려면 이 저장소 폴더에서 서버를 실행합니다. 이전 코드로 실행 중인 서버는 종료 후 새 코드로 다시 실행해야 합니다.

```bash
MPLCONFIGDIR=/tmp/nox-mpl python -m nox.app
```

다른 터미널에서:

```bash
curl -sS http://127.0.0.1:5001/api/copilot/cases
curl -sS http://127.0.0.1:5001/api/reports/drafts \
  -H 'Content-Type: application/json' \
  --data-binary @examples/report_request.json
```

반환된 links.json과 links.html을 사용해 조회합니다. 서버는 보고서 모드에서 기존 Predictor를 새로 실행하거나 Gemini를 호출하지 않습니다.

## 검증 결과

2026년 10월 14일 보고서 신규 테스트 61개와 전체 306개 테스트가 통과했습니다. 실제 로컬 자산을 포함한 전체 실행 시간은 55.34초, 건너뛴 검사는 0개입니다. 기존 불리언 부적합 입력 검사에서 Pandas FutureWarning 1건이 발생했습니다.

```bash
MPLCONFIGDIR=/tmp/nox-mpl python -m pytest -q --require-local-assets
```

검사 범위는 수치·단위·기간·품질·가중치·출처, 결측/0, 관측일 충족 정합성, 중복/비수치 JSON, 경로 입력, 코드·원본·저장 결과 변조, 반복·동시 생성, HTML 이스케이프, JSON/HTML 숫자와 표 순서, 저장 중 원본 변경 및 기존 API입니다.

실제 영동 자료로 JSON/HTML을 생성했고 내용과 생성 시각·파일 수정 시각이 재실행에서 유지되는지 확인했습니다. 임시 localhost HTTP 서버에서도 기존 초안 200, 새 방법 조합 201, JSON/HTML/사례/health/기존 예측 200, 잘못된 요청 400을 확인한 뒤 서버를 종료했습니다. 측정 유효성·기관 집계 규칙은 확인 대기 상태이며 신규 NOx 학습이나 실제 Gemini 호출은 하지 않았습니다.

브라우저 도구의 file:// 접근 정책 때문에 실제 화면 캡처는 확인하지 못했습니다. HTML의 구조·수치·표·문자 이스케이프·HTTP 전달은 검증했습니다.

## 다음 작업

현재 보고서 기반을 RAG의 수치 도구로 사용할 수 있습니다. 먼저 실제 문서·버전·적용 범위·정답 근거를 수집하고, 검색 API와 검색 방식 비교를 구현합니다. 새 조건 추론 어댑터·신규 호기 보고서 지원, Agent 도구 선택·사용자 승인·재개는 따로 검증합니다.

- [전체 확장 계획](COPILOT_IMPLEMENTATION_PLAN.md)
- [등록 사례 계산](COPILOT_CALCULATIONS.md)
- [노션 구조·진행 현황](https://app.notion.com/p/3f8a7839359181ee8286cf13ccbfba31)
- [보고서 검증 코드](../NOx_corrected/nox/copilot/validators.py)
- [보고서 테스트](../NOx_corrected/tests/test_copilot_reports.py)
