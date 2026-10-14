# NOx Copilot 확장 · 구현 과정과 검증 결과

2026년 10월 14일 구현 기록입니다. **검색 보완 → 수치·문서 근거 보고서 → Gemini 설명 계약 → Agent 실행·검토·재개** 순서로 진행했습니다. 실제 발전소 자료의 보고서는 사용자 검토 대기로 준비했습니다. Gemini 실제 호출은 실행 환경에 키와 모델이 없어 `not_run`이며, 모의 응답 검증과 실제 호출용 코드는 완료했습니다.

## 완료 범위

| 요청 | 이번 결과 | 검증 범위 |
| --- | --- | --- |
| 검색 품질 보완 | 문서 범위 선택·한영 용어 확장·중복 페이지 제한, 새 검색 API | 이전 50개로 선정, 추가 30개로 확인 |
| 수치·문서 연결 | 서버 사실 ID·문서 근거 ID·내용 해시·JSON/HTML 보고서 | 실제 영동 사례와 합성 자료 |
| Gemini 설명 | 구조화 응답·참조 검사·수치/단위 보호·기본 설명 전환 | 성공/실패/부적합 응답 모의 검증; 실제 호출 미확인 |
| Agent | LangGraph·SQLite 체크포인트·백그라운드 실행·허용 도구 | 규칙 기반 실제 실행, Gemini 도구 선택은 모의 검증 |
| 승인·재개 | 인증된 검토·보고서 개정·이전 버전 차단·재시작 복구 | 합성 사례 승인/거절; 실제 발전소 보고서는 미승인 |

**전체 419개 테스트 통과 / 건너뛴 검사 0개**입니다. 이번 신규 검사는 검색 17개, 근거 보고서 22개, Agent 32개로 총 71개이며 이전 348개도 통과했습니다. 기존 Pandas FutureWarning 1건이 있고 테스트 실패는 없습니다.

기존 모델·전처리·관측·저장 예측 9개 파일과 기존 보고서 계약을 보존했습니다. 이번 작업에서 NOx 모델을 재학습하거나 서비스 기본 모델을 바꾸지 않았습니다.

## 1. 검색 실패 분석과 개선

### 발견한 문제

한국어 질문을 전체 문서에서 검색하면 한국어 프로젝트 문서가 영문 기술자료보다 앞에 나오는 경우가 많았습니다. 겹치는 청크가 같은 페이지에서 여러 자리를 차지했고, 문서와 관계없는 질문에도 키워드가 우연히 겹친 결과를 반환했습니다.

### 구현한 행동

`nox/copilot/rag/domain.py`를 기존 검색 위에 별도 계층으로 추가했습니다.

1. 연구 타깃·모델·API·전월 계산은 `project_method`, SCR/SNCR 일반 기술은 `general_technical`, CleanSYS 개요는 `korean_tms_general`로 범위를 정합니다. 요청의 명시적 범위가 우선입니다.
2. 촉매→catalyst, 비활성→deactivation, 체류시간→residence time 등 고정 용어표로 기술 검색어를 확장합니다. LLM 번역이나 정답 문장 삽입은 사용하지 않습니다.
3. 같은 페이지/절은 최종 결과에 최대 2개까지만 포함해 겹치는 청크가 모든 자리를 차지하지 않게 합니다.
4. 발전소 실제 수치·고장·법적 판정·명시적 무관 질문을 구분합니다. 이 분기는 제한된 표현에 대한 규칙이며 일반적인 환각 방지 보장이 아닙니다.
5. 기존 `/api/rag/search`와 원래 평가 기록은 보존하고 `/api/rag/search/domain`을 추가했습니다.

### 비교 설계와 결과

이미 결과를 확인했던 50개 질문은 모두 개발 자료로 취급했습니다. 추가 30개 질문은 검색 결과를 보기 전에 원문 문자 구간과 함께 고정했습니다. 이 중 답변 가능 20개·미지원/무관 10개입니다. 추가 질문도 같은 문서의 유사 개념을 AI가 작성한 것이므로 독립 전문가 평가로 부르지 않습니다.

기존 Keyword/Vector/Hybrid와 범위·용어를 보완한 세 방식을 비교했습니다. 개발 자료의 Recall@5와 잘못된 근거 반환률로 `domain:keyword`를 선정하고 잠금 파일을 저장한 뒤 추가 질문을 평가했습니다.

| 동일한 질문 묶음 | 기존 Keyword | 선정한 개선 Keyword | 변화 |
| --- | ---: | ---: | ---: |
| 기존 개발 질문 중 답변 가능 42개 | 32개 / 42개, 76.2% | 35개 / 42개, 83.3% | +7.1%p |
| 추가 질문 중 답변 가능 20개 | 11개 / 20개, 55.0% | 13개 / 20개, 65.0% | **+10.0%p** |
| 추가 미지원/무관 질문 10개 처리 | 4개 / 10개 | 10개 / 10개 | 규칙 분기 포함 |
| 추가 질문의 첫 근거 순위 지표 MRR | 0.379 | 0.567 | 상위 근거 배치 개선 |

추가 질문에서 개선 Vector/Hybrid는 각각 14/20, 70.0%였습니다. 이미 본 결과로 선정 방식을 바꾸지 않고 개발에서 정한 Keyword를 유지했습니다. 새 선택 실험은 추가 질문도 개발 자료로 전환한 뒤 별도 질문으로 확인해야 합니다.

**남은 문제:** 추가 질문의 목표 80%에는 못 미쳤습니다. 전월 누락 질문 한 개는 기존 검색에서 찾았지만 개선 검색에서 놓쳤습니다. 이력 입수 시각·기상 해석·기간 분할·일부 SCR 반응 근거도 놓쳤습니다. 등록 구간 외의 동등한 근거가 존재할 수 있어 담당자의 정답 대안 검토도 필요합니다. 10/10 미지원 처리를 일반적인 의미 이해·답변 정확도 100%로 표현하지 않습니다.

[검색 집계](generated/domain_search_metrics.csv) · [선정/평가 근거](generated/domain_search_evaluation.json) · [추가 질문](generated/domain_search_additional_questions.json) · [질문별 결과](generated/domain_search_question_results.json)

## 2. 수치와 문서 근거를 보고서로 연결

기존 v1 보고서를 변경하면 이전 보고서의 구현 해시가 달라집니다. 따라서 `nox/copilot/evidence/`에 별도 v2 보고서 봉투를 구현하고, 기존 검증 보고서를 `base_report`로 포함했습니다. 원본의 수치·기간·품질 상태는 그대로 유지합니다.

| 구성 | 역할 | 변경 검출 |
| --- | --- | --- |
| `facts.py` | 관측·전월·평균·저장 예측·오차를 서버 사실 ID로 구성 | 검증된 base_report에서 다시 생성해 비교 |
| `citations_for` | 문서·버전·페이지/절·청크·문자 구간에 근거 ID 연결 | 등록 원문·청크·메타데이터 대조 |
| `reports.py` | 수치·검색 문맥·설명·개정·한계를 묶어 저장 | 보고서/문맥/구현 해시·원본 재검사 |
| `render.py` | 수치 표·설명 참조·문서 위치·검토 상태 표시 | JSON 검증 후 HTML 이스케이프 |

사실 ID 예시는 `F_CURRENT`, `F_PREVIOUS`, `F_MEAN`, `F_ERROR_LIGHTGBM`입니다. 문서 근거 ID는 `C01`처럼 서버가 부여합니다. 설명은 `fact_refs`와 `citation_refs`로 참조하며 자유롭게 새 수치를 만들지 않습니다.

### 영동 실제 사례

영동 1호기 2025년 10월의 등록 사례에서 **수치·상태 사실 22개, 중복을 제거한 문서 근거 8개**를 연결했습니다. 이 사례는 C 기간의 저장 평가 재현이며 개선 후보의 새 추론이나 당시 배포된 미래 예측이 아닙니다.

현재 상태는 `waiting_review`, 설명은 `fallback / gemini_not_configured`입니다. 실제 보고서는 승인하지 않았습니다. 측정 유효성도 계속 기관 확인 대기입니다. 보고서 검토 승인이 기관 측정 인증으로 바뀌지 않도록 한계를 고정했습니다.

JSON·HTML·DB·원문·모델·실제 관측값은 비공개 런타임 경로에 보존합니다. 공개 HTML은 [합성 자료 검증 예시](generated/copilot_workflow_demo.html)입니다.

## 3. Gemini 설명 계약과 검증

### 처리 순서

`facts/citations` 구성 → 프롬프트 → Gemini 구조화 JSON → 참조·형식·문장 검사 → 검증 결과 또는 기본 설명 반환 순서입니다.

- 응답은 최대 4개의 짧은 문장과 기존 근거 ID로 제한합니다.
- 관측 설명에는 사실 ID, 일반 점검 설명에는 문서 근거 ID가 있어야 합니다.
- 숫자·단위의 반복, 없는 참조, 측정 인증·고장 확정·운전 변경 지시는 검사에서 차단합니다.
- 검사 실패나 호출 실패 시 수치와 문서 근거를 보존하고 서버의 기본 설명을 반환합니다.
- 기관 확인·원인 미확정·일반 자료의 적용 한계는 LLM이 생략해도 보고서에 유지합니다.
- 문서 본문을 도구 실행 권한으로 취급하지 않습니다. 설명 호출에는 실행 도구를 제공하지 않습니다.

처음 검사에서는 영어 단어 경계가 한국어 접미사와 붙은 `kg으로`를 잡지 못했습니다. 단위 경계를 수정해 한국어와 붙은 단위도 거부하도록 했고 테스트로 재확인했습니다.

**참조 검사·수치 보호는 의미적 근거 충실성을 증명하지 않습니다.** 표현이 실제 원문을 올바르게 요약했는지는 사용자·전문가의 별도 검토가 필요하며 `semantic_faithfulness_verified=false`를 유지합니다.

### 실제 호출 상태

실행 환경에 `GEMINI_API_KEY`와 `GEMINI_MODEL`이 없어 실제 확인 CLI는 다음 상태를 반환했습니다.

```json
{
  "live_requested": true,
  "credentials_configured": false,
  "actual_external_call_verified": false,
  "status": "not_run"
}
```

키가 설정된 터미널에서 아래 명령으로 실제 설명 요청 1회와 수치 보존을 확인할 수 있습니다. 모델 ID는 해당 계정에서 사용 가능한 값을 설정합니다. 키를 코드·노션·공개 로그에 적지 않습니다.

```bash
cd NOx_corrected
python -m nox.copilot.evidence.live_check \
  --case-key yeongdong_u1_2025_10_fold_c_v1 --live \
  --output "data/runtime/copilot/gemini_live_$(date +%Y%m%d_%H%M%S).json"
```

[실제 호출 설정 검사 기록](generated/copilot_gemini_live_check.json)

## 4. Agent 도구 실행·검토·재개

LangGraph 1.2.15와 SQLite 체크포인트 3.1.2를 별도 프로젝트 의존성 경로에 설치했습니다. PostgreSQL·pgvector를 도입한 것으로 표시하지 않습니다.

### 요청에 따른 실행 분기

문서 질문은 검색하고 종료합니다. 사례 보고서는 아래 5개 도구로 구성합니다. 대상이 불명확하면 임의 호기·월을 채우지 않고 입력을 기다립니다.

| 도구 | 반환 | 검사 |
| --- | --- | --- |
| `get_monthly_history` | 현재·전월·직전 3개월 관측 | 등록 대상·달력 월·사례 버전 |
| `get_data_quality` | 품질 사유·기관 확인 상태·한계 | 미확인 상태 보존 |
| `get_saved_predictions` | 저장 평가 예측·모델 | 새 추론·미래 예측으로 표현하지 않음 |
| `search_documents` | 범위가 정해진 보고서 근거 | 문서·색인·검색 구현 버전 |
| `create_report_draft` | 검증된 근거 보고서 ID | 앞선 도구 결과·원본 재확인 |

도구 호출 상한은 6회입니다. 선행 조건을 만족한 허용 도구만 노출하며 알 수 없는 도구·다른 사례 인자·반복 호출을 거부합니다. SQL·셸·원본 파일 경로·설비 제어·승인 도구는 목록에 없습니다.

**기본 `planner=rule`은 규칙 기반 워크플로입니다.** Gemini Function Calling 도구 선택 코드도 추가해 모의 응답으로 검증했습니다. 실제 Gemini가 선택하는 실행은 아직 확인하지 않았으며, `planner=gemini`인데 키가 없으면 규칙으로 조용히 바꾸지 않고 503을 반환합니다.

Gemini의 함수 호출 ID와 불투명 thought signature는 원 응답 그대로 보존해 다음 함수 결과와 연결합니다. 검색 문서의 본문은 도구 선택 대화에 넣지 않고 결과 수·상태·해시만 전달합니다.

### 백그라운드 실행과 지속성

`POST /api/agent/runs`는 202를 반환하고 작업은 별도 스레드에서 실행합니다. 상태·요청·도구 결과·검토 결정은 SQLite, 그래프 상태는 별도 체크포인트 DB에 저장합니다. LangSmith 외부 추적을 사용하지 않고 로컬 해시 연결 실행 기록을 남깁니다.

`queued → running → waiting_input / waiting_review → succeeded / cancelled / failed`를 관리합니다. 도구 결과를 요청·출처 해시에 묶어 캐시하고, 재개 뒤 이미 완료한 도구를 반복하지 않습니다. 요청 키가 같은 재전송도 중복 실행을 만들지 않습니다.

프로세스가 종료돼도 검토 대기 체크포인트를 다시 읽고 재개하는 검사를 통과했습니다. 실행 전 대기 작업도 복구할 수 있습니다. 일시 실패의 명시적 복구는 한 실행에서 1회로 제한하며 원본·계약 오류는 재시도로 덮지 않습니다.

### 사용자 승인과 개정

검토 API는 `X-Review-Token`으로 서버가 검증한 검토자만 받습니다. 검토자 이름을 JSON으로 받아 신뢰하지 않습니다. 토큰은 `COPILOT_REVIEW_TOKEN` 설정 또는 런타임의 권한 제한 파일을 사용합니다. 조직별 로그인·RBAC를 구현한 것은 아닙니다.

검토 결정은 `run_id`, 현재 `state_version`, `report_id`, `revision`, `context_hash`에 연결합니다. 값이 맞지 않거나 이전 개정이면 409입니다. 중복된 같은 결정은 재실행하지 않고 다른 결정과 충돌합니다.

수정은 기존 보고서를 덮어쓰지 않고 새 ID·개정을 만듭니다. 현재 수정 기능은 **사용자 검토 메모 추가**이며 수치나 근거를 자유롭게 편집하는 기능은 아닙니다. 새 개정이 생기면 이전 개정 승인 요청은 무효입니다. 보고서 파일은 불변 초안이며 승인 상태는 실행·검토 기록에서 별도로 관리합니다.

합성 사례에서 승인·거절·동시 상반 결정·권한 없음·이전 개정·원본 변경·재시작·복구·실행 기록 변경을 검사했습니다. 실제 영동 보고서에는 승인 기록을 만들지 않았습니다.

## 5. API와 코드 위치

| API | 목적 |
| --- | --- |
| `POST /api/rag/search/domain` | 개선 검색 |
| `POST /api/reports/evidence-drafts` | 근거 보고서 생성 |
| `GET /api/reports/evidence/{id}` 및 `/html` | 검증된 JSON·HTML |
| `GET /api/agent/status` | 실행 환경·설정 상태 |
| `POST /api/agent/runs` | 작업 생성·백그라운드 실행 |
| `GET /api/agent/runs/{id}` | 상태·도구 기록·대기 정보 |
| `POST /api/agent/runs/{id}/resume` | 대상 보완 또는 인증된 검토 후 재개 |
| `POST /api/agent/runs/{id}/revisions` | 현재 검토 초안의 새 개정 |
| `POST /api/agent/runs/{id}/recover` | 대기·일시 실패 복구 |
| `GET /api/agent/runs/{id}/report` 및 `/html` | 보고서와 검토 상태 조회 |

등록 사례의 요청 예시입니다.

```json
{
  "request": "영동 1호기 2025년 10월 관측과 저장 예측을 비교한 보고서를 작성해 주세요.",
  "case_key": "yeongdong_u1_2025_10_fold_c_v1",
  "task_type": "case_report",
  "planner": "rule",
  "llm_enabled": false,
  "idempotency_key": "my-case-report-request"
}
```

승인 시에는 GET 응답에서 **현재 값**을 복사해 `decision`, `expected_state_version`, `expected_report_id`, `expected_revision`, `expected_context_hash`를 전송합니다. 실제 보고서를 읽고 판단한 사용자가 승인해야 하며 토큰을 LLM 입력이나 공개 예시에 넣지 않습니다.

| 코드 | 책임 |
| --- | --- |
| `rag/domain.py`, `domain_evaluation.py`, `fresh_gold.py` | 검색 계획·추가 질문·비교·선정 잠금 |
| `evidence/facts.py`, `reports.py`, `claims.py` | 수치·인용·보고서·설명 계약 |
| `evidence/gemini.py`, `live_check.py` | 구조화 설명·도구 제안·실제 호출 검사 |
| `agent/tools.py`, `graph.py` | 허용 도구·의존관계·분기·중단 |
| `agent/store.py`, `service.py`, `routes.py` | 원자적 상태·실행 기록·인증 검토·복구 API |

## 6. 검증 결과와 재현

최종 테스트는 **419개 통과**입니다. 실제 HTTP 검사에서는 실제 NOx 예측 모델·실제 등록 문서 검색과 합성 보고서/승인 사례를 함께 사용했습니다. 총 **23개 요청(상태 조회 반복 포함)**이 기대한 응답을 반환했고 인증 없는 승인 요청은 401이었습니다. 실제 발전소 보고서를 테스트 목적으로 승인하지 않았습니다.

```bash
cd NOx_corrected
python -m pytest --require-local-assets -q
MPLCONFIGDIR=/tmp/nox-mpl python scripts/check_copilot_workflow.py
python -m nox.app
```

선택적 Agent 패키지는 아래처럼 별도 경로에 준비합니다. 기존 NOx 전역 패키지를 교체하지 않습니다.

```bash
python -m pip install \
  --target data/runtime/copilot/agent/deps -r requirements-agent.txt
```

재현용 전체 의존성 버전은 [설치 기록](generated/copilot_agent_dependencies.json)에 있습니다. 검색 평가·선정 파일은 덮어쓰지 않습니다. 전체 재평가에는 새 출력 버전을 구현하고, 이미 본 추가 질문을 독립 최종 질문처럼 사용하지 않습니다.

[검증 요약](generated/copilot_workflow_verification.json) · [HTTP 기록](generated/copilot_workflow_http.json)

## 7. 남은 일과 다음 기준

1. 실제 Gemini 설정 후 설명 1회 호출·형식·수치 보존·근거 문장을 검토합니다. 이후 도구 선택도 실제 호출로 확인합니다.
2. 정답 근거의 대안을 담당자가 검토하고, 실패 질문·한국어 표현 변형·여러 근거·모호한 질문을 보완합니다. 새 질문으로 80% 목표와 과도한 거절도 확인합니다.
3. 언어·범위 선택과 청크 다양화가 각각 얼마나 영향을 줬는지 분리 실험하고 재순위 모델을 비교합니다.
4. 실제 사용자의 보고서 검토·수정량·작업 시간을 측정합니다. 아직 시간 절감률·업무 효과·운영 SLA 성과는 없습니다.
5. 새 조건 추론·신규 호기·미래 예측·kg 타깃·PostgreSQL/pgvector·클라우드 배포는 각각 별도로 검증합니다.

NOx 급등 예측 오차·기관 측정 유효성은 이번 검색/워크플로 기능으로 해결된 것이 아닙니다. 일반 문서의 출처 연결, 잘못된 수치 생성 차단, 재개와 검토 기록을 구현한 성과로 구분합니다.

## 공식 설계 참고

- [Gemini Generate Content 구조화 출력](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [Gemini Generate Content 함수 호출](https://ai.google.dev/gemini-api/docs/generate-content/function-calling?hl=ko)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
