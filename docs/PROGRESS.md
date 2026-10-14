# 프로젝트 진행 기록

공모전 원본과 이후 보완 구현을 구분한 기록입니다. 파일명 날짜는 해당 작업 묶음의 날짜이며, 구현 기록은 2026-10-14까지, 저장소 정리는 2026-10-15 기준입니다.

| 단계 | 실제 작업 | 저장 근거 |
| --- | --- | --- |
| 2025년 공모전 | 발전 실적 수집 노트북, 당시 전처리·모델·시스템 설명 | 최상위 원본 노트북, `docs/archive/competition_readme_2025.md` |
| 2026-09-30 보완 | 데이터 감사, ppm/legacy 분리, 달력 특징, 시간 분할, Flask·Gemini 코드, 모의 응답 검증 | `nox/data.py`, `features.py`, `train.py`, `app.py`, `llm.py`, `artifacts/concentration` |
| 타깃 품질·이력 | 원본 보존·검토 이유, 월별 관측 품질, 모델 바인딩 DB·입수 시각·개정·재반입 검사 | `nox/quality.py`, `history.py`, 관련 테스트 |
| 2026-10-01 최신 자료·재학습 | 2025~2026년 8월 공개 자료, 기존 자료와 중첩 대조·SHA-256, 초기/재학습 공통 관측 비교 | `data/retrieved/20261001`, `data/extended_20261001`, `reports/retraining` |
| 2026-10-01 실제 호기 연료 | 영동 호기 실적·사업소 합계 대조, 추정 입력과 분리 | `data/unit_fuel_20261001`, `reports/unit_fuel_retraining` |
| 2026-10-01 기간별 검증 | A/B/C 달력 월 분할, 7개 방법 비교, 신규 호기 분리, 모델·가중치 고정 검증 | `reports/temporal_validation_20261001` |
| 2026-10-02 시간별 운전 | 영동 시간별 전기 발전 패턴·월 실적 대조, 불일치 입력 특징 격리·타깃 유지 | `data/operations_20261002`, `reports/operations_validation_20261002` |
| 2026-10-02 기존 자료 개선 실험 | 큰 오차 기여 분석→변수 묶음 제거 비교→ppm/log 학습 비교 | `reports/pre_request_experiments_20261002` |
| 2026-10-02 기상·후보 기준 | 기상 셀 출처·충족률, 2026년 영동 11,664개 시간 셀 보완, 120개 실험·150회 모델 적합 | `data/readiness_bundle_20261002`, `reports/readiness_20261002` |
| 2026-10-03 기관 타깃 절차 | 급등 4구간 첨부·기관 요청 초안, 공식 일평균 재현·버전·검증 후 학습 코드, 합성 자료 테스트 | `nox/target_validation.py`, `validated_training.py`, `tests/test_target_validation.py` |
| 2026-10-04 저장소 정리 | 원본 보존, 표·그림 재계산, 실행 안내, 로컬 자료 연결·전체 164개 테스트, 공개 합성 자료 검사 | `scripts/summarize_results.py`, `docs/RELEASE_CHECK.md` |
| 2026-10-13 사례 등록 | 영동 1호기 2025년 10월 C 평가 묶음, 9개 파일 해시·기간·관측/예측·선정 잠금 검사, 신규 49개·전체 213개 테스트 통과 | `nox/copilot/registry.py`, `tests/test_copilot_registry.py`, [등록 안내](COPILOT_CASE_REGISTRY.md) |
| 2026-10-13 사례 계산·확장 구조 | 검증 바이트의 전월·직전 3개월·저장 예측 오차, 품질·출처·코드 버전 JSON, 신규 32개·전체 245개 테스트 통과, 노션 구조 정리 | `nox/copilot/archive_repository.py`, `calculations.py`, [계산 안내](COPILOT_CALCULATIONS.md), [노션](https://app.notion.com/p/3f8a7839359181ee8286cf13ccbfba31) |
| 2026-10-14 보고서 JSON·HTML·API | 계약·원본/내용 검증·관측 충족률·초안 저장·4개 Flask API·CLI, 신규 61개·전체 306개 테스트 및 실제 HTTP 정상 응답 | `nox/copilot/report_builder.py`, `validators.py`, `renderers.py`, `repositories.py`, `routes.py`, [보고서 안내](COPILOT_REPORTS.md) |
| 2026-10-14 문서 검색·평가 | 실제 문서 10개·616개 청크·50개 원문 근거 질문, BM25/실제 E5/Hybrid·3개 API, 신규 42개·전체 348개 테스트·실제 HTTP 확인 | `nox/copilot/rag/`, [검색 결과와 한계](RAG_RETRIEVAL.md), `docs/generated/rag_*` |
| 2026-10-14 근거 보고서·Agent | 범위/한영 검색·추가 질문 30개, 수치/문서 참조·Gemini 설명 계약, LangGraph/SQLite·백그라운드·인증 검토/개정/재개·복구, 신규 71개·전체 419개 테스트·실제 HTTP 23개 요청 | `rag/domain.py`, `copilot/evidence/`, `copilot/agent/`, [상세 과정](COPILOT_WORKFLOW_20261014.md), `docs/generated/copilot_*` |
| 2026-10-15 공개 저장소 정리 | 사례·보고서·RAG·Agent 코드, 테스트, 검색 비교·HTTP 검증 집계와 구현 문서 정리. 원천 자료·모델·실제 보고서·검토 토큰은 제외 | [최상위 README](../README.md), `docs/generated/`, `NOx_corrected/nox/copilot/` |

## 구현했지만 실제 검증이 남은 부분

- Gemini: HTTP 호출·프롬프트·스키마·오류 처리 구현, 모의 응답 검증 완료. 실제 외부 호출 성공 기록 없음.
- Gemini 근거 설명/도구 선택: 참조·수치 보호와 함수 호출 ID/서명 보존 코드·모의 검증 완료. 키·모델 미설정으로 실제 확인은 `not_run`.
- Agent: 기본 규칙 기반 흐름·LangGraph 체크포인트·인증된 검토/재개·원본/버전 충돌 검사 완료. 실제 Gemini가 선택하는 실행·조직 로그인/권한·운영 배포는 미확인.
- 기관 확인: 요청 초안·첨부·검증 코드 구현. 메일 발송·회신·실제 일평균 재현 완료 기록 없음.
- 타깃 수정 후 재학습: 실행 경로·차단 조건·합성 자료 통합 테스트 구현. 실제 확인 버전·실제 재학습 결과 없음.
- 개선 후보: 22.28 / 6.48 ppm의 개발 결과 확인. 기본 Flask 모델 교체 없음.
- 새 기간 최종 평가: 자료 확보·정의 확정 후 진행해야 함.
- 월 총배출량 kg: 유량·시간·표준 상태·보정 정의 미확보로 검증 가능한 타깃 미구성.

## 저장소에 포함한 범위

원본 수집 노트북을 변경하지 않고 이후 코드를 `NOx_corrected/`에 추가했습니다. 공개 대상은 코드·문서·집계 지표·그림·가상 입력입니다. 원천 자료·저장 모델·행별 평가 예측·개인 DB는 Git에서 제외했습니다. 기존 작업 폴더를 로컬에 연결하는 스크립트와 데이터 없이 실행 가능한 합성 자료 검사도 제공합니다.

테스트 한 곳이 개발자 이력 DB 파일에 의존하던 문제는 임시 DB를 직접 생성하게 수정했습니다. `collect_operations_evidence.py`가 사용하는 BeautifulSoup을 검증 버전 의존성에 추가했습니다. 로컬 학습 모델·데이터 원본·평가 예측 파일은 원래 저장 값을 유지했습니다. 공개 입력 예시는 가상값으로 바꿨습니다.

## 다음 순서

RAG·Agent 확장은 기관 확인과 병행합니다. 수치/문서 근거 보고서와 기본 규칙 기반 Agent 검토/재개까지 구현했습니다. 동일한 추가 답변 가능 질문 20개에서 기존 Keyword 11개(55%)→개선 Keyword 13개(65%)로 증가했지만 80% 목표에는 못 미쳤습니다. 실제 영동 사례는 사실 22개·문서 근거 8개가 연결된 사용자 검토 대기로 준비했습니다. 실제 Gemini 호출·독립 근거 검토·검색 품질·신규 조건/호기·운영 검증은 남았습니다. [상세 과정](COPILOT_WORKFLOW_20261014.md)과 [전체 계획](COPILOT_IMPLEMENTATION_PLAN.md)에 범위를 구분합니다.

1. 대표 급등과 정상·정지·기동·교정 구간의 기관 정의·상태 코드를 확인합니다.
2. 공식 일평균·유효 시간을 재현하고 타깃 규칙 버전을 고정합니다.
3. 전체 개발 기간의 동일 타깃에서 고정 후보들을 재비교합니다.
4. 유효 급등은 연소·SCR 기록으로 설명 특징을 보완합니다.
5. 아직 결과를 확인하지 않은 새 기간에서 기준 예측·모델·지역별 성능을 평가합니다.
6. Gemini 실제 호출을 점검하고 운영 배포·화면 연결을 별도로 검증합니다.
