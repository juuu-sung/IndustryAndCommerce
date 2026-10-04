# NOx 보완 파이프라인 실행 안내

공모전 이후 전처리·평가·API·관측 이력·기관 타깃 검증을 보완한 코드입니다. 프로젝트 개요와 주요 수치는 [최상위 README](../README.md), 코드별 진행 상황은 [진행 기록](../docs/PROGRESS.md)에 있습니다.

## 설치와 기본 API

데이터·모델은 GitHub에 포함하지 않습니다. 기존 작업 폴더를 가진 사용자는 저장소 최상위에서 아래 명령으로 로컬 파일을 연결합니다. 코드·문서와 공개 가상 예시는 덮어쓰지 않습니다. 기존 폴더가 없다면 필요한 자료를 확보하고 같은 형식으로 학습 모델을 먼저 구성해야 합니다.

```bash
python scripts/restore_local_assets.py --source ../NOx_corrected --dry-run
python scripts/restore_local_assets.py --source ../NOx_corrected
```

`--source`는 기존 프로젝트 경로로 바꿀 수 있습니다. 이후 아래 명령은 `NOx_corrected/` 안에서 실행합니다. 로컬 저장 모델을 연결했다면 추가 학습 없이 실행할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nox.history init
python -m nox.app
```

기본 DB는 `data/runtime/history.sqlite`, 기본 모델은 `artifacts/concentration`입니다. `init`은 기존 DB를 삭제하는 명령이 아닙니다. 새 모델은 새 DB에 연결하고 모델 식별 정보를 검증합니다. 서버는 개발용 `127.0.0.1:5001`에서 실행되며 첫 터미널이 요청을 기다리는 것이 정상입니다.

```bash
# 별도 터미널, 같은 디렉터리
curl -sS http://127.0.0.1:5001/api/health
curl -sS http://127.0.0.1:5001/api/history/status
curl -sS http://127.0.0.1:5001/api/predict \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

## 기존 로컬 저장 모델 선택

| 디렉터리 | 학습 | 가중치 선택 | 저장 평가 | 용도 |
| --- | --- | --- | --- | --- |
| `artifacts/concentration` | 2023년 | 2024년 상반기 | 2024년 하반기 | 기본 API·초기 보완 기준 |
| `artifacts/concentration_20261001` | 2023~2025년 상반기 | 2025년 하반기 | 2026년 1~8월 | 이후 기간 재학습 비교 |
| `artifacts/concentration_unit_fuel_20261001` | 동일 | 동일 | 동일 | 영동 호기별 실제 연료 입력 비교 |
| `artifacts/legacy` | 원래 구성 재현 | 메타데이터 참조 | 진단용 | 단위·환산 미검증; 실서비스 사용 대상 아님 |
| `reports/temporal_validation_20261001/models` | A/B/C | 각 검증 구간 | 각 후속 구간 | 기간별 실험 재현 |
| `reports/operations_validation_20261002/models` | A/B/C | 각 검증 구간 | 각 후속 구간 | 운전 특징 실험·API 정책 검증 |

예를 들어 2025년 상반기까지 학습한 모델과 2026년 8월까지의 관측 이력을 로컬에서 사용하려면:

```bash
python -m nox.history \
  --db data/runtime/retrained.sqlite \
  --artifact-dir artifacts/concentration_20261001 init
python -m nox.history \
  --db data/runtime/retrained.sqlite \
  --artifact-dir artifacts/concentration_20261001 import \
  --raw-dir data/extended_20261001 \
  --start-month 2025-01 --end-month 2026-08 \
  --source official_public_snapshot_20261001 \
  --report-dir reports/local_history_import
NOX_ARTIFACT_DIR=artifacts/concentration_20261001 \
NOX_HISTORY_DB=data/runtime/retrained.sqlite python -m nox.app
```

최신 이력용 `examples/request_latest.json`은 기존 로컬 프로젝트에서 연결하는 예시입니다. 공개 `examples/request.json`은 형식 확인용 가상 운전 조건이며 실제 발전 실적이 아닙니다. 영동 실제 연료 모델은 해당 `fuel_basis`가 필요하고, 운전 특징 실험 모델은 `operation_pattern`을 명시해야 합니다. [API 안내](../docs/API_GEMINI.md)를 확인하세요.

## 테스트와 지표 재생성

```bash
python -m pytest -q
# 로컬 자료가 필요한 전체 통합 검증
python -m pytest -q --require-local-assets
# 저장 예측값에서 표 재계산: 저장소 최상위에서 실행
cd ..
python scripts/summarize_results.py
# 그림도 생성하려면
python -m pip install -r NOx_corrected/requirements-docs.txt
MPLCONFIGDIR=/tmp/nox-mpl python scripts/summarize_results.py --chart
```

공개 코드만 있는 경우 합성 자료로 독립 검사를 실행하며 `requires_assets` 검사는 건너뜁니다. 전체 검증 명령은 필수 로컬 스냅샷이 없으면 수집 단계에서 실패합니다. 테스트는 모의 LLM을 사용하며 실제 Gemini 호출이나 외부 자료 다운로드를 하지 않습니다. SQLite는 임시 경로에서 생성합니다. 전체 테스트에는 소규모 실제 모델 적합을 포함하므로 실행 환경에 따라 시간이 달라집니다.

## 기존 자료로 후보 실험을 다시 실행

저장된 결과 폴더는 덮어쓰지 않으므로 새 출력 경로를 지정합니다.

```bash
python -m nox.candidate_selection \
  --raw data/readiness_bundle_20261002 \
  --protocol reports/readiness_20261002/protocol.json \
  --output "reports/new_candidates_$(date +%Y%m%d_%H%M%S)" \
  --max-seconds 300
```

4개 후보 × 2개 입력 정책 × 3개 기간 × 5개 시드 = 120개 실험, 150개 모델 적합입니다. 300초 제한을 넘기면 워커를 중단하고 부분 결과를 남깁니다. VS Code에서 길게 실행하려면 새 출력 경로와 `--max-seconds 3600`을 사용하세요. 이는 이미 검토한 기간의 **개발 비교**입니다.

## 기관 자료 확인 후 재학습

지금 실행 가능한 것은 요청 자료 준비와 저장 예측 비교입니다. **확인된 타깃 버전 폴더는 아직 없습니다.** `verified_v1/target_version.json`을 임의로 만들거나 템플릿의 상태만 바꾸어 학습을 진행하면 안 됩니다. 기관 원시 구간 자료·공식 일평균·규칙 증빙을 확보한 뒤 [타깃 검증 안내](../docs/TARGET_VALIDATION.md)의 `verify`를 통과해야 합니다.

## 핵심 모듈

| 모듈 | 책임 |
| --- | --- |
| `data.py`, `features.py`, `quality.py` | 데이터 결합, 달력 특징, 타깃·입력 품질 |
| `train.py`, `temporal_validation.py` | 기본 학습·가중치 선정·기간별 검증 |
| `development_experiments.py`, `candidate_selection.py` | 변수 묶음·변환·시드·후보 비교 |
| `unit_fuel.py`, `operating_patterns.py`, `weather_quality.py` | 실제 호기 연료·시간별 발전·기상 출처 관리 |
| `history.py`, `predict.py` | 관측 이력 갱신·시점 선택·저장 모델 추론 |
| `app.py`, `llm.py`, `gemini_check.py` | REST API·Gemini 호출·실제 호출 점검 |
| `target_validation.py`, `validated_training.py` | 기관 일평균 재현·타깃 버전·검증 후 재학습 |

공개 저장소에는 코드·집계 지표·그림을 포함했습니다. 실제 원천 자료·기준 모델·행별 예측·대량 후보 모델·개인 실행 로그는 모두 제외했습니다. 로컬 스냅샷을 연결하면 전체 테스트와 예측을 실행하고, 새 후보 모델은 위 명령으로 재생성할 수 있습니다.
