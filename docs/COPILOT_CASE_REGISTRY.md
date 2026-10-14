# NOx 평가 사례 등록과 재검증

영동 1호기 2025년 10월의 저장 관측·평가 예측·모델·전처리를 하나의 사례 버전으로 등록했습니다. 이 단계는 보고서 작성 전에 사용할 원본과 평가 기간을 고정합니다. 보고서 생성·RAG·Agent 기능은 이후 단계입니다.

## 등록된 사례

| 항목 | 값 |
| --- | --- |
| case_key | yeongdong_u1_2025_10_fold_c_v1 |
| 분석 모드 | evaluation_replay |
| 대상 | 영동 1호기, 2025년 10월 |
| 학습 | 2023년 1월~2024년 12월 |
| 선정 | 2025년 1~6월 |
| 평가 | 2025년 7~12월 |
| 실제 모델 생성 | 2026년 10월 1일 |
| 파일 수 | 9 |
| 관측 품질 | review_required |
| 기관 타깃 확인 | pending_provider_confirmation |

사례는 과거 기간의 평가를 현재 재현하는 기록입니다. 당시 서비스가 운영한 미래 예측 기록이나 공식 측정 유효성 확인 결과로 표현하지 않습니다. 등록 결과에도 이 한계를 보존했습니다.

실제 관측값·예측값과 해시는 로컬 data/runtime/copilot/cases/yeongdong_u1_2025_10_fold_c_v1.json에 있습니다. 이 폴더는 Git에서 제외합니다.

## 고정한 파일과 검증

| 파일 | 검사 |
| --- | --- |
| metadata.json | 연구 농도 타깃·ppm, 기간 순서, 학습 종료, 생성 시각, 알려진 호기 |
| history.csv | 유일한 발전소·호기·월, 관측값, 직전 3개월 이력 |
| test_predictions.csv | 평가 기간·관측 대응·호기 지원·입력 검증 기록 |
| validation_predictions.csv | 선정 기간·관측 대응·호기 지원 |
| validation_weight_search.json | 검증 RMSE로 선택한 가중치와 메타데이터 일치 |
| selection_lock.json | 평가 전 고정 선언과 4개 잠금 파일 해시 |
| preprocessor.json | 학습 기간의 fit_scope, 특징·중앙값 |
| xgb_model.json | 메타데이터 해시 및 특징 이름·순서 |
| lgb_model.txt | 메타데이터 해시 및 특징 이름·순서 |

등록 파일은 각 원본의 SHA-256과 바이트 수를 보존합니다. 모델·전처리에는 기존 메타데이터 및 selection_lock의 해시도 대조합니다. 파일의 등록 후 변경, 중복 키, 타깃 불일치, 기간 중첩, 미학습 호기, 거부된 사례 입력, 비유한 수치, 잘못된 특징 순서를 차단합니다.

모델 파일의 무결성과 특징 헤더를 검사하며, 이 모듈에서 모델을 실행하거나 재학습하지는 않습니다. 등록한 저장 예측의 오차 계산은 [별도 계산 기능](COPILOT_CALCULATIONS.md)으로 구현했습니다. 새로운 조건의 추론 연결과 보고서 구성은 후속 작업입니다. 원천 자료 manifest의 해시는 메타데이터에 기록된 값을 보존하지만 해당 원천 manifest 파일을 재검증한 것으로 표시하지 않습니다.

## 버전과 덮어쓰기 방지

- model_version_id는 모델·전처리와 모델 메타데이터의 파일 내용에서 계산합니다.
- bundle_version_id는 9개 파일의 내용에서 계산합니다.
- case_version_id는 대상, 기간, 타깃 정의, 모델·자료 버전, 고정 관측·예측 및 품질 상태를 묶습니다.
- 생성 시각과 폴더 위치는 내용 버전에서 제외하므로 같은 파일 묶음을 옮겨도 내용 버전은 유지됩니다.
- 같은 case_key를 다시 등록하면 기존 파일·등록 시각·버전을 그대로 반환합니다.
- 기존 원본이 바뀐 상태에서는 같은 키의 재등록으로 해시를 갱신하지 않습니다. 새 자료에는 새 사례 키/버전을 사용합니다.
- 임시 파일과 배타적 생성으로 완성된 JSON만 저장하며 기존 등록부를 덮어쓰지 않습니다.

최초 등록은 현재 파일을 고정하는 기준을 만듭니다. 파일 해시만으로 과거 생성 순서나 실제 측정값의 유효성이 인증되는 것은 아닙니다.

## 실행

이 코드가 추가된 저장소의 NOx_corrected에서 실행합니다. 기본 저장 위치는 해당 프로젝트의 data/runtime/copilot/cases입니다.

```bash
cd NOx_corrected

python -m nox.copilot.registry register \
  --artifact-dir reports/temporal_validation_20261001/models/unit_actual/C \
  --case-key yeongdong_u1_2025_10_fold_c_v1 \
  --plant 영동 --unit 1 --month 2025-10

python -m nox.copilot.registry verify \
  --case-key yeongdong_u1_2025_10_fold_c_v1
```

정상이면 status: ok, 오류가 있으면 status: blocked와 종료 코드 1을 반환합니다. CLI에는 관측·예측 수치를 출력하지 않고 사례·기간·검토 상태를 요약합니다. 다른 폴더를 사용할 때는 register와 verify에 같은 --registry-dir를 지정합니다.

Python에서는 nox.copilot.registry의 CaseRegistry를 사용합니다. register는 검증된 등록 데이터를 반환하고 verify는 등록된 파일과 원본의 대응을 다시 검사합니다. 다음 ReportBuilder는 검증 결과의 snapshot과 파일·모델·기간 정보를 사용해 현재의 다른 관측으로 바꿔 읽는 일을 피합니다.

## 검증 범위와 다음 작업

신규 검증은 실제 자산 없이 실행할 수 있는 합성 검사와 실제 영동 자산 등록 검사를 구분합니다. 합성 모델 파일은 무결성·특징 계약 검사를 위한 최소 헤더이며 모델 추론 결과가 아닙니다. 실제 자산 검사에서는 등록 전후 9개 원본 파일의 해시가 유지되는지 확인합니다.

```bash
python -m pytest -q tests/test_copilot_registry.py
```

2026년 10월 13일 신규 등록·오류 검증 49개와 실제 자산을 포함한 전체 213개 테스트가 통과했습니다. 전체 검사는 --require-local-assets로 실행했고 63.26초가 걸렸으며 건너뛴 검사는 없습니다. 기존의 불리언 부적합 입력 검사에서 Pandas FutureWarning 1건이 발생했습니다.

등록부의 관측·예측으로 전월 변화·3개월 평균·오차를 계산하는 기능과 버전 JSON을 구현했습니다. 2026년 10월 14일에는 보고서 계약·검증·JSON/HTML·4개 Flask API를 추가했고 전체 306개 테스트가 통과했습니다. [계산 안내](COPILOT_CALCULATIONS.md)와 [보고서 안내](COPILOT_REPORTS.md)를 참고하세요. RAG·Agent·새 조건 추론은 후속 단계입니다.

- [등록 구현](../NOx_corrected/nox/copilot/registry.py)
- [검증 코드](../NOx_corrected/tests/test_copilot_registry.py)
- [전체 구현 계획](COPILOT_IMPLEMENTATION_PLAN.md)

