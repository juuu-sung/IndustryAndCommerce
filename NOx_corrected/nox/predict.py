import hashlib
import json
import os
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import build_features, Preprocessor
from .history import DEFAULT_DB, HistoryError, HistoryStore, model_key, timestamp, utc_now
from .schema import FUELS, ROOT
from .train import blend


def validate_input(data, known_units):
    if not isinstance(data, dict):
        raise ValueError('JSON object is required')
    required = {'plant', 'unit', 'month', 'capacity_mw', 'generation_mwh',
                'thermal_efficiency_pct', 'utilization_pct', *FUELS.values()}
    weather = {'temperature_c', 'humidity_pct', 'wind_speed_ms', 'wind_direction_deg'}
    if required - set(data):
        raise ValueError(f'Missing fields: {sorted(required - set(data))}')
    if set(data) - required - weather - {'as_of', 'fuel_basis'}:
        raise ValueError(f'Unknown fields: {sorted(set(data) - required - weather - {"as_of", "fuel_basis"})}')
    if 'fuel_basis' in data and data['fuel_basis'] not in ('provider_reported_unit_month', 'site_generation_share_estimate'):
        raise ValueError('fuel_basis must be provider_reported_unit_month or site_generation_share_estimate')
    if not isinstance(data['plant'], str) or isinstance(data['unit'], bool) or not isinstance(data['unit'], int):
        raise ValueError('plant must be text and unit must be an integer')
    if (data['plant'], data['unit']) not in known_units:
        raise ValueError('Plant/unit was not represented in model training')
    if not isinstance(data['month'], str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', data['month']):
        raise ValueError('month must have YYYY-MM format')
    out = dict(data)
    out['month'] = pd.to_datetime(data['month'], format='%Y-%m')
    if 'as_of' in data:
        if not isinstance(data['as_of'], str):
            raise ValueError('as_of must be an ISO timestamp with timezone')
        try:
            out['as_of'] = timestamp(data['as_of'])
        except HistoryError as error:
            raise ValueError(str(error)) from error
        if pd.Timestamp(out['as_of']) > pd.Timestamp(utc_now()):
            raise ValueError('as_of cannot be in the future')
    for col in (required - {'plant', 'unit', 'month'}) | weather:
        value = data.get(col)
        if value is None and col in weather:
            out[col] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'{col} must be a finite number')
        out[col] = float(value)
        if col not in weather and out[col] < 0:
            raise ValueError(f'{col} must be nonnegative')
    if out['capacity_mw'] <= 0:
        raise ValueError('capacity_mw must be positive')
    if not 0 <= out['thermal_efficiency_pct'] <= 100:
        raise ValueError('thermal_efficiency_pct must be between 0 and 100')
    # Utilization can exceed 100 in the supplied data; no invented 100% cap.
    bounds = {'temperature_c':(-80,60), 'humidity_pct':(0,100),
              'wind_speed_ms':(0,100), 'wind_direction_deg':(0,360)}
    for col, (low, high) in bounds.items():
        if pd.notna(out[col]) and not low <= out[col] <= high:
            raise ValueError(f'{col} must be between {low} and {high}')
    radians = np.deg2rad(out.pop('wind_direction_deg'))
    out['wind_sin'], out['wind_cos'] = np.sin(radians), np.cos(radians)
    return out


def validate_model_input(data, known_units, operation_policy=None):
    if not isinstance(data, dict):
        raise ValueError('JSON object is required')
    from .operating_patterns import FEATURES, POLICY, validate_pattern
    supplied = 'operation_pattern' in data
    cleaned = dict(data)
    pattern = cleaned.pop('operation_pattern', None)
    row = validate_input(cleaned, known_units)
    if operation_policy is None:
        if supplied:
            raise ValueError('This model has no operation_pattern input policy')
        return row
    if operation_policy != POLICY:
        raise ValueError('Unknown operation input policy')
    if row['plant'] in operation_policy['plants']:
        if not supplied:
            raise ValueError('Configured plant requires operation_pattern; use explicit null for unavailable/review-required hourly data')
        row.update({col: np.nan for col in FEATURES} if pattern is None else validate_pattern(pattern, row['month']))
    else:
        if supplied:
            raise ValueError('Operation patterns are only supported for configured plants')
        row.update({col: np.nan for col in FEATURES})
    return row


class Predictor:
    def __init__(self, artifact_dir=None, *, history_db=None):
        artifact_dir = Path(artifact_dir or ROOT / 'artifacts/concentration')
        self.metadata = json.loads((artifact_dir / 'metadata.json').read_text())
        for name, expected in self.metadata['bundle_sha256'].items():
            if hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact checksum mismatch: {name}')
        self.pre = Preprocessor.from_dict(json.loads((artifact_dir / 'preprocessor.json').read_text()))
        self.xgb = xgb.XGBRegressor(n_jobs=1)
        self.xgb.load_model(artifact_dir / 'xgb_model.json')
        self.lgb = lgb.Booster(model_file=str(artifact_dir / 'lgb_model.txt'))
        self.history = pd.read_csv(artifact_dir / 'history.csv', parse_dates=['month'])
        self.history_store = None
        if self.metadata['target_mode'] == 'concentration' and history_db is not False:
            db = Path(history_db or os.environ.get('NOX_HISTORY_DB', str(DEFAULT_DB)))
            identity, _ = model_key(artifact_dir)
            self.history_store = HistoryStore(db, identity)
        self.known_units = {(row['plant'], row['unit']) for row in self.metadata['known_units']}
        self.weight = self.metadata['xgboost_weight']
        if not 0 <= self.weight <= 1 or not np.isclose(self.weight + self.metadata['lightgbm_weight'], 1):
            raise ValueError('Invalid saved ensemble weights')
        if self.lgb.feature_name() != self.pre.columns or list(self.xgb.get_booster().feature_names) != self.pre.columns:
            raise ValueError('Model/preprocessor feature schema mismatch')

    def history_status(self):
        if self.history_store and self.history_store.path.exists():
            return self.history_store.status()
        observed = self.history.loc[self.history['target'].notna()]
        return {'storage':'frozen_snapshot', 'revision':0, 'row_count':len(self.history),
                'latest_observed_month':observed['month'].max().strftime('%Y-%m') if len(observed) else None,
                'bootstrap_availability_assumed':True, 'updates_enabled':self.history_store is not None}

    def predict(self, data):
        operation_policy = self.metadata.get('operation_input_policy')
        row = validate_model_input(data, self.known_units, operation_policy)
        fuel_policy = self.metadata.get('fuel_input_basis')
        fuel_basis = row.pop('fuel_basis', None)
        if fuel_policy:
            expected_basis = ('provider_reported_unit_month' if row['plant'] in fuel_policy['reported_unit_plants']
                              else 'site_generation_share_estimate')
            if fuel_basis is None and expected_basis == 'provider_reported_unit_month':
                raise ValueError('This plant requires actual unit fuel: set fuel_basis=provider_reported_unit_month')
            if fuel_basis is not None and fuel_basis != expected_basis:
                raise ValueError('fuel_basis differs from the frozen model input policy')
            fuel_basis = expected_basis
        as_of = row.pop('as_of', utc_now())
        if self.history_store and self.history_store.path.exists():
            history, history_info = self.history_store.snapshot(as_of)
        else:
            history = self.history
            history_info = {'revision':0, 'as_of':as_of, 'bootstrap_availability_assumed':True}
            # The frozen research snapshot has no confirmed publication timestamps.
            ends = (history['month'] + pd.offsets.MonthBegin(1)).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
            history = history.loc[ends <= pd.Timestamp(as_of)]
        # Even a full history file cannot leak current/future targets into requests.
        history = history.loc[history['month'] < row['month']]
        if fuel_policy:
            validate_fuel_history(history, fuel_policy)
        frame = pd.DataFrame([row])
        X = build_features(frame, history)
        transformed = self.pre.transform(X)
        a, b = self.xgb.predict(transformed), self.lgb.predict(transformed, num_threads=1)
        value = float(blend(a, b, self.weight)[0])
        if not np.isfinite(value):
            raise ValueError('Model returned a non-finite prediction')
        warnings = list(self.metadata['target']['limitations'])
        if operation_policy:
            warnings.append('운전 특징은 해당 월의 실제 발전 실적을 집계한 값입니다. 월 시작 전 사전 예측 검증이 아닙니다.')
            warnings.append('발전 실적 0은 보일러·SCR 정지의 확인값이 아니며, 시간별 자료의 공개 시점과 물리 단위는 미확인입니다.')
        imputed = [col for col in self.pre.columns if X[col].isna().iloc[0]]
        if imputed:
            warnings.append('학습 기간의 중앙값으로 결측 특징을 보완했습니다: ' + ', '.join(imputed))
        if X['nox_history_count'].iloc[0] < 3:
            warnings.append('직전 3개월의 실제 NOx 관측값이 모두 확보되지 않았습니다.')
        prior_months = [row['month'] - pd.DateOffset(months=n) for n in (1,2,3)]
        recent = history.loc[(history['plant']==row['plant']) & (history['unit']==row['unit'])
                             & history['month'].isin(prior_months)]
        quality = []
        for _, observation in recent.iterrows():
            reasons = observation.get('quality_reasons', '[]')
            quality.append({'month':observation['month'].strftime('%Y-%m'),
                            'status':observation.get('quality_status','unverified'),
                            'reasons':json.loads(reasons) if isinstance(reasons,str) else [],
                            'has_target':bool(pd.notna(observation['target'])),
                            'source':observation.get('history_source','saved_research_snapshot'),
                            'revision':int(observation.get('history_revision',0))})
        if any(q['status']=='review_required' for q in quality):
            warnings.append('과거 NOx 이력에 품질 검토가 필요한 월이 있습니다. history_quality를 확인하세요.')
        if history_info['bootstrap_availability_assumed'] and (
                'history_availability_assumed' not in recent or recent['history_availability_assumed'].any()):
            warnings.append('기존 연구 스냅샷의 관측 공개 시점은 월 종료 후로 가정했습니다. 실운영 시점 검증이 아닙니다.')
        outside = [col for col, limits in self.metadata['feature_ranges'].items()
                   if not limits['min'] <= transformed[col].iloc[0] <= limits['max']]
        if outside:
            warnings.append('학습 범위를 벗어난 특징이 있습니다: ' + ', '.join(outside))
        trained_through = self.metadata.get('trained_through', self.metadata['split']['train'][1])
        if row['month'] <= pd.Timestamp(trained_through+'-01'):
            warnings.append('학습 기간의 조건에 대한 추론입니다. 독립적인 미래 평가 결과가 아닙니다.')
        if self.metadata.get('trained_at') and pd.Timestamp(as_of)<pd.Timestamp(self.metadata['trained_at']):
            warnings.append('모델 생성 이전 시점 조회입니다. as_of는 관측 이력만 제한하며 당시 모델 가중치를 복원하지 않습니다.')
        return {
            'prediction': {'value': value, 'unit': self.metadata['target']['unit'],
                           'target': self.metadata['target']['name']},
            'model': {'xgboost_weight': self.weight, 'lightgbm_weight': 1-self.weight,
                      'trained_through': trained_through, 'target_mode': self.metadata['target_mode']},
            'warnings': warnings,
            'input_quality': {'imputed_features': imputed, 'outside_training_range': outside,
                              'fuel_basis':fuel_basis,
                              'operation_pattern_status': ('observed_electrical_generation' if operation_policy and pd.notna(row.get('gen_hour_coverage')) else 'unavailable_or_review_required' if operation_policy and row['plant'] in operation_policy['plants'] else 'not_configured'),
                              'observed_previous_months': int(X['nox_history_count'].iloc[0]),
                              'history_quality':quality,
                              'history_sufficiency':'complete' if X['nox_history_count'].iloc[0]==3 else
                                                    'unavailable' if X['nox_history_count'].iloc[0]==0 else 'partial',
                              'history':history_info},
        }


def validate_fuel_history(history, policy):
    """Keep actual/estimated fuel semantics identical to model training."""
    if history.empty:
        return
    expected = np.where(history.plant.isin(policy['reported_unit_plants']),
                        'provider_reported_unit_month', 'site_generation_share_estimate')
    if 'fuel_basis' not in history or not history.fuel_basis.eq(expected).all():
        raise HistoryError('Observation fuel basis differs from the frozen model input policy')
