import numpy as np
import pandas as pd

from .data import unique
from .schema import CONDITIONS, FUELS, KEYS


def build_features(rows, history=None):
    """The same function for batch training and a single API request.

    History joins require exact calendar months. A missing February cannot make
    January become March's lag1. Current/future targets never enter features.
    """
    rows = rows.copy().reset_index(drop=True)
    history = rows if history is None else history.copy()
    unique(rows, KEYS, 'prediction rows')
    unique(history, KEYS, 'history')
    out = rows[KEYS + CONDITIONS].copy()
    from .weather_quality import QUALITY_FEATURES
    if any(col in rows for col in QUALITY_FEATURES):
        if set(QUALITY_FEATURES) - set(rows):
            raise ValueError('Incomplete weather quality features')
        out[QUALITY_FEATURES] = rows[QUALITY_FEATURES].to_numpy()
    from .operating_patterns import FEATURES as OPERATION_FEATURES
    if any(col in rows for col in OPERATION_FEATURES):
        for col in OPERATION_FEATURES:
            out[col] = rows[col] if col in rows else np.nan
        out['operation_pattern_available'] = out['gen_hour_coverage'].notna().astype(int)
    target_history = history[KEYS + ['target']]
    for lag in (1, 2, 3):
        shifted = target_history.copy()
        shifted['month'] = shifted['month'] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={'target': f'lag{lag}'})
        out = out.merge(shifted, on=KEYS, how='left', validate='one_to_one')
    out['nox_lag1'] = out['lag1']
    out['nox_roll3'] = out[['lag1', 'lag2', 'lag3']].mean(axis=1)
    out['nox_history_count'] = out[['lag1', 'lag2', 'lag3']].notna().sum(axis=1)
    out['nox_lag1_missing'] = out['nox_lag1'].isna().astype(int)
    fuel_cols = list(FUELS.values())
    prior = history[KEYS + fuel_cols].copy()
    prior['month'] += pd.DateOffset(months=1)
    prior = prior.rename(columns={c: f'previous_{c}' for c in fuel_cols})
    out = out.merge(prior, on=KEYS, how='left', validate='one_to_one')
    for col in fuel_cols:
        # Each fuel retains its unit. No total across oil kl and mass ton.
        before = out[f'previous_{col}']
        out[f'{col}_change'] = out[col].div(before.where(before > 0)) - 1
        out[f'{col}_per_mwh'] = out[col].div(out['generation_mwh'].where(out['generation_mwh'] > 0))
    month = out['month'].dt.month
    out['month_sin'] = np.sin(2 * np.pi * month / 12)
    out['month_cos'] = np.cos(2 * np.pi * month / 12)
    for plant in ['분당', '삼천포', '여수', '영동', '영흥']:
        out[f'plant_{plant}'] = (out['plant'] == plant).astype(int)
    for unit in range(1, 9):
        out[f'unit_{unit}'] = (out['unit'] == unit).astype(int)
    return out.drop(columns=[*KEYS, 'lag1', 'lag2', 'lag3',
                              *[f'previous_{c}' for c in fuel_cols]]).replace([np.inf, -np.inf], np.nan)


class Preprocessor:
    """Train-only feature selection and median fill, serializable as plain JSON."""
    def fit(self, X, *, fit_scope='2023 only'):
        self.fit_scope = fit_scope
        self.columns = [c for c in X if X[c].notna().any() and X[c].nunique(dropna=False) > 1]
        if not self.columns:
            raise ValueError('No nonconstant observed features')
        self.medians = {c: float(X[c].median()) for c in self.columns}
        return self

    def transform(self, X):
        missing = set(self.columns) - set(X)
        if missing:
            raise ValueError(f'Missing features: {sorted(missing)}')
        result = X[self.columns].fillna(self.medians).astype(float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Non-finite model inputs')
        return result

    def to_dict(self):
        return {'columns': self.columns, 'medians': self.medians,
                'fit_scope': getattr(self, 'fit_scope', '2023 only'), 'version': 1}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.columns, obj.medians = data['columns'], data['medians']
        obj.fit_scope = data.get('fit_scope', 'unspecified')
        return obj
