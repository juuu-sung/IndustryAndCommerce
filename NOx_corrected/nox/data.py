"""Left joins, explicit calendar keys, and auditable monthly aggregation."""
import json
import re

import numpy as np
import pandas as pd

from .schema import FUELS, KEYS, OPERATIONS, ROOT


def read(name, raw=None):
    return pd.read_csv((raw or ROOT / 'data/raw') / name, encoding='utf-8-sig')


def months(series):
    text = series.astype(str).str.replace('-', '', regex=False)
    return pd.to_datetime(text, format='%Y%m', errors='raise')


def normalize_keys(df, month_col='월'):
    df = df.rename(columns={'사업소': 'plant', '호기': 'unit'}).copy()
    df['plant'] = df['plant'].astype(str).str.strip().replace({'분당화력': '분당'})
    units = pd.to_numeric(df['unit'], errors='raise')
    if not np.isfinite(units).all() or (units <= 0).any() or (units != np.floor(units)).any():
        raise ValueError('unit must be a positive integer')
    df['unit'] = units.astype(int)
    df['month'] = months(df[month_col])
    return df.drop(columns=[month_col])


def unique(df, keys, name):
    if df.duplicated(keys).any():
        raise ValueError(f'{name}: duplicate keys {keys}')


def generation(raw=None):
    df = normalize_keys(read('generation.csv', raw)).rename(columns=OPERATIONS)
    unique(df, KEYS, 'generation')
    return df


def allocate_fuel(gen, raw=None):
    """Use site totals as estimates; no arbitrary mapping of CG/CS API units."""
    site = read('fuel_site.csv', raw).rename(columns={'사업소': 'plant'})
    site['plant'] = site['plant'].replace({'분당화력': '분당'})
    site = site[site['plant'].isin(gen['plant'].unique())].copy()
    site['month'] = months(site['일자'])
    unique(site, ['plant', 'month'], 'site fuel')
    if not np.allclose(site['계(석탄)'], site['유연탄'] + site['무연탄']):
        raise ValueError('Coal total does not match components')
    result = gen[KEYS + ['generation_mwh']].merge(
        site[['plant', 'month', *FUELS]], on=['plant', 'month'],
        how='left', validate='many_to_one')
    denominator = result.groupby(['plant', 'month'])['generation_mwh'].transform('sum')
    ratio = result['generation_mwh'].div(denominator.where(denominator > 0))
    for source, output in FUELS.items():
        result[output] = result[source] * ratio
    # kl and ton stay separate; coal total never becomes a second input.
    result = result[KEYS + list(FUELS.values())]
    unit_path = (raw or ROOT / 'data/raw') / 'fuel_unit.csv'
    if not unit_path.exists():
        return result
    # Overrides are complete provider-reported site/month groups. A partial
    # group would mix actual values with allocated estimates and double count.
    unit = read('fuel_unit.csv', raw)
    required = ['사업소', '호기', '일자', *FUELS, '계(석탄)', 'source_file']
    if set(required) - set(unit) or unit.empty:
        raise ValueError('unit fuel: missing columns or empty source')
    # Audit exports also have derived English month/unit columns. Use original
    # provider keys explicitly so a rename cannot create duplicate columns.
    unit = normalize_keys(unit[required], month_col='일자')
    unique(unit, KEYS, 'unit fuel')
    for col in [*FUELS, '계(석탄)']:
        unit[col] = pd.to_numeric(unit[col], errors='raise')
        if not np.isfinite(unit[col]).all() or (unit[col] < 0).any():
            raise ValueError('unit fuel: all reported values must be finite and nonnegative')
    if unit['source_file'].isna().any() or unit['source_file'].astype(str).str.strip().eq('').any():
        raise ValueError('unit fuel: source_file is required')
    if not np.allclose(unit['계(석탄)'], unit['유연탄'] + unit['무연탄'], rtol=0, atol=1e-6):
        raise ValueError('unit fuel: coal total does not match components')
    matched = unit.merge(gen[KEYS], on=KEYS, how='left', indicator=True, validate='one_to_one')
    if not matched['_merge'].eq('both').all():
        raise ValueError('unit fuel: unrepresented plant/unit/month')
    groups = unit[['plant', 'month']].drop_duplicates()
    expected = gen[KEYS].merge(groups, on=['plant', 'month'], validate='many_to_one')
    if len(expected) != len(unit):
        raise ValueError('unit fuel: incomplete site/month unit coverage')
    totals = unit.groupby(['plant', 'month'], as_index=False)[list(FUELS)].sum()
    control = totals.merge(site[['plant', 'month', *FUELS]], on=['plant', 'month'],
                           how='left', suffixes=('_unit', '_site'), validate='one_to_one')
    if any(not np.allclose(control[c+'_unit'], control[c+'_site'], rtol=0, atol=1e-6)
           for c in FUELS):
        raise ValueError('unit fuel: reported units do not reconcile to site totals')
    actual = unit[KEYS + list(FUELS) + ['source_file']].rename(columns={
        **{c:'actual_'+v for c,v in FUELS.items()}, 'source_file':'fuel_source_file'})
    result = result.merge(actual, on=KEYS, how='left', validate='one_to_one')
    reported = result['fuel_source_file'].notna()
    result['fuel_basis'] = np.where(reported, 'provider_reported_unit_month', 'site_generation_share_estimate')
    for col in FUELS.values():
        result.loc[reported, col] = result.loc[reported, 'actual_'+col]
    return result.drop(columns=['actual_'+c for c in FUELS.values()])


def monthly_weather(raw=None):
    df = read('weather_daily.csv', raw).rename(columns={
        '사업소': 'plant', '호기': 'station', '온도': 'temperature_c',
        '습도': 'humidity_pct', '풍향': 'direction_deg', '풍속': 'wind_speed_ms'})
    df['date'] = pd.to_datetime(df['일자'], errors='raise')
    unique(df, ['plant', 'station', 'date'], 'weather daily')
    ranges = {'temperature_c': (-80, 60), 'humidity_pct': (0, 100),
              'direction_deg': (0, 360), 'wind_speed_ms': (0, 100)}
    invalid = {}
    for col, (low, high) in ranges.items():
        df[col] = pd.to_numeric(df[col], errors='coerce')
        bad = df[col].notna() & ~df[col].between(low, high)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    # 0 C, 0 m/s and 0 degrees are real values. Wind is averaged as a vector.
    angle = np.deg2rad(df['direction_deg'])
    df['wind_sin'], df['wind_cos'] = np.sin(angle), np.cos(angle)
    cols = ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'wind_sin', 'wind_cos']
    daily = df.groupby(['plant', 'date'], as_index=False)[cols].mean()
    daily['month'] = daily['date'].dt.to_period('M').dt.to_timestamp()
    monthly = daily.groupby(['plant', 'month'], as_index=False)[cols].mean()
    # Use the circular mean direction at both training and request time.
    magnitude = np.hypot(monthly['wind_sin'], monthly['wind_cos'])
    monthly['wind_sin'] /= magnitude.where(magnitude > 1e-12)
    monthly['wind_cos'] /= magnitude.where(magnitude > 1e-12)
    counts = daily.groupby(['plant', 'month']).agg(
        weather_days=('date', 'nunique'), temperature_days=('temperature_c', 'count'))
    monthly = monthly.merge(counts.reset_index(), on=['plant', 'month'], validate='one_to_one')
    monthly['weather_coverage'] = monthly['temperature_days'] / monthly['month'].dt.days_in_month
    return monthly, invalid


def monthly_concentration(raw=None, gen=None):
    from .quality import inspect_targets
    daily, out, summary = inspect_targets(read('emissions_daily.csv', raw), gen)
    if summary['structural_error_rows']:
        raise ValueError('Malformed/duplicate emissions: run python -m nox.quality to inspect')
    unmapped = daily.loc[daily['unit'].isna(), ['plant', '호기']].value_counts().reset_index(name='rows')
    return out, unmapped.to_dict('records')


def build_dataset(mode, raw=None):
    gen = generation(raw)
    fuel = allocate_fuel(gen, raw)
    weather, invalid = monthly_weather(raw)
    if mode == 'legacy':
        base = normalize_keys(read('legacy_model.csv', raw)).rename(columns={**OPERATIONS, 'NOX_kg': 'target'})
        base = base[KEYS + list(OPERATIONS.values()) + ['target']]
        unmapped = []
    elif mode == 'concentration':
        concentration, unmapped = monthly_concentration(raw, gen)
        base = gen[KEYS + list(OPERATIONS.values())].merge(concentration, on=KEYS, how='left', validate='one_to_one')
    else:
        raise ValueError(f'Unknown target mode: {mode}')
    unique(base, KEYS, 'model base')
    out = base.merge(fuel, on=KEYS, how='left', validate='one_to_one').merge(
        weather, on=['plant', 'month'], how='left', validate='many_to_one')
    out = out.sort_values(KEYS).reset_index(drop=True)
    bad_efficiency = out['thermal_efficiency_pct'].notna() & ~out['thermal_efficiency_pct'].between(0,100)
    out['reported_thermal_efficiency_pct'] = out['thermal_efficiency_pct'].where(bad_efficiency)
    out.loc[bad_efficiency, 'thermal_efficiency_pct'] = np.nan
    if mode == 'concentration':
        from .quality import update_weather_quality
        out = update_weather_quality(out)
    raw_root = raw or ROOT / 'data/raw'
    manifest_path = raw_root / 'manifest.json'
    if manifest_path.exists():
        from .operating_patterns import attach_patterns
        source_manifest = json.loads(manifest_path.read_text())
        out = attach_patterns(out, raw_root, source_manifest)
        from .weather_quality import attach_metadata
        out = attach_metadata(out, raw_root, source_manifest)
    audit = {
        'mode': mode, 'rows_before_join': len(base), 'rows_after_join': len(out),
        'observed_targets': int(out['target'].notna().sum()),
        'missing_target_rows': out.loc[out['target'].isna(), KEYS].astype(str).to_dict('records'),
        'missing_weather_rows': out.loc[out['temperature_c'].isna(), KEYS].astype(str).to_dict('records'),
        'weather_months': len(weather), 'weather_invalid_values': invalid,
        'weather_low_coverage': weather.loc[weather['weather_coverage'] < .8,
             ['plant', 'month', 'weather_coverage']].astype(str).to_dict('records'),
        'unmapped_emission_units': unmapped,
        'fuel_source': 'site monthly totals allocated by current-month generation; estimates, not measured unit fuel',
        'invalid_efficiency_rows': out.loc[bad_efficiency, KEYS+['reported_thermal_efficiency_pct']].astype(str).to_dict('records'),
    }
    if mode == 'concentration':
        audit['target_quality_status_counts'] = out['quality_status'].value_counts().to_dict()
        audit['review_flags_do_not_remove_targets'] = True
    if 'fuel_basis' in out:
        audit['fuel_source'] = 'Provider-reported unit fuel where complete and reconciled; site generation-share estimates elsewhere'
        audit['fuel_basis_counts'] = out['fuel_basis'].value_counts().to_dict()
        audit['reported_unit_fuel_plants'] = sorted(out.loc[out.fuel_basis.eq('provider_reported_unit_month'), 'plant'].unique())
    if 'gen_hour_coverage' in out:
        audit['operation_unit_months'] = int(out.gen_hour_coverage.notna().sum())
        audit['operation_features_are_electrical_generation_only'] = True
    return out, audit


def mass_from_intervals(concentration_ppm, gas_volume_sm3, *, volume_is_integrated,
                        same_interval, mg_per_sm3_per_ppm):
    """Explicit interval-volume calculation, never guess a factor from daily means.

    The caller must establish gas reference conditions and NOx equivalent first.
    Returns interval masses in kg; total mass is their sum, never their mean.
    """
    if not volume_is_integrated or not same_interval:
        raise ValueError('Matched interval concentration and integrated gas volume are required')
    c, v = np.asarray(concentration_ppm, dtype=float), np.asarray(gas_volume_sm3, dtype=float)
    if c.shape != v.shape or not np.isfinite(c).all() or not np.isfinite(v).all():
        raise ValueError('Non-finite or mismatched interval measurements')
    if (c < 0).any() or (v < 0).any() or not np.isfinite(mg_per_sm3_per_ppm) or mg_per_sm3_per_ppm <= 0:
        raise ValueError('Invalid measurement or conversion coefficient')
    return c * v * mg_per_sm3_per_ppm / 1e6
