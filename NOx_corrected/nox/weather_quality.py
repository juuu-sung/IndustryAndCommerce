"""Weather recovery with cell provenance; never replace an existing observation."""
import json

import numpy as np
import pandas as pd

FIELDS = {'온도': ('temperature', -80, 60), '습도': ('humidity', 0, 100),
          '풍속': ('wind_speed', 0, 100), '풍향': ('wind_direction', 0, 360)}
QUALITY_FEATURES = [f'weather_{name}_{suffix}' for name, _, _ in FIELDS.values()
                    for suffix in ('coverage', 'missing')]
POLICY = {'version': 1, 'source': 'KOEN plant weather CSV; external stations not substituted',
          'features': QUALITY_FEATURES, 'coverage': 'valid days / calendar days; station means per day',
          'publication_time': 'unverified; retrieval is not provider publication',
          'recovery': 'fill missing cells or missing plant/date only; keep existing finite cells'}
KEYS = ['사업소', '호기', '일자']


def normalize(frame):
    required = {*KEYS, *FIELDS}
    if required - set(frame):
        raise ValueError('Missing weather source columns')
    frame = frame.copy()
    frame['사업소'] = frame['사업소'].astype(str).str.strip().replace({'분당화력': '분당'})
    frame['호기'] = frame['호기'].fillna('-').astype(str).str.strip()
    if frame['사업소'].eq('').any() or frame['호기'].eq('').any():
        raise ValueError('Empty weather plant/station key')
    text = frame['일자'].astype(str).str.replace('-', '', regex=False)
    frame['일자'] = pd.to_datetime(text, format='%Y%m%d', errors='raise').dt.strftime('%Y-%m-%d')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate weather plant/station/date')
    for col in FIELDS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame


def recover_weather(original, exports):
    """exports: ordered (source_ref, frame). Conflicts are recorded, not overwritten."""
    combined = normalize(original).set_index(KEYS)
    sources = {(key, col): 'baseline_weather' for key, row in combined.iterrows()
               for col in FIELDS if pd.notna(row[col])}
    conflicts, skipped, recovered = [], [], []
    for source_ref, supplied in exports:
        fresh = normalize(supplied).set_index(KEYS)
        existing_days = {(p, d) for p, _, d in combined.index}
        for key, row in fresh.iterrows():
            if key not in combined.index:
                if (key[0], key[2]) in existing_days:
                    skipped.append({'plant': key[0], 'station': key[1], 'date': key[2],
                                    'source_ref': source_ref, 'reason': 'new station on an existing day; mapping unverified'})
                    continue
                values = {col: row.get(col, np.nan) for col in combined.columns}
                combined.loc[key, :] = [values[c] for c in combined.columns]
                existing_days.add((key[0], key[2]))
                for col in FIELDS:
                    if pd.notna(row[col]):
                        sources[key, col] = source_ref
                        recovered.append({'plant': key[0], 'station': key[1], 'date': key[2], 'field': col,
                                          'kind': 'missing_day', 'source_ref': source_ref})
                continue
            for col, (_, low, high) in FIELDS.items():
                old, new = combined.at[key, col], row[col]
                if pd.isna(old) and pd.notna(new) and low <= new <= high:
                    combined.at[key, col] = new
                    sources[key, col] = source_ref
                    recovered.append({'plant': key[0], 'station': key[1], 'date': key[2], 'field': col,
                                      'kind': 'missing_cell', 'source_ref': source_ref})
                elif pd.notna(old) and pd.notna(new) and not np.isclose(old, new, rtol=0, atol=1e-9):
                    conflicts.append({'plant': key[0], 'station': key[1], 'date': key[2], 'field': col,
                                      'baseline_value': float(old), 'retrieved_value': float(new),
                                      'source_ref': source_ref})
    records = [{'plant': key[0], 'station': key[1], 'date': key[2], 'field': col,
                'source_ref': sources.get((key, col), 'missing')}
               for key in combined.index for col in FIELDS]
    return combined.reset_index(), pd.DataFrame(records), {
        'recovered_cells': recovered, 'preserved_conflicts': conflicts, 'unmapped_stations': skipped,
        'existing_observations_replaced': 0}


def monthly_metadata(weather, generation, cell_sources):
    weather = normalize(weather)
    weather['date'] = pd.to_datetime(weather['일자'])
    for col, (_, low, high) in FIELDS.items():
        weather.loc[~weather[col].between(low, high), col] = np.nan
    daily = weather.groupby(['사업소', 'date'], as_index=False)[list(FIELDS)].mean()
    daily['month'] = daily.date.dt.to_period('M').dt.to_timestamp()
    valid = daily.groupby(['사업소', 'month'])[list(FIELDS)].count()
    observations = daily.groupby(['사업소', 'month']).date.nunique()
    sources = cell_sources.copy()
    sources['month'] = pd.to_datetime(sources.date).dt.to_period('M').dt.to_timestamp()
    source_lists = sources.groupby(['plant', 'month']).source_ref.agg(lambda x: sorted(set(x) - {'missing'}))
    rows = []
    for plant, month in generation[['plant', 'month']].drop_duplicates().itertuples(index=False, name=None):
        month = pd.Timestamp(month)
        key, expected = (plant, month), month.days_in_month
        counts = valid.loc[key] if key in valid.index else pd.Series(0, index=list(FIELDS))
        record = {'plant': plant, 'month': month, 'expected_days': expected,
                  'reported_days': int(observations.get(key, 0)),
                  'source_refs': json.dumps(source_lists.get(key, []), ensure_ascii=False),
                  'source_kind': 'koen_plant_weather' if key in observations.index else 'unavailable',
                  'publication_time_status': 'unverified'}
        for col, (name, _, _) in FIELDS.items():
            record[f'{name}_valid_days'] = int(counts[col])
            record[f'weather_{name}_coverage'] = float(counts[col] / expected)
            record[f'weather_{name}_missing'] = int(counts[col] == 0)
        rows.append(record)
    return pd.DataFrame(rows).sort_values(['plant', 'month']).reset_index(drop=True)


def attach_metadata(frame, raw_dir, manifest):
    if 'weather_quality_policy' not in manifest:
        return frame
    if manifest['weather_quality_policy'] != POLICY:
        raise ValueError('Unknown weather quality policy')
    metadata = pd.read_csv(raw_dir / 'weather_monthly_metadata.csv', parse_dates=['month'])
    if metadata.duplicated(['plant', 'month']).any() or set(QUALITY_FEATURES) - set(metadata):
        raise ValueError('Invalid weather metadata schema/keys')
    expected = frame[['plant', 'month']].drop_duplicates()
    if len(metadata) != len(expected) or not pd.MultiIndex.from_frame(expected).isin(
            pd.MultiIndex.from_frame(metadata[['plant', 'month']])).all():
        raise ValueError('Weather metadata must cover every generator plant/month exactly')
    if not metadata.expected_days.eq(metadata.month.dt.days_in_month).all():
        raise ValueError('Incorrect weather calendar-day counts')
    for name, _, _ in FIELDS.values():
        coverage, missing = metadata[f'weather_{name}_coverage'], metadata[f'weather_{name}_missing']
        days = metadata[f'{name}_valid_days']
        if not coverage.between(0, 1).all() or not missing.isin([0, 1]).all() or not missing.eq(coverage.eq(0)).all():
            raise ValueError('Invalid weather coverage/missing flags')
        if not days.eq(days.round()).all() or not np.allclose(coverage, days / metadata.expected_days):
            raise ValueError('Inconsistent weather valid-day counts')
    return frame.merge(metadata[['plant', 'month', *QUALITY_FEATURES, 'source_kind', 'source_refs']],
                       on=['plant', 'month'], how='left', validate='many_to_one')
