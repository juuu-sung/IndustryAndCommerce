import hashlib
import json

import numpy as np
import pandas as pd

from .data import read, normalize_keys, generation, allocate_fuel, monthly_weather, build_dataset
from .schema import FUELS, ROOT
from .train import write_json


def audit():
    legacy = read('legacy_model.csv')
    fuel = read('legacy_fuel.csv')
    weather = read('legacy_weather.csv')
    weather['월'] = weather['월'].str.replace('-', '', regex=False).astype(int)
    old = legacy.merge(fuel, on=['사업소', '호기', '월'], validate='one_to_one').merge(
        weather.drop(columns=['호기']), on=['사업소', '월'], validate='many_to_one')
    q1, q3 = old['NOX_kg'].quantile([.25, .75])
    filtered = old.loc[(old['NOX_kg'] > 5) & old['NOX_kg'].between(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))].copy()
    filtered.sort_values(['사업소', '호기', '월'], inplace=True)
    grouped = filtered.groupby(['사업소', '호기'])['NOX_kg']
    bad = grouped.apply(lambda s: s.shift().rolling(3, min_periods=1).mean()).reset_index(drop=True)
    filtered['bad_roll'] = bad
    correct = grouped.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    mismatch = ~np.isclose(filtered['bad_roll'], correct, equal_nan=True)
    gap = filtered.groupby(['사업소', '호기'])['월'].transform(
        lambda s: pd.to_datetime(s.astype(str), format='%Y%m').dt.year * 12 +
                  pd.to_datetime(s.astype(str), format='%Y%m').dt.month).groupby(
                      [filtered['사업소'], filtered['호기']]).diff()
    new, new_audit = build_dataset('legacy')
    gen = generation()
    allocated = allocate_fuel(gen)
    old_fuel = normalize_keys(fuel)
    old_fuel = old_fuel.rename(columns={f'{c}_호기별': v for c, v in FUELS.items()})
    compared = allocated.merge(old_fuel, on=['plant','unit','month'], suffixes=('_new','_old'), validate='one_to_one')
    allocation_diff = {col: float((compared[f'{col}_new'] - compared[f'{col}_old']).abs().max())
                       for col in FUELS.values()}
    # An API CSV is not automatically a drop-in replacement: compare site totals.
    api = read('fuel_api.csv')
    api['plant'] = api['orgNm'].replace({'분당화력': '분당'})
    api['month'] = api['ym']
    mapping = {'유연탄':'유연탄','무연탄':'무연탄','중유':'유류','B-C유':'유류','LNG':'LNG','우드펠릿':'우드펠릿'}
    api['fuel'] = api['fuelName'].map(mapping)
    totals = api.pivot_table(index=['plant','month'], columns='fuel', values='fuelVolume', aggfunc='sum', fill_value=0).reset_index()
    site = read('fuel_site.csv').rename(columns={'사업소':'plant','일자':'month'})
    site['plant'] = site['plant'].replace({'분당화력':'분당'})
    joined = site.merge(totals, on=['plant','month'], suffixes=('_site','_api'), validate='one_to_one')
    api_diff = {}
    for col in mapping.values():
        if f'{col}_api' not in joined: continue
        a, b = joined[f'{col}_site'], joined[f'{col}_api']
        api_diff[col] = {'matched_site_months': len(joined), 'different_totals': int((~np.isclose(a,b,atol=.01)).sum()),
                         'max_absolute_difference': float((a-b).abs().max())}
    # A testable interpretation, explicitly not an authoritative code table.
    candidate_codes = {1:'유연탄',2:'무연탄',3:'유류',4:'LNG',5:'고형연료',6:'우드펠릿'}
    api['candidate_fuel_name'] = api['fuelCd'].map(candidate_codes)
    api['mapping_status'] = 'inferred_from_site_totals; official_code_table_required'
    if api['candidate_fuel_name'].isna().any():
        raise ValueError('Unknown API fuel code; do not guess its label')
    candidate_totals = api.pivot_table(index=['plant','month'],columns='candidate_fuel_name',
                                     values='fuelVolume',aggfunc='sum',fill_value=0).reset_index()
    candidate_comparison = site.merge(candidate_totals,on=['plant','month'],suffixes=('_site','_candidate'),validate='one_to_one')
    candidate_diff = {col:{'matched_site_months':len(candidate_comparison),
                          'different_totals':int((~np.isclose(candidate_comparison[f'{col}_site'],candidate_comparison[f'{col}_candidate'],atol=.01)).sum())}
                      for col in candidate_codes.values()}
    api.to_csv(ROOT/'reports/fuel_api_label_review.csv',index=False)
    mass = read('legacy_monthly_mass.csv')
    mass['호기'] = mass['호기'].astype(str).str.extract(r'^(\d+)')[0].astype(int)
    mass_compare = legacy.merge(mass, on=['사업소','호기','월'], suffixes=('_model','_monthly'), validate='one_to_one')
    output = {
        'original_rows': len(legacy), 'old_inner_join_rows': len(old), 'old_filtered_rows': len(filtered),
        'rolling_alignment_wrong_rows': int(mismatch.sum()),
        'rolling_checked_rows': len(filtered), 'old_lag_transitions_skipping_calendar_months': int((gap>1).sum()),
        'coal_total_equals_components_rows': int(np.isclose(fuel['계(석탄)_호기별'], fuel['유연탄_호기별']+fuel['무연탄_호기별']).sum()),
        'coal_positive_doublecount_rows': int((fuel['계(석탄)_호기별']>0).sum()),
        'reallocated_fuel_max_abs_difference_from_original': allocation_diff,
        'api_site_total_comparison': api_diff,
        'candidate_api_code_mapping': candidate_codes,
        'candidate_api_totals_comparison': candidate_diff,
        'candidate_mapping_authoritative': False,
        'api_fuel_units_not_verified': True,
        'api_replacement_status': 'not adopted: candidate code labels match all site totals, but official code table, CG/CS mapping and API units still require confirmation',
        'different_legacy_target_versions': int((~np.isclose(mass_compare['NOX_kg_model'],mass_compare['NOX_kg_monthly'],atol=.001)).sum()),
        'corrected_legacy_data': new_audit,
        'kg_conversion_status': 'blocked: daily mean concentrations and unknown flow time units do not establish monthly mass',
        'original_saved_metrics': {'xgboost_rmse_2024':192.2,'lightgbm_rmse_2024':203.1,'xgboost_r2_2024':.845,
                                  'selected_xgboost_weight':1.,'manual_gemini_weight':.6,
                                  'warning':'Saved notebook output; not directly comparable to new H2 evaluation or different target definitions'},
    }
    manifest = json.loads((ROOT/'data/raw/manifest.json').read_text())
    output['copied_source_hashes_valid'] = all(hashlib.sha256((ROOT/'data/raw'/f['file']).read_bytes()).hexdigest()==f['sha256'] for f in manifest['files'])
    write_json(ROOT/'reports/data_validation.json', output)
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return output


if __name__ == '__main__':
    audit()
