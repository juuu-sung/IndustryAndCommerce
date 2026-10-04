"""Download public KOEN exports and build a new, traceable observation bundle.

These are the CSV buttons on the provider's public pages, not authenticated APIs.
Original files and frozen model artifacts are never replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import requests

from .history import month_key, period_end, utc_now
from .schema import ROOT

SITES = {'분당','삼천포','여수','영동','영흥'}
DOWNLOADS = {
    'generation': ('nfdt01','|','strDateS','strDateE'),
    'fuel': ('nfdt04','|','strMonthS','strMonthE'),
    'emissions': ('nfdt16',',','strDateS','strDateE'),
    'weather': ('nfdt18',',','strDateS','strDateE'),
}
BD_UNITS = {f'CG{n}':n for n in range(1,9)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_export(content, kind):
    frame = pd.read_csv(io.BytesIO(content), sep=DOWNLOADS[kind][1],
                        encoding='utf-8-sig', skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for col in frame.select_dtypes('object'):
        frame[col] = frame[col].str.strip()
    required = {'사업소','호기','일자'} | ({'NOX'} if kind=='emissions' else
        {'온도','습도','풍향','풍속'} if kind=='weather' else
        {'용량(MW)','발전량(MWh)','열효율(%)','이용률(%)'} if kind=='generation' else
        {'유연탄','무연탄','계(석탄)','유류','LNG','고형연료','우드펠릿'})
    if required - set(frame) or frame.empty:
        raise ValueError(f'{kind}: empty export or provider schema changed')
    return frame


def normalize_generation(frame):
    frame = frame.loc[frame['사업소'].isin(SITES)].copy()
    steam = (frame['사업소']=='분당') & frame['호기'].isin(['CS1','CS2'])
    excluded = frame.loc[steam,['사업소','호기','일자']].to_dict('records')
    frame = frame.loc[~steam]
    units = []
    for plant, unit in frame[['사업소','호기']].itertuples(index=False, name=None):
        text = str(unit)
        if plant=='분당' and text in BD_UNITS:
            units.append(BD_UNITS[text])
        elif plant!='분당' and text.isdecimal() and int(text)>0:
            units.append(int(text))
        else:
            raise ValueError(f'Unverified generator mapping: {plant}/{unit}')
    frame['호기'] = units
    frame = frame.rename(columns={'일자':'월'})
    columns = ['사업소','호기','월','용량(MW)','발전량(MWh)','열효율(%)','이용률(%)']
    if '발전원' in frame:
        columns.append('발전원')
    return frame[columns], excluded


def overlap_review(downloads, baseline):
    """Reject changed historical generator mappings rather than guessing codes."""
    generation, _ = normalize_generation(parse_export((downloads/'generation_overlap_2024.csv').read_bytes(),'generation'))
    old = pd.read_csv(baseline/'generation.csv')
    old = old.loc[old['월'].between(202401,202412)]
    joined = old.merge(generation,on=['사업소','호기','월'],suffixes=('_old','_new'),validate='one_to_one')
    cols = ['용량(MW)','발전량(MWh)','열효율(%)','이용률(%)']
    same = len(joined)==len(old)==len(generation) and all(
        np.allclose(joined[c+'_old'],joined[c+'_new'],rtol=0,atol=1e-6) for c in cols)
    if not same:
        raise ValueError('Historical generator mapping/values changed; manual review required')
    fuel = parse_export((downloads/'fuel_overlap_2024.csv').read_bytes(),'fuel')
    old_fuel = pd.read_csv(baseline/'fuel_site.csv')
    old_fuel = old_fuel.loc[old_fuel['일자'].between(202401,202412)]
    joined_fuel = old_fuel.merge(fuel,on=['사업소','일자'],suffixes=('_old','_new'),validate='one_to_one')
    fuel_cols = ['유연탄','무연탄','유류','LNG','고형연료','우드펠릿']
    if len(joined_fuel)!=len(old_fuel) or not all(np.allclose(
            joined_fuel[c+'_old'],joined_fuel[c+'_new'],rtol=0,atol=1e-6) for c in fuel_cols):
        raise ValueError('Historical site fuel values changed; manual review required')
    return {'generation_matched_rows':len(joined),'generation_fields':cols,
            'fuel_matched_rows':len(joined_fuel),'fuel_fields':fuel_cols,
            'bundang_mapping':BD_UNITS, 'steam_generators_excluded':['CS1','CS2'],
            'scope':'2024 values exactly match the existing numeric-unit dataset; physical stack codebook still unverified'}


def collect(output, downloads, *, start_month='2025-01', end_month='2026-08', reuse_downloads=False,
            baseline=ROOT/'data/raw'):
    output, downloads, baseline = map(Path,(output,downloads,baseline))
    start, end = month_key(start_month), month_key(end_month)
    if start.year < 2025 or start > end or period_end(end)>pd.Timestamp(utc_now()):
        raise ValueError('Use complete observation months from 2025 onward')
    if output.exists():
        raise ValueError('Output already exists; choose a new bundle directory')
    downloads.mkdir(parents=True,exist_ok=True)
    baseline_manifest = json.loads((baseline/'manifest.json').read_text())
    for item in baseline_manifest['files']:
        if digest(baseline/item['file']) != item['sha256']:
            raise ValueError('Baseline source changed; restore or explicitly review it')
    provenance = []

    def get(kind,label,first,last):
        page,_,lower,upper = DOWNLOADS[kind]
        params = {'pageIndex':'1','strOrgNo':'','strHokiS':'','strHokiE':'',lower:first,upper:last}
        url = f'https://www.koenergy.kr/kosep/gv/nf/dt/{page}/csvDown.do'
        path = downloads/(label+'.csv')
        if path.exists():
            if not reuse_downloads:
                raise ValueError(f'Export exists: {path}; use --reuse-downloads or another directory')
            content = path.read_bytes()
        else:
            response = requests.post(url,data=params,timeout=45)
            response.raise_for_status()
            content = response.content
            parse_export(content,kind)  # Never save an HTML/error response as data.
            path.write_bytes(content)
        frame = parse_export(content,kind)
        dates = frame['일자'].astype(str)
        parsed = pd.to_datetime(dates,format='%Y%m%d' if kind in ('weather','emissions') else '%Y%m')
        lower_date = pd.to_datetime(first,format='%Y%m%d' if len(first)==8 else '%Y%m')
        upper_date = pd.to_datetime(last,format='%Y%m%d' if len(last)==8 else '%Y%m')
        if not parsed.between(lower_date,upper_date).all():
            raise ValueError(f'{label}: provider/cache returned dates outside the requested period')
        provenance.append({'file':path.name,'url':url,'method':'POST','parameters':params,
            'sha256':digest(path),'bytes':len(content),'rows':len(frame),
            'recorded_file_time':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
            'publication_time':'unknown; file time is retrieval, not provider publication'})
        print(f'{label}: {len(frame)} rows',flush=True)
        return frame

    for kind in ('generation','fuel'):
        get(kind,kind+'_overlap_2024','202401','202412')
    review = overlap_review(downloads,baseline)
    pieces = {name:[] for name in DOWNLOADS}
    for year in range(start.year,end.year+1):
        first = max(start,pd.Timestamp(year,1,1)); last = min(end,pd.Timestamp(year,12,1))
        for kind in DOWNLOADS:
            lo,hi = first.strftime('%Y%m'),last.strftime('%Y%m')
            if kind in ('weather','emissions'):
                lo,hi = first.strftime('%Y%m%d'),(last+pd.offsets.MonthEnd(0)).strftime('%Y%m%d')
            pieces[kind].append(get(kind,f'{kind}_{year}',lo,hi))
    frames = {name:pd.concat(parts,ignore_index=True) for name,parts in pieces.items()}
    generation, excluded = normalize_generation(frames['generation'])
    frames['generation'] = generation
    frames['weather']['일자'] = pd.to_datetime(frames['weather']['일자'].astype(str),format='%Y%m%d').dt.strftime('%Y-%m-%d')
    names = {'generation':'generation.csv','fuel':'fuel_site.csv',
             'weather':'weather_daily.csv','emissions':'emissions_daily.csv'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='bundle_pending_') as temp:
        pending = Path(temp)/'bundle'; pending.mkdir()
        files = []
        for kind,name in names.items():
            old = pd.read_csv(baseline/name,encoding='utf-8-sig')
            new = frames[kind]
            # Same columns as the baseline; new provider statuses are not guessed.
            combined = pd.concat([old,new[old.columns]],ignore_index=True)
            combined.to_csv(pending/name,index=False)
            files.append({'file':name,'sha256':digest(pending/name),'old_rows':len(old),'new_rows':len(new)})
        manifest = {'schema_version':1,'created_at':utc_now(),'requested_months':[start_month,end_month],
            'bootstrap_observation_month':'2024-12',
            'files':files,'official_exports':provenance,'overlap_review':review,
            'baseline_manifest_sha256':digest(baseline/'manifest.json'),
            'excluded_steam_generators':excluded,
            'limitations':['Site fuel allocation remains an estimate, not measured unit fuel.',
                           'Missing provider weather stays missing.',
                           'NOx zero/validity meanings and physical stack mapping require provider confirmation.']}
        (pending/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
        from .data import build_dataset
        dataset,audit = build_dataset('concentration',pending)
        (pending/'collection_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False))
        if dataset.month.max() < end:
            raise ValueError('Requested latest generation month was not delivered')
        pending.rename(output)
    return {'output':str(output),'month_range':[str(dataset.month.min().date()),str(dataset.month.max().date())],
            'rows':len(dataset),'observed_targets':int(dataset.target.notna().sum()),'overlap_review':review}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-month',default='2025-01')
    parser.add_argument('--end-month',default='2026-08')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--downloads',type=Path,required=True)
    parser.add_argument('--reuse-downloads',action='store_true')
    args = parser.parse_args()
    print(json.dumps(collect(args.output,args.downloads,start_month=args.start_month,
        end_month=args.end_month,reuse_downloads=args.reuse_downloads),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
