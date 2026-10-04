"""Prepare reconciled unit fuel and compare each model on its own input basis."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .data import build_dataset
from .features import build_features
from .predict import Predictor
from .schema import CONDITIONS, FUELS, KEYS, ROOT
from .train import blend, metrics, split_masks, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(baseline_raw, audit_dir, output):
    baseline_raw, audit_dir, output = map(Path, (baseline_raw, audit_dir, output))
    if output.exists():
        raise ValueError('Refusing to overwrite an existing observation bundle')
    source = audit_dir/'results/unit_fuel_202301_202608.csv'
    delivery = json.loads((audit_dir/'delivery_manifest.json').read_text())
    items = {i['file']:i for i in delivery['files']}
    if digest(source) != items['results/unit_fuel_202301_202608.csv']['sha256']:
        raise ValueError('Audited unit fuel checksum mismatch')
    unit = pd.read_csv(source)
    if len(unit) != 88 or set(unit['사업소']) != {'영동'} or set(unit['호기']) != {1,2}:
        raise ValueError('Expected 44 months of audited Yeongdong units 1 and 2')
    manifest = json.loads((baseline_raw/'manifest.json').read_text())
    for item in manifest['files']:
        if digest(baseline_raw/item['file']) != item['sha256']:
            raise ValueError('Baseline observation bundle checksum mismatch')
    originals = []
    for name in sorted(unit.source_file.unique()):
        path = audit_dir/'sources'/name
        provenance = json.loads(Path(str(path)+'.provenance.json').read_text())
        if digest(path) != provenance['sha256']:
            raise ValueError('Provider unit export checksum mismatch')
        originals.append({'file':'sources/'+name, 'sha256':digest(path), 'provenance':provenance})
    output.mkdir(parents=True)
    for item in manifest['files']:
        shutil.copy2(baseline_raw/item['file'], output/item['file'])
    shutil.copy2(source, output/'fuel_unit.csv')
    (output/'sources').mkdir()
    for item in originals:
        name = Path(item['file']).name
        shutil.copy2(audit_dir/'sources'/name, output/'sources'/name)
        shutil.copy2(audit_dir/'sources'/(name+'.provenance.json'), output/'sources'/(name+'.provenance.json'))
    manifest['created_at'] = datetime.now(timezone.utc).isoformat()
    manifest['parent_manifest_sha256'] = digest(baseline_raw/'manifest.json')
    manifest['files'].append({'file':'fuel_unit.csv', 'sha256':digest(output/'fuel_unit.csv'), 'rows':88})
    manifest['unit_fuel_exports'] = originals
    manifest['fuel_input_basis'] = {
        'version':1, 'reported_unit_plants':['영동'],
        'other_plants':'site_generation_share_estimate',
        'reported_months':['2023-01','2026-08'],
        'validation':'complete unit coverage and equality to site totals for every fuel',
        'publication_latency':'unverified; newly retrieved historical data used for retrospective research',
    }
    write_json(output/'manifest.json', manifest)
    try:
        frame, audit = build_dataset('concentration', output)
        old, _ = build_dataset('concentration', baseline_raw)
        pd.testing.assert_frame_equal(old[KEYS+['target']], frame[KEYS+['target']])
    except Exception:
        shutil.rmtree(output)
        raise
    write_json(output/'unit_fuel_preparation.json', {'rows':len(frame), 'unit_fuel_rows':88,
        'unit_months':44, 'target_unchanged':True, 'dataset_audit':audit,
        'source_integrity_verified':True})
    return {'output':str(output), 'rows':len(frame), 'unit_fuel_rows':88, 'target_unchanged':True}


def compare(candidate_dir, baseline_dir, candidate_raw, baseline_raw, output):
    predictors = {name:Predictor(path, history_db=False)
                  for name,path in [('candidate',candidate_dir), ('baseline',baseline_dir)]}
    candidate, baseline = predictors['candidate'], predictors['baseline']
    if candidate.metadata['target'] != baseline.metadata['target'] or candidate.metadata['split'] != baseline.metadata['split']:
        raise ValueError('Comparison requires identical targets and time splits')
    frames = {name:build_dataset('concentration', Path(path))[0]
              for name,path in [('candidate',candidate_raw),('baseline',baseline_raw)]}
    pd.testing.assert_frame_equal(frames['candidate'][KEYS+['target']], frames['baseline'][KEYS+['target']])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {'protocol':candidate.metadata['split'], 'target':candidate.metadata['target'],
        'input_comparison':'Each model receives its own training fuel basis, including previous-month fuel history',
        'target_unchanged':True, 'weights_selected_on_validation_only':True,
        'previous_test_results_already_seen':True,
        'evaluation_limit':'Same frozen test used as a follow-up diagnostic; this is not a new untouched holdout',
        'default_model_replaced':False, 'scores':{}, 'by_plant':{}, 'relative_error_reduction_pct':{}}
    predictions = {}
    for split in ('validation','test'):
        tables = {}
        for name, predictor in predictors.items():
            frame = frames[name]
            mask = split_masks(frame, candidate.metadata['split'])[split]
            table = frame.loc[mask, KEYS+['target']].copy()
            X = predictor.pre.transform(build_features(frame).loc[mask])
            table['pred_'+name] = blend(predictor.xgb.predict(X), predictor.lgb.predict(X,num_threads=1), predictor.weight)
            saved = pd.read_csv(Path(candidate_dir if name=='candidate' else baseline_dir)/(split+'_predictions.csv'), parse_dates=['month'])
            check = table.merge(saved[KEYS+['pred_ensemble']], on=KEYS, validate='one_to_one')
            if len(check) != len(table):
                raise ValueError('Saved prediction key coverage mismatch')
            np.testing.assert_allclose(check['pred_'+name], check.pred_ensemble, rtol=1e-7, atol=1e-8)
            tables[name] = table
        joined = tables['baseline'].merge(tables['candidate'], on=KEYS, suffixes=('_baseline','_candidate'), validate='one_to_one')
        np.testing.assert_allclose(joined.target_baseline, joined.target_candidate, rtol=0, atol=0)
        joined = joined.rename(columns={'target_baseline':'target'}).drop(columns='target_candidate')
        joined['abs_error_baseline'] = abs(joined.target-joined.pred_baseline)
        joined['abs_error_candidate'] = abs(joined.target-joined.pred_candidate)
        joined.to_csv(output/(split+'_comparison.csv'), index=False, date_format='%Y-%m-%d')
        predictions[split] = joined
        report['scores'][split] = {name:metrics(joined.target, joined['pred_'+name]) for name in predictors}
        report['by_plant'][split] = {plant:{name:metrics(group.target, group['pred_'+name]) for name in predictors}
                                      for plant,group in joined.groupby('plant')}
        report['relative_error_reduction_pct'][split] = {
            key:100*(report['scores'][split]['baseline'][key]-report['scores'][split]['candidate'][key])/report['scores'][split]['baseline'][key]
            for key in ('rmse','mae')}
    audit = json.loads((Path(candidate_dir)/'data_audit.json').read_text())
    report['fuel_basis_counts'] = audit['fuel_basis_counts']
    test = predictions['test']
    report['largest_error_rows'] = test.sort_values('abs_error_candidate',ascending=False).head(10).assign(
        month=lambda d:d.month.dt.strftime('%Y-%m')).to_dict('records')
    same = frames['baseline'][KEYS+list(FUELS.values())].merge(frames['candidate'][KEYS+list(FUELS.values())],
              on=KEYS, suffixes=('_baseline','_candidate'), validate='one_to_one')
    changed = np.zeros(len(same), dtype=bool)
    for col in FUELS.values():
        changed |= ~np.isclose(same[col+'_baseline'],same[col+'_candidate'],equal_nan=True)
    report['changed_fuel_rows'] = int(changed.sum())
    same.loc[changed].to_csv(output/'changed_fuel_rows.csv', index=False)
    write_json(output/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--baseline-raw',type=Path,default=ROOT/'data/extended_20261001')
    p.add_argument('--audit-dir',type=Path,default=ROOT.parent/'NOx_yeongdong_audit_20261001')
    p.add_argument('--output',type=Path,required=True)
    p = commands.add_parser('compare')
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/concentration_20261001')
    p.add_argument('--candidate-raw',type=Path,required=True)
    p.add_argument('--baseline-raw',type=Path,default=ROOT/'data/extended_20261001')
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        result = prepare(args.baseline_raw,args.audit_dir,args.output)
    else:
        result = compare(args.candidate,args.baseline,args.candidate_raw,args.baseline_raw,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
