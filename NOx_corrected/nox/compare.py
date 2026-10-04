"""Evaluate two frozen bundles on identical rows and identical observed history."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import build_dataset
from .features import build_features
from .predict import Predictor
from .schema import KEYS, ROOT
from .train import blend, metrics, split_masks, write_json


def compare(candidate_dir, baseline_dir, raw_dir, output):
    candidate = Predictor(candidate_dir,history_db=False)
    baseline = Predictor(baseline_dir,history_db=False)
    frame,_ = build_dataset('concentration',Path(raw_dir))
    if candidate.metadata['target'] != baseline.metadata['target']:
        raise ValueError('Target definitions differ; comparison would be invalid')
    mask = split_masks(frame,candidate.metadata['split'])['test']
    test = frame.loc[mask].copy()
    common = [(p,u) in baseline.known_units for p,u in test[['plant','unit']].itertuples(index=False,name=None)]
    test['baseline_supported'] = common
    features = build_features(frame).loc[mask]
    for name,predictor in [('candidate',candidate),('baseline',baseline)]:
        inputs = predictor.pre.transform(features)
        test['pred_'+name] = blend(predictor.xgb.predict(inputs),
            predictor.lgb.predict(inputs,num_threads=1),predictor.weight)
    # Do not claim baseline service support for a new plant/unit.
    supported = test.loc[test['baseline_supported']]
    comparison = {name:metrics(supported.target,supported['pred_'+name]) for name in ('candidate','baseline')}
    by_plant = {p:{name:metrics(g.target,g['pred_'+name]) for name in ('candidate','baseline')}
                for p,g in supported.groupby('plant')}
    improvements = {key:100*(comparison['baseline'][key]-comparison['candidate'][key])/comparison['baseline'][key]
                    for key in ('rmse','mae')}
    report = {'protocol':candidate.metadata['split'],'target':candidate.metadata['target'],
        'common_rows':len(supported),'new_unit_rows':len(test)-len(supported),
        'common_scores':comparison,'relative_error_reduction_pct':improvements,'common_by_plant':by_plant,
        'candidate_all_rows':metrics(test.target,test.pred_candidate),
        'new_unit_keys':test.loc[~test.baseline_supported,['plant','unit']].drop_duplicates().to_dict('records'),
        'evaluation_mode':'retrospective monthly rolling origin with realized conditions and prior observed targets; publication latency unverified',
        'baseline_history':'same extended observed history for both; not a comparison with stale 2024 history',
        'model_parameters':'fixed before test; ensemble weight selected on 2025-H2 only',
        'default_model_replaced':False}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/'comparison.json',report)
    test[KEYS+['target','baseline_supported','pred_baseline','pred_candidate']].to_csv(output/'common_period_predictions.csv',index=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/concentration')
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'reports/retraining')
    args=p.parse_args()
    print(json.dumps(compare(args.candidate,args.baseline,args.raw_dir,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
