"""Run one fixed-protocol candidate with a wall-clock training limit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .schema import ROOT
from .train import write_json
from .unit_fuel import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/concentration_unit_fuel_20261001')
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/unit_fuel_20261001')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/unit_fuel_retraining')
    parser.add_argument('--max-training-seconds',type=float,default=300.)
    args = parser.parse_args()
    if args.max_training_seconds<=0:
        parser.error('--max-training-seconds must be positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new --output and --report-dir')
    args.report_dir.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-m','nox.train','--target','concentration','--raw-dir',str(args.raw_dir),
        '--output',str(args.output),'--train-start','2023-01','--train-end','2025-06',
        '--validation-start','2025-07','--validation-end','2025-12','--test-start','2026-01','--test-end','2026-08']
    started = time.perf_counter()
    try:
        with (args.report_dir/'training_console.txt').open('w') as log:
            process = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=args.max_training_seconds,check=False)
    except subprocess.TimeoutExpired:
        incomplete = str(args.output)+'_incomplete_'+str(int(time.time()))
        if args.output.exists(): args.output.rename(incomplete)
        write_json(args.report_dir/'timing.json',{'training_seconds':time.perf_counter()-started,
            'status':'timeout','limit_seconds':args.max_training_seconds,'incomplete_output':incomplete})
        parser.exit(2,'Training exceeded the time limit. Partial output was preserved; run again with a larger limit.\n')
    seconds = time.perf_counter()-started
    write_json(args.report_dir/'timing.json',{'training_seconds':seconds,'status':'success' if process.returncode==0 else 'failed',
        'limit_seconds':args.max_training_seconds,'returncode':process.returncode,'command':command})
    if process.returncode:
        parser.exit(process.returncode,'Training failed; inspect training_console.txt\n')
    report = compare(args.output,ROOT/'artifacts/concentration_20261001',args.raw_dir,
                     ROOT/'data/extended_20261001',args.report_dir)
    print(json.dumps({'training_seconds':seconds,'output':str(args.output),
        'scores':report['scores'],'relative_error_reduction_pct':report['relative_error_reduction_pct']},
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
