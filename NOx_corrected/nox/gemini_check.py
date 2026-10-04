"""Credential-safe, one-request Gemini smoke check with explicit live execution."""
import argparse
import json
import os
from pathlib import Path
import time

from .llm import Gemini, explain, validate_analysis
from .predict import Predictor
from .schema import ROOT
from .train import write_json


def check(*, live=False, artifact=None, request=None, client=None):
    configured = bool(os.environ.get('GEMINI_API_KEY')) and bool(os.environ.get('GEMINI_MODEL'))
    base = {'live_requested': live, 'credentials_configured': configured,
            'secrets_in_output': False, 'actual_external_call_verified': False}
    if not live:
        return {**base, 'status': 'not_run', 'reason': 'use --live for one external request'}
    if client is None and not configured:
        return {**base, 'status': 'not_run', 'reason': 'GEMINI_API_KEY and GEMINI_MODEL are required'}
    artifact = Path(artifact or ROOT / 'artifacts/concentration_20261001')
    request = Path(request or ROOT / 'examples/request_latest.json')
    data = json.loads(request.read_text())
    predictor = Predictor(artifact, history_db=False)
    prediction = predictor.predict(data)
    original = dict(prediction['prediction'])
    started = time.perf_counter()
    result = explain(data, prediction, enabled=True, client=client or Gemini())
    elapsed = time.perf_counter() - started
    unchanged = original == prediction['prediction']
    valid = result['status'] == 'gemini'
    if valid:
        validate_analysis(result['content'])
    return {**base, 'status': 'passed' if valid and unchanged else 'failed_or_fallback',
            'analysis_status': result['status'], 'prediction_unchanged': unchanged,
            'analysis_schema_valid': valid, 'elapsed_seconds': elapsed,
            'backend_limitations_preserved': set(prediction['warnings']).issubset(result['content']['limitations']),
            'actual_external_call_verified': client is None and valid,
            'mock_client_used': client is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError('Refusing to overwrite Gemini check evidence')
    result = check(live=args.live, artifact=args.artifact, request=args.request)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
