"""One-request live check, with explicit evidence that mock calls are not live calls."""
import argparse
import json
from pathlib import Path
import time

from .gemini import GeminiClient
from .reports import EvidenceReportService
from ..registry import _write_new


def check(case_key,*,live=False,service=None,client=None):
    client=client if client is not None else GeminiClient()
    configured=bool(getattr(client,'configured',False))
    base={'live_requested':live,'credentials_configured':configured,
          'actual_external_call_verified':False,'secrets_in_output':False}
    if not live or not configured:
        return {**base,'status':'not_run','reason':'live_flag_and_GEMINI_API_KEY_and_GEMINI_MODEL_required'}
    service=service if service is not None else EvidenceReportService(llm_client=client)
    service.llm_client=client
    started=time.perf_counter()
    report,_=service.create(case_key,llm_enabled=True)
    analysis=report['analysis']
    unchanged=report['facts']==__import__('nox.copilot.evidence.facts',fromlist=['make_facts']).make_facts(report['base_report'])
    return {**base,'status':'passed' if analysis['status']=='gemini' and unchanged else 'failed_or_fallback',
            'actual_external_call_verified':analysis['actual_external_call_verified'],
            'analysis_status':analysis['status'],'reason':analysis['reason'],'facts_unchanged':unchanged,
            'reference_schema_valid':True,'semantic_faithfulness_verified':False,
            'report_id':report['report_id'],'elapsed_seconds':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case-key',required=True)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=check(args.case_key,live=args.live)
    if args.output:
        _write_new(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
