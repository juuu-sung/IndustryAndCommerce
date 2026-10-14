"""HTTP integration: real ML/retrieval plus explicitly synthetic review decisions."""
import json
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))

from werkzeug.serving import make_server
from nox.app import create_app
from nox.predict import Predictor
from nox.copilot.registry import CaseRegistry
from nox.copilot.repositories import ReportService,ReportRepository
from nox.copilot.rag.retrieval import RetrievalService
from nox.copilot.rag.domain import DomainRetrieval
from nox.copilot.evidence.reports import EvidenceReportService,EvidenceRepository
from nox.copilot.evidence.render import render_html
from nox.copilot.agent.common import Reviewer
from nox.copilot.agent.service import AgentService
from test_copilot_registry import bundle,CASE,KEY


def main(directory=None):
    directory=Path(directory) if directory is not None else ROOT/'data/runtime/copilot'/('integration_validation_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=True)
    # Fixture values are synthetic and contain no real plant observations or approvals.
    source=directory/'synthetic_source'
    source.mkdir(exist_ok=True)
    artifact=bundle.__wrapped__(source)
    registry=CaseRegistry(directory/'synthetic_cases')
    registry.register(artifact,**CASE)
    base_reports=ReportService(registry,ReportRepository(directory/'base_reports'))
    baseline=RetrievalService()
    domain=DomainRetrieval(baseline)
    evidence=EvidenceReportService(base_reports,domain,EvidenceRepository(directory/'evidence_reports'))
    token='synthetic-http-review-token-123456789'
    agent=AgentService(evidence,directory=directory/'agent',reviewer=Reviewer(token,identity='synthetic_http_reviewer'))
    app=create_app(Predictor(history_db=False),copilot_service=base_reports,rag_service=baseline,
                   evidence_service=evidence,agent_service=agent)
    server=make_server('127.0.0.1',0,app)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    origin=f'http://127.0.0.1:{server.server_port}'
    checks=[]
    def call(path,payload=None,*,credential=None,expected=200,raw=False):
        data=json.dumps(payload).encode() if payload is not None else None
        headers={'Content-Type':'application/json'}
        if credential is not None:headers['X-Review-Token']=credential
        req=urllib.request.Request(origin+path,data=data,headers=headers)
        try:
            response=urllib.request.urlopen(req,timeout=30)
        except urllib.error.HTTPError as error:response=error
        with response:
            body=response.read()
            assert response.code==expected,(path,response.code)
            checks.append({'path':path,'status':response.code})
            return body if raw else json.loads(body)
    def wait(identifier,status):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            run=call('/api/agent/runs/'+identifier)
            if run['status']==status:return run
            if run['status'] in ('failed','cancelled'):raise AssertionError(run['error_code'])
            time.sleep(.2)
        raise TimeoutError('Synthetic workflow did not reach the expected state')
    try:
        assert call('/api/health')['status']=='ok'
        assert call('/api/agent/status')['engine']=='langgraph'
        query=call('/api/rag/search/domain',{'query':'SCR 촉매 비활성과 반응 온도 점검 근거','method':'keyword'})
        assert query['results'] and all(h['scope']=='general_technical' for h in query['results'])
        assert query['results'][0]['source_sha256']
        prediction=call('/api/predict',json.loads((ROOT/'examples/request.json').read_text()))
        assert prediction['prediction']['unit']=='ppm'
        draft=call('/api/reports/evidence-drafts',{'case_key':KEY},expected=201)['report']
        assert draft['facts'][0]['value']==12 and draft['citations']
        assert call('/api/reports/evidence/'+draft['report_id'])['context_hash']==draft['context_hash']
        assert b'ppm' in call('/api/reports/evidence/'+draft['report_id']+'/html',raw=True)
        request={'request':'합성 자료의 관측과 저장 예측을 비교해 NOx 보고서를 작성해 주세요.',
                 'case_key':KEY,'task_type':'case_report','idempotency_key':'synthetic-http-report'}
        created=call('/api/agent/runs',request,expected=202)['run']
        run=wait(created['run_id'],'waiting_review')
        assert len(run['tool_calls'])==5
        path='/api/agent/runs/'+run['run_id']
        assert call(path+'/report')['workflow']['status']=='waiting_review'
        assert b'ppm' in call(path+'/report/html',raw=True)
        approve={'decision':'approve','expected_state_version':run['state_version'],
                 'expected_report_id':run['report_id'],'expected_revision':run['report_revision'],
                 'expected_context_hash':run['context_hash']}
        assert call(path+'/resume',approve,expected=401)['error']['code']=='review_authentication_required'
        call(path+'/resume',approve,credential=token,expected=202)
        done=wait(run['run_id'],'succeeded')
        assert done['report_workflow_status']=='approved' and len(done['tool_calls'])==5
        final,view=agent.report(run['run_id'])
        html=render_html(final,view).replace('<h1>','<p class="warning"><strong>합성 자료의 HTTP 검증 예시입니다. 실제 발전소 분석이나 사용자 승인이 아닙니다.</strong></p><h1>',1)
        (ROOT.parent/'docs/generated/copilot_workflow_demo.html').write_text(html)
        summary={'all_passed':True,'checks':checks,'request_count':len(checks),
                 'real_ml_api_used':True,'real_document_retrieval_used':True,
                 'approval_case':'synthetic_only','real_plant_report_approved':False,
                 'rule_tool_calls':len(done['tool_calls']),'review_status':done['report_workflow_status'],
                 'gemini_actual_call_verified':False,'facts_unchanged':final['facts'][0]['value']==12,
                 'source_and_revision_bound':True,'no_tool_reexecution_after_review':len(done['tool_calls'])==len(run['tool_calls'])}
        (directory/'http_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=3)
        agent.close()
        if baseline.encoder is not None:baseline.encoder.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path)
    main(parser.parse_args().directory)
