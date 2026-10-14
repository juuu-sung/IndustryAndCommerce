from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import sqlite3

import pytest

from nox.app import create_app
from nox.copilot.agent.common import AgentError,Reviewer,RunRequest
from nox.copilot.agent.service import AgentService
from nox.copilot.evidence.gemini import GeminiClient
from nox.copilot.evidence.render import render_html
from test_copilot_registry import bundle,registry,CASE,KEY
from test_copilot_reports import service,StubPredictor
from test_copilot_rag import rag
from test_copilot_evidence import evidence

TOKEN='synthetic-review-token-123456789'
REQUEST={'request':'등록 사례의 NOx 관측과 저장 예측을 비교해 근거 보고서를 작성해 주세요.',
         'case_key':KEY,'task_type':'case_report'}


@pytest.fixture
def agent(evidence,tmp_path):
    value=AgentService(evidence,directory=tmp_path/'agent',background=False,
                       reviewer=Reviewer(TOKEN,identity='synthetic_reviewer'))
    yield value
    value.close()


def decision(run,action='approve'):
    return {'decision':action,'expected_state_version':run['state_version'],
            'expected_report_id':run['report_id'],'expected_revision':run['report_revision'],
            'expected_context_hash':run['context_hash']}


def test_rule_report_uses_five_tools_then_waits_for_human_review(agent):
    run,created=agent.create(REQUEST)
    assert created and run['status']=='waiting_review' and run['report_revision']==1
    assert [c['tool'] for c in run['tool_calls']]==[
        'get_monthly_history','get_data_quality','get_saved_predictions','search_documents','create_report_draft']
    assert run['planner']=='rule' and run['planner_live_verified'] is False
    assert run['review_decision'] is None and run['checkpoint_id']
    report,view=agent.report(run['run_id'])
    assert report['facts'][0]['value']==12 and '사용자 검토 대기' in render_html(report,view)


def test_authenticated_approval_resumes_without_repeating_tools(agent):
    run,_=agent.create(REQUEST)
    done,created=agent.resume(run['run_id'],decision(run),TOKEN)
    assert created and done['status']=='succeeded' and done['report_workflow_status']=='approved'
    assert len(done['tool_calls'])==5 and done['review_decision']['actor']=='synthetic_reviewer'
    again,created=agent.resume(run['run_id'],decision(run),TOKEN)
    assert not created and again['status']=='succeeded' and len(again['tool_calls'])==5


def test_rejection_cancels_run_and_is_not_an_approval(agent):
    run,_=agent.create(REQUEST)
    rejected,_=agent.resume(run['run_id'],decision(run,'reject'),TOKEN)
    assert rejected['status']=='cancelled' and rejected['report_workflow_status']=='rejected'


@pytest.mark.parametrize('credential',[None,'invalid-token'])
def test_approval_needs_server_verified_reviewer(agent,credential):
    run,_=agent.create(REQUEST)
    with pytest.raises(AgentError) as error:
        agent.resume(run['run_id'],decision(run),credential)
    assert error.value.status==401 and agent.get(run['run_id'])['status']=='waiting_review'


@pytest.mark.parametrize('field',['expected_state_version','expected_report_id','expected_revision','expected_context_hash'])
def test_stale_or_wrong_report_decision_cannot_resume(agent,field):
    run,_=agent.create(REQUEST)
    data=decision(run)
    data[field]=data[field]+1 if type(data[field]) is int else 'wrong'
    with pytest.raises(AgentError):agent.resume(run['run_id'],data,TOKEN)
    assert agent.get(run['run_id'])['status']=='waiting_review'


def test_revision_invalidates_previous_review_without_changing_facts(agent):
    original,_=agent.create(REQUEST)
    revised=agent.revise(original['run_id'],{'review_note':'추가 기관 확인이 필요합니다.',
        'expected_state_version':original['state_version'],'expected_report_id':original['report_id'],
        'expected_revision':original['report_revision']},TOKEN)
    assert revised['status']=='waiting_review' and revised['report_revision']==2
    assert revised['report_id']!=original['report_id'] and len(revised['tool_calls'])==5
    with pytest.raises(AgentError):agent.resume(original['run_id'],decision(original),TOKEN)
    done,_=agent.resume(revised['run_id'],decision(revised),TOKEN)
    assert done['status']=='succeeded' and done['report_revision']==2


def test_pending_review_survives_service_restart(evidence,tmp_path):
    directory=tmp_path/'restart'
    first=AgentService(evidence,directory=directory,background=False,reviewer=Reviewer(TOKEN))
    run,_=first.create(REQUEST)
    first.close()
    second=AgentService(evidence,directory=directory,background=False,reviewer=Reviewer(TOKEN))
    try:
        recovered=second.get(run['run_id'])
        assert recovered['status']=='waiting_review' and recovered['checkpoint_id']==run['checkpoint_id']
        done,_=second.resume(run['run_id'],decision(recovered),TOKEN)
        assert done['status']=='succeeded' and len(done['tool_calls'])==5
    finally:second.close()


def test_missing_case_pauses_and_user_input_resumes(agent):
    run,_=agent.create({'request':'NOx 보고서를 만들어 주세요.','task_type':'case_report'})
    assert run['status']=='waiting_input' and not run['tool_calls']
    ready,_=agent.resume(run['run_id'],{'action':'provide_case','case_key':KEY,
                                       'expected_state_version':run['state_version']},TOKEN)
    assert ready['status']=='waiting_review' and ready['case_key']==KEY and len(ready['tool_calls'])==5


def test_document_only_request_does_not_create_or_approve_report(agent):
    run,_=agent.create({'request':'NOx 연구 타깃 농도 정의 alpha','task_type':'document_search'})
    assert run['status']=='succeeded' and run['report_id'] is None
    assert [c['tool'] for c in run['tool_calls']]==['search_documents','finish_search']
    assert run['search']['results']


def test_idempotency_key_does_not_duplicate_run(agent):
    data={**REQUEST,'idempotency_key':'same-request'}
    first,_=agent.create(data)
    second,created=agent.create(data)
    assert not created and first['run_id']==second['run_id']
    with pytest.raises(AgentError):agent.create({**data,'request':'different'})


def test_concurrent_opposite_decisions_allow_only_one_result(agent):
    run,_=agent.create(REQUEST)
    def decide(action):
        try:return agent.resume(run['run_id'],decision(run,action),TOKEN)[0]['status']
        except AgentError:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(decide,['approve','reject']))
    assert results.count('conflict')==1
    assert agent.get(run['run_id'])['status'] in ('succeeded','cancelled')


def test_changed_sources_block_pending_approval(agent,bundle):
    run,_=agent.create(REQUEST)
    path=bundle/'history.csv'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):agent.resume(run['run_id'],decision(run),TOKEN)
    assert agent.store.decision(run['run_id'],1) is None


def test_audit_receipt_corruption_is_detected(agent):
    run,_=agent.create(REQUEST)
    with agent.store.connect() as db:
        db.execute('UPDATE events SET event_hash=? WHERE run_id=? AND seq=1',('bad',run['run_id']))
    with pytest.raises(AgentError,match='integrity'):agent.get(run['run_id'])


def test_transient_tool_failure_has_one_explicit_recovery(agent):
    original=agent.tools.execute
    failed=[False]
    def fail_once(name,args,state):
        if name=='get_data_quality' and not failed[0]:
            failed[0]=True
            raise TimeoutError('synthetic failure')
        return original(name,args,state)
    agent.tools.execute=fail_once
    run,_=agent.create(REQUEST)
    assert run['status']=='failed' and len(run['tool_calls'])==1
    recovered=agent.recover(run['run_id'],{'expected_state_version':run['state_version']},TOKEN)
    assert recovered['status']=='waiting_review' and len(recovered['tool_calls'])==5


def test_unstarted_queued_request_can_recover_after_restart(agent):
    queued,_=agent.store.create(asdict(RunRequest.parse(REQUEST)))
    assert agent.get(queued['run_id'])['status']=='queued'
    run=agent.recover(queued['run_id'],{'expected_state_version':queued['state_version']},TOKEN)
    assert run['status']=='waiting_review' and len(run['tool_calls'])==5


def test_failed_claim_does_not_change_another_workers_running_run(agent):
    queued,_=agent.store.create(asdict(RunRequest.parse(REQUEST)))
    agent.store.transition(queued['run_id'],'running',allowed={'queued'},owner_pid=1234)
    agent._drive(queued['run_id'])
    assert agent.store.get(queued['run_id'])['status']=='running'


def test_gemini_planner_mock_selects_tools_but_cannot_approve(evidence,tmp_path):
    class Mock:
        def choose_tool(self,contents,declarations):
            name=declarations[-1]['name']
            props=declarations[-1]['parameters']['properties']
            args={k:v['enum'][0] for k,v in props.items()}
            return {'name':name,'args':args,'call_id':'mock-call','model_content':{
                'role':'model','parts':[{'functionCall':{'name':name,'args':args,'id':'mock-call'},'thoughtSignature':'mock-signature'}]}}
    agent=AgentService(evidence,directory=tmp_path/'gemini',background=False,reviewer=Reviewer(TOKEN),planner=Mock())
    try:
        run,_=agent.create({**REQUEST,'planner':'gemini'})
        assert run['status']=='waiting_review' and run['review_decision'] is None
        assert run['planner']=='gemini' and run['planner_live_verified'] is False
        assert len(run['tool_calls'])==5
    finally:agent.close()


@pytest.mark.parametrize('bad_name,bad_args',[('approve_report',{}),('run_shell',{'command':'anything'}),('get_monthly_history',{'case_key':'other'})])
def test_llm_unknown_or_out_of_scope_tools_fail_closed(evidence,tmp_path,bad_name,bad_args):
    class Mock:
        def choose_tool(self,*args):return {'name':bad_name,'args':bad_args,'call_id':None,'model_content':None}
    agent=AgentService(evidence,directory=tmp_path/bad_name,background=False,reviewer=Reviewer(TOKEN),planner=Mock())
    try:
        run,_=agent.create({**REQUEST,'planner':'gemini'})
        assert run['status']=='failed' and not run['tool_calls'] and run['report_id'] is None
    finally:agent.close()


def test_gemini_planner_without_credentials_is_not_silently_rule_based(evidence,tmp_path):
    agent=AgentService(evidence,directory=tmp_path/'no-key',background=False,reviewer=Reviewer(TOKEN),
                       planner=GeminiClient(api_key='',model=''))
    try:
        with pytest.raises(AgentError,match='configured'):agent.create({**REQUEST,'planner':'gemini'})
    finally:agent.close()


@pytest.mark.parametrize('data',[{},[],{'request':True},{'request':'q','llm_enabled':'yes'},
                                {'request':'q','planner':'unknown'},{'request':'q','max_tool_calls':999}])
def test_invalid_run_requests_are_rejected(data):
    with pytest.raises(AgentError):RunRequest.parse(data)


def test_agent_http_review_authentication_and_report_response(agent):
    client=create_app(StubPredictor(),agent_service=agent,evidence_service=agent.evidence).test_client()
    response=client.post('/api/agent/runs',json=REQUEST)
    assert response.status_code==202
    run=response.json['run']
    path='/api/agent/runs/'+run['run_id']
    assert client.get(path+'/report').json['report']['facts'][0]['value']==12
    assert client.get(path+'/report/html').status_code==200
    assert client.post(path+'/resume',json=decision(run)).status_code==401
    done=client.post(path+'/resume',json=decision(run),headers={'X-Review-Token':TOKEN})
    assert done.status_code==202 and done.json['run']['status']=='succeeded'
    assert client.get('/api/health').status_code==200
    assert client.post('/api/predict',json={}).json['prediction']['value']==7
    assert client.post('/api/agent/runs',data='{"request":"q","request":"other"}',content_type='application/json').status_code==400
