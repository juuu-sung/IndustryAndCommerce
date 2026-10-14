"""Background execution, durable review/resume and report revision coordination."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import threading

from .common import DIRECTORY,AgentError,Reviewer,RunRequest,dependency,implementation_hash
from .graph import build_graph
from .store import WorkflowStore
from .tools import ToolSet
from ..evidence.gemini import GeminiClient
from ..evidence.reports import EvidenceReportService
from ..registry import _digest


class AgentService:
    def __init__(self,evidence=None,*,directory=DIRECTORY,planner=None,reviewer=None,background=True):
        self.evidence=evidence if evidence is not None else EvidenceReportService()
        self.store=WorkflowStore(directory)
        self.reviewer=reviewer if reviewer is not None else Reviewer(directory=directory)
        self.planner=planner if planner is not None else GeminiClient()
        checkpoint_path=self.store.directory/'checkpoints.sqlite'
        if checkpoint_path.is_symlink():raise AgentError('agent_storage_invalid','Checkpoint storage is invalid',503)
        self.connection=sqlite3.connect(checkpoint_path,check_same_thread=False)
        serializer=dependency('langgraph.checkpoint.serde.jsonplus').JsonPlusSerializer(
            pickle_fallback=False,allowed_json_modules=(),allowed_msgpack_modules=())
        self.saver=dependency('langgraph.checkpoint.sqlite').SqliteSaver(self.connection,serde=serializer)
        self.tools=ToolSet(self.evidence)
        self.graph=build_graph(self.tools,self.store,self.saver,self.planner)
        os.chmod(checkpoint_path,0o600)
        self.background=background
        self.executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='nox-agent') if background else None
        self.active=set()
        self.lock=threading.RLock()
        self.closed=False

    def status(self):
        return {'ready':True,'engine':'langgraph','checkpoint_storage':'sqlite',
                'planner_default':'rule','gemini_configured':bool(getattr(self.planner,'configured',False)),
                'review_authentication':'local_token','max_tool_calls':6,
                'supported_analysis_modes':['evaluation_replay'],
                'live_verification_status':'status_endpoint_does_not_make_external_calls'}

    def config(self,identifier):
        return {'configurable':{'thread_id':identifier},'recursion_limit':30,'callbacks':[]}

    def _schedule(self,identifier):
        with self.lock:
            if self.closed:raise AgentError('agent_unavailable','Workflow runner is closed',503)
            if identifier in self.active:return
            self.active.add(identifier)
        if self.background:
            self.executor.submit(self._drive,identifier)
        else:self._drive(identifier)

    def create(self,data):
        request=RunRequest.parse(data)
        if request.planner=='gemini' and isinstance(self.planner,GeminiClient) and not self.planner.configured:
            raise AgentError('gemini_not_configured','Gemini planner requires configured credentials and model',503)
        run,created=self.store.create(asdict(request))
        if created:self._schedule(run['run_id'])
        return self.get(run['run_id']),created

    def _drive(self,identifier):
        claimed=False
        try:
            run=self.store.transition(identifier,'running',allowed={'queued'},owner_pid=os.getpid())
            claimed=True
            config=self.config(identifier)
            snapshot=self.graph.get_state(config)
            if snapshot.values and snapshot.values.get('graph_implementation_hash')!=implementation_hash():
                raise AgentError('workflow_version_conflict','Workflow implementation changed')
            pending=run['pending']
            if pending is not None:
                command=dependency('langgraph.types').Command
                if pending['action']=='review':initial=command(resume={'decision_id':pending['decision_id']})
                elif pending['action']=='provide_case':initial=command(resume={'case_key':pending['case_key']})
                else:initial=None
            else:
                initial={'run_id':identifier,'request':run['request'],'graph_implementation_hash':implementation_hash()}
            # Traces stay in our local receipts, even if unrelated environment tracing is enabled.
            with dependency('langsmith.run_helpers').tracing_context(enabled=False):
                result=self.graph.invoke(initial,config)
            snapshot=self.graph.get_state(config)
            values=snapshot.values
            if result.get('__interrupt__'):
                interrupt=result['__interrupt__'][0].value
                status='waiting_input' if interrupt['kind']=='missing_case' else 'waiting_review'
                self.store.transition(identifier,status,allowed={'running'},
                    report_id=values.get('report_id'),report_revision=values.get('report_revision'),
                    context_hash=values.get('context_hash'),pins_json=json.dumps(values.get('pins'),ensure_ascii=False),
                    pending_json=None,owner_pid=None,error_code=None)
            else:
                decision=values.get('decision')
                status='cancelled' if decision and decision['decision']=='reject' else 'succeeded'
                self.store.transition(identifier,status,allowed={'running'},pending_json=None,owner_pid=None,error_code=None)
        except Exception as error:
            code=getattr(error,'code','workflow_execution_failed')
            if claimed:
                try:self.store.transition(identifier,'failed',allowed={'running'},error_code=code,owner_pid=None)
                except AgentError:pass
        finally:
            with self.lock:self.active.discard(identifier)

    def get(self,identifier):
        run=self.store.get(identifier)
        snapshot=self.graph.get_state(self.config(identifier))
        events=self.store.events(identifier)
        completed=[e['payload'] for e in events if e['payload']['kind']=='tool_completed']
        value={'run_id':identifier,'status':run['status'],'state_version':run['state_version'],
               'planner':run['request']['planner'],
               'planner_live_verified':any(e.get('planner_live_verified') is True for e in completed),
               'request':run['request']['request'],'case_key':snapshot.values.get('case_key'),
               'task_type':snapshot.values.get('task_type'),'report_id':run['report_id'],
               'report_revision':run['report_revision'],'context_hash':run['context_hash'],
               'tool_calls':completed,'checkpoint_id':snapshot.config.get('configurable',{}).get('checkpoint_id') if snapshot.config else None,
               'error_code':run['error_code'],'created_at':run['created_at'],'updated_at':run['updated_at']}
        if snapshot.interrupts:value['pending_input']=[i.value for i in snapshot.interrupts]
        if run['report_id'] is not None:
            decision=self.store.decision(identifier,run['report_revision'])
            value['review_decision']=decision
            value['report_workflow_status']=('approved' if decision and decision['decision']=='approve' and run['status']=='succeeded'
                else 'rejected' if decision and decision['decision']=='reject' and run['status']=='cancelled'
                else 'decision_recorded' if decision else 'waiting_review')
            value['links']={'json':'/api/agent/runs/'+identifier+'/report',
                            'html':'/api/agent/runs/'+identifier+'/report/html'}
        if run['status']=='succeeded' and value['task_type']=='document_search':
            value['search']=snapshot.values['results']['search_documents']
        return value

    def _assert_current(self,run):
        state=self.graph.get_state(self.config(run['run_id'])).values
        if not state and run['status']=='queued':
            RunRequest.parse(run['request'])
            return state
        if state.get('graph_implementation_hash')!=implementation_hash():
            raise AgentError('workflow_version_conflict','Workflow implementation changed')
        pins=run['pins'] if run['pins'] is not None else state.get('pins')
        if pins is not None:self.tools.verify_pins(pins)
        if run['report_id'] is not None:
            report=self.evidence.get(run['report_id'])
            if report['revision']!=run['report_revision'] or report['context_hash']!=run['context_hash']:
                raise AgentError('review_version_conflict','Stored report and workflow version differ')
        return state

    def resume(self,identifier,data,credential):
        actor=self.reviewer.authenticate(credential)
        with self.lock:
            run=self.store.get(identifier)
            self._assert_current(run)
            if isinstance(data,dict) and data.get('action')=='provide_case':
                if set(data)!={'action','case_key','expected_state_version'} or type(data['expected_state_version']) is not int:
                    raise AgentError('invalid_resume_input','Provide a case key and current state version',400)
                self.tools.resolve_case({**run['request'],'case_key':data['case_key']})
                self.store.transition(identifier,'queued',allowed={'waiting_input'},expected_version=data['expected_state_version'],
                    pending_json=json.dumps({'action':'provide_case','case_key':data['case_key']}))
                created=True
            else:
                _,created=self.store.decide(identifier,data,actor)
        if created:self._schedule(identifier)
        return self.get(identifier),created

    def revise(self,identifier,data,credential):
        actor=self.reviewer.authenticate(credential)
        required={'review_note','expected_state_version','expected_report_id','expected_revision'}
        if (not isinstance(data,dict) or set(data)!=required or type(data['expected_state_version']) is not int
                or type(data['expected_revision']) is not int):
            raise AgentError('invalid_revision_request','Provide a review note and current report version',400)
        with self.lock:
            run=self.store.get(identifier)
            self._assert_current(run)
            if (run['status']!='waiting_review' or run['state_version']!=data['expected_state_version']
                    or run['report_id']!=data['expected_report_id'] or run['report_revision']!=data['expected_revision']):
                raise AgentError('review_version_conflict','Only the current waiting-review report can be revised')
            revised,_=self.evidence.revise(run['report_id'],data['review_note'])
            config=self.config(identifier)
            self.graph.update_state(config,{'report_id':revised['report_id'],'report_revision':revised['revision'],
                                          'context_hash':revised['context_hash']},as_node='execute')
            with dependency('langsmith.run_helpers').tracing_context(enabled=False):
                result=self.graph.invoke(None,config)
            if not result.get('__interrupt__'):raise AgentError('revision_checkpoint_invalid','Revised report did not pause for review')
            self.store.transition(identifier,'waiting_review',allowed={'waiting_review'},expected_version=data['expected_state_version'],
                report_id=revised['report_id'],report_revision=revised['revision'],context_hash=revised['context_hash'],
                event={'kind':'report_revised','actor':actor,'previous_report_id':run['report_id'],
                       'report_id':revised['report_id'],'revision':revised['revision']})
        return self.get(identifier)

    def recover(self,identifier,data,credential):
        self.reviewer.authenticate(credential)
        if not isinstance(data,dict) or set(data)!={'expected_state_version'} or type(data['expected_state_version']) is not int:
            raise AgentError('invalid_recovery_request','Provide the current state version',400)
        with self.lock:
            run=self.store.get(identifier)
            self._assert_current(run)
            if identifier in self.active:raise AgentError('run_active','Workflow is currently running')
            if run['status']=='running':
                try:os.kill(run['owner_pid'],0)
                except ProcessLookupError:pass
                else:raise AgentError('run_active','The previous workflow process is still active')
            elif run['status']=='queued':
                pass
            elif run['status']!='failed' or run['error_code'] not in ('workflow_execution_failed','gemini_transport_error','gemini_http_error'):
                raise AgentError('recovery_not_allowed','This failure cannot be retried without correcting sources or input')
            if any(e['payload']['kind']=='recovery_requested' for e in self.store.events(identifier)):
                raise AgentError('recovery_budget_exceeded','One explicit recovery attempt is permitted for this run')
            pending=run['pending'] if run['status']=='queued' else {'action':'recover'}
            self.store.transition(identifier,'queued',allowed={'queued','running','failed'},expected_version=data['expected_state_version'],
                                  pending_json=json.dumps(pending) if pending is not None else None,owner_pid=None,error_code=None,
                                  event={'kind':'recovery_requested','state_version':data['expected_state_version']})
        self._schedule(identifier)
        return self.get(identifier)

    def report(self,identifier):
        run=self.store.get(identifier)
        if run['report_id'] is None:raise AgentError('report_not_available','This workflow has no report yet',404)
        self._assert_current(run)
        return self.evidence.get(run['report_id']),self.get(identifier)

    def close(self):
        with self.lock:self.closed=True
        if self.executor is not None:self.executor.shutdown(wait=True)
        self.connection.close()
