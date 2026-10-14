"""LangGraph tool dispatch with persistent human-input and review interrupts."""
import copy
import time
from typing import TypedDict

from .common import AgentError,dependency,implementation_hash
from ..evidence.gemini import GeminiClient
from ..registry import _digest


class State(TypedDict,total=False):
    run_id:str
    request:dict
    task_type:str
    case_key:str|None
    pins:dict
    completed:list
    results:dict
    contents:list
    proposed:dict
    phase:str
    report_id:str|None
    report_revision:int|None
    context_hash:str|None
    decision:dict
    graph_implementation_hash:str


def build_graph(tools,store,saver,planner=None):
    graph_module=dependency('langgraph.graph')
    types=dependency('langgraph.types')
    interrupt=types.interrupt
    planner=planner if planner is not None else GeminiClient()

    def prepare(state):
        if state['graph_implementation_hash']!=implementation_hash():
            raise AgentError('workflow_version_conflict','Workflow implementation changed')
        request=state['request']
        task=request['task_type']
        if task=='auto':
            task='case_report' if request['case_key'] is not None or '보고서' in request['request'] or 'report' in request['request'].lower() else 'document_search'
        if task=='case_report' and ('kg' in request['request'].lower() or '미래 예측' in request['request'] or '실시간' in request['request']):
            raise AgentError('unsupported_analysis_mode','Only registered retrospective concentration reports are supported',400)
        case_key=tools.resolve_case(request) if task=='case_report' else None
        if task=='case_report' and case_key is None:
            answer=interrupt({'kind':'missing_case','message':'분석할 등록 사례를 지정해 주세요.',
                              'available_case_keys':[c['case_key'] for c in tools.base.cases() if c['status']=='available']})
            if not isinstance(answer,dict) or set(answer)!={'case_key'}:
                raise AgentError('invalid_resume_input','Provide a registered case key',400)
            request={**request,'case_key':answer['case_key']}
            case_key=tools.resolve_case(request)
        pins=tools.pin(case_key)
        initial={'role':'user','parts':[{'text':'요청에 맞게 허용된 서버 도구만 선택하세요. 승인·설비 제어·임의 코드 실행은 도구에 없습니다. '+request['request']}]}
        return {'request':request,'task_type':task,'case_key':case_key,'pins':pins,'completed':[],
                'results':{},'contents':[initial],'phase':'tools'}

    def choose(state):
        tools.verify_pins(state['pins'])
        allowed=tools.allowed(state)
        if not allowed:raise AgentError('no_tool_available','No valid next tool exists')
        if state['request']['planner']=='gemini':
            proposal=planner.choose_tool(copy.deepcopy(state['contents']),[tools.declaration(n,state) for n in allowed])
        else:
            name=allowed[0]
            proposal={'name':name,'args':tools.arguments(name,state),'call_id':None,'model_content':None}
        tools.validate_call(proposal['name'],proposal['args'],state)
        return {'proposed':proposal}

    def execute(state):
        proposal=state['proposed']
        started=time.perf_counter()
        tools.validate_call(proposal['name'],proposal['args'],state)
        result=store.cached_tool(state['run_id'],proposal['name'],proposal['args'],state['pins'])
        if result is None:
            result=tools.execute(proposal['name'],proposal['args'],state)
            store.cache_tool(state['run_id'],proposal['name'],proposal['args'],state['pins'],result,
                             (time.perf_counter()-started)*1000,state['request']['planner'],
                             planner_live_verified=state['request']['planner']=='gemini' and type(planner) is GeminiClient
                             and not planner.mock_transport and proposal.get('external_call_verified') is True)
        contents=copy.deepcopy(state['contents'])
        if proposal.get('model_content') is not None:
            contents.append(proposal['model_content'])
            response={'name':proposal['name'],'response':tools.planner_result(proposal['name'],result)}
            if proposal.get('call_id') is not None:response['id']=proposal['call_id']
            contents.append({'role':'user','parts':[{'functionResponse':response}]})
        update={'completed':[*state['completed'],proposal['name']],
                'results':{**state['results'],proposal['name']:result},'contents':contents}
        if proposal['name']=='create_report_draft':
            update.update(report_id=result['report_id'],report_revision=result['revision'],context_hash=result['context_hash'])
        return update

    def after_tool(state):
        name=state['proposed']['name']
        return 'review' if name=='create_report_draft' else 'finish' if name=='finish_search' else 'choose'

    def review(state):
        # No mutation before interrupt: resuming restarts this node.
        tools.verify_pins(state['pins'])
        tools.evidence.get(state['report_id'])
        answer=interrupt({'kind':'report_review','report_id':state['report_id'],
                          'revision':state['report_revision'],'context_hash':state['context_hash']})
        decision=store.decision(state['run_id'],state['report_revision'])
        if (not isinstance(answer,dict) or decision is None or answer.get('decision_id')!=decision['decision_id']
                or decision['report_id']!=state['report_id'] or decision['context_hash']!=state['context_hash']):
            raise AgentError('review_receipt_invalid','Resume requires a source-bound authenticated review receipt')
        return {'decision':decision,'phase':'reviewed'}

    def finish(state):
        tools.verify_pins(state['pins'])
        return {'phase':'finished'}

    builder=graph_module.StateGraph(State)
    for name,node in [('prepare',prepare),('choose',choose),('execute',execute),('review',review),('finish',finish)]:
        builder.add_node(name,node)
    builder.add_edge(graph_module.START,'prepare')
    builder.add_edge('prepare','choose')
    builder.add_edge('choose','execute')
    builder.add_conditional_edges('execute',after_tool,{'choose':'choose','review':'review','finish':'finish'})
    builder.add_edge('review','finish')
    builder.add_edge('finish',graph_module.END)
    return builder.compile(checkpointer=saver)
