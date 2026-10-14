"""Source-pinned, bounded server tools; no arbitrary SQL, shell or approval tool."""
import copy
import re

from .common import AgentError
from ..evidence.common import implementation_hash as report_hash
from ..rag.domain import implementation_hash as retrieval_hash
from ..registry import _digest

CASE_TOOLS=('get_monthly_history','get_data_quality','get_saved_predictions')


class ToolSet:
    def __init__(self,evidence):
        self.evidence=evidence
        self.base=evidence.base_reports

    def resolve_case(self,request):
        available=[c for c in self.base.cases() if c['status']=='available']
        key=request.get('case_key')
        plant=next((p for p in ('영동','삼천포','여수','영흥','분당') if p in request['request']),None)
        unit=re.search(r'(\d+)\s*호기',request['request'])
        month=re.search(r'(20\d{2})\s*(?:년\s*|-)(\d{1,2})\s*월?',request['request'])
        expected={'plant':plant,'unit':int(unit[1]) if unit else None,
                  'month':f'{month[1]}-{int(month[2]):02d}' if month else None}
        if key is not None:
            selected=next((c for c in available if c['case_key']==key),None)
            if selected is None:raise AgentError('case_not_available','Use an available registered case',400)
            if any(v is not None and selected['case'][k]!=v for k,v in expected.items()):
                raise AgentError('case_request_mismatch','Request coordinates differ from the registered case',400)
            return key
        if all(v is not None for v in expected.values()):
            matches=[c['case_key'] for c in available if c['case']==expected]
            if len(matches)==1:return matches[0]
        return None

    def pin(self,case_key):
        status=self.evidence.retrieval.status()
        if not status['ready']:raise AgentError('retrieval_assets_unavailable','Search assets are unavailable',503)
        pins={'index_version_id':status['index_version_id'],'retrieval_hash':retrieval_hash(),
              'evidence_report_hash':report_hash()}
        if case_key is not None:
            calculation=self.base.calculation(case_key)
            pins.update(case_key=case_key,case_version_id=calculation['provenance']['case_version_id'])
        return pins

    def verify_pins(self,pins):
        actual=self.pin(pins.get('case_key'))
        if actual!=pins:raise AgentError('source_version_conflict','Source or implementation versions changed')

    def allowed(self,state):
        completed=state.get('completed',[])
        if state['task_type']=='document_search':
            return ['search_documents'] if 'search_documents' not in completed else ['finish_search']
        missing=[name for name in CASE_TOOLS if name not in completed]
        if missing:return missing
        if 'search_documents' not in completed:return ['search_documents']
        if 'create_report_draft' not in completed:return ['create_report_draft']
        return []

    def declaration(self,name,state):
        if name in CASE_TOOLS or name=='create_report_draft':
            properties={'case_key':{'type':'string','enum':[state['case_key']]}}
            required=['case_key']
        elif name=='search_documents' and state['task_type']=='case_report':
            properties={'purpose':{'type':'string','enum':['report_evidence']}}
            required=['purpose']
        elif name=='search_documents':
            properties={'query':{'type':'string','enum':[state['request']['request']]}}
            required=['query']
        else:
            properties,required={},[]
        return {'name':name,'description':{
            'get_monthly_history':'Read the exact calendar observations of the pinned saved evaluation case.',
            'get_data_quality':'Read unresolved source validity and quality flags; do not certify validity.',
            'get_saved_predictions':'Read saved retrospective predictions; this is not a new or future inference.',
            'search_documents':'Retrieve bounded general/method evidence from versioned documents.',
            'create_report_draft':'Persist a numeric/source-bound draft after prerequisite tools complete.',
            'finish_search':'Finish the document-only workflow without creating or approving a report.',
        }[name], 'parameters':{'type':'object','properties':properties,'required':required}}

    def arguments(self,name,state):
        declaration=self.declaration(name,state)
        return {k:v['enum'][0] for k,v in declaration['parameters']['properties'].items()}

    def validate_call(self,name,args,state):
        if name not in self.allowed(state) or name in state.get('completed',[]):
            raise AgentError('tool_not_allowed','Tool is unknown, repeated or its prerequisites are incomplete')
        if args!=self.arguments(name,state):
            raise AgentError('invalid_tool_arguments','Tool arguments must match the pinned case or original request')
        if len(state.get('completed',[]))>=6:
            raise AgentError('tool_budget_exceeded','Tool call budget was reached')
        self.verify_pins(state['pins'])

    def execute(self,name,args,state):
        self.validate_call(name,args,state)
        if name in CASE_TOOLS:
            calculation=self.base.calculation(state['case_key'])
            if name=='get_monthly_history':
                return {'current':calculation['metrics']['current_observation'],
                        'previous_3months':calculation['metrics']['previous_3months'],
                        'previous_month':calculation['metrics']['previous_month']}
            if name=='get_data_quality':
                return {'target_verification_status':calculation['target_verification_status'],
                        'current_quality':calculation['metrics']['current_observation']['quality_status'],
                        'quality_reasons':calculation['metrics']['current_observation']['quality_reasons'],
                        'limitations':calculation['limitations']}
            return {'prediction_kind':'saved_evaluation','predictions':calculation['metrics']['prediction_errors'],
                    'model':calculation['model']}
        if name=='search_documents':
            if state['task_type']=='case_report':return {'sections':self.evidence.retrieve()}
            return self.evidence.retrieval.search({'query':args['query']})
        if name=='create_report_draft':
            report,created=self.evidence.create(state['case_key'],llm_enabled=state['request']['llm_enabled'],
                expected_case_version_id=state['pins']['case_version_id'],
                contexts=state['results']['search_documents']['sections'])
            return {'report_id':report['report_id'],'revision':report['revision'],
                    'context_hash':report['context_hash'],'created':created,
                    'analysis_status':report['analysis']['status']}
        if name=='finish_search':
            search=state['results']['search_documents']
            return {'status':search['status'],'results':len(search['results']),
                    'retrieval_context_hash':search['retrieval_context_hash']}
        raise AgentError('tool_not_allowed','Tool is not supported')

    def planner_result(self,name,result):
        # Raw retrieved document instructions do not enter the tool-selection conversation.
        if name=='search_documents':
            if 'sections' in result:
                return {'sections':[{ 'section_id':s['section_id'],'status':s['response']['status'],
                                      'evidence_count':len(s['response']['results'])} for s in result['sections']]}
            return {'status':result['status'],'evidence_count':len(result['results'])}
        if name=='create_report_draft':return result
        return {'tool_result_hash':_digest(result),'source_bound':True}
