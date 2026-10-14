import copy
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pytest

from nox.copilot.contracts import ReportError
from nox.copilot.registry import _digest
from nox.copilot.evidence.common import semantic
from nox.copilot.evidence.facts import make_facts
from nox.copilot.evidence.claims import fallback,validate_analysis
from nox.copilot.evidence.gemini import GeminiClient, GeminiError
from nox.copilot.evidence.reports import EvidenceReportService,EvidenceRepository,validate_envelope
from nox.copilot.evidence.render import render_html
from nox.copilot.rag.domain import DomainRetrieval
from test_copilot_registry import bundle,registry,CASE,KEY
from test_copilot_reports import service
from test_copilot_rag import rag


@pytest.fixture
def evidence(service,rag,tmp_path):
    return EvidenceReportService(service,DomainRetrieval(rag[0],default_method='keyword'),
                                 EvidenceRepository(tmp_path/'evidence'))


def test_numeric_facts_and_document_scope_remain_separate(evidence):
    report,created=evidence.create(KEY)
    assert created and report['schema_version']==2
    facts={f['fact_id']:f for f in report['facts']}
    assert facts['F_CURRENT']['value']==12 and facts['F_PREVIOUS']['value']==10
    assert facts['F_MEAN']['value']==9 and facts['F_ERROR_LIGHTGBM']['value']==1
    assert facts['F_PROVIDER']['value']=='pending_provider_confirmation'
    assert report['analysis']['status']=='disabled' and report['citations']
    assert all(c['plant_specific_evidence'] is False for c in report['citations'])
    assert evidence.get(report['report_id'])==report
    assert make_facts(report['base_report'])==report['facts']


def test_unchanged_report_request_is_idempotent(evidence):
    one,_=evidence.create(KEY)
    two,created=evidence.create(KEY)
    assert one==two and created is False


@pytest.mark.parametrize('field',['facts','citations','limitations','base_report','analysis','context_hash'])
def test_rehashed_report_tampering_is_rejected(evidence,field):
    report,_=evidence.create(KEY)
    if field=='facts': report[field][0]['value']=999
    elif field=='citations': report[field][0]['locator']['char_end']-=1
    elif field=='limitations': report[field]=[]
    elif field=='base_report': report[field]['observed']['value']=999
    elif field=='analysis': report[field]['content']['statements'][0]['fact_refs']=['FAKE']
    else: report[field]='a'*64
    report['report_id']='er_'+_digest(semantic(report))
    with pytest.raises((ReportError,ValueError)):
        validate_envelope(report,evidence.retrieval)


def test_revision_preserves_original_facts_sources_and_parent(evidence):
    first,_=evidence.create(KEY)
    second,created=evidence.revise(first['report_id'],'측정 유효성 확인 전 연구 초안으로 검토했습니다.')
    assert created and second['revision']==2 and second['parent_report_id']==first['report_id']
    assert second['facts']==first['facts'] and second['citations']==first['citations']
    assert second['report_id']!=first['report_id'] and evidence.get(first['report_id'])==first


@pytest.mark.parametrize('text',['관측은 12ppm입니다.','예측을 kg으로 환산하세요.','측정 유효성을 확정했습니다.','온도 설정을 높이세요.'])
def test_generated_numeric_and_unsupported_claims_are_rejected(evidence,text):
    report,_=evidence.create(KEY)
    value=fallback(report['facts'],report['citations'])
    value['statements'][0]['text']=text
    with pytest.raises(ValueError):
        validate_analysis(value,report['facts'],report['citations'])


def test_mock_gemini_success_does_not_claim_live_verification(evidence):
    class Mock:
        def generate_analysis(self,text):
            assert 'fact_refs' in text and '신뢰할 수 없는' in text
            return {'statements':[{'kind':'observation','text':'현재 관측과 이전 이력을 대조해 주세요.',
                                 'fact_refs':['F_CURRENT','F_PREVIOUS'],'citation_refs':[]}]}
    evidence.llm_client=Mock()
    report,_=evidence.create(KEY,llm_enabled=True)
    assert report['analysis']['status']=='gemini'
    assert report['analysis']['actual_external_call_verified'] is False
    assert report['analysis']['semantic_faithfulness_verified'] is False
    assert {f['fact_id']:f['value'] for f in report['facts']}['F_CURRENT']==12


@pytest.mark.parametrize('mode',['timeout','invalid_reference','number','bad_fields'])
def test_llm_failure_falls_back_with_numbers_and_warnings_preserved(evidence,mode):
    class Mock:
        def generate_analysis(self,text):
            if mode=='timeout':raise TimeoutError('secret should never leave server')
            if mode=='bad_fields':return {'prediction':900}
            return {'statements':[{'kind':'observation','text':'관측은 900입니다.' if mode=='number' else '확인하세요.',
                                 'fact_refs':['FAKE' if mode=='invalid_reference' else 'F_CURRENT'],'citation_refs':[]}]}
    evidence.llm_client=Mock()
    report,_=evidence.create(KEY,llm_enabled=True)
    assert report['analysis']['status']=='fallback'
    assert 'secret' not in json.dumps(report)
    assert report['base_report']['observed']['value']==12 and report['limitations']


def test_html_escapes_review_text_and_keeps_units_refs(evidence):
    first,_=evidence.create(KEY)
    report,_=evidence.revise(first['report_id'],'<script>alert("test")</script>')
    html=render_html(report)
    soup=BeautifulSoup(html,'html.parser')
    assert not soup.find('script') and 'ppm' in soup.get_text()
    assert 'F_CURRENT' in soup.get_text() and report['citations'][0]['citation_id'] in soup.get_text()
    assert '기관 확인 대기' in soup.get_text()


def test_gemini_transport_uses_header_and_structured_schema_without_printing_key():
    requests=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,limit):
            return json.dumps({'candidates':[{'finishReason':'STOP','content':{'role':'model',
                     'parts':[{'text':json.dumps({'statements':[]})}]}}]}).encode()
    def opener(request,timeout):
        requests.append(request)
        return Response()
    client=GeminiClient(api_key='test-secret',model='test-model',opener=opener)
    assert client.generate_analysis('test')=={'statements':[]}
    body=json.loads(requests[0].data)
    assert 'responseFormat' in body['generationConfig']
    assert 'test-secret' not in requests[0].full_url and requests[0].get_header('X-goog-api-key')=='test-secret'


def test_gemini_function_call_preserves_ids_and_thought_signature():
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,limit):
            return json.dumps({'candidates':[{'finishReason':'STOP','content':{'role':'model','parts':[
                {'functionCall':{'name':'search_documents','args':{'query':'q'},'id':'opaque-id'},
                 'thoughtSignature':'opaque-signature'}]}}]}).encode()
    client=GeminiClient(api_key='test-secret',model='test-model',opener=lambda *a,**k:Response())
    result=client.choose_tool([], [{'name':'search_documents'}])
    assert result['call_id']=='opaque-id' and result['model_content']['parts'][0]['thoughtSignature']=='opaque-signature'


def test_missing_gemini_configuration_is_explicit():
    client=GeminiClient(api_key='',model='')
    assert not client.configured
    with pytest.raises(GeminiError,match='gemini_not_configured'):
        client.generate_analysis('test')
