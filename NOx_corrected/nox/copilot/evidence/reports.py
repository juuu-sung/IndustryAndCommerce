"""Immutable report envelopes joining numeric facts, source spans and explanations."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

from .claims import validate_analysis
from .common import DIRECTORY, VERSION, identifier, implementation_hash, semantic
from .facts import citations_for, make_facts
from .gemini import explain
from ..contracts import ReportError, ReportRequest
from ..registry import RegistrationError, _canonical, _digest, _json, _write_new
from ..repositories import ReportService
from ..rag.common import RagError
from ..rag.domain import DomainRetrieval, implementation_hash as retrieval_hash
from ..validators import validate_report

SECTIONS = [
    {'section_id':'target', 'label':'연구 타깃과 단위', 'query':'NOx 연구 타깃 일평균 월 산술평균 ppm 질량 kg', 'scope':'project_method'},
    {'section_id':'quality', 'label':'측정 유효성·집계 확인', 'query':'측정 유효성 보정 무효 플래그 유효 시간 일평균 집계 확인', 'scope':'project_method'},
    {'section_id':'technical', 'label':'일반 SCR 점검 근거', 'query':'SCR catalyst performance reaction temperature residence time catalyst deactivation management', 'scope':'general_technical'},
]


def context_hash(base, contexts, facts, citations):
    return _digest({'base_report_id':base['report_id'], 'facts':facts, 'citations':citations,
                    'retrieval_contexts':contexts, 'retrieval_implementation_hash':retrieval_hash()})


def validate_envelope(report, retrieval=None):
    try:
        fields = {'schema_version','report_version','implementation_hash','report_id','generated_at',
                  'revision','parent_report_id','review_note','status','base_report','facts','citations',
                  'retrieval_contexts','retrieval_implementation_hash','context_hash','analysis','limitations','validation'}
        if not isinstance(report,dict) or set(report) != fields:
            raise ValueError('Evidence report fields differ')
        if (report['schema_version'] != 2 or report['report_version'] != VERSION
                or report['implementation_hash'] != implementation_hash() or report['status'] != 'draft'
                or type(report['revision']) is not int or not 1 <= report['revision'] <= 20
                or not isinstance(report['review_note'],str) or len(report['review_note']) > 1000):
            raise ValueError('Evidence report version differs')
        identifier(report['report_id'])
        if report['parent_report_id'] is not None:
            identifier(report['parent_report_id'])
        if (report['revision'] == 1) != (report['parent_report_id'] is None):
            raise ValueError('Report revision ancestry differs')
        validate_report(report['base_report'])
        if report['facts'] != make_facts(report['base_report']) or report['citations'] != citations_for(report['retrieval_contexts']):
            raise ValueError('Facts or citations differ from verified inputs')
        if len(report['retrieval_contexts']) != len(SECTIONS):
            raise ValueError('Required evidence sections differ')
        for section, expected in zip(report['retrieval_contexts'],SECTIONS):
            if set(section) != set(expected) | {'response'} or any(section[k] != v for k,v in expected.items()):
                raise ValueError('Evidence section query differs')
            response = section['response']
            if response['query'] != expected['query'] or response['filters']['scope'] != expected['scope']:
                raise ValueError('Search scope differs')
            if retrieval is not None:
                retrieval.verify_response(response)
        if report['retrieval_implementation_hash'] != retrieval_hash():
            raise ValueError('Search implementation changed')
        if report['context_hash'] != context_hash(report['base_report'],report['retrieval_contexts'],report['facts'],report['citations']):
            raise ValueError('Context hash differs')
        analysis = report['analysis']
        if (set(analysis) != {'status','content','reason','actual_external_call_verified','semantic_faithfulness_verified'}
                or analysis['status'] not in ('disabled','gemini','fallback')
                or type(analysis['actual_external_call_verified']) is not bool
                or analysis['semantic_faithfulness_verified'] is not False
                or analysis['actual_external_call_verified'] and analysis['status'] != 'gemini'):
            raise ValueError('LLM analysis status differs')
        validate_analysis(analysis['content'],report['facts'],report['citations'])
        if report['limitations'] != limitations_for(report['base_report']):
            raise ValueError('Mandatory limitations differ')
        if report['validation'] != validation_for():
            raise ValueError('Validation declaration differs')
        if report['report_id'] != 'er_' + _digest(semantic(report)):
            raise ValueError('Evidence report content ID differs')
        when = datetime.fromisoformat(report['generated_at'])
        if when.tzinfo is None:
            raise ValueError('Report timestamp must include timezone')
    except (RegistrationError,KeyError,TypeError,ValueError,IndexError) as error:
        if isinstance(error,(ReportError,RagError)):
            raise
        raise ReportError('evidence_report_invalid','Evidence report content or source could not be verified') from error
    return report


def limitations_for(base):
    original = [s for s in base['limitations'] if s != '문서 RAG와 LLM 설명은 이 초안에 사용하지 않았습니다.']
    return list(dict.fromkeys(original + [
        '문서의 일반 기술 근거는 해당 발전소·호기의 실제 고장 원인이나 측정 유효성을 증명하지 않습니다.',
        '참조·형식·수치 검증을 통과해도 자연어의 의미적 근거 충실성은 사용자 검토가 필요합니다.',
        '검색 품질은 소규모 평가이며 전문가가 독립 검증한 결과가 아닙니다.',
        '사용자 승인은 이 보고서 초안의 검토 결정이며 기관 측정 유효성 인증이나 설비 제어 승인이 아닙니다.',
    ]))


def validation_for():
    return {'passed':['base_report_verified','backend_facts_equal','source_span_references','context_hash',
                      'analysis_schema_and_numeric_guard'],
            'unverified':['provider_measurement_validity','official_aggregation','semantic_faithfulness',
                          'plant_specific_operating_evidence']}


class EvidenceRepository:
    def __init__(self,directory=DIRECTORY):
        self.directory = Path(directory).resolve()

    def read(self,report_id):
        path = self.directory / (identifier(report_id) + '.json')
        try:
            if path.is_symlink():
                raise ReportError('evidence_report_invalid','Linked report files are not supported')
            value = _json(path.read_bytes(),'evidence report')
        except FileNotFoundError as error:
            raise ReportError('report_not_found','Evidence report was not found',404) from error
        except (OSError,RegistrationError) as error:
            raise ReportError('evidence_report_invalid','Evidence report storage could not be verified') from error
        validate_envelope(value)
        if value['report_id'] != report_id:
            raise ReportError('evidence_report_invalid','Report identifier differs')
        return value

    def save(self,report):
        validate_envelope(report)
        path = self.directory / (identifier(report['report_id'])+'.json')
        if path.exists() or path.is_symlink():
            existing = self.read(report['report_id'])
            if _canonical(semantic(existing)) != _canonical(semantic(report)):
                raise ReportError('report_conflict','Existing evidence report differs')
            return existing,False
        try:
            _write_new(path,report)
        except FileExistsError:
            return self.read(report['report_id']),False
        return report,True


class EvidenceReportService:
    def __init__(self,base_reports=None,retrieval=None,repository=None,llm_client=None):
        self.base_reports = base_reports if base_reports is not None else ReportService()
        self.retrieval = retrieval if retrieval is not None else DomainRetrieval()
        self.repository = repository if repository is not None else EvidenceRepository()
        self.llm_client = llm_client
        self.lock = threading.RLock()
        if self.repository.directory.is_relative_to(self.base_reports.registry.directory):
            raise ValueError('Evidence report storage must be outside source registry')

    def retrieve(self):
        contexts=[]
        for section in SECTIONS:
            response=self.retrieval.search({'query':section['query'],'scope':section['scope'],'top_k':3})
            response.pop('elapsed_ms',None)
            self.retrieval.verify_response(response)
            contexts.append({**section,'response':response})
        return contexts

    def create(self,case_key,*,llm_enabled=False,expected_case_version_id=None,contexts=None):
        if type(llm_enabled) is not bool:
            raise ReportError('invalid_input','llm_enabled must be boolean',400)
        with self.lock:
            base,_=self.base_reports.create(ReportRequest(case_key,expected_case_version_id=expected_case_version_id))
            contexts=copy.deepcopy(contexts) if contexts is not None else self.retrieve()
            facts,citations=make_facts(base),citations_for(contexts)
            report={'schema_version':2,'report_version':VERSION,'implementation_hash':implementation_hash(),
                    'revision':1,'parent_report_id':None,'review_note':'','status':'draft','base_report':base,
                    'facts':facts,'citations':citations,'retrieval_contexts':contexts,
                    'retrieval_implementation_hash':retrieval_hash(),
                    'context_hash':context_hash(base,contexts,facts,citations),
                    'analysis':explain(facts,citations,base,enabled=llm_enabled,client=self.llm_client),
                    'limitations':limitations_for(base),'validation':validation_for()}
            report['report_id']='er_'+_digest(semantic(report))
            report['generated_at']=datetime.now(timezone.utc).isoformat()
            validate_envelope(report,self.retrieval)
            # Source checks run again immediately before persistence.
            self.base_reports.get(base['report_id'])
            manifest=self.base_reports.registry.verify(case_key)
            source=(self.base_reports.registry.directory/manifest['artifact_path']).resolve()
            if self.repository.directory.is_relative_to(source):
                raise ReportError('reports_unavailable','Report storage must be outside frozen sources',503)
            return self.repository.save(report)

    def get(self,report_id):
        report=self.repository.read(report_id)
        actual=self.base_reports.get(report['base_report']['report_id'])
        if actual != report['base_report']:
            raise ReportError('source_conflict','Base report differs from registered sources')
        validate_envelope(report,self.retrieval)
        return report

    def revise(self,report_id,note):
        if not isinstance(note,str) or not note.strip() or len(note)>1000:
            raise ReportError('invalid_review_note','Review note must contain 1-1000 characters',400)
        with self.lock:
            original=self.get(report_id)
            if original['revision']>=20:
                raise ReportError('revision_limit','Report revision limit was reached')
            revised=copy.deepcopy(original)
            revised.update(revision=original['revision']+1,parent_report_id=report_id,review_note=note,
                           generated_at=datetime.now(timezone.utc).isoformat())
            revised['report_id']='er_'+_digest(semantic(revised))
            validate_envelope(revised,self.retrieval)
            return self.repository.save(revised)
