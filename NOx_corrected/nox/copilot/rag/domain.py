"""Explicit scope selection and bilingual expansion, separate from the frozen baseline."""
from dataclasses import dataclass
from pathlib import Path
import re

from .common import RUNTIME, RagError, read_json
from .retrieval import RetrievalService, SearchRequest, required_evidence
from ..registry import _digest, _sha

VERSION = 'scoped_bilingual_v1'
STOP_WORDS = r'어떻게|무엇인가요|무엇인가|알려주세요|찾아주세요|확인하나요|하나요|되나요|인가요|주세요'
TECHNICAL = {
    '암모니아': 'ammonia', '슬립': 'slip unreacted ammonia', '요소': 'urea',
    '환원제': 'reducing reagent ammonia urea', '생성물': 'nitrogen N2 water H2O',
    '질소산화물': 'NOx nitrogen oxides', '반응': 'reaction',
    '온도': 'temperature range', '체류시간': 'residence time',
    '촉매': 'catalyst', '관리': 'management', '비활성': 'deactivation', '열화': 'deactivation',
    '성능': 'performance operational factors', '저감': 'reduction',
    '저부하': 'low load', '부하': 'load', '연료': 'different fuels',
    '분포': 'temperature profile', '낮거나 높': 'low high slow kinetics oxidation',
    '낮으면': 'low slow kinetics', '높으면': 'high oxidation additional NOx',
}
PROJECT_TERMS = (
    'nox', 'ppm', 'kg', '타깃', '농도', '배출', '관측', '연료', '발전', '기상',
    '전월', '개월', '월별', '월평균', '이동평균', '결측', '중앙값', '시계열',
    '학습', '모델', '검증', '평가', 'rmse', 'mae', 'llm', 'gemini', 'api', 'as_of',
    '보정', '이력', '해시', '보고서', '측정', '정답', '호기', '풍향', '풍속',
    '운전', '촉매', '환원', '암모니아', '체류', 'scr', 'sncr', 'cleansys', 'tms',
)


def implementation_hash():
    directory = Path(__file__).parent
    files = ('domain.py', 'retrieval.py', 'index.py', 'ingest.py', 'embeddings.py',
             'worker_encoder.py', 'embedding_worker.py', 'common.py')
    return _digest({name: _sha((directory / name).read_bytes()) for name in files})


@dataclass(frozen=True)
class DomainRequest:
    query: str
    method: str = 'hybrid'
    top_k: int = 5
    document_ids: tuple[str, ...] | None = None
    scope: str | None = None
    as_of: str | None = None

    def __post_init__(self):
        SearchRequest(self.query, self.method, self.top_k, self.document_ids, self.scope, self.as_of)

    @classmethod
    def parse(cls, value):
        fields = {'query', 'method', 'top_k', 'document_ids', 'scope', 'as_of'}
        if not isinstance(value, dict) or 'query' not in value or set(value) - fields:
            raise RagError('invalid_input', 'Provide only supported domain search fields', 400)
        return cls(**value)


def query_plan(query, explicit_scope=None):
    collapsed = re.sub(r'\s+', '', query).lower()
    needed = required_evidence(query)
    plant = any(p in query for p in ('영동', '삼천포', '여수', '영흥', '분당'))
    if needed is None and plant and re.search(r'촉매.*수명|작동\s*여부|보정\s*전|실제.*환원제|실제.*설비', query):
        needed = 'requires_provider_evidence'
    if needed is None and re.search(r'법정|법적.*기준|위반.*판정|위반.*여부', query):
        needed = 'requires_verified_regulation'
    off_topic = re.search(r'레시피|볶음밥|케이크|조리\s*순서|여행지|축구\s*결과|복권|주식\s*추천', query)
    if needed is None and (off_topic or not any(term in collapsed for term in PROJECT_TERMS)):
        needed = 'insufficient_evidence'
    project_marker = re.search(r'타깃|학습|평가|검증|모델|예측|이력|보고서|api|gemini|llm|해시|특징|전월|개월|기상|풍향|결측|발전.*패턴', query, re.I)
    technical_marker = re.search(r'\bS(?:N)?CR\b|촉매|암모니아|환원제|체류|slip|urea', query, re.I)
    if explicit_scope is not None:
        scope, reason = explicit_scope, 'explicit_scope'
    elif project_marker:
        scope, reason = 'project_method', 'project_method_terms'
    elif technical_marker:
        scope, reason = 'general_technical', 'general_technical_terms'
    elif re.search(r'cleansys|tms|굴뚝원격감시체계', query, re.I):
        scope, reason = 'korean_tms_general', 'tms_overview_terms'
    else:
        scope, reason = 'project_method', 'project_domain_default'
    normalized = re.sub(STOP_WORDS, ' ', query).strip()
    if scope == 'general_technical':
        english = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', query)
        english += [word for key, phrase in TECHNICAL.items() if key in collapsed for word in phrase.split()]
        # No answers or gold spans are inserted; this is a fixed engineering glossary.
        normalized = ' '.join(dict.fromkeys(english)) or normalized
    return {'original_query': query, 'retrieval_query': normalized, 'scope': scope,
            'scope_reason': reason, 'required_status': needed, 'version': VERSION}


class DomainRetrieval:
    def __init__(self, base=None, *, default_method=None):
        self.base = base if base is not None else RetrievalService()
        self.index = self.base.index
        if default_method is not None and default_method not in ('keyword', 'vector', 'hybrid'):
            raise RagError('invalid_search_configuration', 'Unsupported default retrieval method', 400)
        self.configured_method = default_method

    def status(self):
        return {**self.base.status(), 'domain_profile': VERSION,
                'domain_implementation_hash': implementation_hash()}

    def search(self, request):
        if not isinstance(request, DomainRequest):
            if isinstance(request, dict) and 'method' not in request:
                request = {**request, 'method': self.default_method()}
            request = DomainRequest.parse(request)
        plan = query_plan(request.query, request.scope)
        # Keep the pre-ranking filters in the verified baseline service.
        response = self.base.search(SearchRequest(
            plan['retrieval_query'] if plan['required_status'] is None else request.query,
            method='keyword' if plan['required_status'] is not None else request.method,
            top_k=10, document_ids=request.document_ids,
            scope=plan['scope'], as_of=request.as_of))
        if plan['required_status'] is not None:
            response['results'] = []
            response['status'] = plan['required_status']
        else:
            # Limit overlapping chunks from one page/section so five slots cover more evidence.
            selected, per_unit = [], {}
            for hit in response['results']:
                if per_unit.get(hit['unit_id'], 0) >= 2:
                    continue
                selected.append(hit)
                per_unit[hit['unit_id']] = per_unit.get(hit['unit_id'], 0) + 1
                if len(selected) == request.top_k:
                    break
            response['results'] = selected
            response['status'] = 'evidence_found' if selected else 'insufficient_evidence'
        response.update(query=request.query, top_k=request.top_k, query_plan=plan,
                        method=request.method,
                        domain_profile=VERSION, domain_implementation_hash=implementation_hash())
        response['filters']['scope'] = plan['scope']
        response.pop('retrieval_context_hash', None)
        response['retrieval_context_hash'] = _digest({k: v for k, v in response.items() if k != 'elapsed_ms'})
        return response

    def default_method(self):
        if self.configured_method is not None:
            return self.configured_method
        path = RUNTIME / 'domain_evaluation_v1/selection_lock.json'
        if not path.exists():
            return 'hybrid'
        lock = read_json(path)
        status = self.base.status()
        if (lock.get('domain_implementation_hash') != implementation_hash()
                or lock.get('selection_id') != _digest({k:v for k,v in lock.items() if k != 'selection_id'})
                or lock.get('selected_profile') not in ('domain:keyword', 'domain:vector', 'domain:hybrid')
                or lock.get('index_version_id') != status.get('index_version_id')):
            raise RagError('retrieval_configuration_conflict', 'Search selection is stale or unsupported')
        return lock['selected_profile'].split(':')[1]

    def verify_response(self, response):
        index, corpus, chunks, _ = self.index.load()
        expected = _digest({k: v for k, v in response.items() if k not in ('elapsed_ms', 'retrieval_context_hash')})
        if (response.get('retrieval_context_hash') != expected
                or response.get('index_version_id') != index['index_version_id']
                or response.get('corpus_version_id') != corpus['corpus_version_id']
                or response.get('domain_implementation_hash') != implementation_hash()):
            raise RagError('retrieval_version_conflict', 'Search context or implementation changed')
        by_chunk = {c['chunk_id']: c for c in chunks}
        docs = {d['document_id']: d for d in corpus['documents']}
        units = {u['unit_id']:u for u in self.index.corpus.verify()[1]}
        for hit in response['results']:
            chunk = by_chunk.get(hit['chunk_id'])
            if chunk is None:
                raise RagError('retrieval_version_conflict', 'Unknown source chunk')
            doc = docs[chunk['document_id']]
            locator = {**chunk['locator'], 'char_start':chunk['char_start'], 'char_end':chunk['char_end']}
            if locator['kind'] in ('markdown_lines', 'html_extracted_lines'):
                unit = units[chunk['unit_id']]
                locator['chunk_line_start'] = locator['line_start'] + unit['text'][:chunk['char_start']].count('\n')
                locator['chunk_line_end'] = locator['line_start'] + unit['text'][:chunk['char_end']-1].count('\n')
            meta = ('document_id','document_version_id','source_sha256','title','source_url',
                    'version_label','language','country','scope')
            if (doc['access'] not in self.base.allowed_access or hit['document_version_id'] != doc['document_version_id']
                    or hit['source_sha256'] != doc['source_sha256'] or hit['text'] != chunk['text']
                    or hit['unit_id'] != chunk['unit_id'] or hit['scope'] != doc['scope']
                    or hit['document_id'] != doc['document_id'] or hit['source_url'] != doc['source_url']
                    or any(hit.get(field) != doc[field] for field in meta)
                    or hit['locator'] != locator or hit.get('plant_specific_evidence') is not False):
                raise RagError('retrieval_version_conflict', 'Search evidence differs from its source')
        return response
