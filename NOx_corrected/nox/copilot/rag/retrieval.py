"""BM25, exact E5 cosine and RRF retrieval with pre-ranking scope/ACL filters."""
from collections import Counter
from dataclasses import dataclass
from datetime import date
import math
import re
import threading
import time
import unicodedata
import numpy as np

from .common import INDEX, RagError, identifier
from .worker_encoder import WorkerEncoder
from .index import VectorIndex
from ..registry import _digest

ALIASES = {
    '선택적촉매환원': 'scr', '선택적비촉매환원': 'sncr', '질소산화물': 'nox',
    '암모니아': 'ammonia', '촉매': 'catalyst', '환원제': 'reagent',
    '온도': 'temperature', '체류시간': 'residence time', '슬립': 'slip',
    '굴뚝원격감시체계': 'cleansys', '질량': 'mass', '농도': 'concentration',
}


def terms(text):
    text = unicodedata.normalize('NFKC', text).lower()
    collapsed = re.sub(r'\s+', '', text)
    extra = ' '.join(value for key, value in ALIASES.items() if key in collapsed)
    tokens = re.findall(r'[a-z0-9_]+|[가-힣]+', text + ' ' + extra)
    result = []
    for token in tokens:
        if re.fullmatch(r'[가-힣]+', token) and len(token) > 1:
            result.extend(token[i:i + 2] for i in range(len(token) - 1))
        result.append(token)
    return result


def bm25(chunks, query):
    documents = [Counter(terms(chunk['passage'])) for chunk in chunks]
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) if lengths else 1
    query_terms = set(terms(query))
    frequency = {term: sum(term in doc for doc in documents) for term in query_terms}
    scores = []
    for doc, length in zip(documents, lengths):
        score = 0.
        for term in query_terms:
            tf = doc[term]
            if tf:
                idf = math.log(1 + (len(documents) - frequency[term] + .5) / (frequency[term] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / average))
        scores.append(score)
    return np.asarray(scores)


def required_evidence(query):
    """Explicit limited rules; not an LLM answer correctness classifier."""
    plant = any(name in query for name in ('영동', '삼천포', '분당', '영흥', '여수'))
    period = bool(re.search(r'20\d{2}|최근|이번\s*달|지난\s*달', query))
    if plant and period and re.search(r'농도|예측값|배출량|발전량|투입량|사용량', query):
        return 'requires_numeric_tool'
    if plant and re.search(r'고장|정비\s*일시|촉매\s*교체|경보\s*기록|센서\s*무효', query):
        return 'requires_provider_evidence'
    if re.search(r'법정\s*허용|위반\s*확정|위법|허용기준\s*수치', query):
        return 'requires_verified_regulation'
    return None


@dataclass(frozen=True)
class SearchRequest:
    query: str
    method: str = 'hybrid'
    top_k: int = 5
    document_ids: tuple[str, ...] | None = None
    scope: str | None = None
    as_of: str | None = None

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 2000:
            raise RagError('invalid_query', 'Query must contain 1-2000 characters', 400)
        if self.method not in ('keyword', 'vector', 'hybrid') or type(self.top_k) is not int or not 1 <= self.top_k <= 10:
            raise RagError('invalid_search_configuration', 'Invalid retrieval method or top_k', 400)
        if self.document_ids is not None:
            if (not isinstance(self.document_ids, (list, tuple)) or not self.document_ids
                    or len(self.document_ids) > 20 or any(not isinstance(x, str) for x in self.document_ids)
                    or len(set(self.document_ids)) != len(self.document_ids)):
                raise RagError('invalid_filter', 'Document IDs must be unique', 400)
            for value in self.document_ids:
                identifier(value)
        if self.scope is not None and self.scope not in ('project_method', 'general_technical', 'korean_tms_general'):
            raise RagError('invalid_filter', 'Unsupported source scope', 400)
        if self.as_of is not None:
            try:
                date.fromisoformat(self.as_of)
            except (TypeError, ValueError) as error:
                raise RagError('invalid_filter', 'as_of must be a known publication cutoff date', 400) from error

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or 'query' not in data or set(data) - {'query', 'method', 'top_k', 'document_ids', 'scope', 'as_of'}:
            raise RagError('invalid_input', 'Provide a query and only supported search fields', 400)
        return cls(**data)


class RetrievalService:
    def __init__(self, index=None, encoder=None, *, minimum_cosine=.80, allowed_access=('public',)):
        if isinstance(minimum_cosine, bool) or not isinstance(minimum_cosine, (int, float)) or not math.isfinite(minimum_cosine) or not 0 <= minimum_cosine < 1:
            raise RagError('invalid_search_configuration', 'Cosine threshold must be finite and in [0,1)', 400)
        self.index = index if index is not None else VectorIndex()
        self.encoder = encoder
        self.minimum_cosine = minimum_cosine
        self.allowed_access = frozenset(allowed_access)
        self.lock = threading.Lock()
        self._query_cache = {}

    def _encoder(self, descriptor):
        with self.lock:
            if self.encoder is None:
                self.encoder = WorkerEncoder()
            if self.encoder.descriptor != descriptor:
                raise RagError('embedding_version_conflict', 'Query embedding and stored document vectors differ')
        return self.encoder

    def status(self):
        try:
            index, corpus, chunks, _ = self.index.load()
            return {'ready': True, 'backend': index['backend'], 'index_version_id': index['index_version_id'],
                    'corpus_version_id': corpus['corpus_version_id'], 'document_count': len(corpus['documents']),
                    'chunk_count': len(chunks), 'embedding': index['embedding'],
                    'generation_enabled': False, 'minimum_cosine': self.minimum_cosine}
        except (RagError, OSError, ValueError, KeyError, TypeError):
            return {'ready': False, 'code': 'retrieval_assets_unavailable', 'generation_enabled': False}

    def documents(self):
        manifest, _ = self.index.corpus.verify()
        return [doc for doc in manifest['documents'] if doc['access'] in self.allowed_access]

    def search(self, request):
        if not isinstance(request, SearchRequest):
            request = SearchRequest.parse(request)
        started = time.perf_counter()
        index, corpus, chunks, vectors = self.index.load()
        documents = {doc['document_id']: doc for doc in corpus['documents']}
        selected = []
        for i, chunk in enumerate(chunks):
            doc = documents[chunk['document_id']]
            if (doc['access'] not in self.allowed_access
                    or request.document_ids is not None and doc['document_id'] not in request.document_ids
                    or request.scope is not None and doc['scope'] != request.scope
                    or request.as_of is not None and (doc['published_date'] is None or doc['published_date'] > request.as_of)):
                continue
            selected.append(i)
        needed = required_evidence(request.query)
        results = []
        if selected and needed is None:
            filtered = [chunks[i] for i in selected]
            keyword = bm25(filtered, request.query) if request.method in ('keyword', 'hybrid') else np.zeros(len(filtered))
            cosine = np.zeros(len(filtered))
            if request.method in ('vector', 'hybrid'):
                encoder = self._encoder(index['embedding'])
                cache_key = (index['index_version_id'], request.query)
                with self.lock:
                    query = self._query_cache.get(cache_key)
                if query is None:
                    query = encoder.encode([request.query], kind='query')[0]
                    with self.lock:
                        if len(self._query_cache) >= 256:
                            self._query_cache.clear()
                        self._query_cache[cache_key] = query
                if (query.shape != (index['embedding']['dimension'],) or not np.isfinite(query).all()
                        or not np.isclose(np.linalg.norm(query), 1., atol=1e-5)):
                    raise RagError('invalid_query_vector', 'Query embedding is invalid')
                cosine = vectors[selected] @ query
            ranks = {}
            for name, scores, threshold in [('keyword', keyword, 0.), ('vector', cosine, self.minimum_cosine)]:
                if request.method != 'hybrid' and request.method != name:
                    continue
                order = sorted(range(len(filtered)), key=lambda i: (-float(scores[i]), filtered[i]['chunk_id']))
                ranks[name] = {i: rank for rank, i in enumerate([i for i in order if scores[i] > threshold][:20], 1)}
            # Lexical coincidences alone cannot override an empty semantic branch.
            # This is a retrieval abstention heuristic, not an answer verifier.
            if request.method == 'hybrid' and not ranks.get('vector'):
                ranks['keyword'] = {}
            candidates = set().union(*(set(values) for values in ranks.values()))
            ranked = sorted(candidates, key=lambda i: (
                -sum(1 / (60 + values[i]) for values in ranks.values() if i in values)
                if request.method == 'hybrid' else -float(keyword[i] if request.method == 'keyword' else cosine[i]),
                filtered[i]['chunk_id']))
            for i in ranked[:request.top_k]:
                chunk, doc = filtered[i], documents[filtered[i]['document_id']]
                locator = {**chunk['locator'], 'char_start': chunk['char_start'], 'char_end': chunk['char_end']}
                if locator['kind'] in ('markdown_lines', 'html_extracted_lines'):
                    # Chunk positions are relative to the preserved section text.
                    unit = next(unit for unit in self.index.corpus.verify()[1] if unit['unit_id'] == chunk['unit_id'])
                    locator['chunk_line_start'] = locator['line_start'] + unit['text'][:chunk['char_start']].count('\n')
                    locator['chunk_line_end'] = locator['line_start'] + unit['text'][:chunk['char_end'] - 1].count('\n')
                results.append({
                    'chunk_id': chunk['chunk_id'], 'unit_id': chunk['unit_id'], 'document_id': doc['document_id'],
                    'document_version_id': doc['document_version_id'], 'source_sha256': doc['source_sha256'],
                    'title': doc['title'], 'source_url': doc['source_url'], 'version_label': doc['version_label'],
                    'language': doc['language'], 'country': doc['country'], 'scope': doc['scope'],
                    'plant_specific_evidence': False, 'locator': locator, 'text': chunk['text'],
                    'scores': {'keyword': float(keyword[i]), 'cosine': float(cosine[i]),
                               'rrf': sum(1 / (60 + values[i]) for values in ranks.values() if i in values)},
                    'warnings': ['일반/연구 문서이며 해당 호기의 실제 상태를 증명하지 않습니다.',
                                 '자동 추출된 표·수치는 원문에서 재확인하세요.'],
                })
        content = {'query': request.query, 'method': request.method, 'top_k': request.top_k,
                   'status': needed or ('evidence_found' if results else 'insufficient_evidence'),
                   'results': results, 'index_version_id': index['index_version_id'],
                   'corpus_version_id': corpus['corpus_version_id'],
                   'generation_status': 'not_enabled', 'score_is_answer_probability': False,
                   'filters': {'document_ids': request.document_ids, 'scope': request.scope, 'as_of': request.as_of},
                   'configuration': {'minimum_cosine': self.minimum_cosine, 'rrf_k': 60, 'branch_candidates': 20,
                                     'hybrid_requires_semantic_candidate': True},
                   'limitations': ['검색 근거 반환 단계이며 LLM 답변·원인 확정·기관 측정 유효성 인증을 수행하지 않습니다.',
                                   'as_of는 알려진 발행일 필터이며 설비 적용일/법령 효력을 확인한 것이 아닙니다.']}
        content['retrieval_context_hash'] = _digest(content)
        content['elapsed_ms'] = (time.perf_counter() - started) * 1000
        return content
