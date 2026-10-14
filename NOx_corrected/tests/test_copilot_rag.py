import copy
import json
import re

import numpy as np
import pytest

from nox.app import create_app
from nox.copilot.registry import _digest
from nox.copilot.rag.common import CORPUS, INDEX, RagError, read_json
from nox.copilot.rag.evaluation import score_question, select_method, verify_gold
from nox.copilot.rag.ingest import Corpus, DocumentSpec, markdown_units
from nox.copilot.rag.index import VectorIndex, chunks_for
from nox.copilot.rag.retrieval import RetrievalService, SearchRequest, required_evidence, terms
from test_copilot_reports import StubPredictor


class FakeEncoder:
    descriptor = {'model': 'test_only', 'revision': 'fixed', 'dimension': 4, 'maximum_tokens': 512}

    def __init__(self):
        self.inputs = []

    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]

    def count(self, text):
        return len(self.offsets(text))

    def encode(self, texts, kind):
        self.inputs.extend((kind, text) for text in texts)
        vectors = []
        for text in texts:
            vector = np.asarray([float(word in text.lower()) for word in ('alpha', 'beta', 'gamma', 'delta')], dtype=np.float32)
            if not vector.any():
                vector[:] = .5
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def rag(tmp_path):
    corpus = Corpus(tmp_path / 'corpus')
    sources = []
    for name, access, published, scope in (
        ('alpha', 'public', '2020-01-01', 'project_method'),
        ('beta', 'restricted', '2019-01-01', 'general_technical'),
        ('gamma', 'public', '2025-01-01', 'general_technical'),
        ('delta', 'public', None, 'general_technical'),
    ):
        path = tmp_path / (name + '.md')
        path.write_text(f'# {name} reference\n\n{name} evidence: NOx 농도 ppm 연구 월별 평균 촉매 ammonia catalyst.\n', encoding='utf-8')
        spec = DocumentSpec(name, name + ' reference', 'https://example.test/' + name,
                            'synthetic test', 'v1', 'ko', 'KR', scope, 'md', published, access)
        sources.append((spec, path))
    corpus.register(sources)
    encoder = FakeEncoder()
    index = VectorIndex(tmp_path / 'index', corpus)
    index.build(encoder, tokens=32, overlap=4)
    service = RetrievalService(index, encoder)
    return service, sources


def test_code_fence_hash_comments_are_not_sections():
    text = '# Top\ncontent long enough here\n```python\n# not a heading\n```\n## End\nmore content long enough\n'
    units = markdown_units(text)
    assert len(units) == 2 and '# not a heading' in units[0]['text']
    assert units[1]['locator']['line_start'] == 6


def test_corpus_versions_and_idempotent_registration(rag):
    service, sources = rag
    before = service.index.corpus.verify()[0]
    assert service.index.corpus.register(sources) == before
    assert len(service.index.corpus.verify()[1]) == 4
    sources[0][1].write_text('# Changed\nA different source after registration.\n')
    with pytest.raises(RagError, match='differs'):
        service.index.corpus.register(sources)


@pytest.mark.parametrize('asset', ['raw', 'unit', 'chunk', 'vector', 'manifest'])
def test_asset_corruption_blocks_search(rag, asset):
    service, _ = rag
    if asset == 'raw':
        path = service.index.corpus.directory / 'raw/alpha.md'
        path.write_bytes(path.read_bytes() + b'changed')
    elif asset == 'unit':
        path = service.index.corpus.directory / 'units.json'
        data = read_json(path)
        data[0]['text'] += 'changed'
        path.write_text(json.dumps(data))
    elif asset == 'chunk':
        path = service.index.directory / 'chunks.json'
        data = read_json(path)
        data[0]['text'] += 'changed'
        path.write_text(json.dumps(data))
    elif asset == 'vector':
        path = service.index.directory / 'vectors.npy'
        path.write_bytes(path.read_bytes() + b'changed')
    else:
        path = service.index.directory / 'manifest.json'
        path.write_text('[]')
    assert service.status()['ready'] is False
    with pytest.raises(RagError):
        service.search({'query': 'alpha', 'method': 'keyword'})


def test_symlink_asset_is_rejected(rag, tmp_path):
    service, _ = rag
    path = service.index.directory / 'manifest.json'
    backup = tmp_path / 'backup.json'
    path.rename(backup)
    path.symlink_to(backup)
    with pytest.raises(RagError):
        service.index.load()


@pytest.mark.parametrize('method', ['keyword', 'vector', 'hybrid'])
def test_search_has_source_spans_and_repeatable_context(rag, method):
    service, _ = rag
    first = service.search({'query': 'alpha', 'method': method})
    second = service.search({'query': 'alpha', 'method': method})
    assert first['status'] == 'evidence_found'
    assert first['results'][0]['document_id'] == 'alpha'
    assert first['retrieval_context_hash'] == second['retrieval_context_hash']
    hit = first['results'][0]
    assert hit['locator']['chunk_line_start'] == 1 and hit['source_sha256']
    assert hit['plant_specific_evidence'] is False
    assert first['generation_status'] == 'not_enabled'


def test_acl_scope_document_and_date_filters_precede_ranking(rag):
    service, _ = rag
    assert all(d['document_id'] != 'beta' for d in service.documents())
    assert service.search({'query': 'beta', 'method': 'vector'})['results'] == []
    assert service.search({'query': 'gamma', 'scope': 'project_method'})['results'] == []
    assert service.search({'query': 'alpha', 'document_ids': ['gamma']})['results'] == []
    result = service.search({'query': 'alpha', 'as_of': '2021-01-01', 'method': 'keyword'})
    assert {r['document_id'] for r in result['results']} == {'alpha'}
    assert service.search({'query': 'delta', 'as_of': '2021-01-01', 'method': 'vector'})['results'] == []
    # The denied source is absent from embedding query/passage inputs after index creation.
    service.encoder.inputs.clear()
    service.search({'query': 'alpha', 'method': 'hybrid'})
    assert all('beta evidence' not in text for _, text in service.encoder.inputs)


def test_hybrid_abstains_on_lexical_only_matches(rag):
    service, _ = rag
    assert service.search({'query': 'NOx', 'method': 'keyword'})['results']
    assert service.search({'query': 'NOx', 'method': 'hybrid'})['status'] == 'insufficient_evidence'


def test_encoder_version_mismatch_blocks_vector_query(rag):
    service, _ = rag
    service.encoder = FakeEncoder()
    service.encoder.descriptor = {**FakeEncoder.descriptor, 'revision': 'different'}
    with pytest.raises(RagError, match='differ'):
        service.search({'query': 'alpha', 'method': 'vector'})


def test_chunk_offsets_cover_exact_source_and_overlap(rag):
    service, _ = rag
    manifest, units = service.index.corpus.verify()
    units = [copy.deepcopy(units[0])]
    units[0]['text'] = ' '.join(f'w{i}' for i in range(70))
    chunks = chunks_for(manifest, units, FakeEncoder(), tokens=32, overlap=8)
    assert len(chunks) == 3 and chunks[1]['char_start'] < chunks[0]['char_end']
    for chunk in chunks:
        assert chunk['text'] == units[0]['text'][chunk['char_start']:chunk['char_end']]


@pytest.mark.parametrize('payload', [
    None, [], {}, {'query': ''}, {'query': True}, {'query': 'a' * 2001},
    {'query': 'alpha', 'method': 'reranker'}, {'query': 'alpha', 'top_k': True},
    {'query': 'alpha', 'top_k': 0}, {'query': 'alpha', 'top_k': 11},
    {'query': 'alpha', 'document_ids': ['../secret']}, {'query': 'alpha', 'document_ids': ['alpha', 'alpha']},
    {'query': 'alpha', 'document_ids': 'alpha'}, {'query': 'alpha', 'scope': 'plant_manual'},
    {'query': 'alpha', 'as_of': 'yesterday'}, {'query': 'alpha', 'unexpected': 7},
])
def test_invalid_requests_are_rejected(payload):
    with pytest.raises(RagError) as caught:
        SearchRequest.parse(payload)
    assert caught.value.status == 400


@pytest.mark.parametrize('query,status', [
    ('영동 2025년 NOx 예측값 알려주세요', 'requires_numeric_tool'),
    ('영동 SCR 고장 확정해 주세요', 'requires_provider_evidence'),
    ('국내 NOx 법정 허용기준 수치는?', 'requires_verified_regulation'),
])
def test_unsupported_plant_and_regulatory_questions_return_no_evidence(rag, query, status):
    service, _ = rag
    result = service.search({'query': query})
    assert result['status'] == status and result['results'] == []
    assert required_evidence(query) == status


def test_korean_aliases_include_technical_english():
    assert 'scr' in terms('선택적 촉매 환원')
    assert 'ammonia' in terms('암모니아')
    assert 'nox' in terms('질소산화물')


def test_chunk_end_on_newline_stays_inside_section_lines(rag):
    service, _ = rag
    # Simulate a tokenizer including a trailing newline in its final token.
    manifest, units = service.index.corpus.verify()
    unit = units[0]
    index_manifest, _, chunks, vectors = service.index.load()
    chunk = next(c for c in chunks if c['unit_id'] == unit['unit_id'])
    chunk['char_end'] = len(unit['text'])
    chunk['text'] = unit['text']
    service.index.load = lambda: (index_manifest, manifest, chunks, vectors)
    response = service.search({'query': 'alpha', 'method': 'keyword'})
    locator = response['results'][0]['locator']
    assert locator['chunk_line_end'] == unit['locator']['line_end']


def test_gold_scoring_requires_full_span_and_counts_groups_once():
    q = {'question_id': 'D1', 'query': 'q', 'category': 'multi', 'expected_status': 'answerable',
         'required_evidence_groups': [{'group_id': 'g1', 'alternatives': [{'unit_id': 'a',
              'document_version_id': 'v', 'char_start': 10, 'char_end': 20}]},
              {'group_id': 'g2', 'alternatives': [{'unit_id': 'b', 'document_version_id': 'v',
                                               'char_start': 10, 'char_end': 20}]}]}
    hit = {'unit_id': 'a', 'document_version_id': 'v', 'chunk_id': 'c', 'document_id': 'doc',
           'locator': {'char_start': 0, 'char_end': 20}}
    response = {'status': 'evidence_found', 'results': [hit, hit], 'elapsed_ms': 1., 'retrieval_context_hash': 'h'}
    row = score_question(q, response)
    assert row['recall_at_5'] == .5 and row['reciprocal_rank'] == 1.
    hit['locator']['char_end'] = 19
    assert score_question(q, response)['recall_at_5'] == 0
    hit['locator']['char_end'] = 20
    hit['document_version_id'] = 'new'
    assert score_question(q, response)['hit_at_5'] == 0


def test_method_selection_uses_declared_development_rule():
    metrics = {'keyword': {'recall_at_5': .9, 'false_acceptance_rate': 1., 'mrr': 1.},
               'vector': {'recall_at_5': .8, 'false_acceptance_rate': 0., 'mrr': .7},
               'hybrid': {'recall_at_5': .8, 'false_acceptance_rate': 0., 'mrr': .7}}
    assert select_method(metrics) == 'vector'


def test_gold_wrong_source_version_is_rejected(rag):
    service, _ = rag
    manifest, units = service.index.corpus.verify()
    unit = units[0]
    gold = {'schema_version': 1, 'corpus_version_id': manifest['corpus_version_id'], 'questions': [{
        'question_id': 'D1', 'split': 'development', 'query': 'alpha', 'expected_status': 'answerable',
        'required_evidence_groups': [{'group_id': 'g', 'alternatives': [{
            'unit_id': unit['unit_id'], 'unit_sha256': unit['unit_sha256'], 'document_version_id': 'wrong',
            'char_start': 0, 'char_end': 5}]}]}]}
    gold['gold_version_id'] = _digest(gold)
    with pytest.raises(RagError, match='version'):
        verify_gold(gold, service.index.corpus)


def test_rag_http_routes_and_predict_are_independent(rag):
    service, _ = rag
    client = create_app(StubPredictor(), enable_copilot=False, rag_service=service).test_client()
    assert client.get('/api/rag/status').json['ready'] is True
    assert len(client.get('/api/rag/documents').json['documents']) == 3
    response = client.post('/api/rag/search', json={'query': 'alpha', 'method': 'vector'})
    assert response.status_code == 200 and response.json['results'][0]['document_id'] == 'alpha'
    assert response.headers['Cache-Control'] == 'no-store'
    assert client.post('/api/rag/search', data='alpha').status_code == 415
    for text in ('{"query":"alpha","query":"beta"}', '{"query":NaN}', '{broken', '[]'):
        assert client.post('/api/rag/search', data=text, content_type='application/json').status_code == 400
    assert client.get('/api/health').status_code == 200
    assert client.post('/api/predict', json={}).json['prediction']['value'] == 7


def test_missing_rag_assets_do_not_break_existing_api(tmp_path):
    service = RetrievalService(VectorIndex(tmp_path / 'index', Corpus(tmp_path / 'corpus')))
    client = create_app(StubPredictor(), rag_service=service, enable_copilot=False).test_client()
    assert client.get('/api/rag/status').json['ready'] is False
    assert client.post('/api/rag/search', json={'query': 'q'}).status_code == 503
    assert client.get('/api/rag/documents').status_code == 503
    assert client.get('/api/health').status_code == 200


def test_real_e5_index_and_source_query():
    if not (INDEX / 'manifest.json').exists():
        pytest.skip('Optional private retrieval assets are absent')
    service = RetrievalService()
    status = service.status()
    assert status['ready'] and status['document_count'] == 10 and status['chunk_count'] == 616
    assert status['embedding']['dimension'] == 384
    assert status['embedding']['model_id'] == 'intfloat/multilingual-e5-small'
    result = service.search({'query': 'NOx 연구 타깃의 월평균 농도 ppm 정의', 'method': 'vector',
                             'document_ids': ['project_data']})
    assert result['results'] and all(r['document_id'] == 'project_data' for r in result['results'])
    assert all(r['locator']['char_end'] > r['locator']['char_start'] for r in result['results'])
    # A crashed optional worker fails closed without taking down the prediction process.
    service.encoder.close()
    service._query_cache.clear()
    client = create_app(StubPredictor(), enable_copilot=False, rag_service=service).test_client()
    assert client.post('/api/rag/search', json={'query': 'different', 'method': 'vector'}).status_code == 503
    assert client.post('/api/predict', json={}).json['prediction']['value'] == 7
