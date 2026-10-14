import copy
import pytest

from nox.copilot.rag.common import RagError
from nox.copilot.rag.domain import DomainRetrieval, query_plan
from nox.copilot.registry import _digest
from test_copilot_rag import rag


@pytest.mark.parametrize('query,scope', [
    ('전월 NOx와 이동평균을 어떻게 계산하나요?', 'project_method'),
    ('SCR 반응 온도와 체류시간은 왜 중요한가요?', 'general_technical'),
    ('암모니아 슬립의 의미는?', 'general_technical'),
    ('굴뚝원격감시체계 CleanSYS란?', 'korean_tms_general'),
    ('시간별 발전 패턴으로 SCR 운전을 판단할 수 있나요?', 'project_method'),
])
def test_scope_routing(query, scope):
    assert query_plan(query)['scope'] == scope


def test_glossary_expands_terms_without_gold_answers():
    plan = query_plan('SCR 촉매 비활성과 관리·체류시간·암모니아 슬립')
    assert all(term in plan['retrieval_query'] for term in ('catalyst', 'deactivation', 'management', 'residence', 'unreacted'))


@pytest.mark.parametrize('query,status', [
    ('영동 2025년 10월 NOx 예측값', 'requires_numeric_tool'),
    ('영동 촉매의 실제 수명은?', 'requires_provider_evidence'),
    ('법적 NOx 기준과 위반 여부', 'requires_verified_regulation'),
    ('초콜릿 케이크 레시피', 'insufficient_evidence'),
    ('NOx 문서와 무관한 김치볶음밥 조리 순서', 'insufficient_evidence'),
    ('오늘 서울의 여행지는?', 'insufficient_evidence'),
])
def test_unsupported_queries_do_not_return_context(rag, query, status):
    service = DomainRetrieval(rag[0], default_method='hybrid')
    result = service.search({'query': query})
    assert result['status'] == status and result['results'] == []


def test_explicit_scope_and_acl_survive_domain_layer(rag):
    service = DomainRetrieval(rag[0], default_method='hybrid')
    response = service.search({'query': 'NOx alpha', 'method': 'keyword', 'scope': 'project_method'})
    assert {r['document_id'] for r in response['results']} == {'alpha'}
    assert service.verify_response(response) == response
    denied = service.search({'query': 'NOx beta', 'method': 'keyword', 'document_ids': ['beta']})
    assert denied['results'] == []


@pytest.mark.parametrize('field', ['text', 'source_url', 'locator', 'document_version_id'])
def test_context_cannot_be_rehashed_to_replace_evidence(rag, field):
    service = DomainRetrieval(rag[0], default_method='hybrid')
    response = service.search({'query': 'NOx alpha', 'method': 'keyword', 'scope': 'project_method'})
    hit = response['results'][0]
    if field == 'locator':
        hit[field]['char_end'] -= 1
    else:
        hit[field] = 'tampered'
    response['retrieval_context_hash'] = _digest({k:v for k,v in response.items() if k not in ('elapsed_ms','retrieval_context_hash')})
    with pytest.raises(RagError):
        service.verify_response(response)
