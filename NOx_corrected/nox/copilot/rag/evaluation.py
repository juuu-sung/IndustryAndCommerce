"""Source-span retrieval benchmark, development selection before sealed test scoring."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .common import RUNTIME, RagError, asset_bytes, read_json, write_new
from .gold import GOLD_PATH
from .retrieval import RetrievalService, SearchRequest
from ..registry import _digest, _sha

METHODS = ('keyword', 'vector', 'hybrid')
POLICY = {
    'top_k': 5, 'minimum_cosine': .80, 'rrf_k': 60, 'branch_candidates': 20,
    'hybrid_requires_semantic_candidate': True,
    'selection': 'development mean group Recall@5 - 0.25 * unsupported false acceptance rate; '
                 'ties: MRR, then keyword/vector/hybrid simplicity order',
    'threshold_policy': '0.80 fixed before scoring; no test tuning',
    'latency': 'warm CPU encoder; query cache cleared before each measured request; asset verification included',
}


def verify_gold(gold, corpus):
    manifest, units = corpus.verify()
    if (not isinstance(gold, dict) or gold.get('schema_version') != 1
            or gold.get('corpus_version_id') != manifest['corpus_version_id']
            or gold.get('gold_version_id') != _digest({k: v for k, v in gold.items() if k != 'gold_version_id'})
            or not isinstance(gold.get('questions'), list) or not gold['questions']):
        raise RagError('invalid_gold', 'Benchmark corpus or version differs')
    by_unit, seen = {u['unit_id']: u for u in units}, set()
    try:
        for q in gold['questions']:
            if (q['question_id'] in seen or q['split'] not in ('development', 'test')
                    or not isinstance(q['query'], str) or not q['query'].strip()
                    or q['expected_status'] not in ('answerable', 'insufficient_evidence', 'requires_numeric_tool',
                                                  'requires_provider_evidence', 'requires_verified_regulation')):
                raise RagError('invalid_gold', 'Question contract differs')
            seen.add(q['question_id'])
            groups = q['required_evidence_groups']
            if not isinstance(groups, list) or bool(groups) != (q['expected_status'] == 'answerable'):
                raise RagError('invalid_gold', 'Question evidence contract differs')
            group_ids = set()
            for group in groups:
                if group['group_id'] in group_ids or not group['alternatives']:
                    raise RagError('invalid_gold', 'Evidence groups must be unique and nonempty')
                group_ids.add(group['group_id'])
                for ref in group['alternatives']:
                    unit = by_unit[ref['unit_id']]
                    if (ref['unit_sha256'] != unit['unit_sha256']
                            or ref['document_version_id'] != unit['document_version_id']
                            or type(ref['char_start']) is not int or type(ref['char_end']) is not int
                            or not 0 <= ref['char_start'] < ref['char_end'] <= len(unit['text'])):
                        raise RagError('invalid_gold', 'Evidence span or source version differs')
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, RagError):
            raise
        raise RagError('invalid_gold', 'Benchmark structure is invalid') from error
    return gold


def score_question(question, response):
    groups = question['required_evidence_groups']
    first_ranks = []
    for group in groups:
        ranks = []
        for rank, hit in enumerate(response['results'], 1):
            for ref in group['alternatives']:
                if (hit['unit_id'] == ref['unit_id']
                        and hit['document_version_id'] == ref['document_version_id']
                        and hit['locator']['char_start'] <= ref['char_start']
                        and hit['locator']['char_end'] >= ref['char_end']):
                    ranks.append(rank)
        first_ranks.append(min(ranks) if ranks else None)
    answerable = question['expected_status'] == 'answerable'
    return {
        'question_id': question['question_id'], 'query': question['query'], 'category': question['category'],
        'expected_status': question['expected_status'], 'actual_status': response['status'],
        'answerable': answerable, 'group_first_ranks': first_ranks,
        'hit_at_5': float(any(rank is not None for rank in first_ranks)) if answerable else None,
        'recall_at_5': sum(rank is not None for rank in first_ranks) / len(groups) if answerable else None,
        'reciprocal_rank': 1 / min(rank for rank in first_ranks if rank is not None)
                           if any(rank is not None for rank in first_ranks) else (0. if answerable else None),
        'unsupported_handled': not response['results'] and response['status'] == question['expected_status']
                               if not answerable else None,
        'false_acceptance': bool(response['results']) if not answerable else None,
        'retrieved': [{'chunk_id': r['chunk_id'], 'document_id': r['document_id'],
                       'document_version_id': r['document_version_id'], 'locator': r['locator']}
                      for r in response['results']],
        'elapsed_ms': response['elapsed_ms'], 'context_hash': response['retrieval_context_hash'],
    }


def summarize(rows):
    answerable = [row for row in rows if row['answerable']]
    unsupported = [row for row in rows if not row['answerable']]
    def average(field, items):
        return sum(row[field] for row in items) / len(items) if items else None
    return {
        'questions': len(rows), 'answerable_questions': len(answerable), 'unsupported_questions': len(unsupported),
        'hit_at_5': average('hit_at_5', answerable), 'recall_at_5': average('recall_at_5', answerable),
        'mrr': average('reciprocal_rank', answerable),
        'hit_questions': sum(bool(r['hit_at_5']) for r in answerable),
        'unsupported_handled_count': sum(bool(r['unsupported_handled']) for r in unsupported),
        'unsupported_handled_rate': average('unsupported_handled', unsupported),
        'false_acceptance_count': sum(bool(r['false_acceptance']) for r in unsupported),
        'false_acceptance_rate': average('false_acceptance', unsupported),
        'latency_p50_ms': float(np.percentile([r['elapsed_ms'] for r in rows], 50)),
        'latency_p95_ms': float(np.percentile([r['elapsed_ms'] for r in rows], 95)),
    }


def select_method(metrics):
    return max(METHODS, key=lambda method: (
        metrics[method]['recall_at_5'] - .25 * metrics[method]['false_acceptance_rate'],
        metrics[method]['mrr'], -METHODS.index(method)))


def implementation_hash():
    directory = Path(__file__).parent
    files = ('common.py', 'ingest.py', 'index.py', 'embeddings.py', 'embedding_worker.py',
             'worker_encoder.py', 'retrieval.py', 'gold.py', 'evaluation.py')
    return _digest({name: _sha(asset_bytes(directory / name)) for name in files})


def evaluate(output, *, service=None, gold_path=GOLD_PATH, prior_evaluation=None):
    service = service if service is not None else RetrievalService(minimum_cosine=POLICY['minimum_cosine'])
    if service.minimum_cosine != POLICY['minimum_cosine']:
        raise RagError('benchmark_configuration_conflict', 'Benchmark cosine threshold differs')
    gold = verify_gold(read_json(gold_path), service.index.corpus)
    status = service.status()
    if not status['ready']:
        raise RagError('retrieval_assets_unavailable', 'Benchmark requires a verified index', 503)
    # Load the encoder once before measured requests; no query vector reuse across methods.
    service._encoder(status['embedding']).encode(['검색 지연 측정 준비'], kind='query')
    output = Path(output)
    source = implementation_hash()
    prior = read_json(prior_evaluation) if prior_evaluation is not None else None
    results, metrics = {}, {}
    def run(split):
        questions = [q for q in gold['questions'] if q['split'] == split]
        rows = {method: [] for method in METHODS}
        for pos, q in enumerate(questions):
            # Rotate method order to reduce systematic order/CPU heat bias.
            order = METHODS[pos % 3:] + METHODS[:pos % 3]
            for method in order:
                with service.lock:
                    service._query_cache.clear()
                response = service.search(SearchRequest(q['query'], method=method, top_k=5))
                rows[method].append(score_question(q, response))
        return rows, {method: summarize(values) for method, values in rows.items()}
    results['development'], metrics['development'] = run('development')
    selected = select_method(metrics['development'])
    selection = {'schema_version': 1, 'gold_version_id': gold['gold_version_id'],
                 'corpus_version_id': status['corpus_version_id'], 'index_version_id': status['index_version_id'],
                 'implementation_hash': source, 'policy': POLICY,
                 'selected_method': selected, 'development_metrics': metrics['development']}
    selection['selection_id'] = _digest(selection)
    # This write must complete before the test loop. A conflicting rerun needs a new directory.
    write_new(output / 'selection_lock.json', selection)
    print(f'Development selection locked: {selected}', flush=True)
    results['test'], metrics['test'] = run('test')
    if implementation_hash() != source:
        raise RagError('benchmark_implementation_changed', 'Search implementation changed during evaluation')
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'selection_id': selection['selection_id'], 'selected_method': selected,
              'gold_version_id': gold['gold_version_id'], 'corpus_version_id': status['corpus_version_id'],
              'index_version_id': status['index_version_id'], 'annotation_status': gold['annotation_status'],
              'policy': POLICY, 'metrics': metrics, 'results': results,
              'generation_evaluated': False, 'expert_review_completed': False,
              'prior_evaluation_id': prior['evaluation_id'] if prior is not None else None,
              'test_results_previously_observed': prior is not None,
              'limitations': ['Small assistant-authored benchmark; not independent expert evaluation.',
                              'Test results are held out from method selection for this run only.',
                              'No LLM answer faithfulness or NOx model performance is measured.',
                              'Unsupported handling includes explicit routing rules, not a general semantic guarantee.']}
    report['evaluation_id'] = _digest(report)
    write_new(output / 'evaluation.json', report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUNTIME / 'evaluation_v1')
    parser.add_argument('--prior-evaluation', type=Path)
    arguments = parser.parse_args()
    started = time.monotonic()
    value = evaluate(arguments.output, prior_evaluation=arguments.prior_evaluation)
    print(json.dumps({'selected_method': value['selected_method'], 'metrics': value['metrics'],
                      'seconds': time.monotonic() - started}, ensure_ascii=False, indent=2))
