"""Select with previously observed development questions, then score frozen additions."""
from pathlib import Path
import json
import time

from .common import RUNTIME, read_json, write_new
from .domain import DomainRetrieval, implementation_hash
from .evaluation import score_question, summarize, verify_gold
from .fresh_gold import PATH
from .gold import GOLD_PATH
from .retrieval import RetrievalService
from ..registry import _digest

PROFILES = ['baseline:keyword', 'baseline:vector', 'baseline:hybrid',
            'domain:keyword', 'domain:vector', 'domain:hybrid']


def main():
    base = RetrievalService()
    domain = DomainRetrieval(base)
    old = verify_gold(read_json(GOLD_PATH), base.index.corpus)
    fresh = verify_gold(read_json(PATH), base.index.corpus)
    output = RUNTIME / 'domain_evaluation_v1'
    status = base.status()
    base._encoder(status['embedding']).encode(['검색 준비'], kind='query')
    results, metrics = {}, {}
    def run(questions):
        rows = {profile: [] for profile in PROFILES}
        for pos, question in enumerate(questions):
            for profile in PROFILES[pos % 6:] + PROFILES[:pos % 6]:
                kind, method = profile.split(':')
                with base.lock:
                    base._query_cache.clear()
                request = {'query': question['query'], 'method': method, 'top_k': 5}
                response = domain.search(request) if kind == 'domain' else base.search(request)
                rows[profile].append(score_question(question, response))
        return rows, {p: summarize(v) for p, v in rows.items()}
    started = time.monotonic()
    results['observed_development'], metrics['observed_development'] = run(old['questions'])
    # The deployable domain layer is chosen only from domain candidates on observed development.
    selected = max(PROFILES[3:], key=lambda p: (
        metrics['observed_development'][p]['recall_at_5'] - .25 * metrics['observed_development'][p]['false_acceptance_rate'],
        metrics['observed_development'][p]['mrr'], -PROFILES.index(p)))
    lock = {'schema_version': 1, 'domain_implementation_hash': implementation_hash(),
            'selected_profile': selected, 'index_version_id': status['index_version_id'],
            'old_gold_version_id': old['gold_version_id'], 'fresh_gold_version_id': fresh['gold_version_id'],
            'development_metrics': metrics['observed_development'],
            'selection_policy': 'domain development Recall@5 - 0.25 * unsupported false acceptance; tie MRR/simplicity',
            'old_questions_previously_observed': True}
    lock['selection_id'] = _digest(lock)
    write_new(output / 'selection_lock.json', lock)
    print('Development configuration locked: ' + selected, flush=True)
    results['additional_questions'], metrics['additional_questions'] = run(fresh['questions'])
    assert lock['domain_implementation_hash'] == implementation_hash()
    report = {'schema_version': 1, 'selection': lock, 'metrics': metrics, 'results': results,
              'generation_evaluated': False, 'expert_review_completed': False,
              'additional_questions': len(fresh['questions']), 'elapsed_seconds': time.monotonic() - started,
              'limitations': ['Previous 50 questions are development data now.',
                              'Additional 30 questions share source concepts and were authored by the assistant.',
                              'Source-span retrieval is not answer faithfulness or plant diagnosis accuracy.']}
    report['evaluation_id'] = _digest(report)
    write_new(output / 'evaluation.json', report)
    print(json.dumps({'selected': selected, 'metrics': metrics,
                      'seconds': report['elapsed_seconds']}, ensure_ascii=False, indent=2))
    base.encoder.close()


if __name__ == '__main__':
    main()
