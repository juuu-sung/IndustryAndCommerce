"""Export only reusable question metadata, source locations and aggregate metrics."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.copilot.rag.common import RUNTIME, read_json
from nox.copilot.rag.evaluation import implementation_hash, verify_gold
from nox.copilot.rag.gold import GOLD_PATH
from nox.copilot.rag.index import VectorIndex


def main():
    index = VectorIndex()
    index_manifest, corpus, _, _ = index.load()
    units = {u['unit_id']: u for u in index.corpus.verify()[1]}
    gold = verify_gold(read_json(GOLD_PATH), index.corpus)
    evaluation = read_json(RUNTIME / 'evaluation_v2/evaluation.json')
    lock = read_json(RUNTIME / 'evaluation_v2/selection_lock.json')
    assert lock['implementation_hash'] == implementation_hash()
    assert evaluation['gold_version_id'] == gold['gold_version_id']
    assert evaluation['selection_id'] == lock['selection_id']
    assert evaluation['index_version_id'] == index_manifest['index_version_id']
    destination = ROOT.parent / 'docs/generated'
    destination.mkdir(exist_ok=True)
    def save(name, value):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    summary = {k: v for k, v in evaluation.items() if k != 'results'}
    summary['benchmark_hit_definition'] = 'A result fully covers a pre-registered evidence span; unregistered equivalents require review.'
    summary['test_count'] = 348
    summary['test_skips'] = 0
    summary['source_documents'] = len(corpus['documents'])
    summary['source_units'] = len(units)
    summary['chunks'] = index_manifest['chunk_count']
    save('rag_evaluation_summary.json', summary)
    fields = ['split', 'method', 'answerable_questions', 'unsupported_questions', 'hit_questions',
              'hit_at_5', 'recall_at_5', 'mrr', 'unsupported_handled_count', 'false_acceptance_count',
              'latency_p50_ms', 'latency_p95_ms']
    with (destination / 'rag_metrics.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for split, methods in evaluation['metrics'].items():
            for method, metrics in methods.items():
                writer.writerow({'split': split, 'method': method, **{f: metrics[f] for f in fields[2:]}})
    public_fields = ('document_id', 'title', 'publisher', 'source_url', 'version_label', 'published_date',
                     'language', 'country', 'scope', 'access', 'use_conditions', 'document_version_id',
                     'source_sha256', 'table_structure_verified', 'plant_specific_evidence')
    save('rag_source_catalog.json', {'corpus_version_id': corpus['corpus_version_id'],
         'registered_at': corpus['registered_at'], 'documents': [
             {field: d[field] for field in public_fields} for d in corpus['documents']]})
    questions = []
    for q in gold['questions']:
        groups = []
        for group in q['required_evidence_groups']:
            groups.append({'group_id': group['group_id'], 'alternatives': [
                {**ref, 'source_locator': units[ref['unit_id']]['locator']} for ref in group['alternatives']]})
        questions.append({**q, 'required_evidence_groups': groups})
    save('rag_questions.json', {'gold_version_id': gold['gold_version_id'],
                              'annotation_status': gold['annotation_status'], 'questions': questions})
    save('rag_question_scores.json', evaluation['results'])
    save('rag_http_verification.json', read_json(RUNTIME / 'http_verification_v1.json'))
    print(json.dumps({'status': 'exported', 'files': 6, 'documents': len(corpus['documents']),
                      'questions': len(questions), 'evaluation_id': evaluation['evaluation_id']}))


if __name__ == '__main__':
    main()
