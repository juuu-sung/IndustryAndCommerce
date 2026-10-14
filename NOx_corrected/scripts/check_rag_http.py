"""Real loopback HTTP smoke check, with existing local models and frozen reports."""
import json
from pathlib import Path
import sys
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from werkzeug.serving import make_server
from nox.app import create_app
from nox.predict import Predictor
from nox.copilot.rag.retrieval import RetrievalService
from nox.copilot.rag.common import RUNTIME


def main():
    rag = RetrievalService()
    app = create_app(Predictor(history_db=False), rag_service=rag)
    server = make_server('127.0.0.1', 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    checks = []
    def call(path, data=None, raw=False):
        payload = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(base + path, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            checks.append({'path': path, 'status': response.status,
                           'content_type': response.headers['Content-Type']})
            return body if raw else json.loads(body)
    try:
        assert call('/api/health')['status'] == 'ok'
        assert call('/api/rag/status')['document_count'] == 10
        assert len(call('/api/rag/documents')['documents']) == 10
        project = call('/api/rag/search', {'query': 'NOx 연구 타깃은 무엇을 월별 평균한 값인가요?', 'method': 'keyword'})
        assert project['results'] and any(r['document_id'] == 'project_data' for r in project['results'])
        technical = call('/api/rag/search', {'query': 'What operational factors affect SCR catalyst performance?',
                                           'method': 'hybrid', 'scope': 'general_technical'})
        assert technical['results'] and all(r['document_id'].startswith('epa_') for r in technical['results'])
        units = {u['unit_id']: u for u in rag.index.corpus.verify()[1]}
        for hit in project['results'] + technical['results']:
            unit = units[hit['unit_id']]
            assert hit['text'] == unit['text'][hit['locator']['char_start']:hit['locator']['char_end']]
        numeric = call('/api/rag/search', {'query': '영동 2025년 10월 NOx 예측값', 'method': 'hybrid'})
        assert numeric['status'] == 'requires_numeric_tool' and not numeric['results']
        prediction = call('/api/predict', json.loads((ROOT / 'examples/request.json').read_text()))
        assert prediction['prediction']['unit'] == 'ppm'
        case = call('/api/copilot/cases')['cases'][0]['case_key']
        report = call('/api/reports/drafts', {'case_key': case})['report']
        assert report['target_unit'] == 'ppm' and report['rag_status'] == 'not_enabled'
        assert call('/api/reports/' + report['report_id'])['report_id'] == report['report_id']
        assert b'ppm' in call('/api/reports/' + report['report_id'] + '/html', raw=True)
        def brief(value):
            return {'status': value['status'], 'top_document': value['results'][0]['document_id'],
                    'locator': value['results'][0]['locator']}
        # No observation rows, predictions or reports are copied into the public check summary.
        result = {'checks': checks, 'all_passed': True, 'retrieval_examples': {
            'project': brief(project), 'technical': brief(technical),
            'numeric': {'status': numeric['status'], 'results': len(numeric['results'])}},
            'predict_unit': prediction['prediction']['unit'], 'report_rag_status': report['rag_status']}
        path = RUNTIME / 'http_verification_v1.json'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=3)
        if rag.encoder is not None:
            rag.encoder.close()


if __name__ == '__main__':
    main()
