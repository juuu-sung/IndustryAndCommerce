import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .llm import explain
from .history import HistoryError
from .predict import Predictor
from .schema import ROOT


def create_app(predictor=None, *, enable_gemini=False, llm_client=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=32_768, JSON_SORT_KEYS=False)
    app.json.ensure_ascii = False
    predictor = predictor or Predictor()

    @app.get('/api/health')
    def health():
        return jsonify({'status':'ok','target':predictor.metadata['target'],
                        'gemini_enabled':enable_gemini, 'history':predictor.history_status()})

    @app.get('/api/history/status')
    def history_status():
        return jsonify(predictor.history_status())

    @app.post('/api/predict')
    def predict():
        if not request.is_json:
            return jsonify({'error':{'code':'invalid_content_type','message':'application/json is required'}}), 415
        data = request.get_json()
        try:
            result = predictor.predict(data)
        except HistoryError:
            raise
        except ValueError as error:
            return jsonify({'error':{'code':'invalid_input','message':str(error)}}), 400
        result['analysis'] = explain(data, result, enabled=enable_gemini, client=llm_client)
        return jsonify(result)

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify({'error':{'code':error.name.lower().replace(' ','_'),
                                'message':error.description}}), error.code

    @app.errorhandler(HistoryError)
    @app.errorhandler(sqlite3.Error)
    def history_error(error):
        return jsonify({'error':{'code':'history_unavailable',
                                'message':'Observation history could not be verified'}}), 503

    @app.errorhandler(Exception)
    def server_error(error):
        return jsonify({'error':{'code':'internal_error','message':'Prediction could not be completed'}}), 500

    return app


if __name__ == '__main__':
    mode = os.environ.get('NOX_TARGET_MODE', 'concentration')
    if mode not in ('legacy', 'concentration'):
        raise ValueError('NOX_TARGET_MODE must be legacy or concentration')
    artifact = Path(os.environ.get('NOX_ARTIFACT_DIR', str(ROOT / 'artifacts' / mode)))
    app = create_app(Predictor(artifact), enable_gemini=os.environ.get('ENABLE_GEMINI')=='1')
    app.run(host='127.0.0.1',port=5001,debug=False)
