"""Optional retrieval Blueprint, independent of prediction and report generation."""
from flask import Blueprint, jsonify, request

from .common import RagError
from .retrieval import RetrievalService, SearchRequest
from ..registry import RegistrationError, _json


def create_blueprint(service=None):
    service = service if service is not None else RetrievalService()
    blueprint = Blueprint('rag_retrieval', __name__)

    @blueprint.after_request
    def private(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @blueprint.errorhandler(RagError)
    def error(value):
        return jsonify({'error': {'code': value.code, 'message': str(value)}}), value.status

    @blueprint.get('/api/rag/status')
    def status():
        return jsonify(service.status())

    @blueprint.get('/api/rag/documents')
    def documents():
        return jsonify({'documents': service.documents()})

    @blueprint.post('/api/rag/search')
    def search():
        if not request.is_json:
            raise RagError('invalid_content_type', 'application/json is required', 415)
        try:
            payload = _json(request.get_data(), 'search request')
        except RegistrationError as value:
            raise RagError('invalid_json', 'Search request must be valid JSON without duplicate keys', 400) from value
        return jsonify(service.search(SearchRequest.parse(payload)))

    return blueprint
