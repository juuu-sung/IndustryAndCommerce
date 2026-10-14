"""Optional improved search endpoint, leaving the original benchmark API intact."""
from flask import Blueprint, jsonify, request
from .common import RagError
from .domain import DomainRequest, DomainRetrieval
from ..registry import RegistrationError, _json


def create_blueprint(service=None):
    service = service if service is not None else DomainRetrieval()
    blueprint = Blueprint('domain_retrieval', __name__)

    @blueprint.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @blueprint.errorhandler(RagError)
    def failure(error):
        return jsonify({'error': {'code': error.code, 'message': str(error)}}), error.status

    @blueprint.post('/api/rag/search/domain')
    def search():
        if not request.is_json:
            raise RagError('invalid_content_type', 'application/json is required', 415)
        try:
            payload = _json(request.get_data(), 'domain search')
        except RegistrationError as error:
            raise RagError('invalid_json', 'Search JSON is invalid or contains duplicate keys', 400) from error
        return jsonify(service.search(DomainRequest.parse(payload)))

    return blueprint
