"""Short synchronous report endpoints, isolated from prediction and LLM calls."""
from flask import Blueprint, jsonify, request, Response

from .contracts import ReportError, ReportRequest
from .renderers import render_html
from .repositories import ReportService
from .registry import RegistrationError, _json


def create_blueprint(service=None):
    service = service if service is not None else ReportService()
    blueprint = Blueprint('copilot_reports', __name__)

    @blueprint.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @blueprint.errorhandler(ReportError)
    def report_error(error):
        return jsonify({'error': {'code': error.code, 'message': str(error)}}), error.status

    @blueprint.get('/api/copilot/cases')
    def cases():
        return jsonify({'cases': service.cases(), 'supported_analysis_modes': ['evaluation_replay']})

    @blueprint.post('/api/reports/drafts')
    def create_draft():
        if not request.is_json:
            raise ReportError('invalid_content_type', 'application/json is required', 415)
        try:
            payload = _json(request.get_data(), 'report request')
        except RegistrationError as error:
            raise ReportError('invalid_json', 'Report request must be valid JSON without duplicate keys', 400) from error
        report, created = service.create(ReportRequest.parse(payload))
        identifier = report['report_id']
        result = jsonify({'report': report, 'created': created,
                          'links': {'json': '/api/reports/' + identifier,
                                    'html': '/api/reports/' + identifier + '/html'}})
        result.status_code = 201 if created else 200
        result.headers['Location'] = '/api/reports/' + identifier
        return result

    @blueprint.get('/api/reports/<identifier>')
    def get_report(identifier):
        return jsonify(service.get(identifier))

    @blueprint.get('/api/reports/<identifier>/html')
    def get_html(identifier):
        response = Response(render_html(service.get(identifier)), content_type='text/html; charset=utf-8')
        response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    return blueprint
