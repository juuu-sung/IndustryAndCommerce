from flask import Blueprint,Response,jsonify,request
from .reports import EvidenceReportService
from .render import render_html
from ..contracts import ReportError
from ..registry import RegistrationError,_json
from ..rag.common import RagError


def create_blueprint(service=None):
    service=service if service is not None else EvidenceReportService()
    blueprint=Blueprint('evidence_reports',__name__)

    @blueprint.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    @blueprint.errorhandler(ReportError)
    @blueprint.errorhandler(RagError)
    def failure(error):
        return jsonify({'error':{'code':error.code,'message':str(error)}}),error.status

    @blueprint.post('/api/reports/evidence-drafts')
    def create():
        if not request.is_json:
            raise ReportError('invalid_content_type','application/json is required',415)
        try:
            data=_json(request.get_data(),'evidence draft')
        except RegistrationError as error:
            raise ReportError('invalid_json','Report JSON is invalid or contains duplicate keys',400) from error
        allowed={'case_key','llm_enabled','expected_case_version_id'}
        if not isinstance(data,dict) or 'case_key' not in data or set(data)-allowed:
            raise ReportError('invalid_input','Provide a registered case and supported report fields',400)
        value,created=service.create(**data)
        return jsonify({'report':value,'created':created,'links':{
            'json':'/api/reports/evidence/'+value['report_id'],
            'html':'/api/reports/evidence/'+value['report_id']+'/html'}}),201 if created else 200

    @blueprint.get('/api/reports/evidence/<identifier>')
    def get(identifier):
        return jsonify(service.get(identifier))

    @blueprint.get('/api/reports/evidence/<identifier>/html')
    def html(identifier):
        response=Response(render_html(service.get(identifier)),content_type='text/html; charset=utf-8')
        response.headers['Content-Security-Policy']="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
        response.headers['Referrer-Policy']='no-referrer'
        return response

    return blueprint
