"""Background Agent endpoints; reviewer identity never comes from JSON input."""
import threading
from flask import Blueprint,Response,jsonify,request

from .common import AgentError
from .service import AgentService
from ..contracts import ReportError
from ..evidence.render import render_html
from ..rag.common import RagError
from ..registry import RegistrationError,_json


def create_blueprint(service=None,*,evidence=None):
    current=service
    lock=threading.Lock()
    blueprint=Blueprint('agent_workflow',__name__)

    def get_service():
        nonlocal current
        with lock:
            if current is None:current=AgentService(evidence)
        return current

    def body():
        if not request.is_json:raise AgentError('invalid_content_type','application/json is required',415)
        try:return _json(request.get_data(),'workflow request')
        except RegistrationError as error:raise AgentError('invalid_json','Workflow JSON is invalid or contains duplicate keys',400) from error

    @blueprint.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    @blueprint.errorhandler(AgentError)
    @blueprint.errorhandler(ReportError)
    @blueprint.errorhandler(RagError)
    def failure(error):
        return jsonify({'error':{'code':error.code,'message':str(error)}}),error.status

    @blueprint.get('/api/agent/status')
    def status():
        try:return jsonify(get_service().status())
        except AgentError as error:
            return jsonify({'ready':False,'code':error.code,'actual_gemini_call_verified':False})

    @blueprint.post('/api/agent/runs')
    def create():
        run,created=get_service().create(body())
        return jsonify({'run':run,'created':created}),202 if created else 200

    @blueprint.get('/api/agent/runs/<identifier>')
    def get(identifier):
        return jsonify(get_service().get(identifier))

    @blueprint.post('/api/agent/runs/<identifier>/resume')
    def resume(identifier):
        run,created=get_service().resume(identifier,body(),request.headers.get('X-Review-Token'))
        return jsonify({'run':run,'new_decision_or_input':created}),202 if created else 200

    @blueprint.post('/api/agent/runs/<identifier>/revisions')
    def revise(identifier):
        return jsonify({'run':get_service().revise(identifier,body(),request.headers.get('X-Review-Token'))})

    @blueprint.post('/api/agent/runs/<identifier>/recover')
    def recover(identifier):
        return jsonify({'run':get_service().recover(identifier,body(),request.headers.get('X-Review-Token'))}),202

    @blueprint.get('/api/agent/runs/<identifier>/report')
    def report(identifier):
        value,run=get_service().report(identifier)
        return jsonify({'report':value,'workflow':run})

    @blueprint.get('/api/agent/runs/<identifier>/report/html')
    def html(identifier):
        value,run=get_service().report(identifier)
        response=Response(render_html(value,run),content_type='text/html; charset=utf-8')
        response.headers['Content-Security-Policy']="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
        response.headers['Referrer-Policy']='no-referrer'
        return response

    return blueprint
