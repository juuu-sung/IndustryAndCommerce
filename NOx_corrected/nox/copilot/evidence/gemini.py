"""Bounded Gemini REST calls for report explanation and validated tool proposals."""
import copy
import json
import os
import re
import urllib.error
import urllib.request

from .claims import SCHEMA, fallback, prompt, validate_analysis
from ..registry import _json, RegistrationError


class GeminiError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class GeminiClient:
    def __init__(self, *, api_key=None, model=None, opener=None, timeout=15):
        self.api_key = api_key if api_key is not None else os.getenv('GEMINI_API_KEY')
        self.model = model if model is not None else os.getenv('GEMINI_MODEL')
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout
        self.mock_transport = opener is not None

    @property
    def configured(self):
        return bool(self.api_key and self.model and re.fullmatch(r'[a-zA-Z0-9._-]+', self.model))

    def _generate(self, body):
        if not self.configured:
            raise GeminiError('gemini_not_configured')
        request = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={'Content-Type':'application/json','x-goog-api-key':self.api_key}, method='POST')
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw = response.read(1_000_001)
        except urllib.error.HTTPError as error:
            raise GeminiError('gemini_http_error') from None
        except (TimeoutError, urllib.error.URLError, OSError):
            raise GeminiError('gemini_transport_error') from None
        try:
            if len(raw) > 1_000_000:
                raise GeminiError('gemini_response_too_large')
            envelope = _json(raw, 'Gemini envelope')
            candidate = envelope['candidates'][0]
            if candidate.get('finishReason') != 'STOP':
                raise GeminiError('gemini_incomplete_response')
            content = candidate['content']
            if content.get('role') != 'model' or not isinstance(content.get('parts'), list):
                raise GeminiError('gemini_invalid_response')
            return copy.deepcopy(content)
        except (RegistrationError, KeyError, TypeError, IndexError):
            raise GeminiError('gemini_invalid_response') from None

    def generate_analysis(self, text):
        content = self._generate({'contents':[{'role':'user','parts':[{'text':text}]}],
            'generationConfig':{'responseFormat':{'text':{'mimeType':'application/json','schema':SCHEMA}},
                                'temperature':.1,'maxOutputTokens':2500}})
        try:
            generated = ''.join(p.get('text','') for p in content['parts'] if not p.get('thought'))
            return _json(generated.encode(), 'Gemini analysis')
        except (RegistrationError, TypeError):
            raise GeminiError('gemini_invalid_analysis') from None

    def choose_tool(self, contents, declarations):
        names = [d['name'] for d in declarations]
        content = self._generate({'contents':contents, 'tools':[{'functionDeclarations':declarations}],
            'toolConfig':{'functionCallingConfig':{'mode':'ANY','allowedFunctionNames':names}},
            'generationConfig':{'temperature':0,'maxOutputTokens':1500}})
        calls = [p['functionCall'] for p in content['parts'] if 'functionCall' in p]
        if len(calls) != 1 or calls[0].get('name') not in names or not isinstance(calls[0].get('args',{}), dict):
            raise GeminiError('gemini_invalid_tool_proposal')
        # Preserve opaque thought signatures and functionCall IDs when returning the full turn.
        return {'name':calls[0]['name'], 'args':calls[0].get('args',{}),
                'call_id':calls[0].get('id'), 'model_content':content,
                'external_call_verified':not self.mock_transport}


def explain(facts, citations, base, *, enabled=False, client=None):
    basic = fallback(facts, citations)
    if not enabled:
        return {'status':'disabled','content':basic,'reason':None,
                'actual_external_call_verified':False,'semantic_faithfulness_verified':False}
    client = client or GeminiClient()
    try:
        content = validate_analysis(client.generate_analysis(prompt(facts,citations,base)), facts,citations)
        return {'status':'gemini','content':content,'reason':None,
                'actual_external_call_verified':isinstance(client,GeminiClient) and not client.mock_transport,
                'semantic_faithfulness_verified':False}
    except Exception as error:
        code = error.code if isinstance(error,GeminiError) else 'gemini_analysis_invalid_or_unavailable'
        return {'status':'fallback','content':basic,'reason':code,
                'actual_external_call_verified':False,'semantic_faithfulness_verified':False}
