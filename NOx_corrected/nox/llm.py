"""Gemini REST integration; prediction remains a backend-owned numeric field."""
import json
import os
import re
import urllib.request


SCHEMA = {
    'type': 'OBJECT', 'required': ['summary', 'observations', 'suggested_checks', 'limitations'],
    'properties': {
        'summary': {'type':'STRING'},
        **{key:{'type':'ARRAY','items':{'type':'STRING'}}
           for key in ['observations','suggested_checks','limitations']},
    },
}


def validate_analysis(data):
    if not isinstance(data, dict) or set(data) != set(SCHEMA['properties']):
        raise ValueError('Invalid analysis object fields')
    if not isinstance(data['summary'], str) or not data['summary'].strip() or len(data['summary'])>2000:
        raise ValueError('Invalid summary')
    for key in ['observations','suggested_checks','limitations']:
        items = data[key]
        if not isinstance(items, list) or len(items)>10 or any(not isinstance(x,str) or not x.strip() or len(x)>2000 for x in items):
            raise ValueError(f'Invalid analysis list: {key}')
    return data


def prompt(data, result):
    labels = {
        'capacity_mw':'설비용량(MW)', 'generation_mwh':'월 발전량(MWh)',
        'thermal_efficiency_pct':'열효율(%)', 'utilization_pct':'이용률(%)',
        'bituminous_ton':'유연탄(ton)', 'anthracite_ton':'무연탄(ton)',
        'oil_kl':'유류(kl)', 'lng_ton':'LNG(ton)', 'solid_ton':'고형연료(ton)',
        'pellet_ton':'우드펠릿(ton)', 'temperature_c':'외기 온도(섭씨; 연소 온도가 아님)',
        'humidity_pct':'외기 상대습도(%)', 'wind_speed_ms':'외기 풍속(m/s)',
        'wind_direction_deg':'월 평균 외기 풍향(도; 원형평균)',
    }
    facts = {'plant':data['plant'],'unit':data['unit'],'month':data['month'],
             'operating_conditions':{labels[k]:v for k,v in data.items() if k in labels},
             'prediction':result['prediction'],'known_limitations':result['warnings']}
    if 'operation_pattern' in data:
        facts['electrical_generation_patterns'] = data['operation_pattern']
        facts['operation_pattern_status'] = result['input_quality']['operation_pattern_status']
    return (
        '당신은 NOx 연구 예측 결과의 검토를 돕습니다. 아래 사실만 사용해 한국어로 작성하세요. '
        '예측 수치와 단위를 바꾸거나 재계산하지 말고 수치의 반복은 피하세요. '
        '월 배출 질량이 검증되지 않은 값이나 농도를 kg로 변환하지 마세요. '
        '외기 온도를 연소 온도로, LNG ton을 MWh로 혼동하지 마세요. '
        '발전 실적 0을 보일러 또는 SCR 정지로 단정하지 마세요. 실제 SCR 운전·환원제 사용량은 미확보입니다. '
        '상관관계를 인과관계로 단정하지 마세요. 현재 특징만으로 예측 원인을 알 수 없음을 밝히세요. '
        '배출허용기준, 건강 위험, 안전 등급, 구체적 운전 설정값을 만들어내지 마세요. '
        '관측 정보, 추가 확인 항목, 알려진 한계를 나눠 정리하세요. '
        '설비 운전 변경 지시 대신 실제 계측치와 담당자 검토를 권하세요. '
        'JSON 형식의 summary, observations, suggested_checks, limitations 필드만 반환하세요.\n'
        + json.dumps(facts, ensure_ascii=False, allow_nan=False)
    )


class Gemini:
    def __init__(self, api_key=None, model=None, timeout=12, opener=None):
        self.api_key = api_key if api_key is not None else os.environ.get('GEMINI_API_KEY')
        self.model = model if model is not None else os.environ.get('GEMINI_MODEL')
        self.timeout, self.opener = timeout, opener or urllib.request.urlopen

    def generate(self, text):
        if not self.api_key or not self.model or not re.fullmatch(r'[a-zA-Z0-9._-]+', self.model):
            raise ValueError('Gemini credentials/model are not configured')
        body = {
            'contents':[{'role':'user','parts':[{'text':text}]}],
            'generationConfig':{'responseMimeType':'application/json', 'responseSchema':SCHEMA,
                                'temperature':.2, 'maxOutputTokens':1500},
        }
        req = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            data=json.dumps(body,ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type':'application/json','x-goog-api-key':self.api_key}, method='POST')
        with self.opener(req, timeout=self.timeout) as response:
            raw = response.read(1_000_001)
        if len(raw)>1_000_000:
            raise ValueError('Oversized Gemini response')
        envelope = json.loads(raw)
        candidate = envelope['candidates'][0]
        if candidate.get('finishReason') != 'STOP':
            raise ValueError('Gemini response did not finish normally')
        generated = ''.join(p.get('text','') for p in candidate['content']['parts'] if not p.get('thought'))
        return validate_analysis(json.loads(generated))


def explain(data, result, *, enabled=False, client=None):
    fallback = {
        'summary':'예측값은 입력 조건에 대한 연구용 추정입니다.',
        'observations':['예측 원인을 이 입력만으로 확정할 수 없습니다.'],
        'suggested_checks':['실제 NOx 계측값과 월별 입력 조건을 대조하세요.'],
        'limitations':list(result['warnings']),
    }
    if not enabled:
        return {'status':'disabled','content':fallback}
    try:
        content = validate_analysis((client or Gemini()).generate(prompt(data, result)))
        # Backend limitations survive even if the LLM omits them.
        content = dict(content)
        content['limitations'] = list(dict.fromkeys(content['limitations'] + result['warnings']))
        return {'status':'gemini','content':content}
    except Exception:
        # Avoid returning upstream error text, requests, or credentials.
        return {'status':'fallback','reason':'gemini_unavailable_or_invalid_response','content':fallback}
