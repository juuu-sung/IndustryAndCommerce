"""Private local embedding worker: no network, no ML predictor/native libraries."""
import json
import sys

from .common import RagError
from .embeddings import E5Encoder


def main():
    try:
        encoder = E5Encoder()
        print(json.dumps({'ready': True, 'descriptor': encoder.descriptor}), flush=True)
    except Exception:
        print(json.dumps({'ready': False}), flush=True)
        return 1
    for line in sys.stdin:
        try:
            data = json.loads(line)
            operation = data['operation']
            if operation == 'encode':
                if not isinstance(data['texts'], list) or len(data['texts']) > 64:
                    raise RagError('invalid_embedding_input', 'Invalid embedding batch', 400)
                result = encoder.encode(data['texts'], kind=data['kind']).tolist()
            elif operation == 'offsets':
                result = encoder.offsets(data['text'])
            elif operation == 'count':
                result = encoder.count(data['text'])
            else:
                raise RagError('invalid_embedding_operation', 'Invalid embedding operation', 400)
            print(json.dumps({'ok': True, 'result': result}), flush=True)
        except RagError as error:
            print(json.dumps({'ok': False, 'code': error.code, 'status': error.status}), flush=True)
        except Exception:
            print(json.dumps({'ok': False, 'code': 'embedding_worker_failed', 'status': 503}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
