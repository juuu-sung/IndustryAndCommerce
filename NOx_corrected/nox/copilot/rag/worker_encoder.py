"""Keep optional PyTorch native runtimes outside the existing NOx process."""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading

import numpy as np

from .common import RagError
from ...schema import ROOT


class WorkerEncoder:
    def __init__(self):
        self.lock = threading.Lock()
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT)
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'nox.copilot.rag.embedding_worker'], cwd=ROOT,
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        atexit.register(self.close)
        try:
            ready = self._receive()
            if not ready.get('ready'):
                raise RagError('embedding_unavailable', 'Local embedding worker could not start', 503)
            self.descriptor = ready['descriptor']
        except Exception:
            self.close()
            raise

    def _receive(self):
        if not select.select([self.process.stdout], [], [], 60)[0]:
            self.close()
            raise RagError('embedding_timeout', 'Local embedding worker timed out', 503)
        try:
            return json.loads(self.process.stdout.readline())
        except (ValueError, OSError) as error:
            raise RagError('embedding_unavailable', 'Local embedding worker is unavailable', 503) from error

    def _request(self, operation, **data):
        with self.lock:
            try:
                if self.process.poll() is not None:
                    raise RagError('embedding_unavailable', 'Local embedding worker stopped', 503)
                self.process.stdin.write(json.dumps({'operation': operation, **data}) + '\n')
                self.process.stdin.flush()
                response = self._receive()
            except (BrokenPipeError, OSError) as error:
                raise RagError('embedding_unavailable', 'Local embedding worker stopped', 503) from error
            if not response.get('ok'):
                raise RagError(response.get('code', 'embedding_worker_failed'),
                               'Local embedding operation failed', response.get('status', 503))
            return response['result']

    def encode(self, texts, *, kind):
        return np.asarray(self._request('encode', texts=list(texts), kind=kind), dtype=np.float32)

    def offsets(self, text):
        return self._request('offsets', text=text)

    def count(self, text):
        return self._request('count', text=text)

    def close(self):
        process = getattr(self, 'process', None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
