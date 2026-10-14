"""Shared, strict RAG asset contracts and private storage defaults."""
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import re
import sys

from ...schema import ROOT
from ..registry import _canonical, _digest, _json, _sha, _write_new

RUNTIME = ROOT / 'data/runtime/copilot/rag'
CORPUS = RUNTIME / 'corpus_v2'
INDEX = RUNTIME / 'index_v2'
MODEL_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
MODEL_PATH = RUNTIME / 'models' / MODEL_REVISION


class RagError(ValueError):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status


def dependency(name):
    """Optional workspace packages are isolated from the existing NOx environment."""
    local = RUNTIME / 'deps'
    if local.is_dir() and str(local) not in sys.path:
        sys.path.insert(0, str(local))
    os.environ.setdefault('HF_HOME', str(RUNTIME / 'hf_cache'))
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise RagError('rag_dependency_unavailable', 'Optional retrieval dependencies are unavailable', 503) from error


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,79}', value):
        raise RagError('invalid_document_id', 'Invalid document identifier', 400)
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return _json(asset_bytes(path), 'RAG asset')
    except (OSError, ValueError) as error:
        raise RagError('rag_asset_invalid', 'Retrieval asset is missing or invalid', 503) from error


def write_new(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _canonical(read_json(path)) != _canonical(value):
            raise RagError('rag_version_conflict', 'Existing retrieval asset differs; use a new version')
        return
    try:
        _write_new(path, value)
    except FileExistsError:
        if _canonical(read_json(path)) != _canonical(value):
            raise RagError('rag_version_conflict', 'Concurrent retrieval asset differs')


def asset_bytes(path):
    path = Path(path)
    try:
        if path.is_symlink():
            raise RagError('rag_asset_invalid', 'Linked retrieval assets are not supported')
        return path.read_bytes()
    except OSError as error:
        raise RagError('rag_asset_unavailable', 'Retrieval assets are unavailable', 503) from error
