"""Exact local vector index, with a portable corpus/chunk/embedding contract."""
import io
from pathlib import Path
import time
import numpy as np

from .common import CORPUS, INDEX, RagError, asset_bytes, read_json, write_new
from .ingest import Corpus
from ..registry import _digest, _sha


def chunks_for(manifest, units, encoder, *, tokens=256, overlap=32):
    if type(tokens) is not int or type(overlap) is not int or not 32 <= tokens <= 384 or not 0 <= overlap < tokens:
        raise RagError('invalid_chunk_configuration', 'Invalid token window or overlap', 400)
    documents = {doc['document_id']: doc for doc in manifest['documents']}
    chunks = []
    for unit in units:
        offsets = [pair for pair in encoder.offsets(unit['text']) if pair[1] > pair[0]]
        for first in range(0, len(offsets), tokens - overlap):
            window = offsets[first:first + tokens]
            if not window:
                continue
            start, end = window[0][0], window[-1][1]
            text = unit['text'][start:end]
            doc = documents[unit['document_id']]
            passage = doc['title'] + '\n' + unit['locator'].get('section', '') + '\n' + text
            if encoder.count('passage: ' + passage) > encoder.descriptor['maximum_tokens']:
                raise RagError('chunk_too_long', 'Heading and body exceed embedding input limits')
            value = {'unit_id': unit['unit_id'], 'unit_sha256': unit['unit_sha256'],
                     'document_id': unit['document_id'], 'document_version_id': unit['document_version_id'],
                     'char_start': start, 'char_end': end, 'text': text, 'passage': passage,
                     'locator': unit['locator']}
            value['chunk_id'] = 'ch_' + _digest(value)
            chunks.append(value)
            if first + tokens >= len(offsets):
                break
    if not chunks:
        raise RagError('empty_index', 'No searchable text chunks were created', 400)
    return chunks


class VectorIndex:
    def __init__(self, directory=INDEX, corpus=None):
        self.directory = Path(directory).resolve()
        self.corpus = corpus if corpus is not None else Corpus()

    def build(self, encoder, *, tokens=256, overlap=32, deadline_seconds=300):
        manifest, units = self.corpus.verify()
        chunks = chunks_for(manifest, units, encoder, tokens=tokens, overlap=overlap)
        started = time.monotonic()
        arrays = []
        for start in range(0, len(chunks), 32):
            if time.monotonic() - started > deadline_seconds:
                raise RagError('index_deadline_exceeded', 'Embedding inference exceeded the configured time budget', 503)
            arrays.append(encoder.encode([c['passage'] for c in chunks[start:start + 32]], kind='passage'))
        vectors = np.concatenate(arrays, axis=0).astype(np.float32)
        if vectors.shape != (len(chunks), encoder.descriptor['dimension']) or not np.isfinite(vectors).all():
            raise RagError('invalid_embeddings', 'Embedding vectors differ from the declared schema')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise RagError('invalid_embeddings', 'Cosine vectors must be L2 normalized')
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        binary = buffer.getvalue()
        core = {'schema_version': 1, 'backend': 'local_exact_cosine',
                'corpus_version_id': manifest['corpus_version_id'], 'chunks_sha256': _digest(chunks),
                'vectors_sha256': _sha(binary), 'chunk_count': len(chunks),
                'embedding': encoder.descriptor, 'chunking': {'body_tokens': tokens, 'overlap_tokens': overlap}}
        core['index_version_id'] = _digest(core)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / 'vectors.npy'
        if path.exists() and asset_bytes(path) != binary:
            raise RagError('rag_version_conflict', 'Existing vector index differs; use a new directory')
        if not path.exists():
            with path.open('xb') as output:
                output.write(binary)
        write_new(self.directory / 'chunks.json', chunks)
        write_new(self.directory / 'manifest.json', core)
        self.load()
        return core

    def load(self):
        try:
            return self._load()
        except RagError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
            raise RagError('rag_asset_invalid', 'Retrieval index structure is invalid', 503) from error

    def _load(self):
        corpus, units = self.corpus.verify()
        manifest = read_json(self.directory / 'manifest.json')
        chunks = read_json(self.directory / 'chunks.json')
        binary = asset_bytes(self.directory / 'vectors.npy')
        if (not isinstance(manifest, dict) or not isinstance(chunks, list)
                or manifest.get('backend') != 'local_exact_cosine' or type(manifest.get('schema_version')) is not int
                or manifest['schema_version'] != 1 or manifest.get('corpus_version_id') != corpus['corpus_version_id']
                or manifest.get('vectors_sha256') != _sha(binary) or manifest.get('chunks_sha256') != _digest(chunks)
                or manifest.get('index_version_id') != _digest({k: v for k, v in manifest.items() if k != 'index_version_id'})):
            raise RagError('rag_version_conflict', 'Index files or document corpus changed')
        by_unit = {unit['unit_id']: unit for unit in units}
        by_document = {doc['document_id']: doc for doc in corpus['documents']}
        seen = set()
        for chunk in chunks:
            unit = by_unit.get(chunk['unit_id'])
            if (chunk['chunk_id'] in seen or unit is None or chunk['unit_sha256'] != unit['unit_sha256']
                    or chunk['document_id'] != unit['document_id']
                    or chunk['document_version_id'] != unit['document_version_id']
                    or type(chunk['char_start']) is not int or type(chunk['char_end']) is not int
                    or not 0 <= chunk['char_start'] < chunk['char_end'] <= len(unit['text'])
                    or chunk['text'] != unit['text'][chunk['char_start']:chunk['char_end']]
                    or chunk['passage'] != by_document[unit['document_id']]['title'] + '\n'
                       + unit['locator'].get('section', '') + '\n' + chunk['text']
                    or chunk['locator'] != unit['locator']
                    or chunk['chunk_id'] != 'ch_' + _digest({k: v for k, v in chunk.items() if k != 'chunk_id'})):
                raise RagError('rag_version_conflict', 'Chunk text or provenance differs')
            seen.add(chunk['chunk_id'])
        vectors = np.load(io.BytesIO(binary), allow_pickle=False)
        if (not isinstance(chunks, list) or len(chunks) != manifest['chunk_count']
                or vectors.dtype != np.float32 or vectors.shape != (len(chunks), manifest['embedding']['dimension'])
                or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise RagError('rag_version_conflict', 'Vector contents differ from their contract')
        return manifest, corpus, chunks, vectors
