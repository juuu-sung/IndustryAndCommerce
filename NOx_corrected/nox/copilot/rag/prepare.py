"""Freeze the real source corpus and optionally build an E5 vector index."""
import argparse
import json
from pathlib import Path

from .catalog import default_sources
from .common import CORPUS, INDEX, RagError
from .embeddings import E5Encoder
from .index import VectorIndex
from .ingest import Corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=CORPUS)
    parser.add_argument('--index-dir', type=Path, default=INDEX)
    parser.add_argument('--build-index', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    try:
        corpus = Corpus(args.corpus_dir)
        manifest = corpus.register(default_sources())
        _, units = corpus.verify()
        print(json.dumps({'status': 'corpus_registered', 'documents': len(manifest['documents']),
                          'text_units': len(units), 'corpus_version_id': manifest['corpus_version_id']}, ensure_ascii=False), flush=True)
        if args.build_index:
            result = VectorIndex(args.index_dir, corpus).build(E5Encoder(), deadline_seconds=args.deadline_seconds)
            print(json.dumps({'status': 'index_built', **result}, ensure_ascii=False), flush=True)
    except (RagError, OSError, ValueError) as error:
        message = str(error) if isinstance(error, RagError) else 'Source preparation failed'
        print(json.dumps({'status': 'blocked', 'message': message}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
